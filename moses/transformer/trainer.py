import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from build.lib.moses.utils import WordVocab
from moses.distribute_utils import is_main_process, reduce_value
from moses.interfaces import MosesTrainer
from moses.utils import CharVocab, Logger
import torch.distributed as dist


class TransformerTrainer(MosesTrainer):

    def __init__(self, config):
        self.config = config

    def get_loop_data(self, source, i):
        data = source[:, :i]
        return data

    def _train_epoch(self, model, data, criterion, optimizer=None):

        if optimizer is None:
            model.eval()
        else:
            model.train()

        postfix = {'loss': 0,
                   'running_loss': 0}

        for i, (prevs, nexts, lens) in enumerate(data):
            prevs = prevs.to(model.device)
            nexts = nexts.to(model.device)
            lens = lens.to(model.device)

            outputs = model(prevs, lens)
            loss = criterion(outputs.view(-1, outputs.shape[-1]),
                             nexts.view(-1))

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                loss = reduce_value(loss, average=True)
                optimizer.step()

            if is_main_process():
                postfix['loss'] = loss.item()
                postfix['running_loss'] += (loss.item() -
                                            postfix['running_loss']) / (i + 1)
                data.set_postfix(postfix)

        postfix['mode'] = 'Eval' if optimizer is None else 'Train'
        # 等待所有进程计算完毕
        if model.device != torch.device("cpu"):
            torch.cuda.synchronize(model.device)
        return postfix

    def _train(self, model, train_loader, val_loader=None, logger=None):

        def get_params():

            return (p for p in model.parameters() if p.requires_grad)

        device = model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              self.config.step_size,
                                              self.config.gamma)

        model.zero_grad()
        for epoch in range(self.config.train_epochs):
            scheduler.step()

            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, tqdm_data, criterion, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, tqdm_data, criterion)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(
                    model.state_dict(),
                    self.config.model_save[:-3] + '_{0:03d}.pt'.format(epoch)
                )
                model = model.to(device)

    def _distribute_train(self, model, train_loader, val_loader=None, logger=None):

        def get_params():

            return (p for p in model.parameters() if p.requires_grad)

        device = model.device
        print(self.config.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.config.gpu])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              self.config.step_size,
                                              self.config.gamma)

        model.zero_grad()
        for epoch in range(self.config.train_epochs):
            scheduler.step()
            # 在进程0中打印训练进度
            if is_main_process():
                train_loader = tqdm(train_loader,
                                    desc='Training (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, train_loader, criterion, optimizer)
            if logger is not None and is_main_process():
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                if is_main_process():
                    val_loader = tqdm(val_loader,
                                      desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, val_loader, criterion)
                if logger is not None and is_main_process():
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0) \
                    and is_main_process():
                # model = model.to('cpu')
                torch.save(
                    model.module.state_dict(),
                    self.config.model_save[:-3] + '_{0:03d}.pt'.format(epoch)
                )
                model = model.to(device)

    def get_vocabulary(self, data):
        return WordVocab.from_data(data)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)
        device = 'cpu'

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device)
                       for string in data]

            pad = model.vocabulary.pad
            prevs = pad_sequence([t[:-1] for t in tensors],
                                 batch_first=True, padding_value=pad)
            nexts = pad_sequence([t[1:] for t in tensors],
                                 batch_first=True, padding_value=pad)
            lens = torch.tensor([len(t) - 1 for t in tensors],
                                dtype=torch.long, device=device)
            return prevs, nexts, lens

        return collate

    def fit(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None

        """train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(
            model, val_data, shuffle=False
        )"""

        # self._train(model, train_loader, val_loader, logger)
        ###
        train_loader = self.get_distribute_dataloader(model, train_data, config=self.config, shuffle=True)
        val_loader = None if val_data is None else self.get_distribute_dataloader(
            model, val_data, config=self.config, shuffle=False)
        self._distribute_train(model, train_loader, val_loader, logger)
        ###
        return model
