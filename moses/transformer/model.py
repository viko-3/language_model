import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.nn import MultiheadAttention
import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP


class Transformer1(nn.Module):

    def __init__(self, vocabulary, config):
        super(Transformer1, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            from torch.nn import TransformerDecoder, TransformerDecoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pad_mask = None
        self.vocabulary = vocabulary
        self.hidden_size = config.hidden
        self.nhead = config.head
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.ninp = config.ninp
        self.vocab_size = len(vocabulary)

        self.pos_encoder = PositionalEncoding(self.ninp, self.dropout)
        encoder_layers = TransformerEncoderLayer(self.ninp, self.nhead, self.hidden_size, self.dropout,
                                                 batch_first=True)
        """encoder_layers = TransformerEncoderLayer(self.ninp, self.nhead, self.hidden_size, self.dropout,
                                                 batch_first=True)"""
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.num_layers)
        self.encoder = nn.Embedding(self.vocab_size, self.ninp, padding_idx=vocabulary.pad)
        self.decoder = nn.Linear(self.ninp, self.vocab_size)
        self.init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.masked_fill(mask == 0, True).masked_fill(mask == 1, False)
        return mask

    def _generate_square_padding_mask(self, sz, lengths):
        # e.g When lengths is torch.tensor([1,3,2,4])
        lengths = lengths.cpu()
        row_vector = torch.arange(0, sz, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector >= matrix
        return mask

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, lengths, has_mask=True):
        if has_mask:
            device = x.device
            if self.src_mask is None or self.src_mask.size(0) != len(x[0]):
                mask = self._generate_square_subsequent_mask(len(x[0])).to(device)
                self.src_mask = mask
                padding_mask = self._generate_square_padding_mask(len(x[0]), lengths).to(device)
                self.pad_mask = padding_mask
        else:
            self.src_mask = None
            self.pad_mask = None

        src = self.encoder(x)  # * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask, self.pad_mask)
        output = self.decoder(output)

        return F.log_softmax(output, dim=-1), lengths

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long,
                              device=self.device
                              if device == 'model' else device)

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        """try:
            end_first_index = ids.index(self.vocabulary.eos)
            ids = ids[:end_first_index + 1]
        except:
            pass"""
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def sample1(self, n_batch, max_length=200):
        with torch.no_grad():
            starts = [torch.tensor([self.vocabulary.bos],
                                   dtype=torch.long,
                                   device=self.device)
                      for _ in range(n_batch)]

            starts = torch.tensor(starts, dtype=torch.long,
                                  device=self.device).unsqueeze(1)

            new_smiles_list = [
                torch.tensor(self.vocabulary.pad, dtype=torch.long,
                             device=self.device).repeat(max_length + 2)
                for _ in range(n_batch)]

            for i in range(n_batch):
                new_smiles_list[i][0] = self.vocabulary.bos

            len_smiles_list = [1 for _ in range(n_batch)]
            lens = torch.tensor([1 for _ in range(n_batch)],
                                dtype=torch.long, device=self.device)
            end_smiles_list = [False for _ in range(n_batch)]

            hiddens = None
            for i in range(1, max_length + 1):
                output, _ = self.forward(starts, lens)

                # probabilities
                probs = [F.softmax(o, dim=-1) for o in output]

                # sample from probabilities
                ind_tops = [torch.multinomial(p, 1) for p in probs]

                for j, top in enumerate(ind_tops):
                    if not end_smiles_list[j]:
                        top_elem = top[0].item()
                        if top_elem == self.vocabulary.eos:
                            end_smiles_list[j] = True

                        new_smiles_list[j][i] = top_elem
                        len_smiles_list[j] = len_smiles_list[j] + 1

                starts = torch.tensor(ind_tops, dtype=torch.long,
                                      device=self.device).unsqueeze(1)

            new_smiles_list = [new_smiles_list[i][:l]
                               for i, l in enumerate(len_smiles_list)]
            return [self.tensor2string(t) for t in new_smiles_list]

    def sample(self, n_batch, max_length=100):
        print(max_length)
        starts = [torch.tensor([self.vocabulary.bos],
                               dtype=torch.long,
                               device=self.device)
                  for _ in range(n_batch)]

        starts = torch.tensor(starts, dtype=torch.long,
                              device=self.device).unsqueeze(1)
        input = starts
        with torch.no_grad():
            for i in range(max_length):
                output, _ = self.forward(input, lengths=0, has_mask=False)
                word_weights = output[:, -1].cpu()
                # word_weights = word_weights.squeeze().cpu()

                word_weights = torch.exp(word_weights)
                word_idx = torch.multinomial(word_weights, 1)[:, 0]
                # word_tensor = torch.Tensor([[word_idx]]).long().cuda()
                input = torch.cat([input, word_idx.unsqueeze(1).cuda()], 1)
            out = [self.tensor2string(t) for t in input]
        return out


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)  # batch_first
        self.register_buffer('pe', pe)

    def forward(self, x, length):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:, :x[0].size(0)]  # batch_first
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self, vocabulary, config):
        super(Transformer, self).__init__()
        self.max_len = 2300
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pad_mask = None
        self.vocabulary = vocabulary
        self.hidden_size = config.hidden
        self.nhead = config.head
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.ninp = config.ninp
        self.vocab_size = len(vocabulary)
        self.encoder = nn.Embedding(self.vocab_size, self.ninp, padding_idx=vocabulary.pad)
        # self.pos_emb = PositionalEncoding(self.ninp, self.dropout)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.ninp))
        self.drop = nn.Dropout(config.dropout)
        # transformer
        self.encoder_blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.ninp)
        self.decoder = nn.Linear(self.ninp, self.vocab_size)
        self.init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.masked_fill(mask == 0, True).masked_fill(mask == 1, False)
        return mask

    def _generate_square_padding_mask(self, sz, lengths):
        # e.g When lengths is torch.tensor([1,3,2,4])
        lengths = lengths.cpu()
        row_vector = torch.arange(0, sz, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector >= matrix
        return mask

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, lengths, has_mask=True):
        b, t = x.size()
        self.src_mask = None
        self.pad_mask = None

        x = self.encoder(x)  # 字符编码+位置编码
        x = self.drop(x + self.pos_emb[:, :t, :])

        for layer in self.encoder_blocks:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.decoder(x)
        return logits

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long,
                              device=self.device
                              if device == 'model' else device)

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        try:
            end_first_index = ids.index(self.vocabulary.eos)
            ids = ids[:end_first_index + 1]
        except:
            pass
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def sample(self, n_batch, max_length):
        max_length = self.max_len
        starts = [torch.tensor([self.vocabulary.bos],
                               dtype=torch.long,
                               device=self.device)
                  for _ in range(n_batch)]

        starts = torch.tensor(starts, dtype=torch.long,
                              device=self.device).unsqueeze(1)
        input = starts
        with torch.no_grad():
            for i in range(max_length):
                output = self.forward(input, lengths=0, has_mask=False)
                output = F.softmax(output, dim=-1)
                word_weights = output[:, -1].cpu()
                word_idx = torch.multinomial(word_weights, 1)
                input = torch.cat([input, word_idx.cuda()], 1)
            out = [self.tensor2string(t) for t in input]
        return out


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.ninp)
        self.ln2 = nn.LayerNorm(config.ninp)
        # self.self_attn = MultiheadAttention(config.ninp, config.head, dropout=config.dropout, batch_first=True)
        self.self_attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.ninp, 4 * config.ninp),
            nn.GELU(),
            nn.Linear(4 * config.ninp, config.ninp),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = self.ln1(x)
        y, attn = self.self_attn(x)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.ninp % config.head == 0
        self.max_len = 2300
        # key, query, value projections for all heads
        self.key = nn.Linear(config.ninp, config.ninp)
        self.query = nn.Linear(config.ninp, config.ninp)
        self.value = nn.Linear(config.ninp, config.ninp)
        # regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        # output projection
        self.proj = nn.Linear(config.ninp, config.ninp)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # mask.shape torch.Size([1, 1, len, len])
        self.register_buffer("mask", torch.tril(torch.ones(self.max_len, self.max_len))
                             .view(1, 1, self.max_len, self.max_len))

        self.n_head = config.head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save
