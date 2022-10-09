import csv
def deal_generated_csv(path):
    res = []
    with open(path,'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            row = row[0].replace('<pad>','')
            res.append(row)
    with open('{}'.format(path),'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for r in res:
            f_csv.writerow([r])

path = '/home/chenyy/Code/moses/checkpoints/aae_generated.csv'
deal_generated_csv(path)