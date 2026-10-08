import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from fedeave import FedEAVE
from fedeave.data import StudentDataset,partition_dirichlet,make_demo
from fedeave.federated import federated_train
from fedeave.metrics import evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='demo.npz'); p.add_argument('--demo',action='store_true')
    p.add_argument('--rounds',type=int,default=2); p.add_argument('--clients',type=int,default=4); p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=8); p.add_argument('--private',action='store_true'); p.add_argument('--noise',type=float,default=1.1)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--out',default='outputs'); args=p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    if args.demo: make_demo(args.data)
    dataset=StudentDataset(args.data)
    n_items=max(101,int(dataset.data['assessment'][:,:,0].max())+1,int(dataset.data['item'].max())+1)
    n_concepts=max(8,int(dataset.data['concept'].max())+1,int(dataset.data['assessment'][:,:,1].max())+1)
    model=FedEAVE(n_items,n_concepts)
    parts=partition_dirichlet(dataset.data['label'].numpy(),args.clients,.5,args.seed)
    hist=federated_train(model,dataset,parts,rounds=args.rounds,client_fraction=max(.1,1/args.clients),epochs=args.epochs,batch_size=args.batch_size,private=args.private,noise=args.noise,seed=args.seed)
    result=evaluate(model,dataset); print('TRAIN-SET sanity metrics (NOT held-out validation):',result)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/'model.pt')
    (out/'history.json').write_text(json.dumps({'history':hist,'train_metrics':result},indent=2))
    if args.private: print('WARNING: This is DP-SGD-style noise injection, NOT a certified (epsilon,delta)-DP claim; accountant and sampling protocol required.')
if __name__=='__main__':main()
