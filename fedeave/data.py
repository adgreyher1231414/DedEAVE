import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Subset

class StudentDataset(Dataset):
    """Input arrays loaded from a normalized .npz file; no missing evidence is imputed."""
    def __init__(self,path):
        z=np.load(path); self.data={k:torch.from_numpy(z[k]) for k in z.files}
        required=['assessment','behaviour','temporal','peer','adj','available','concept','item','label','history']
        if any(k not in self.data for k in required): raise ValueError('Missing keys: '+str(set(required)-set(self.data)))
    def __len__(self): return len(self.data['label'])
    def __getitem__(self,i):
        return {k:v[i] for k,v in self.data.items() if k!='adj'} | {'index':torch.tensor(i)}

def collate_with_graph(dataset):
    def collate(examples):
        ids=torch.stack([e['index'] for e in examples]); out={k:torch.stack([e[k] for e in examples]) for k in examples[0] if k!='index'}
        out['adj']=dataset.data['adj'][ids[:,None],ids[None,:]]
        return out
    return collate

def partition_dirichlet(labels,n_clients=20,alpha=.5,seed=42):
    rng=np.random.default_rng(seed); labels=np.asarray(labels)
    partitions=[[] for _ in range(n_clients)]
    for cls in np.unique(labels):
        inds=rng.permutation(np.flatnonzero(labels==cls)); proportions=rng.dirichlet([alpha]*n_clients)
        chunks=np.split(inds,np.cumsum((proportions*len(inds)).astype(int))[:-1])
        for k,chunk in enumerate(chunks): partitions[k].extend(chunk.tolist())
    return [np.asarray(p,dtype=np.int64) for p in partitions]

def make_demo(path,n=96,seq=8,n_items=101,n_concepts=8,seed=42):
    rng=np.random.default_rng(seed); concept=rng.integers(0,n_concepts,n,dtype=np.int64)
    item=rng.integers(1,n_items,n,dtype=np.int64)
    a=np.zeros((n,seq,5),np.int64)
    a[:,:,0]=rng.integers(1,n_items,(n,seq)); a[:,:,1]=rng.integers(0,n_concepts,(n,seq)); a[:,:,2]=rng.integers(0,2,(n,seq)); a[:,:,3]=rng.integers(0,16,(n,seq)); a[:,:,4]=rng.integers(0,16,(n,seq))
    behaviour=rng.normal(size=(n,seq,6)).astype('float32'); temporal=rng.normal(size=(n,seq,6)).astype('float32')
    peer=rng.normal(size=(n,6)).astype('float32'); available=np.ones((n,4),dtype=bool); available[:,2]=False
    adj=np.eye(n,dtype='float32'); history=np.zeros((n,4),dtype='float32')
    label=(a[:,:,2].mean(1)+behaviour[:,:,0].mean(1)*.3+rng.normal(0,.2,n)>.5).astype('float32')
    np.savez_compressed(path,assessment=a,behaviour=behaviour,temporal=temporal,peer=peer,available=available,adj=adj,concept=concept,item=item,label=label,history=history)
