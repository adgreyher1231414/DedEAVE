import numpy as np
import torch
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,mean_absolute_error
from torch.utils.data import DataLoader
from .data import collate_with_graph

@torch.no_grad()
def evaluate(model,dataset,batch_size=32,device='cpu'):
    model.eval(); y=[]; p=[]
    for b in DataLoader(dataset,batch_size=batch_size,collate_fn=collate_with_graph(dataset)):
        b={k:v.to(device) for k,v in b.items()}; out=model(b)
        y.extend(b['label'].cpu().numpy()); p.extend(torch.sigmoid(out['logit']).cpu().numpy())
    y=np.asarray(y); p=np.asarray(p); pred=p>=.5
    return {'accuracy':float(accuracy_score(y,pred)),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'f1':float(f1_score(y,pred,zero_division=0)),'mae':float(mean_absolute_error(y,p))}
