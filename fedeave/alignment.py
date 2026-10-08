import torch
from torch import nn
import torch.nn.functional as F

class EvidenceAlignment(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.project=nn.ModuleList([nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,d)) for _ in range(4)])
        self.source=nn.Parameter(torch.zeros(1,4,d))
        layer=nn.TransformerEncoderLayer(d,4,256,.1,activation='gelu',batch_first=True)
        self.cross=nn.TransformerEncoder(layer,2)
        self.reliability=nn.Sequential(nn.Linear(d,64),nn.Tanh(),nn.Linear(64,1))
        self.norm=nn.LayerNorm(d)
    def forward(self,features,available):
        available=available.bool(); assert available.any(1).all(), 'At least one stream required'
        v=torch.stack([F.normalize(head(features[:,j]),dim=-1) for j,head in enumerate(self.project)],1)
        v=v*available.unsqueeze(-1)
        u=self.cross(v+self.source,src_key_padding_mask=~available)
        u=u*available.unsqueeze(-1)
        logits=self.reliability(u).squeeze(-1).masked_fill(~available,-1e4)
        beta=torch.softmax(logits,1)
        fused=self.norm((u*beta.unsqueeze(-1)).sum(1))
        return fused,v,u,beta

def alignment_loss(v,u,beta,available,temp=.07,conf_weight=.5):
    terms=[]; conflicts=[]; batch=v.size(0)
    for a in range(4):
        for b in range(a+1,4):
            valid=available[:,a].bool() & available[:,b].bool()
            if valid.sum()<2: continue
            ids=valid.nonzero(as_tuple=True)[0]
            logits=v[ids,a]@v[ids,b].T/temp
            label=torch.arange(len(ids),device=v.device)
            terms.append((F.cross_entropy(logits,label,reduction='none')+F.cross_entropy(logits.T,label,reduction='none'))/2)
            diff=(u[ids,a]-u[ids,b]).norm(dim=-1)
            cos=F.cosine_similarity(u[ids,a],u[ids,b],dim=-1)
            conflicts.append(beta[ids,a]*beta[ids,b]*diff*(1-cos))
    if not terms: return v.sum()*0
    return torch.cat(terms).mean()+conf_weight*torch.cat(conflicts).mean()
