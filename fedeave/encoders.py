import torch
from torch import nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.score=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,1))
    def forward(self,x,mask=None):
        a=self.score(x).squeeze(-1)
        if mask is not None: a=a.masked_fill(~mask,-1e4)
        return (x*torch.softmax(a,dim=1).unsqueeze(-1)).sum(1)

class AssessmentEncoder(nn.Module):
    def __init__(self,n_items,n_concepts,d=128,max_len=128):
        super().__init__()
        self.emb=nn.ModuleList([nn.Embedding(n, d) for n in (n_items,n_concepts,3,16,16)])
        self.pos=nn.Embedding(max_len,d)
        layer=nn.TransformerEncoderLayer(d,4,256,0.1,activation='gelu',batch_first=True,norm_first=False)
        self.net=nn.TransformerEncoder(layer,2); self.pool=AttentionPool(d)
    def forward(self,x):
        b,t,_=x.shape; z=sum(e(x[:,:,j].long().clamp(0,e.num_embeddings-1)) for j,e in enumerate(self.emb))
        z=z+self.pos(torch.arange(t,device=x.device))[None]
        causal=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),1)
        return self.pool(self.net(z,mask=causal),x[:,:,0]!=0)

class ResidualTCN(nn.Module):
    def __init__(self,ci,co,k,dil):
        super().__init__(); pad=(k-1)*dil//2
        self.c1=nn.Conv1d(ci,co,k,padding=pad,dilation=dil)
        self.c2=nn.Conv1d(co,co,k,padding=pad,dilation=dil)
        self.skip=nn.Conv1d(ci,co,1) if ci!=co else nn.Identity()
        self.norm=nn.LayerNorm(co); self.drop=nn.Dropout(.1)
    def forward(self,x):
        y=self.c2(F.gelu(self.c1(x)))
        return self.norm((self.skip(x)+self.drop(y)).transpose(1,2)).transpose(1,2)

class BehaviourEncoder(nn.Module):
    def __init__(self,features=6):
        super().__init__(); self.proj=nn.Conv1d(features,64,1)
        self.net=nn.Sequential(ResidualTCN(64,64,3,1),ResidualTCN(64,128,5,2),ResidualTCN(128,128,7,4))
        self.pool=AttentionPool(128)
    def forward(self,x): return self.pool(self.net(F.gelu(self.proj(x.transpose(1,2)))).transpose(1,2))

class TemporalEncoder(nn.Module):
    def __init__(self,features=6):
        super().__init__(); self.proj=nn.Linear(features,64)
        self.rnn=nn.LSTM(64,64,2,batch_first=True,bidirectional=True,dropout=.1); self.pool=AttentionPool(128)
    def forward(self,x): return self.pool(self.rnn(self.proj(x))[0])

class WeightedGAT(nn.Module):
    """Dense minibatch subgraph GAT, adjacency [B,B]; suitable for small client graphs."""
    def __init__(self,features=6,d=128):
        super().__init__(); self.in_proj=nn.Linear(features,d)
        self.projs=nn.ModuleList([nn.Linear(d,d) for _ in range(2)])
        self.a_src=nn.ParameterList([nn.Parameter(torch.randn(4,32)*.02),nn.Parameter(torch.randn(1,128)*.02)])
        self.a_dst=nn.ParameterList([nn.Parameter(torch.randn(4,32)*.02),nn.Parameter(torch.randn(1,128)*.02)])
        self.norm=nn.LayerNorm(d)
    def forward(self,x,adj):
        h=F.elu(self.in_proj(x)); n=h.size(0)
        adj=adj.to(h.device).float(); adj=adj+torch.eye(n,device=h.device)
        for layer in range(2):
            heads=4 if layer==0 else 1; width=128//heads
            v=self.projs[layer](h).view(n,heads,width)
            e=F.leaky_relu((v*self.a_src[layer]).sum(-1)[:,None,:]+(v*self.a_dst[layer]).sum(-1)[None,:,:],.2)
            e=e+torch.log(adj.clamp_min(1e-8))[:,:,None]
            att=torch.softmax(e,dim=1)
            out=torch.einsum('ijh,jhd->ihd',att,v).reshape(n,128)
            h=F.elu(out) if layer==0 else self.norm(h+out)
        return h
