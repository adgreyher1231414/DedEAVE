import torch
from torch import nn
import torch.nn.functional as F

class CausalDiagnosis(nn.Module):
    def __init__(self,n_concepts,n_items,edges,d=128):
        super().__init__(); self.c=n_concepts
        adj=torch.zeros(n_concepts,n_concepts)
        for parent,child in edges:
            if parent>=child: raise ValueError('Edges must follow topological concept ordering (parent < child)')
            adj[child,parent]=1
        counts=adj.sum(0).clamp_min(1); self.register_buffer('parents',adj/counts[None,:])
        self.embedding=nn.Embedding(n_concepts,64)
        self.init=nn.Linear(d+64,d)
        self.msg=nn.ModuleList([nn.Linear(d,d) for _ in range(2)])
        self.gate=nn.ModuleList([nn.Linear(3*d,d) for _ in range(2)])
        self.update=nn.ModuleList([nn.Linear(3*d,d) for _ in range(2)])
        self.mastery=nn.Linear(d,1); self.uncertainty=nn.Linear(d,1)
        self.item_embedding=nn.Embedding(n_items,d)
        self.outcome=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Dropout(.1),nn.Linear(d,1))
    def _propagate(self,h,q,intervention=None):
        # Topological update allows interventions to influence all descendants.
        layers=[]
        for l in range(2):
            new=[]
            for c in range(self.c):
                w=self.parents[c,:c]
                if c: parents=torch.stack(new if False else [q[:,p] for p in range(c)],dim=1)
                msg=(parents*w[None,:,None]).sum(1) if c else torch.zeros_like(h)
                msg=self.msg[l](msg)
                inp=torch.cat([q[:,c],msg,h],-1)
                g=torch.sigmoid(self.gate[l](inp))
                qc=g*torch.tanh(self.update[l](inp))+(1-g)*q[:,c]
                if intervention is not None and intervention[0]==c:
                    # Setting hidden state by shifting its mastery logit is an operational approximation.
                    qc=qc + (2*intervention[1]-1)*self.mastery.weight.squeeze(0)[None,:]*4
                new.append(qc)
            q=torch.stack(new,1)
        return q
    def forward(self,h,concept,item):
        emb=self.embedding(torch.arange(self.c,device=h.device))[None].expand(h.size(0),-1,-1)
        q=F.gelu(self.init(torch.cat([h[:,None,:].expand(-1,self.c,-1),emb],-1)))
        q=self._propagate(h,q)
        z=torch.sigmoid(self.mastery(q)).squeeze(-1)
        u=F.softplus(self.uncertainty(q)).squeeze(-1)
        qc=q[torch.arange(h.size(0),device=h.device),concept]
        logit=self.outcome(torch.cat([h,qc,self.item_embedding(item)],-1)).squeeze(-1)
        return {'logit':logit,'mastery':z,'uncertainty':u,'states':q}
    @torch.no_grad()
    def root_causes(self,h,concept,item):
        original=self.forward(h,concept,item)
        z,u=original['mastery'],original['uncertainty']; effects=[]
        for c in range(self.c):
            emb=self.embedding(torch.arange(self.c,device=h.device))[None].expand(h.size(0),-1,-1)
            q=F.gelu(self.init(torch.cat([h[:,None,:].expand(-1,self.c,-1),emb],-1)))
            values=[]
            for value in (0,1):
                states=self._propagate(h,q,(c,value))
                qc=states[torch.arange(h.size(0),device=h.device),concept]
                values.append(torch.sigmoid(self.outcome(torch.cat([h,qc,self.item_embedding(item)],-1)).squeeze(-1)))
            effects.append(values[1]-values[0])
        ice=torch.stack(effects,1)
        rho=(1-z)*F.relu(ice)*torch.exp(-u)
        return {'ice':ice,'root_score':rho,'root_concept':rho.argmax(1)}
