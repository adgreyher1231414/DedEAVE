import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from .data import collate_with_graph
from .policy import policy_loss

# Caution: explicit per-record gradients are costly; private SGD requires accounting
# all local mini-batch steps and using suitable subsampling assumptions.
def dp_step(model,batch,optimizer,stage=2,clip=1.0,noise=1.1,lambdas=(.2,.1,.1,.1)):
    params=[p for p in model.parameters() if p.requires_grad]
    accum=[torch.zeros_like(p) for p in params]
    b=batch['label'].shape[0]
    for i in range(b):
        single={k:(v[i:i+1] if k!='adj' else v[i:i+1,i:i+1]) for k,v in batch.items()}
        # Single-record DP gradients cannot meaningfully use cross-record InfoNCE;
        # the alignment term is zero and non-private contrastive learning is omitted.
        out=model(single,stage); loss=F.binary_cross_entropy_with_logits(out['logit'],single['label'].float())
        if 'mastery_target' in single:
            loss=loss+lambdas[1]*F.binary_cross_entropy(out['mastery'],single['mastery_target'].float())
        if 'effect_target' in single:
            loss=loss+lambdas[2]*F.mse_loss(out['mastery'].mean(1),single['effect_target'].float())
        if stage>=3 and 'reward' in single and 'action' in single:
            loss=loss+lambdas[3]*policy_loss(out['policy_logits'],out['value'],single['action'].long(),single['reward'].float())
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        norm=torch.sqrt(sum(g.detach().square().sum() for g in grads if g is not None)).clamp_min(1e-12)
        scale=min(1.,float(clip/norm))
        for a,g in zip(accum,grads):
            if g is not None: a.add_(g.detach(),alpha=scale)
    optimizer.zero_grad(set_to_none=True)
    for p,g in zip(params,accum):
        p.grad=(g+torch.randn_like(g)*(noise*clip))/b if noise>0 else g/b
    optimizer.step()
    return float(loss.detach())

def nonprivate_step(model,batch,optimizer,stage=2,lambdas=(.2,.1,.1,.1)):
    optimizer.zero_grad(); out=model(batch,stage)
    loss=F.binary_cross_entropy_with_logits(out['logit'],batch['label'].float())+lambdas[0]*out['align_loss']
    if 'mastery_target' in batch: loss=loss+lambdas[1]*F.binary_cross_entropy(out['mastery'],batch['mastery_target'].float())
    if 'effect_target' in batch: loss=loss+lambdas[2]*F.mse_loss(out['mastery'].mean(1),batch['effect_target'].float())
    if stage>=3 and 'reward' in batch and 'action' in batch:
        loss=loss+lambdas[3]*policy_loss(out['policy_logits'],out['value'],batch['action'].long(),batch['reward'].float())
    loss.backward(); optimizer.step(); return float(loss.detach())

def federated_train(model,dataset,partitions,rounds=2,client_fraction=.1,epochs=1,batch_size=8,private=False,noise=1.1,clip=1.,device='cpu',seed=42):
    rng=np.random.default_rng(seed); history=[]; model.to(device)
    for rnd in range(rounds):
        active=[k for k,p in enumerate(partitions) if len(p)>0]
        chosen=rng.choice(active,size=max(1,int(np.ceil(client_fraction*len(partitions)))),replace=False)
        states=[]; sizes=[]; losses=[]
        for k in chosen:
            local=copy.deepcopy(model).to(device); local.train()
            opt=torch.optim.AdamW(local.parameters(),lr=1e-3,weight_decay=1e-5)
            subset=Subset(dataset,partitions[k].tolist()); loader=DataLoader(subset,batch_size=batch_size,shuffle=True,collate_fn=collate_with_graph(dataset))
            for ep in range(epochs):
                for batch in loader:
                    batch={key:val.to(device) for key,val in batch.items()}
                    # Stage schedule is illustrative and can be changed in real experiments.
                    stage=1 if rnd<rounds//4 else (2 if rnd<3*rounds//4 else 3)
                    fn=dp_step if private else nonprivate_step
                    kwargs={'clip':clip,'noise':noise} if private else {}
                    losses.append(fn(local,batch,opt,stage=stage,**kwargs))
            states.append({key:value.detach().cpu().clone() for key,value in local.state_dict().items()}); sizes.append(len(partitions[k]))
        total=sum(sizes); merged={}
        for name in states[0]:
            if not states[0][name].is_floating_point():
                merged[name]=states[0][name]
            else:
                merged[name]=sum(s[name]*(n/total) for s,n in zip(states,sizes))
        model.load_state_dict(merged)
        entry={'round':rnd+1,'train_loss':float(np.mean(losses)),'clients':len(chosen)}
        print(entry,flush=True); history.append(entry)
    return history
