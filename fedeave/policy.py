import torch
from torch import nn
import torch.nn.functional as F

class InterventionPolicy(nn.Module):
    def __init__(self,n_concepts,n_actions=4,d=128):
        super().__init__(); self.state=nn.Linear(d+3*n_concepts+n_actions,256)
        self.actor=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU(),nn.Linear(128,n_actions))
        self.critic=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,h,z,u,rho,history,action_mask=None):
        s=F.gelu(self.state(torch.cat([h,z,u,rho,history],-1)))
        logits=self.actor(s)
        if action_mask is not None: logits=logits.masked_fill(~action_mask.bool(),-1e4)
        return logits,self.critic(s).squeeze(-1)

def policy_loss(logits,values,actions,rewards,lambda_v=.5,lambda_h=.01):
    dist=torch.distributions.Categorical(logits=logits)
    advantage=rewards-values
    return -(advantage.detach()*dist.log_prob(actions)).mean()+lambda_v*advantage.square().mean()-lambda_h*dist.entropy().mean()
