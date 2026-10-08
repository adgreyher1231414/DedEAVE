import torch
from torch import nn
from .encoders import AssessmentEncoder,BehaviourEncoder,TemporalEncoder,WeightedGAT
from .alignment import EvidenceAlignment,alignment_loss
from .causal import CausalDiagnosis
from .policy import InterventionPolicy

class FedEAVE(nn.Module):
    def __init__(self,n_items=101,n_concepts=8,edges=None,n_actions=4):
        super().__init__(); edges=edges if edges is not None else [(i,i+1) for i in range(n_concepts-1)]
        self.assess=AssessmentEncoder(n_items,n_concepts)
        self.behaviour=BehaviourEncoder(); self.temporal=TemporalEncoder(); self.peer=WeightedGAT()
        self.align=EvidenceAlignment(); self.diagnosis=CausalDiagnosis(n_concepts,n_items,edges)
        self.policy=InterventionPolicy(n_concepts,n_actions)
        self.n_actions=n_actions
    def forward(self,batch,stage=2):
        mask=batch['available'].bool(); b=mask.shape[0]
        peer=self.peer(batch['peer'],batch['adj']) if mask[:,2].any() else torch.zeros(b,128,device=mask.device)
        features=torch.stack([self.assess(batch['assessment']),self.behaviour(batch['behaviour']),peer,self.temporal(batch['temporal'])],1)
        h,v,u,beta=self.align(features,mask)
        diag=self.diagnosis(h,batch['concept'],batch['item'])
        result={**diag,'aligned':h,'weights':beta,'align_loss':alignment_loss(v,u,beta,mask)}
        if stage>=3:
            # During training, differentiable proxy; expensive graph counterfactuals for evaluation only.
            rho=(1-diag['mastery'])*torch.exp(-diag['uncertainty'])
            result['policy_logits'],result['value']=self.policy(h,diag['mastery'],diag['uncertainty'],rho,batch['history'])
        return result
