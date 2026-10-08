import tempfile
from pathlib import Path
import torch
from fedeave.data import make_demo,StudentDataset,collate_with_graph
from fedeave.model import FedEAVE
from torch.utils.data import DataLoader

def test_forward():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'demo.npz';make_demo(p,n=8,seq=4)
        ds=StudentDataset(p);batch=next(iter(DataLoader(ds,batch_size=4,collate_fn=collate_with_graph(ds))))
        model=FedEAVE(); model.eval()
        with torch.no_grad():
            out=model(batch,stage=3)
        assert out['logit'].shape==(4,)
        assert out['mastery'].shape==(4,8)
        assert out['policy_logits'].shape==(4,4)
        assert torch.isfinite(out['logit']).all()
        assert torch.allclose(out['weights'].sum(1),torch.ones(4),atol=1e-5)
        assert torch.all(out['weights'][:,2]==0)
