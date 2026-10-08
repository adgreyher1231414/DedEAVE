# FedEAVE


## Run

```bash
pip install -r requirements.txt
python train.py --demo --rounds 2 --clients 4 --epochs 1
python train.py --demo --rounds 2 --clients 4 --epochs 1 --private
```

**Important:** The demo uses synthetic data solely to test software plumbing. Its numbers are not the paper's results. This training script evaluates on the training set only; for a publishable study, construct student-level chronological train/validation/test splits before federated partitioning and use validation AUC for early stopping.

## Data format

Provide a `.npz` containing:

| key | shape | meaning |
|---|---|---|
| assessment | N,T,5 | item, concept, correctness, response-time bin, attempt index |
| behaviour | N,T,6 | six standardized behavioural features |
| temporal | N,T,6 | six temporal engagement features |
| peer | N,6 | six peer node features |
| adj | N,N | sparse-in-practice weighted peer adjacency, stored densely in demo |
| available | N,4 | source availability boolean in a,b,p,t order |
| concept | N | prediction target concept |
| item | N | prediction target item |
| label | N | binary response label |
| history | N,4 | intervention history |

Optional: `mastery_target` (N,C), `effect_target` (N), `action` (N), `reward` (N). These targets require independent measurement / verification; **do not invent them**. Use separate verified intervention logs to supply policy rewards.

## Scope and known methodological gaps

- The manuscript does not give complete raw-to-feature scripts, exact prerequisite edges, treatment logs, independently labeled mastery targets, or all five task loss weights; supply these for real datasets.
- The weighted peer GAT operates on **within-batch subgraphs**, an approximation to whole-client graph propagation. Missing peer evidence is explicitly masked.
- Concept counterfactuals are implemented as interventions on learned hidden-state projections; they are **model-internal sensitivity analyses**, not established causal effects without identifying assumptions. The `root_causes()` method is inference-time and potentially expensive.
- Matched DiD requires common-support / parallel-trends assumptions; the utility estimates effects but does not establish causality on its own.
- The `--private` path clips individual gradients and adds Gaussian noise per local mini-batch. It omits pairwise cross-record contrastive loss when private, because doing that without privacy-aware treatment breaks per-example accounting.
- **No certified epsilon is claimed**: an RDP/PRV accountant, exact sampling scheme, handling of cross-record graph dependence, all optimizer steps, and treatment-statistic release accounting are necessary. The article's stated `(epsilon=2, delta=1e-5)` cannot be inferred solely from 50 rounds, 10% client sampling, noise 1.1; do not report this as achieved.
- This is a functional starting-point implementation, not a claim to have reproduced the published benchmark scores. Production secure aggregation (cryptographic masking) and PySyft deployment are not included. FedAvg is run in-process.
- Stage 1/2/3 schedule is illustrative; for strict three-stage study use appropriately measured labels and tune warm-up boundaries.
