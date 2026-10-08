import numpy as np

def propensity_did(X,treatment,pre,post,caliper=.2,seed=42):
    """Nearest-neighbour propensity matching without replacement, then DiD.
    Parallel trends and ignorability are assumptions, not verified by this code.
    """
    from sklearn.linear_model import LogisticRegression
    X=np.asarray(X); treatment=np.asarray(treatment).astype(bool)
    if treatment.sum()<2 or (~treatment).sum()<2: return np.nan,0
    clf=LogisticRegression(max_iter=1000,random_state=seed).fit(X,treatment.astype(int))
    ps=np.clip(clf.predict_proba(X)[:,1],1e-5,1-1e-5)
    logit=np.log(ps/(1-ps)); threshold=caliper*np.std(logit)
    treated=np.where(treatment)[0]; controls=set(np.where(~treatment)[0].tolist()); pairs=[]
    for i in treated:
        if not controls: break
        j=min(controls,key=lambda j:abs(logit[i]-logit[j]))
        if abs(logit[i]-logit[j])<=threshold: pairs.append((i,j)); controls.remove(j)
    if not pairs: return np.nan,0
    diff=np.asarray(post)-np.asarray(pre)
    return float(np.mean([diff[i]-diff[j] for i,j in pairs])),len(pairs)

def clipped_noisy_effect(effect,n,limit=1.,noise_std=0.,seed=None):
    rng=np.random.default_rng(seed)
    return float(np.clip(effect,-limit,limit)+rng.normal(0,noise_std)),int(n)

def aggregate_effects(effects):
    valid=[(e,n) for e,n in effects if np.isfinite(e) and n>0]
    return float(np.average([e for e,n in valid],weights=[n for e,n in valid])) if valid else float('nan')
