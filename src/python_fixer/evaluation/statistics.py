"""Paired family-cluster inference using deterministic standard-library RNGs."""
from __future__ import annotations
import itertools
import math
import random
from statistics import mean
from .metrics import quantile


def holm_adjust(pvalues: dict[str,float]) -> dict[str,float]:
    if not pvalues or any(type(p) not in (int,float) or not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("nonempty finite p-values in [0,1] required")
    adjusted={}
    previous=0.0
    for i,(name,p) in enumerate(sorted(pvalues.items(),key=lambda item:(item[1],item[0]))):
        previous=max(previous,min(1.0,p*(len(pvalues)-i)))
        adjusted[name]=previous
    return adjusted


def paired_cluster_analysis(differences: dict[str,dict[str,list[float]]], *, replicates=10000, seed=987) -> dict:
    if type(replicates) is not int or replicates < 100:
        raise ValueError("at least 100 bootstrap replicates required")
    families=sorted(differences)
    if not families:
        raise ValueError("at least one family required")
    seeds=sorted(differences[families[0]])
    if not seeds:
        raise ValueError("at least one paired seed required")
    for family in families:
        if sorted(differences[family])!=seeds:
            raise ValueError("paired seed sets differ across families")
        sizes={len(differences[family][s]) for s in seeds}
        if len(sizes)!=1 or 0 in sizes or any(not math.isfinite(v) or not -1 <= v <= 1
                for s in seeds for v in differences[family][s]):
            raise ValueError("paired family/seed sizes or differences invalid")
    sums={f:{s:math.fsum(differences[f][s]) for s in seeds} for f in families}
    sizes={f:len(differences[f][seeds[0]]) for f in families}
    def effect(family_sample,seed_sample):
        numerator=math.fsum(math.fsum(sums[f][s] for s in seed_sample)/len(seed_sample) for f in family_sample)
        return numerator/sum(sizes[f] for f in family_sample)
    observed=effect(families,seeds)
    rng=random.Random(seed)
    samples=[effect(rng.choices(families,k=len(families)),rng.choices(seeds,k=len(seeds)))
             for _ in range(replicates)]
    family_means=[mean(mean(differences[f][s]) for s in seeds) for f in families]
    weights=[len(differences[f][seeds[0]]) for f in families]
    denominator=sum(weights)
    if len(families)<=16:
        permutations=itertools.product((-1,1),repeat=len(families))
        total=2**len(families)
        exact=True
    else:
        permutation_rng=random.Random(seed+1)
        permutations=([permutation_rng.choice((-1,1)) for _ in families] for _ in range(replicates))
        total=replicates
        exact=False
    exceed=sum(abs(sum(sign*value*weight for sign,value,weight in zip(signs,family_means,weights))/denominator)>=abs(observed)-1e-12
               for signs in permutations)
    pvalue=exceed/total if exact else (exceed+1)/(total+1)
    return {"difference":observed,"family_count":len(families),"paired_seed_count":len(seeds),
        "bootstrap_replicates":replicates,"seed":seed,
        "ci95":[quantile(samples,.025),quantile(samples,.975)],
        "simultaneous_ci975":[quantile(samples,.0125),quantile(samples,.9875)],
        "permutation_p":pvalue,"permutation_exact":exact,"permutation_count":total,
        "per_seed_difference":{s:effect(families,[s]) for s in seeds},
        "warnings":(["fewer than 20 independent families"] if len(families)<20 else [])+
                   (["fewer than three paired training seeds or a fixed control"] if len(seeds)<3 else [])}
