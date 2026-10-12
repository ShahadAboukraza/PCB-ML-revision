"""01 - Feature-leakage analysis (Section 3.1, Table III, Fig. 21).

Compares four input feature sets under repeated stratified 5-fold CV (10 repeats),
tests full vs restricted per model, and compares models pairwise with Holm correction.
"""
from pcb_pipeline import *
OUT = {'n': int(len(df)), 'n_high': int(y.sum()), 'thr60': float(thr60), 'detection': det}
MN = ['LR', 'LR-EN', 'RF', 'XGB', 'MLP']

# 0. trivial reference: Set-B sum as only predictor
OUT['setb_sum_auc'] = float(roc_auc_score(y, setb_sum))

# 1. Feature-set analysis (5-fold x 10 repeats)
main = {}
for fs, feats in FEATS.items():
    r = cv_eval(feats, MN, n_rep=10)
    main[fs] = r
    print(fs, {m: round(np.mean(r[m]['auc']), 3) for m in MN}, flush=True)
summ = {fs: {m: {k: corrected_ci(v) for k, v in main[fs][m].items()} for m in MN} for fs in FEATS}
OUT['main'] = summ

# 2. Pairwise model comparison (restricted set) with Holm correction
pairs = list(itertools.combinations(MN, 2)); pv = []
R = main['Restricted (Set-B removed)']
for a, b in pairs:
    t, p = corrected_ttest(R[a]['auc'], R[b]['auc']); pv.append((a, b, np.mean(R[a]['auc'])-np.mean(R[b]['auc']), t, p))
order = np.argsort([x[4] for x in pv]); m = len(pv); holm = [None]*m; running = 0
for rank, i in enumerate(order):
    adj = min(1.0, (m-rank)*pv[i][4]); running = max(running, adj); holm[i] = running
OUT['pairwise'] = [dict(a=a, b=b, diff=d, t=t, p=p, p_holm=h) for (a, b, d, t, p), h in zip(pv, holm)]
# full vs restricted, per model
OUT['full_vs_restricted'] = {mn: corrected_ttest(main['Full (as published)'][mn]['auc'], R[mn]['auc']) for mn in MN}

OUT['fold_auc'] = {fs: {m: main[fs][m]['auc'] for m in MN} for fs in FEATS}
save_json(OUT, '01_feature_leakage.json')
