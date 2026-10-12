"""02 - Sensitivity analyses for the revision.

Usage:  python src/02_sensitivity_analyses.py [task ...]
Tasks (default: all, in this order):
  ablation   augmentation ablation, expansion 2-9x and noise 0.01-0.10 SD (Fig. 22)   [resumable]
  threshold  label threshold q = 0.50, 0.60, 0.67, 0.75 (Supplementary Fig. S1)
  imputation LOD substitution rules: LOD/2, LOD/sqrt2, LOD, zero
  vif        variance inflation factors, elastic-net coefficients, total-lipid collinearity check
  perm       fold-wise permutation importance (Supplementary Fig. S2)
  nested     nested CV (inner 3-fold grid search, outer 5-fold x 2)
  shap       SHAP for XGB on the restricted feature set (Supplementary Fig. S3)
  fidelity   classifier two-sample test (C2ST), correlation preservation and KS tests for augmented data
"""
import sys
from pcb_pipeline import *
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

FR = FEATS['Restricted (Set-B removed)']; FF = FEATS['Full (as published)']
X = X_all[FR].values.astype(float)
REPS = 1 if SMOKE else 5
ALL = ['ablation', 'threshold', 'imputation', 'vif', 'perm', 'nested', 'shap', 'fidelity']
tasks = sys.argv[1:] or ALL


def ablation():
    """Augmentation applied inside training folds only; test folds are real rows only."""
    path = OUT_DIR / '02_ablation.json'
    done = json.load(open(path)) if path.exists() else []
    grid = [(1, 0.0)] + [(f, s) for f in [2, 4, 6, 9] for s in [0.01, 0.025, 0.05, 0.10]]
    if SMOKE: grid = grid[:2]
    for factor, sigma in grid:
        if any(d['factor'] == factor and d['sigma'] == sigma for d in done):
            continue  # resume: skip configurations already computed
        r = cv_eval(FR, ['LR', 'RF', 'XGB'], n_rep=REPS, aug=(factor, sigma) if factor > 1 else None)
        row = dict(factor=factor, sigma=sigma)
        for m in ['LR', 'RF', 'XGB']:
            row[m + '_auc'] = corrected_ci(r[m]['auc']); row[m + '_prauc'] = corrected_ci(r[m]['prauc'])
        done.append(row); save_json(done, '02_ablation.json')
        print(f'  factor {factor}x sigma {sigma}:', {m: round(row[m + "_auc"][0], 3) for m in ['LR', 'RF', 'XGB']}, flush=True)


def threshold():
    out = []
    for q in [0.5, 0.6, 0.67, 0.75]:
        yq, t = label(q)
        for fs in ['Full (as published)', 'Restricted (Set-B removed)']:
            r = cv_eval(FEATS[fs], ['LR', 'RF'], n_rep=REPS, yv=yq)
            out.append(dict(q=q, thr=t, prev=float(yq.mean()), fs=fs, LR=corrected_ci(r['LR']['auc']), RF=corrected_ci(r['RF']['auc'])))
    save_json(out, '02_threshold.json')


def imputation():
    out = []
    for rule in ['half', 'sqrt2', 'lod', 'zero']:
        r = cv_eval(FR, ['LR', 'RF'], n_rep=REPS, Xdf=build(rule))
        out.append(dict(rule=rule, LR=corrected_ci(r['LR']['auc']), RF=corrected_ci(r['RF']['auc'])))
    save_json(out, '02_imputation.json')


def vif():
    def _vif(feats):
        Xv = X_all[feats].fillna(X_all[feats].median()); Xv = (Xv - Xv.mean()) / Xv.std(); Xv.insert(0, 'const', 1.0)
        return {f: float(variance_inflation_factor(Xv.values, i + 1)) for i, f in enumerate(feats)}
    p = pipe(models()['LR-EN']).fit(X, y)
    d = X_all[['TRIGL_m_mmol/L', 'CHOL_m_mmol/L', 'total lipid g/L_mother']].dropna()
    lin = LinearRegression().fit(d.iloc[:, :2], d.iloc[:, 2])
    save_json({'vif_full': _vif(FF), 'vif_restricted': _vif(FR),
               'en_coef': dict(zip(FR, p.named_steps['m'].coef_[0].tolist())),
               'total_lipid_from_TG_CHOL_R2': lin.score(d.iloc[:, :2], d.iloc[:, 2]),
               'spearman_with_setb_sum': X_all[['PCB-' + k for k in NONB]].corrwith(setb_sum, method='spearman').to_dict()},
              '02_vif.json')


def perm():
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1 if SMOKE else 2, random_state=SEED); pim = {'RF': [], 'LR': []}
    for tr, te in cv.split(X, y):
        for mn in ['RF', 'LR']:
            p = pipe(models()[mn]).fit(X[tr], y[tr])
            pim[mn].append(permutation_importance(p, X[te], y[te], scoring='roc_auc', n_repeats=10, random_state=SEED).importances_mean)
    save_json({mn: {f: corrected_ci(np.array(pim[mn])[:, j]) for j, f in enumerate(FR)} for mn in pim}, '02_permutation.json')


def nested():
    grids = {'LR': {'m__C': [0.01, 0.1, 1, 10]},
             'RF': {'m__min_samples_leaf': [1, 3, 5], 'm__max_features': ['sqrt', 0.5]},
             'XGB': {'m__max_depth': [2, 3, 4], 'm__learning_rate': [0.03, 0.1]}}
    out = {}
    for mn, g in grids.items():
        a = []
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=1 if SMOKE else 2, random_state=SEED).split(X, y):
            base = models()[mn]
            if mn in ('RF', 'XGB'): base.set_params(n_estimators=200)
            gs = GridSearchCV(pipe(base), g, scoring='roc_auc', cv=StratifiedKFold(3, shuffle=True, random_state=SEED)).fit(X[tr], y[tr])
            a.append(roc_auc_score(y[te], gs.predict_proba(X[te])[:, 1]))  # outer fold never used for selection
        out[mn] = corrected_ci(a)
    save_json(out, '02_nested.json')


def shap_task():
    import shap
    Xs = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X_all[FR]), columns=FR)
    sv = shap.TreeExplainer(models()['XGB'].fit(Xs, y)).shap_values(Xs)
    np.save(OUT_DIR / '02_shap_restricted.npy', {'sv': sv, 'X': Xs.values, 'cols': FR}, allow_pickle=True)
    save_json({'mean_abs_shap': dict(zip(FR, np.abs(sv).mean(0).tolist()))}, '02_shap.json')


def fidelity():
    def c2st(cols, sigma=0.025, reps=1 if SMOKE else 3):
        Xi = X_all[cols].fillna(X_all[cols].median()).values; a = []
        for rep in range(reps):
            rng = np.random.default_rng(rep); Xa, _ = augment(Xi, y, 2, sigma, rng); syn = Xa[len(y):]
            Z = np.vstack([Xi, syn]); lab = np.r_[np.zeros(len(Xi)), np.ones(len(syn))]; pr = np.zeros(len(lab))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(Z, lab):
                pr[te] = RandomForestClassifier(200, random_state=rep).fit(Z[tr], lab[tr]).predict_proba(Z[te])[:, 1]
            a.append(roc_auc_score(lab, pr))
        return float(np.mean(a))
    cont = [c for c in FR if c not in ['Maternal_education', 'Maternal_smoking', 'Maternal_alcohol', 'Maternal_GDM', 'Maternal_BP']]
    well = ['PCB-' + k for k in ['99', '118', '138', '153', '156', '170', '180', '187']]
    cens = ['PCB-' + k for k in ['28', '74', '101']]
    out = {'c2st_continuous_restricted': c2st(cont), 'c2st_well_detected': c2st(well), 'c2st_censored': c2st(cens)}
    Xi = X_all[well].values; Xa, _ = augment(Xi, y, 2, 0.025, np.random.default_rng(0)); syn = Xa[len(y):]
    d = np.abs(pd.DataFrame(Xi).corr('spearman').values - pd.DataFrame(syn).corr('spearman').values)[np.triu_indices(len(well), 1)]
    out['spearman_diff_well_detected'] = {'mean': float(d.mean()), 'max': float(d.max())}
    pcbs = ['PCB-' + k for k in NONB + SETB]
    Xi = X_all[pcbs].values; Xa, _ = augment(Xi, y, 6, 0.025, np.random.default_rng(0)); syn = Xa[len(y):]
    out['ks_pvalue_real_vs_augmented'] = {c: float(ks_2samp(Xi[:, j], syn[:, j]).pvalue) for j, c in enumerate(pcbs)}
    save_json(out, '02_fidelity.json')


RUN = {'ablation': ablation, 'threshold': threshold, 'imputation': imputation, 'vif': vif,
       'perm': perm, 'nested': nested, 'shap': shap_task, 'fidelity': fidelity}
for t in tasks:
    print(f'== {t}', flush=True); RUN[t]()
