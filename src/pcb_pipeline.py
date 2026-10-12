"""Shared data loading, preprocessing, models and statistics for the PCB-ML revision analyses.

Label: high exposure = sum of Set-B congeners (PCB-138, 153, 156, 170, 180, 183, 187) above the
60th percentile. Values reported as "<LOD" are substituted by LOD/2 (other rules in build()).
"""
import pandas as pd, numpy as np, json, warnings, itertools
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
#   PCB_DATA   path to the cohort CSV (default: data/cMother_Chemical_data_25_3_2025.csv)
#   PCB_OUT    output folder for JSON results (default: results/)
#   SMOKE=1    quick test run (1 CV repeat) to check the code works end to end
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('PCB_DATA', ROOT / 'data' / 'cMother_Chemical_data_25_3_2025.csv'))
SMOKE = os.environ.get('SMOKE') == '1'
# smoke runs never overwrite the published results
OUT_DIR = Path(os.environ.get('PCB_OUT', ROOT / ('results_smoke' if SMOKE else 'results'))); OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

if not DATA.exists():
    raise FileNotFoundError(f'Cohort data not found at {DATA}. See data/README.md.')
df = pd.read_csv(DATA)

PCB = {'28':'mPCB 28_ng/l','52':'mPCB 52_ng/l','74':'mPCB 74_ng/l','99':'mPCB 99_ng/l',
       '101':'mPCB 101_ng/l','118':'mPCB118_ng/l','138':'mPCB 138_ng/l','153':'mPCB 153_ng/l',
       '156':'mPCB 156_ng/l','170':'mPCB 170_ng/l','180':'mPCB 180_ng/l','183':'mPCB 183_ng/l',
       '187':'mPCB 187_ng/l'}
SETB = ['138','153','156','170','180','183','187']
NONB = ['28','74','99','101','118']   # PCB-52 dropped: 100% <LOD (zero variance)
MAT = ['m_age','prepregancy_bmi','Maternal_education','Maternal_smoking','Maternal_alcohol','Maternal_GDM','Maternal_BP']
LIP = ['LDLC_m_mmol/L','HDLC_m_mmol/L','TRIGL_m_mmol/L','CHOL_m_mmol/L','total lipid g/L_mother']

def parse(col, rule='half'):
    out, lodflag = [], []
    for v in df[col].astype(str):
        v = v.strip()
        if v.startswith('<'):
            lod = float(v[1:].replace(',', '.'))
            f = {'half':0.5,'sqrt2':1/np.sqrt(2),'lod':1.0,'zero':0.0}[rule]
            out.append(lod*f); lodflag.append(1)
        elif v in ('nan','NA',''):
            out.append(np.nan); lodflag.append(0)
        else:
            out.append(max(float(v.replace(',', '.')), 0.0)); lodflag.append(0)
    return np.array(out), np.array(lodflag)

def build(rule='half'):
    X = pd.DataFrame(index=df.index)
    for k, c in PCB.items():
        X['PCB-'+k], _ = parse(c, rule)
    for c in MAT + LIP:
        X[c] = pd.to_numeric(df[c], errors='coerce')
    return X

X_all = build('half')
setb_sum = X_all[['PCB-'+k for k in SETB]].sum(axis=1)

# detection frequencies
det = {k: float(1 - parse(c)[1].mean()) for k, c in PCB.items()}

def label(q=0.6):
    thr = np.quantile(setb_sum, q)
    return (setb_sum > thr).astype(int).values, thr

y, thr60 = label(0.6)
FEATS = {
 'Full (as published)': MAT + LIP + ['PCB-'+k for k in NONB + SETB],
 'Restricted (Set-B removed)': MAT + LIP + ['PCB-'+k for k in NONB],
 'Covariates only (maternal + lipids)': MAT + LIP,
 'Non-Set-B PCBs only': ['PCB-'+k for k in NONB],
}

def models():
    """Model set used in the revised analysis (hyper-parameters as reported in Section 2.3)."""
    return {
     'LR': LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000),
     'LR-EN': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.5,
                                 class_weight='balanced', max_iter=20000),
     'RF': RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                  min_samples_leaf=2, random_state=SEED, n_jobs=-1),
     'XGB': XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, eval_metric='logloss', random_state=SEED,
                          n_jobs=4, verbosity=0),
     'MLP': MLPClassifier(hidden_layer_sizes=(32,16), alpha=1e-2, max_iter=3000,
                          early_stopping=False, random_state=SEED),
    }

def pipe(m):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('m', m)])

def augment(Xtr, ytr, factor, sigma, rng):
    """label-aware bootstrap + Gaussian micro-noise (sigma x feature SD), training fold only"""
    if factor <= 1: return Xtr, ytr
    Xtr = Xtr.copy()
    med = np.nanmedian(Xtr, axis=0)
    idx = np.where(np.isnan(Xtr)); Xtr[idx] = np.take(med, idx[1])
    sd = Xtr.std(axis=0)
    n_new = int(len(ytr)*(factor-1))
    Xs, ys = [Xtr], [ytr]
    for cls in (0, 1):
        ci = np.where(ytr == cls)[0]
        k = int(round(n_new*len(ci)/len(ytr)))
        pick = rng.choice(ci, k, replace=True)
        Xn = Xtr[pick] + rng.normal(0, 1, (k, Xtr.shape[1]))*sd*sigma
        Xs.append(np.clip(Xn, 0, None)); ys.append(np.full(k, cls))
    return np.vstack(Xs), np.concatenate(ys)

def cv_eval(feats, model_names, n_splits=5, n_rep=10, yv=None, Xdf=None, aug=None, seed=SEED):
    """Repeated stratified k-fold CV. Imputation/scaling are fitted inside each training fold
    (Pipeline); optional augmentation (factor, sigma) is applied to the training fold only;
    every test fold contains real rows only."""
    if SMOKE: n_rep = 1
    yv = y if yv is None else yv
    Xdf = X_all if Xdf is None else Xdf
    X = Xdf[feats].values.astype(float)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_rep, random_state=seed)
    rng = np.random.default_rng(seed)
    res = {m: {'auc':[], 'prauc':[], 'brier':[], 'f1':[]} for m in model_names}
    for tr, te in cv.split(X, yv):
        Xtr, ytr = X[tr], yv[tr]
        if aug: Xtr, ytr = augment(Xtr, ytr, aug[0], aug[1], rng)
        for mn in model_names:
            p = pipe(models()[mn]).fit(Xtr, ytr).predict_proba(X[te])[:, 1]
            res[mn]['auc'].append(roc_auc_score(yv[te], p))
            res[mn]['prauc'].append(average_precision_score(yv[te], p))
            res[mn]['brier'].append(brier_score_loss(yv[te], p))
            res[mn]['f1'].append(f1_score(yv[te], (p >= 0.5).astype(int)))
    return res

def corrected_ci(a, test_frac=0.2):
    """Mean and 95% CI using the Nadeau & Bengio (2003) corrected resampled t-statistic."""
    a = np.asarray(a); n = len(a); m = a.mean(); v = a.var(ddof=1)
    se = np.sqrt((1/n + test_frac/(1-test_frac)) * v)
    t = stats.t.ppf(0.975, n-1)
    return m, m - t*se, min(m + t*se, 1.0)

def corrected_ttest(a, b, test_frac=0.2):
    """Paired corrected resampled t-test on fold-level scores (Nadeau & Bengio, 2003)."""
    d = np.asarray(a) - np.asarray(b); n = len(d)
    if d.var(ddof=1) == 0: return 0.0, 1.0
    tstat = d.mean() / np.sqrt((1/n + test_frac/(1-test_frac)) * d.var(ddof=1))
    return tstat, 2*stats.t.sf(abs(tstat), n-1)



def save_json(obj, name):
    with open(OUT_DIR / name, 'w') as f:
        json.dump(obj, f, indent=1, default=float)
    print('saved', OUT_DIR / name)
