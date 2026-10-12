"""03 - Figures for the revision (Figs. 21-22 and Supplementary Figs. S1-S5) from the JSON results."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, shap, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SMOKE = os.environ.get('SMOKE') == '1'
RES = Path(os.environ.get('PCB_OUT', ROOT / ('results_smoke' if SMOKE else 'results')))
FIG = ROOT / ('figures_smoke' if SMOKE else 'figures'); FIG.mkdir(exist_ok=True)
L = lambda n: json.load(open(RES / n))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})
R1 = L('01_feature_leakage.json')
R2 = {'ablation': L('02_ablation.json'), 'threshold': L('02_threshold.json'), 'perm': L('02_permutation.json')}
R2.update(L('02_vif.json'))
AB = {m: [a for a in R2['ablation'] if a['factor'] == 1][0][m + '_auc'] for m in ['LR', 'RF', 'XGB']}
MODELS = ['LR', 'LR-EN', 'RF', 'XGB', 'MLP']
COL = {'LR': '#1f4e79', 'LR-EN': '#5b9bd5', 'RF': '#2e7d32', 'XGB': '#c55a11', 'MLP': '#7f6000'}
SETS = list(R1['main'].keys())

# Fig A: feature-set (circularity) forest plot
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ypos = 0; ticks = []; labels = []
for si, fs in enumerate(SETS):
    for mi, m in enumerate(MODELS):
        mu, lo, hi = R1['main'][fs][m]['auc']
        ax.errorbar(mu, ypos, xerr=[[mu-lo], [hi-mu]], fmt='o', color=COL[m], capsize=3, ms=5,
                    label=m if si == 0 else None)
        ypos -= 1
    ticks.append(ypos + len(MODELS)/2 + 0.5); labels.append(fs.replace('Full (as published)','Full (original\nfeature set)').replace(' (', '\n(') if 'Full' not in fs else 'Full (original\nfeature set)')
    ypos -= 1
ax.axvline(0.5, ls=':', c='grey', lw=1); ax.axvline(R1['setb_sum_auc'], ls='--', c='firebrick', lw=1)
ax.text(R1['setb_sum_auc']-0.005, ypos+1.5, 'Target sum itself\n(AUC = %.2f)' % R1['setb_sum_auc'], ha='right', va='bottom', color='firebrick', fontsize=8)
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('ROC-AUC (mean, 95% CI; repeated 5-fold CV, 10 repeats)')
ax.set_xlim(0.45, 1.03); ax.legend(loc='lower left', fontsize=8, frameon=False, ncol=5, bbox_to_anchor=(0, -0.22))
ax.set_title('Feature-leakage sensitivity analysis (real-data cohort, n = %d)' % R1['n'], fontsize=11, pad=10)
plt.savefig(FIG / 'Fig21_feature_leakage.png'); plt.close()

# Fig B: augmentation ablation
ab = R2['ablation']; base = [a for a in ab if a['factor'] == 1][0]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, m in zip(axs, ['LR', 'RF', 'XGB']):
    b = AB[m]
    ax.axhspan(b[1], b[2], color='grey', alpha=0.15, label='No augmentation (95% CI)')
    ax.axhline(b[0], color='grey', lw=1)
    for s, c in zip([0.01, 0.025, 0.05, 0.10], ['#c6dbef', '#6baed6', '#2171b5', '#08306b']):
        pts = [a for a in ab if a['sigma'] == s]
        ax.plot([a['factor'] for a in pts], [a[m+'_auc'][0] for a in pts], 'o-', color=c, label='σ = %.3g×SD' % s, ms=4)
    ax.set_title(m); ax.set_xlabel('Training-set expansion factor (×)'); ax.set_xticks([2, 4, 6, 9])
axs[0].set_ylabel('ROC-AUC on real-only test folds')
axs[0].set_ylim(0.70, 0.97)
h,l=axs[2].get_legend_handles_labels(); fig.legend(h,l,fontsize=8,frameon=False,loc='lower center',ncol=5,bbox_to_anchor=(0.5,-0.08))
fig.suptitle('Augmentation ablation (restricted feature set; augmentation applied inside training folds only)', fontsize=10)
plt.tight_layout(); plt.savefig(FIG / 'Fig22_augmentation_ablation.png'); plt.close()

# S1 threshold sensitivity
th = R2['threshold']
fig, ax = plt.subplots(figsize=(6, 3.6))
for fs, mk in [('Full (as published)', 's'), ('Restricted (Set-B removed)', 'o')]:
    for m, c in [('LR', COL['LR']), ('RF', COL['RF'])]:
        pts = [t for t in th if t['fs'] == fs]
        q = [t['q'] for t in pts]; mu = [t[m][0] for t in pts]
        err = [[t[m][0]-t[m][1] for t in pts], [t[m][2]-t[m][0] for t in pts]]
        ax.errorbar(q, mu, yerr=err, marker=mk, color=c, capsize=3, ls='-' if mk == 'o' else '--', label='%s – %s' % (m, fs.split(' (')[0]))
ax.set_xlabel('Quantile used to define "high exposure" (q)'); ax.set_ylabel('ROC-AUC (95% CI)')
ax.set_xticks([0.5, 0.6, 0.67, 0.75]); ax.legend(fontsize=7, frameon=False); ax.set_title('Threshold sensitivity', fontsize=11)
plt.savefig(FIG / 'FigS1_threshold_sensitivity.png'); plt.close()

# S2 permutation importance
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, m in zip(axs, ['RF', 'LR']):
    d = R2['perm'][m]; feats = sorted(d, key=lambda f: d[f][0])
    mu = [d[f][0] for f in feats]; lo = [d[f][0]-d[f][1] for f in feats]; hi = [d[f][2]-d[f][0] for f in feats]
    ax.barh(range(len(feats)), mu, xerr=[lo, hi], color=COL[m], alpha=0.8, capsize=2)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels([f.replace('_m_mmol/L', '').replace('_mother', '').replace('prepregancy_bmi', 'BMI').replace('m_age', 'Age') for f in feats], fontsize=8)
    ax.axvline(0, c='k', lw=0.8); ax.set_xlabel('Decrease in ROC-AUC when permuted'); ax.set_title(m + ' (restricted features)')
plt.tight_layout(); plt.savefig(FIG / 'FigS2_permutation_importance.png'); plt.close()

# S3 SHAP
# SHAP values file holds individual-level data, so it is not distributed; regenerate with
# `python 02_sensitivity_analyses.py shap` before drawing this figure.
if not (RES / '02_shap_restricted.npy').exists():
    print('skipping SHAP figure (run task "shap" first)')
else:
  d = np.load(RES / '02_shap_restricted.npy', allow_pickle=True).item()
  names = [c.replace('_m_mmol/L', '').replace('_mother', '').replace('prepregancy_bmi', 'BMI').replace('m_age', 'Age') for c in d['cols']]
  shap.summary_plot(d['sv'], d['X'], feature_names=names, show=False, max_display=15, plot_size=(7, 4.5))
  plt.title('SHAP (XGB, restricted feature set)', fontsize=11); plt.savefig(FIG / 'FigS3_shap_restricted.png'); plt.close()

# S4 VIF
vf = {k: min(v, 1e4) for k, v in R2['vif_full'].items()}; vr = {k: min(v, 1e4) for k, v in R2['vif_restricted'].items()}
feats = sorted(vf, key=lambda f: -vf[f])
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(feats))
ax.bar(x-0.2, [vf[f] for f in feats], 0.4, label='Full feature set', color='#c55a11')
ax.bar(x+0.2, [vr.get(f, np.nan) for f in feats], 0.4, label='Restricted feature set', color='#1f4e79')
ax.axhline(10, ls='--', c='k', lw=0.8); ax.text(len(feats)-1, 11, 'VIF = 10', ha='right', fontsize=8)
ax.set_yscale('log'); ax.text(0, 1.3e4, 'TG, CHOL and total lipid: exact linear dependence (VIF → ∞, capped)', fontsize=7); ax.set_xticks(x); ax.set_xticklabels([f.replace('_m_mmol/L', '').replace('_mother', '').replace('prepregancy_bmi', 'BMI').replace('m_age', 'Age').replace('total lipid g/L', 'Total lipid') for f in feats], rotation=60, ha='right', fontsize=8)
ax.set_ylabel('Variance inflation factor (log scale)'); ax.legend(frameon=False, fontsize=8); ax.set_title('Multicollinearity diagnostics', fontsize=11)
plt.savefig(FIG / 'FigS4_vif.png'); plt.close()

# S5 detection frequency
det = R1['detection']; ks = sorted(det, key=lambda k: det[k])
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(['PCB-'+k for k in ks], [100*det[k] for k in ks], color=['#c00000' if det[k] < 0.7 else '#1f4e79' for k in ks])
ax.axhline(70, ls='--', c='k', lw=0.8); ax.set_ylabel('Detection frequency (%)'); ax.set_ylim(0, 105)
plt.xticks(rotation=45, ha='right', fontsize=8); ax.set_title('Congener detection frequency (> LOD)', fontsize=11)
plt.savefig(FIG / 'FigS5_detection_frequency.png'); plt.close()
print('figs done')
