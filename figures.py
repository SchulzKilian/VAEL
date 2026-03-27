#!/usr/bin/env python3
"""
figures.py — Generate all figures for the VAEL + CFM workshop paper.

Usage:
    python figures.py

Outputs to ./paper_figures/ as both .pdf (for LaTeX) and .png (for preview).
Figures that depend on experiments not yet completed are skipped gracefully.
"""

import os
import csv
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.image import imread

matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

# ── paths ──────────────────────────────────────────────────────────────────────
EXP_ROOT  = './experiments'
OUT_DIR   = './paper_figures'
Path(OUT_DIR).mkdir(exist_ok=True)

COLLAPSE_THRESHOLD = 0.1   # acc_gen below this → posterior collapse

# colour palette (colour-blind friendly)
C = {
    'baseline_low_kl':  '#4C72B0',   # blue    — VAEL, kl=1e-5
    'flow_low_kl':      '#55A868',   # green   — VAEL+flow, kl=1e-5
    'baseline_high_kl': '#C44E52',   # red     — VAEL, kl=1e-3
    'flow_high_kl':     '#DD8452',   # orange  — VAEL+flow, kl=1e-3
    'no_symbolic':      '#8172B2',   # purple  — VAE+flow (no symbolic)
    'collapsed':        '#BBBBBB',   # grey    — collapsed runs
}


# ── helpers ────────────────────────────────────────────────────────────────────

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  saved {name}')


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def is_collapsed(row):
    return float(row['acc_gen']) < COLLAPSE_THRESHOLD


def strip_jitter(n, centre=0, width=0.15):
    """Return n jittered x-positions centred at centre."""
    np.random.seed(42)
    return centre + np.random.uniform(-width, width, n)


def load_npy(path):
    """Load a train_info / validation_info .npy dict. Returns None if missing."""
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


# ── load all experiment data ───────────────────────────────────────────────────

early_rows       = load_csv(f'{EXP_ROOT}/vael_2digitMNIST/vael_2digitMNIST.csv')
early_flow_rows  = load_csv(f'{EXP_ROOT}/vael_2digitMNIST_flow/vael_2digitMNIST_flow.csv')
comparison_rows  = load_csv(f'{EXP_ROOT}/vael_2digitMNIST_comparison/vael_2digitMNIST_comparison.csv')
fixed_rows       = load_csv(f'{EXP_ROOT}/vael_2digitMNIST_fixed/vael_2digitMNIST_fixed.csv')
flow_abl_rows    = load_csv(f'{EXP_ROOT}/vael_2digitMNIST_flow_ablation/vael_2digitMNIST_flow_ablation.csv')
sym_abl_rows     = load_csv(f'{EXP_ROOT}/vael_2digitMNIST_symbolic_ablation/vael_2digitMNIST_symbolic_ablation.csv')
final_rows       = load_csv(f'{EXP_ROOT}/vael_2digitMNIST_final_comparison/vael_2digitMNIST_final_comparison.csv')

# label each row with a condition name
# early_rows have no flow_w column — default to 0.0 (pure baseline)
def label_rows(rows):
    for r in rows:
        fw  = float(r.get('flow_w', 0.0))
        kw  = float(r['kl_w'])
        ns  = r.get('no_symbolic', 'False') in ('True', '1', 'true')
        if ns:
            r['_cond'] = 'no_symbolic'
        elif fw == 0.0 and kw < 1e-4:
            r['_cond'] = 'baseline_low_kl'
        elif fw  > 0.0 and kw < 1e-4:
            r['_cond'] = 'flow_low_kl'
        elif fw == 0.0 and kw >= 1e-4:
            r['_cond'] = 'baseline_high_kl'
        elif fw  > 0.0 and kw >= 1e-4:
            r['_cond'] = 'flow_high_kl'
        else:
            r['_cond'] = 'other'
    return rows

label_rows(early_rows)
label_rows(early_flow_rows)
label_rows(comparison_rows)
label_rows(fixed_rows)
label_rows(flow_abl_rows)
label_rows(sym_abl_rows)
label_rows(final_rows)

# Pool all runs per condition:
#   baseline_low_kl : early(3) + comparison(5)   = 8 runs
#   flow_low_kl     : early_flow(1) + comparison(5) = 6 runs
#   baseline_high_kl: fixed baseline(5) + final(10) = 15 runs
#   flow_high_kl    : flow_w=2 only — flow_abl(3) + final(10) = 13 runs
#     (fixed flow runs used flow_w=1, not the recommended flow_w=2, so excluded
#      from the main comparison; they appear in the flow-weight ablation only)
fixed_baseline_rows = [r for r in fixed_rows if r['_cond'] == 'baseline_high_kl']
flow_abl_fw2_rows   = [r for r in flow_abl_rows if float(r.get('flow_w', 0.0)) == 2.0]

all_rows = (early_rows + early_flow_rows + comparison_rows
            + fixed_baseline_rows + final_rows + flow_abl_fw2_rows
            + sym_abl_rows)


def rows_for(cond, source=None):
    src = source if source is not None else all_rows
    return [r for r in src if r['_cond'] == cond]


def acc_gen(rows):
    return [float(r['acc_gen']) for r in rows]


def acc_gen_noncollapsed(rows):
    return [float(r['acc_gen']) for r in rows if not is_collapsed(r)]


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Main results: acc_gen per condition (strip + mean bar)
# ══════════════════════════════════════════════════════════════════════════════

def fig_main_results():
    conditions = [
        ('baseline_low_kl',  'VAEL\n(baseline)',         'kl=1e-5'),
        ('flow_low_kl',      'VAEL + CFM\n(low KL)',     'kl=1e-5'),
        ('baseline_high_kl', 'VAEL\n(high KL)',          'kl=1e-3'),
        ('flow_high_kl',     'VAEL + CFM\n(high KL)',    'kl=1e-3'),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(1/19, color='#999', lw=1, ls='--', label='Random chance (1/19)')

    for x, (cond, label, _) in enumerate(conditions):
        rows = rows_for(cond)
        if not rows:
            continue
        accs = acc_gen(rows)
        color = C[cond]

        for i, a in enumerate(accs):
            marker = 'x' if a < COLLAPSE_THRESHOLD else 'o'
            fc     = C['collapsed'] if a < COLLAPSE_THRESHOLD else color
            ax.scatter(strip_jitter(1, x)[0], a, marker=marker,
                       color=fc, s=60, zorder=3, linewidths=1.5)

        good = [a for a in accs if a >= COLLAPSE_THRESHOLD]
        if good:
            ax.plot([x - 0.25, x + 0.25], [np.mean(good)] * 2,
                    color=color, lw=2.5, zorder=4)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c[1] for c in conditions])
    ax.set_ylabel('Generative accuracy (acc_gen)')
    ax.set_ylim(-0.02, 0.70)
    ax.set_title('Generative accuracy across conditions\n(× = collapsed run, — = mean of non-collapsed)')

    collapsed_patch = mpatches.Patch(color=C['collapsed'], label='Collapsed run')
    chance_line     = plt.Line2D([0], [0], color='#999', ls='--', label='Random chance')
    ax.legend(handles=[collapsed_patch, chance_line], loc='upper left')

    save(fig, '1_main_results')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — KL interaction: opposite effects on baseline vs flow
# ══════════════════════════════════════════════════════════════════════════════

def fig_kl_interaction():
    groups = {
        'baseline_low_kl':  1e-5,
        'baseline_high_kl': 1e-3,
        'flow_low_kl':      1e-5,
        'flow_high_kl':     1e-3,
    }

    fig, ax = plt.subplots(figsize=(5, 4))

    # baseline line
    bl_low  = acc_gen_noncollapsed(rows_for('baseline_low_kl'))
    bl_high = acc_gen_noncollapsed(rows_for('baseline_high_kl'))
    fl_low  = acc_gen_noncollapsed(rows_for('flow_low_kl'))
    fl_high = acc_gen_noncollapsed(rows_for('flow_high_kl'))

    xs = [1e-5, 1e-3]

    if bl_low and bl_high:
        ax.plot(xs, [np.mean(bl_low), np.mean(bl_high)],
                'o-', color=C['baseline_low_kl'], lw=2, ms=8, label='VAEL (no flow)')
        ax.fill_between(xs,
                        [np.mean(bl_low)  - np.std(bl_low),  np.mean(bl_high)  - np.std(bl_high)],
                        [np.mean(bl_low)  + np.std(bl_low),  np.mean(bl_high)  + np.std(bl_high)],
                        color=C['baseline_low_kl'], alpha=0.15)

    if fl_low and fl_high:
        ax.plot(xs, [np.mean(fl_low), np.mean(fl_high)],
                's-', color=C['flow_high_kl'], lw=2, ms=8, label='VAEL + CFM')
        ax.fill_between(xs,
                        [np.mean(fl_low)  - np.std(fl_low),  np.mean(fl_high)  - np.std(fl_high)],
                        [np.mean(fl_low)  + np.std(fl_low),  np.mean(fl_high)  + np.std(fl_high)],
                        color=C['flow_high_kl'], alpha=0.15)

    ax.set_xscale('log')
    ax.set_xlabel('KL weight (kl_w)')
    ax.set_ylabel('Mean acc_gen (non-collapsed runs)')
    ax.set_title('KL weight interacts oppositely\nwith flow vs. baseline')
    ax.legend()
    ax.set_xticks(xs)
    ax.set_xticklabels(['1e-5', '1e-3'])
    ax.annotate('Baseline ↓', xy=(1e-3, np.mean(bl_high) if bl_high else 0.3),
                xytext=(6e-4, (np.mean(bl_high) + 0.04) if bl_high else 0.34),
                fontsize=9, color=C['baseline_low_kl'])
    ax.annotate('Flow ↑', xy=(1e-3, np.mean(fl_high) if fl_high else 0.45),
                xytext=(6e-4, (np.mean(fl_high) + 0.04) if fl_high else 0.49),
                fontsize=9, color=C['flow_high_kl'])

    save(fig, '2_kl_interaction')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Collapse rate across conditions
# ══════════════════════════════════════════════════════════════════════════════

def fig_collapse_rate():
    conditions = [
        ('baseline_low_kl',  'VAEL\n(kl=1e-5)'),
        ('flow_low_kl',      'VAEL+CFM\n(kl=1e-5)'),
        ('baseline_high_kl', 'VAEL\n(kl=1e-3)'),
        ('flow_high_kl',     'VAEL+CFM\n(kl=1e-3)'),
    ]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    for x, (cond, label) in enumerate(conditions):
        rows = rows_for(cond)
        if not rows:
            continue
        rate = sum(1 for r in rows if is_collapsed(r)) / len(rows)
        ax.bar(x, rate * 100, color=C[cond], width=0.5, zorder=3)
        ax.text(x, rate * 100 + 1.5, f'{rate:.0%}', ha='center', fontsize=10)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c[1] for c in conditions])
    ax.set_ylabel('Collapse rate (%)')
    ax.set_ylim(0, 85)
    ax.set_title('Posterior collapse rate by condition')
    ax.axhline(0, color='k', lw=0.5)

    save(fig, '3_collapse_rate')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Image panel: collapsed / baseline-good / flow-good
# ══════════════════════════════════════════════════════════════════════════════

def fig_image_panel():
    # Hand-picked runs: (exp_folder, exp_ID, run_ID, label, acc_gen)
    panels = [
        ('vael_2digitMNIST_comparison', '1', '22-03-2026-22-32-24',
         'Collapsed\n(acc_gen=0.053)', 'cond_generation_'),
        ('vael_2digitMNIST_comparison', '2', '23-03-2026-01-52-22',
         'VAEL baseline\n(acc_gen=0.485)', 'cond_generation_'),
        ('vael_2digitMNIST_fixed',      '1', '23-03-2026-11-49-42',
         'VAEL + CFM\n(acc_gen=0.578)', 'cond_generation_'),
    ]

    images = []
    labels = []
    for exp, eid, rid, label, fname in panels:
        path = f'{EXP_ROOT}/{exp}/{eid}/{rid}/images/{fname}.png'
        if os.path.exists(path):
            images.append(imread(path))
            labels.append(label)

    if not images:
        print('  skipping fig_image_panel — image files not found')
        return

    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    if len(images) == 1:
        axes = [axes]

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis('off')

    fig.suptitle('Conditional generation samples (10 sums × 10 rows)', y=1.01)
    save(fig, '4_image_panel')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Flow weight ablation (skipped if data unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def fig_flow_ablation():
    if not flow_abl_rows:
        print('  skipping fig_flow_ablation — experiment not yet complete')
        return

    from collections import defaultdict
    by_fw = defaultdict(list)
    for r in flow_abl_rows:
        by_fw[float(r['flow_w'])].append(float(r['acc_gen']))

    # add baseline from fixed experiment (flow_w=0)
    baseline_good = acc_gen_noncollapsed(rows_for('baseline_high_kl'))
    if baseline_good:
        by_fw[0.0] = acc_gen(rows_for('baseline_high_kl'))

    fw_vals  = sorted(by_fw.keys())
    means    = []
    stds     = []
    means_nc = []   # mean of non-collapsed only

    for fw in fw_vals:
        accs    = by_fw[fw]
        nc      = [a for a in accs if a >= COLLAPSE_THRESHOLD]
        means.append(np.mean(accs))
        stds.append(np.std(accs))
        means_nc.append(np.mean(nc) if nc else np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.errorbar(fw_vals, means, yerr=stds, fmt='o-',
                color=C['flow_high_kl'], lw=2, ms=7, capsize=4,
                label='Mean ± std (all runs)')
    ax.plot(fw_vals, means_nc, 's--',
            color=C['baseline_low_kl'], lw=1.5, ms=6,
            label='Mean (non-collapsed only)')
    ax.axhline(1/19, color='#999', lw=1, ls=':', label='Random chance')

    ax.set_xlabel('Flow weight (flow_w)')
    ax.set_ylabel('acc_gen')
    ax.set_title('Effect of flow weight on generation accuracy\n(kl_w=1e-3, 3 seeds each)')
    ax.legend()
    ax.set_xticks(fw_vals)

    save(fig, '5_flow_ablation')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Symbolic ablation: VAEL+CFM vs VAE+CFM (no_symbolic)
# ══════════════════════════════════════════════════════════════════════════════

def fig_symbolic_ablation():
    ns_rows   = rows_for('no_symbolic')
    flow_rows = rows_for('flow_high_kl')

    if not ns_rows:
        print('  skipping fig_symbolic_ablation — experiment not yet complete')
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    for x, (rows, label, color) in enumerate([
        (flow_rows, 'VAEL + CFM\n(symbolic)', C['flow_high_kl']),
        (ns_rows,   'VAE + CFM\n(no symbolic)', C['no_symbolic']),
    ]):
        accs = acc_gen(rows)
        for a in accs:
            marker = 'x' if a < COLLAPSE_THRESHOLD else 'o'
            fc     = C['collapsed'] if a < COLLAPSE_THRESHOLD else color
            ax.scatter(strip_jitter(1, x)[0], a, marker=marker,
                       color=fc, s=60, zorder=3, linewidths=1.5)
        good = [a for a in accs if a >= COLLAPSE_THRESHOLD]
        if good:
            ax.plot([x - 0.25, x + 0.25], [np.mean(good)] * 2,
                    color=color, lw=2.5, zorder=4)

    ax.axhline(1/19, color='#999', lw=1, ls='--', label='Random chance')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['VAEL + CFM\n(symbolic)', 'VAE + CFM\n(no symbolic)'])
    ax.set_ylabel('acc_gen')
    ax.set_title('Does symbolic conditioning matter?\n(— = mean of non-collapsed)')
    ax.legend()

    save(fig, '6_symbolic_ablation')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Learning curves: good run vs collapsed run
# ══════════════════════════════════════════════════════════════════════════════

def fig_learning_curves():
    # Best run: fixed exp, flow_w=1.0, acc_gen=0.578
    best_run_path = f'{EXP_ROOT}/vael_2digitMNIST_fixed/1/23-03-2026-11-49-42'
    # Collapsed run: comparison exp, flow_w=1.0, acc_gen=0.053
    coll_run_path = f'{EXP_ROOT}/vael_2digitMNIST_comparison/1/22-03-2026-22-32-24'

    # Try .npy first; fall back to existing learning curve PNGs
    best_train = load_npy(f'{best_run_path}/train_info.npy')
    coll_train = load_npy(f'{coll_run_path}/train_info.npy')

    if best_train and coll_train:
        # Re-plot from raw data
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, train, title, color in [
            (axes[0], best_train, 'Non-collapsed (acc_gen=0.578)', C['flow_high_kl']),
            (axes[1], coll_train, 'Collapsed (acc_gen=0.053)',     C['collapsed']),
        ]:
            epochs = range(1, len(train['elbo']) + 1)
            ax.plot(epochs, train['elbo'], color=color, lw=1.5, label='Train ELBO')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('ELBO loss')
            ax.set_title(title)
            ax.legend()
        fig.suptitle('Training dynamics: non-collapsed vs collapsed', fontsize=12)
        plt.tight_layout()
        save(fig, '7_learning_curves')
        return

    # Fallback: stitch together the ELBO PNG images that already exist
    best_elbo = f'{best_run_path}/learning_curve/ELBO (MA 50).png'
    coll_elbo = f'{coll_run_path}/learning_curve/ELBO (MA 50).png'

    if not os.path.exists(best_elbo) or not os.path.exists(coll_elbo):
        print('  skipping fig_learning_curves — no data found')
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, path, title in [
        (axes[0], best_elbo, 'Non-collapsed run (acc_gen=0.578)\nVAEL + CFM, kl_w=1e-3'),
        (axes[1], coll_elbo, 'Collapsed run (acc_gen=0.053)\nVAEL + CFM, kl_w=1e-5'),
    ]:
        ax.imshow(imread(path))
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    fig.suptitle('Training dynamics: non-collapsed vs collapsed (ELBO, MA 50)', fontsize=12)
    plt.tight_layout()
    save(fig, '7_learning_curves')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Summary table (for quick reference, not for paper body)
# ══════════════════════════════════════════════════════════════════════════════

def fig_summary_table():
    conditions = [
        ('baseline_low_kl',  'VAEL (kl=1e-5)',        all_rows),
        ('flow_low_kl',      'VAEL+CFM (kl=1e-5)',     all_rows),
        ('baseline_high_kl', 'VAEL (kl=1e-3)',         all_rows),
        ('flow_high_kl',     'VAEL+CFM (kl=1e-3)',     all_rows),
        ('no_symbolic',      'VAE+CFM (no symbolic)',  all_rows),
    ]

    rows_data = []
    for cond, label, src in conditions:
        rows = rows_for(cond, src)
        if not rows:
            continue
        accs   = acc_gen(rows)
        good   = [a for a in accs if a >= COLLAPSE_THRESHOLD]
        discrs = [float(r['acc_discr_test']) for r in rows]
        collapse_rate = sum(1 for a in accs if a < COLLAPSE_THRESHOLD) / len(accs)
        rows_data.append([
            label,
            str(len(rows)),
            f'{collapse_rate:.0%}',
            f'{np.mean(good):.3f} ± {np.std(good):.3f}' if good else 'N/A',
            f'{max(good):.3f}' if good else 'N/A',
            f'{np.mean(discrs):.3f}',
        ])

    if not rows_data:
        print('  skipping fig_summary_table — no data')
        return

    col_labels = ['Condition', 'N', 'Collapsed', 'acc_gen (non-coll.)', 'Best acc_gen', 'acc_discr']

    fig, ax = plt.subplots(figsize=(11, 1.5 + 0.5 * len(rows_data)))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#DDDDDD')
            cell.set_text_props(weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#F7F7F7')
        cell.set_edgecolor('#CCCCCC')

    fig.suptitle('Results summary', fontsize=12, y=0.98)
    save(fig, '8_summary_table')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Generation image grid: best run full cond_generation
# ══════════════════════════════════════════════════════════════════════════════

def fig_best_generation():
    path = f'{EXP_ROOT}/vael_2digitMNIST_fixed/1/23-03-2026-11-49-42/images/cond_generation_.png'
    if not os.path.exists(path):
        print('  skipping fig_best_generation — image not found')
        return

    img = imread(path)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('VAEL + CFM: conditional generation (best run, acc_gen=0.578)\n'
                 'Rows = digit pair sums 0–18, columns = samples', fontsize=11)
    save(fig, '9_best_generation')


# ══════════════════════════════════════════════════════════════════════════════
# run all
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'Saving figures to {OUT_DIR}/\n')
    fig_main_results()
    fig_kl_interaction()
    fig_collapse_rate()
    fig_image_panel()
    fig_flow_ablation()
    fig_symbolic_ablation()
    fig_learning_curves()
    fig_summary_table()
    fig_best_generation()
    print('\nDone.')
