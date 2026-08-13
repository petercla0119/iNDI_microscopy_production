#!/usr/bin/env python3
"""Generate the four phase A-D walkthrough notebooks."""
import json, textwrap
from pathlib import Path

NB_DIR = Path(__file__).parent


def _src(code: str):
    """Convert a multi-line string to a list of '\n'-terminated strings."""
    lines = code.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text: str):
    return {"cell_type": "markdown", "id": f"m{abs(hash(text))%100000:05d}",
            "metadata": {}, "source": _src(text)}


def code(text: str):
    return {"cell_type": "code", "execution_count": None,
            "id": f"c{abs(hash(text))%100000:05d}", "metadata": {}, "outputs": [],
            "source": _src(text)}


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"}
        },
        "cells": cells
    }


STYLE_HELPER = '''
import matplotlib.colors as mc, matplotlib.pyplot as plt, numpy as np

def _fg_for(bg):
    r, g, b = mc.to_rgb(bg)
    return "white" if (0.299*r + 0.587*g + 0.114*b) < 0.5 else "black"

BG = "black"; FG = _fg_for(BG); DPI = 150

def styled_fig(nrows, ncols, figsize, title=""):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG, dpi=DPI)
    if title:
        fig.suptitle(title, color=FG, y=0.99)
    for ax in np.ravel(axs):
        ax.set_facecolor(BG); ax.tick_params(colors=FG)
        ax.title.set_color(FG)
        for s in ax.spines.values(): s.set_color(FG)
        ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    return fig, axs
'''

# ─────────────────────────────────────────────────────────────────────────────
# PHASE A
# ─────────────────────────────────────────────────────────────────────────────

PHASE_A_CELLS = [
    md("""# Phase A — Preprocessing & Corrections

**Dataset:** WARD00034-R2, D28 AB Rep1 (`5c6bf125`)
**Zarr store:** `WARD00034_R2.zarr`  |  field shape `[T=1, C=4, Z=10, Y=2160, X=2160]` uint16
**Pixel size:** 0.2967 µm/px, Z-spacing: 1.997 µm

Covers: zarr store structure · raw channel overview · vignetting · MIP vs best-focus · calibration."""),

    code("""\
%matplotlib inline
from pathlib import Path
import zarr, numpy as np, tifffile
import matplotlib.pyplot as plt, matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap
from skimage.filters import laplace

REPO    = Path('/Users/pmihack/claire/hs-array')
ZARR    = REPO / 'data/WARD00034_R2.zarr'
FIGDIR  = Path(__file__).parent / 'figures/phase_A' if '__file__' in dir() else Path(
    '/Users/pmihack/claire/hs-array/resources/iNDI_microscopy_production/notebooks/figures/phase_A')
FIGDIR.mkdir(parents=True, exist_ok=True)

PX_UM = 0.2967; Z_UM = 1.997; DPI = 150; BG = 'black'; FG = 'white'

def _p(img, lo=0.5, hi=99.5):
    v = img.astype(np.float32)
    return np.percentile(v, lo), np.percentile(v, hi)

store = zarr.open(str(ZARR), mode='r')
print(f"Zarr store opened: {ZARR.name}")
print(f"Top-level keys (row letters): {sorted(store.keys())}")
"""),

    md("## Step 1 — Zarr Store Structure"),

    code("""\
# Plate metadata from zarr.json
attrs = dict(store.attrs)
ome   = attrs['ome']
plate = ome['plate']

print(f"Plate name:  {attrs['experiment_metadata']['plate_name']}")
print(f"Rows:        {[r['name'] for r in plate['rows']]}")
print(f"Columns:     {[c['name'] for c in plate['columns']]}")
print(f"Wells:       {len(plate['wells'])} total")
print(f"Acquisitions:{plate.get('acquisitions', [])}")
print()

# Channel metadata
print("Channel metadata (from zarr.json channels_metadata):")
for ch in attrs['channels_metadata']:
    print(f"  ch{ch['channel_id']}: {ch['name']:10s}  ex={ch['excitation_nm']}nm  em={ch['emission_nm']}nm")
print()

# One field shape
arr = store['C']['7']['4']['0']
print(f"Field C/7/f4 array shape: {arr.shape}  dtype: {arr.dtype}")
print("Axes: [T, C, Z, Y, X] = [1, 4, 10, 2160, 2160]")
print()

# Flatfield profiles
fp = attrs['flatfield_profiles']
print(f"flatfield_profiles: {len(fp)} channel profiles (present in zarr.json, NOT applied at load time)")
for p in fp:
    ptype = type(p['profile']).__name__
    print(f"  channel_id={p['channel_id']}  profile type={ptype}")
print("→ Flatfield profiles are stored as metadata; correction is applied in analysis scripts, not pre-baked into pixel data.")
"""),

    md("## Step 2 — Raw Channel Overview\n\nAll 4 channels, z=5 (mid-stack), field C/7/f4."),

    code("""\
# shape: (1, 4, 10, 2160, 2160)  →  slice [T=0, all_C, Z=5, :, :]
data = arr[0, :, 5, :, :]  # (4, 2160, 2160) uint16

CH_NAMES  = ['DAPI\\n(405/456 nm)', 'Alexa 488\\n(Tuj1+HDGFL2-CE)', 'TDP-43\\n(CF568/546 nm)', 'TGN-46\\n(Alexa647)']
K2G       = LinearSegmentedColormap.from_list('k2g', ['black', 'lime'])

fig, axs = plt.subplots(1, 4, figsize=(18, 5), facecolor=BG, dpi=DPI)
fig.suptitle('Raw channels — well C/7 field 4, z=5  (middle of stack)', color=FG, y=1.01)
plt.subplots_adjust(hspace=0.02, wspace=0.02)

cmaps = ['gray', 'gray', K2G, 'magma']
for i, (ax, name, cmap) in enumerate(zip(axs, CH_NAMES, cmaps)):
    plane = data[i].astype(np.float32)
    vmin, vmax = _p(plane)
    ax.imshow(plane, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
    ax.set_title(f'ch{i}  {name}', color=FG, fontsize=9)
    ax.set_facecolor(BG); ax.axis('off')
    for s in ax.spines.values(): s.set_color(FG)

fig.savefig(FIGDIR / '01_raw_channels.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: 01_raw_channels.png")
"""),

    md("""\
## Step 3 — Vignetting Demonstration

Why flatfield correction matters: intensity falls off toward the image corners due to
uneven illumination (vignetting). This biases per-cell measurements in edge-of-field positions.

Load all 9 fields of well C/7 and compute mean intensity per field per channel.
Then compute corner-vs-center ratio on the DAPI channel."""),

    code("""\
well = store['C']['7']
field_keys = sorted(well.keys())  # 0–8
print(f"Fields in well C/7: {field_keys}")

# Per-field mean intensity for each channel (one z-plane per field)
Z_DEMO = 5
field_means = np.zeros((len(field_keys), 4))
for fi, fk in enumerate(field_keys):
    plane4 = well[fk]['0'][0, :, Z_DEMO, :, :]  # (4, 2160, 2160)
    for ch in range(4):
        field_means[fi, ch] = plane4[ch].mean()

ch_labels = ['DAPI', 'Alexa488', 'TDP-43', 'TGN-46']
fig, axs = plt.subplots(1, 4, figsize=(18, 4), facecolor=BG, dpi=DPI)
fig.suptitle('Per-field mean intensity (well C/7, z=5) — spatial variation reveals vignetting',
             color=FG, y=1.01)
plt.subplots_adjust(wspace=0.35)

for ch in range(4):
    ax = axs[ch]
    vals = field_means[:, ch]
    ax.bar(range(len(field_keys)), vals, color='steelblue', alpha=0.85, edgecolor='none')
    ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.title.set_color(FG)
    for s in ax.spines.values(): s.set_color(FG)
    ax.set_xlabel('field index (0=corner, 4=center, 8=corner)', color=FG, fontsize=8)
    ax.set_ylabel('mean intensity (ADU)', color=FG, fontsize=8)
    ax.set_title(ch_labels[ch], color=FG, fontsize=9)
    ax.axhline(vals.mean(), color='red', ls='--', lw=1, label=f'mean={vals.mean():.0f}')
    ax.legend(fontsize=7, labelcolor='white', facecolor='#222')

fig.savefig(FIGDIR / '02_per_field_mean_intensity.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()

# Corner vs center ratio (DAPI channel, center field = index 4)
center_img = well['4']['0'][0, 0, Z_DEMO, :, :]  # (2160, 2160)
H, W = center_img.shape
P = 256
corners = [center_img[:P, :P], center_img[:P, W-P:],
           center_img[H-P:, :P], center_img[H-P:, W-P:]]
corner_mean = np.mean([c.mean() for c in corners])
center_mean  = center_img[H//2-P//2:H//2+P//2, W//2-P//2:W//2+P//2].mean()
ratio = corner_mean / center_mean

print(f"Corner mean (4 × 256² patches):  {corner_mean:.1f} ADU")
print(f"Center mean (256² central patch): {center_mean:.1f} ADU")
print(f"Corner/Center ratio: {ratio:.3f}  →  {'vignetting present (corners dimmer)' if ratio < 1 else 'no obvious vignetting'}")

# Show the patches
fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG, dpi=DPI)
fig2.suptitle(f'Vignetting: corner vs center patches (DAPI, center field)\\n'
              f'ratio={ratio:.3f}  → corners are {abs(1-ratio)*100:.1f}% {"dimmer" if ratio<1 else "brighter"}',
              color=FG, y=1.02)
for ax in axs2:
    ax.set_facecolor(BG); ax.tick_params(colors=FG)
    for s in ax.spines.values(): s.set_color(FG)

vmin, vmax = _p(center_img)
axs2[0].imshow(center_img, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
axs2[0].set_title('DAPI, center field (2160×2160)', color=FG, fontsize=9)
# overlay corner rectangles
import matplotlib.patches as mpatches
for (y0r, x0r) in [(0,0),(0,W-P),(H-P,0),(H-P,W-P)]:
    axs2[0].add_patch(mpatches.Rectangle((x0r,y0r), P, P, lw=2, edgecolor='red', facecolor='none'))
axs2[0].add_patch(mpatches.Rectangle((W//2-P//2, H//2-P//2), P, P, lw=2, edgecolor='lime', facecolor='none'))
axs2[0].text(30, 80, 'red=corner (dimmer)', color='red', fontsize=8)
axs2[0].text(30, 160, 'green=center (reference)', color='lime', fontsize=8)
axs2[0].axis('off')

# Bar chart summary
labels = ['TL corner', 'TR corner', 'BL corner', 'BR corner', 'CENTER']
vals_bar = [c.mean() for c in corners] + [center_mean]
colors_bar = ['#cc4444']*4 + ['#44cc44']
axs2[1].bar(labels, vals_bar, color=colors_bar, alpha=0.9, edgecolor='none')
axs2[1].set_title(f'Mean intensity per patch\\nCorner/Center = {ratio:.3f}', color=FG, fontsize=9)
axs2[1].set_ylabel('mean intensity (ADU)', color=FG, fontsize=8)
axs2[1].tick_params(colors=FG, axis='both')
axs2[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=7)

fig2.savefig(FIGDIR / '03_vignetting_corner_center.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print(f"Saved: 03_vignetting_corner_center.png")
"""),

    md("""\
## Step 4 — MIP vs Best-Focus Plane Selection

For **detection** (segmentation): Maximum-intensity projection (MIP) over all 10 z-planes.
MIP ensures every nucleus — regardless of which z-plane it peaks at — contributes a strong signal.

For **intensity measurement** (TDP-43 quantification): Best-focus z-plane selected per field
using the variance-of-Laplacian (VoL) focus score: `np.var(laplace(plane))`.
MIP inflates intensity by accumulating signal across planes; the single best-focus plane
gives the cleanest per-compartment measurement."""),

    code("""\
# Load DAPI z-stack and TDP-43 z-stack for center field of well C/7
field_arr = well['4']['0']  # (1, 4, 10, 2160, 2160)

dapi_stack = field_arr[0, 0, :, :, :].astype(np.float32)   # (10, 2160, 2160)
tdp_stack  = field_arr[0, 2, :, :, :].astype(np.float32)   # (10, 2160, 2160)

# MIP
mip_dapi = dapi_stack.max(axis=0)
mip_tdp  = tdp_stack.max(axis=0)

# Best-focus: max variance-of-Laplacian
focus_dapi = np.array([np.var(laplace(p)) for p in dapi_stack])
focus_tdp  = np.array([np.var(laplace(p)) for p in tdp_stack])
bz_dapi = int(focus_dapi.argmax())
bz_tdp  = int(focus_tdp.argmax())

# ── Plot 1: Focus scores
fig, axs = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG, dpi=DPI)
fig.suptitle('Variance-of-Laplacian focus score per z-plane (well C/7 field 4)', color=FG, y=1.01)
for ax in axs: ax.set_facecolor(BG); ax.tick_params(colors=FG); [s.set_color(FG) for s in ax.spines.values()]

z_arr = np.arange(10) * Z_UM
axs[0].bar(z_arr, focus_dapi, width=1.6, color='#4499ff', alpha=0.85, edgecolor='none')
axs[0].axvline(z_arr[bz_dapi], color='cyan', lw=2, label=f'best focus z={bz_dapi+1} ({z_arr[bz_dapi]:.1f} µm)')
axs[0].set_xlabel('z position (µm)', color=FG); axs[0].set_ylabel('VoL score', color=FG)
axs[0].set_title('DAPI focus score', color=FG, fontsize=9); axs[0].legend(fontsize=8, labelcolor=FG, facecolor='#222')

axs[1].bar(z_arr, focus_tdp, width=1.6, color='#44cc88', alpha=0.85, edgecolor='none')
axs[1].axvline(z_arr[bz_tdp], color='lime', lw=2, label=f'best focus z={bz_tdp+1} ({z_arr[bz_tdp]:.1f} µm)')
axs[1].set_xlabel('z position (µm)', color=FG); axs[1].set_ylabel('VoL score', color=FG)
axs[1].set_title('TDP-43 (CF568) focus score', color=FG, fontsize=9); axs[1].legend(fontsize=8, labelcolor=FG, facecolor='#222')

fig.savefig(FIGDIR / '04_focus_scores.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()

# ── Plot 2: MIP vs best-focus side-by-side (DAPI and TDP-43)
fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10), facecolor=BG, dpi=DPI)
fig2.suptitle('MIP vs best-focus plane — DAPI (top) and TDP-43 (bottom)', color=FG, y=1.01)
plt.subplots_adjust(hspace=0.05, wspace=0.05)

panels = [
    (mip_dapi,             'gray',  'DAPI — MIP (max over 10 z-planes)'),
    (dapi_stack[bz_dapi],  'gray',  f'DAPI — best-focus z={bz_dapi+1} ({z_arr[bz_dapi]:.1f} µm)'),
    (mip_dapi - dapi_stack[bz_dapi], 'hot', 'DAPI: MIP − best-focus (extra signal)'),
    (mip_tdp,              LinearSegmentedColormap.from_list('k2g', ['black','lime']),
                                    'TDP-43 — MIP'),
    (tdp_stack[bz_tdp],   LinearSegmentedColormap.from_list('k2g', ['black','lime']),
                                    f'TDP-43 — best-focus z={bz_tdp+1} ({z_arr[bz_tdp]:.1f} µm)'),
    (mip_tdp - tdp_stack[bz_tdp],  'hot', 'TDP-43: MIP − best-focus\\n(inflated by off-focus planes)'),
]

for ax, (img, cmap, title) in zip(axs2.flat, panels):
    vmin, vmax = _p(img)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
    ax.set_title(title, color=FG, fontsize=8)
    ax.set_facecolor(BG); ax.axis('off')
    for s in ax.spines.values(): s.set_color(FG)

fig2.savefig(FIGDIR / '05_mip_vs_bestfocus.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print(f"DAPI best-focus: plane {bz_dapi+1} of 10  (z={z_arr[bz_dapi]:.1f} µm)")
print(f"TDP-43 best-focus: plane {bz_tdp+1} of 10  (z={z_arr[bz_tdp]:.1f} µm)")
print("→ TDP-43 quantification uses the best-focus plane, NOT MIP, to avoid intensity inflation.")
"""),

    md("## Step 5 — Calibration Confirmation"),

    code("""\
# Pixel size from zarr metadata
res_x = attrs['channels_metadata'][0]['image_size_x']  # may be pixel count, not µm
# res_x from parquet inspection: 0.296688 µm/px
# Confirmed from experiment metadata
em = attrs['experiment_metadata']
print("Experiment metadata keys:", list(em.keys()))
print(f"Plate: {em['plate_name']}  Instrument: {em['instrument']}")
print()
print(f"Pixel size (from index.xml / experiment metadata): {PX_UM} µm/px")
print(f"Z-spacing: {Z_UM} µm/plane")
print()

# Nucleus size at this pixel size
# iNeuron nuclei: ~10–20 µm diameter
for diam_um in [10, 15, 20]:
    area_px2 = np.pi * (diam_um / 2 / PX_UM)**2
    area_um2 = np.pi * (diam_um / 2)**2
    print(f"Nucleus diameter {diam_um:2d} µm → {diam_um/PX_UM:.0f} px diameter → area {area_um2:.0f} µm² = {area_px2:.0f} px²")

print()
print("MIN_AREA = 400 px² at 0.2967 µm/px:")
min_area_um2 = 400 * PX_UM**2
print(f"  400 px² = {min_area_um2:.1f} µm² = ~{2*(min_area_um2/np.pi)**0.5:.1f} µm diameter")
print(f"  This excludes debris/fragments (<{min_area_um2:.0f} µm²) while keeping small neurons")
print(f"  Minimum detectable nucleus: ~{2*(min_area_um2/np.pi)**0.5:.1f} µm diameter")
print()
print("Old MIN_AREA = 3500 px² at 0.2967 µm/px:")
old_um2 = 3500 * PX_UM**2
print(f"  3500 px² = {old_um2:.0f} µm² = ~{2*(old_um2/np.pi)**0.5:.0f} µm diameter")
print(f"  → Was eliminating isolated neurons with diameter < {2*(old_um2/np.pi)**0.5:.0f} µm")
print(f"  → Calibrated for Panel_1 at 0.094 µm/px where 3500 px² = 31 µm² (correct floor)")
print(f"  → At 0.2967 µm/px, 3500 px² = {old_um2:.0f} µm²: far too aggressive")

# Visual summary
diams = np.linspace(5, 30, 200)
areas_um2 = np.pi * (diams/2)**2
areas_px2 = areas_um2 / PX_UM**2

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG, dpi=DPI)
ax.set_facecolor(BG); ax.tick_params(colors=FG)
for s in ax.spines.values(): s.set_color(FG)
ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)

ax.plot(diams, areas_px2, color='cyan', lw=2, label='nucleus area (px²) vs diameter (µm)')
ax.axhline(400,  color='lime',  ls='--', lw=2, label='MIN_AREA=400 px² (35 µm²) — CORRECT')
ax.axhline(3500, color='red',   ls='--', lw=2, label='MIN_AREA=3500 px² (308 µm²) — OLD/WRONG')
ax.fill_between(diams, 0, areas_px2, where=areas_px2 < 400,
                color='gray', alpha=0.3, label='below floor (debris)')
ax.fill_between(diams, 400, areas_px2, where=(areas_px2 >= 400) & (areas_px2 < 3500),
                color='orange', alpha=0.3, label='recovered by MIN_AREA=400 fix')
ax.set_xlabel('nucleus diameter (µm)', fontsize=9)
ax.set_ylabel('nucleus area (px²)', fontsize=9)
ax.set_title('Calibration: nucleus size at 0.2967 µm/px\\nOld MIN_AREA=3500 eliminated most isolated iNeurons',
             color=FG, fontsize=9)
ax.legend(fontsize=8, labelcolor=FG, facecolor='#222', framealpha=0.8)
ax.set_xlim(5, 30); ax.set_ylim(0, 5000)

fig.savefig(FIGDIR / '06_calibration.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: 06_calibration.png")
"""),

    code("""\
# Verify figures
pngs = sorted(FIGDIR.glob('*.png'))
empty = [p for p in pngs if p.stat().st_size == 0]
assert pngs,  f"No figures written to {FIGDIR}"
assert not empty, f"0-byte figures: {empty}"
print(f"OK: {len(pngs)} figures in {FIGDIR.name}/")
for p in pngs:
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# PHASE B
# ─────────────────────────────────────────────────────────────────────────────

PHASE_B_CELLS = [
    md("""\
# Phase B — Nucleus Segmentation

**Dataset:** WARD00034-R2 (`5c6bf125`), 249,465 nuclei (MIN_AREA=400)
**Parquet:** `outputs/run_20260806_141727_cyq9/nuclei_features/`

> **Note:** `hs_array_nucleus_segmentation_walkthrough.ipynb` in this directory
> contains the full algorithmic deep-dive (per-object Otsu, all intermediates,
> MIN_AREA diagnostic). This notebook focuses on the production outputs and QC.

Steps: metadata loading · representative field · segmentation sub-steps · QC summary"""),

    code("""\
%matplotlib inline
from pathlib import Path
import numpy as np, pandas as pd, tifffile
import matplotlib.pyplot as plt, matplotlib.colors as mc
from scipy.stats import norm
from skimage import filters, morphology
from skimage.measure import label as sklabel, regionprops_table
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb

REPO    = Path('/Users/pmihack/claire/hs-array')
UUID    = '5c6bf125-6e0f-41be-a447-b03ff294c9cc'
RUN_DIR = REPO / 'outputs/run_20260806_141727_cyq9'
NUC_PAR = RUN_DIR / f'nuclei_features/{UUID}_nuclei_features.parquet'
QC_PAR  = RUN_DIR / f'nuclei_features/{UUID}_image_qc.parquet'
IMG_DIR = REPO / 'data/D28_AB_Rep1' / UUID / 'images'
FIGDIR  = Path('/Users/pmihack/claire/hs-array/resources/iNDI_microscopy_production/notebooks/figures/phase_B')
FIGDIR.mkdir(parents=True, exist_ok=True)

PX_UM = 0.2967; Z_UM = 1.997; DPI = 150; BG = 'black'; FG = 'white'
MIN_AREA = 400   # px² (corrected for 0.2967 µm/px)

def _p(img, lo=0.5, hi=99.5):
    return np.percentile(img.astype(np.float32), lo), np.percentile(img.astype(np.float32), hi)
"""),

    md("## Step 4 — Load Metadata and Pick a Representative Field"),

    code("""\
df = pd.read_parquet(NUC_PAR)
df['area_um2'] = df['area'] * PX_UM**2

print(f"Total nuclei: {len(df):,}")
print(f"Unique wells: {df[['Row','Column']].drop_duplicates().shape[0]}")
print(f"Unique sites: {df[['Row','Column','Frame']].drop_duplicates().shape[0]}")
print(f"Channels in parquet: {df['Channel_name'].unique()}")
print()
print(df[['Row','Column','Frame','area','area_um2','centroid-0','centroid-1',
          'intensity_max','solidity']].head(3).to_string())

site_counts = df.groupby(['Row','Column','Frame']).size().reset_index(name='n')
well_totals = df.groupby(['Row','Column']).size().reset_index(name='well_n')

# Pick center-plate well (rows 5-12, cols 5-18), frame 5 (center of 3x3 grid), ≥10 nuclei/site
cand = (site_counts
        .merge(well_totals, on=['Row','Column'])
        .query("Frame==5 and n>=10 and well_n>=65 and Row>=5 and Row<=12 and Column>=5 and Column<=18")
        .sort_values('n', ascending=False))

if cand.empty:
    cand = site_counts.merge(well_totals, on=['Row','Column']).query("n>=10 and well_n>=65").sort_values('n', ascending=False)

r0 = cand.iloc[0]
WT_ROW, WT_COL, WT_FRAME = int(r0.Row), int(r0.Column), int(r0.Frame)
print(f"\\nWalkthrough site: r{WT_ROW:02d}c{WT_COL:02d} frame={WT_FRAME}  "
      f"({r0.n} nuclei/site, {r0.well_n} nuclei/well)")
"""),

    md("## Step 5 — Nucleus Segmentation Pipeline"),

    code("""\
# Load DAPI z-stack for the chosen site
subdir = IMG_DIR / f'r{WT_ROW:02d}c{WT_COL:02d}'
plane_paths = sorted(subdir.glob(f'r{WT_ROW:02d}c{WT_COL:02d}f{WT_FRAME:02d}p*-ch01t01.tiff'))
planes = [tifffile.imread(p).astype(np.float32) for p in plane_paths]
print(f"Loaded {len(planes)} z-planes for r{WT_ROW:02d}c{WT_COL:02d} f{WT_FRAME:02d}")

# ── Sub-step a: MIP
mip = np.max(np.stack(planes), axis=0)

# ── Sub-step b: Gaussian-fit contrast stretch [µ-1σ … µ+7σ → 0…1]
m, s = norm.fit(mip.flatten())
stretch_min = max(m - 1*s, mip.min())
stretch_max = min(m + 7*s, mip.max())
image_norm = (np.clip(mip, stretch_min, stretch_max) - stretch_min) / (stretch_max - stretch_min)

# ── Sub-step c: Gaussian blur σ=1
blurred = filters.gaussian(image_norm, sigma=1)

# ── Sub-step d: Otsu threshold (production uses (triangle+p50)/2; Otsu shown for clarity)
th_otsu  = filters.threshold_otsu(blurred)
triangle = filters.threshold_triangle(blurred)
p50      = np.percentile(blurred, 50)
th_low   = (triangle + p50) / 2
mask_low = morphology.remove_small_objects(blurred > th_low, min_size=MIN_AREA)
mask_low = morphology.dilation(mask_low, footprint=morphology.disk(2))

# ── Sub-step e: label + regionprops
labeled_raw = sklabel(mask_low)
n_raw = labeled_raw.max()

# ── Sub-step f: QC filters (area, edge exclusion, max-DAPI)
from skimage.measure import regionprops
props = regionprops(labeled_raw, intensity_image=mip)
H, W = mip.shape
EDGE = 20
MAX_AREA_PX2 = 2270  # 200 µm² at 0.2967 µm/px

keep_labels = []
for p in props:
    y0, x0, y1, x1 = p.bbox
    if (y0 < EDGE or x0 < EDGE or y1 > H-EDGE or x1 > W-EDGE):
        continue
    if p.area < MIN_AREA or p.area > MAX_AREA_PX2:
        continue
    if p.intensity_max < 500:
        continue
    keep_labels.append(p.label)

mask_final = np.isin(labeled_raw, keep_labels)
labeled_final = sklabel(mask_final)
n_final = labeled_final.max()

print(f"\\nSub-step a (MIP): {mip.shape} uint16")
print(f"Sub-step b (stretch): raw [{mip.min():.0f},{mip.max():.0f}] → clip [{stretch_min:.0f},{stretch_max:.0f}] (µ={m:.0f}, σ={s:.0f})")
print(f"Sub-step c (blur): σ=1")
print(f"Sub-step d (threshold): (triangle={triangle:.3f} + p50={p50:.3f})/2 = {th_low:.3f}")
print(f"Sub-step e (label): {n_raw} raw blobs")
print(f"Sub-step f (QC): {n_raw} → {n_final} nuclei  (area {MIN_AREA}–{MAX_AREA_PX2} px², edge excl., max-DAPI>500)")
"""),

    code("""\
# Show all 6 sub-steps in one figure
from skimage.color import label2rgb as l2rgb

panels = [
    ('a: MIP', mip, 'gray'),
    ('b: Contrast stretch', image_norm, 'gray'),
    ('c: Gaussian blur σ=1', blurred, 'gray'),
    ('d: Threshold mask', mask_low.astype(float), 'gray'),
    ('e: Labeled (raw)', None, None),
    ('f: After QC filters', None, None),
]

fig, axs = plt.subplots(2, 3, figsize=(15, 10), facecolor=BG, dpi=DPI)
fig.suptitle(f'Segmentation sub-steps — r{WT_ROW:02d}c{WT_COL:02d} f{WT_FRAME:02d}  '
             f'(final: {n_final} nuclei)', color=FG, y=1.01)
plt.subplots_adjust(hspace=0.05, wspace=0.05)

vmin_raw, vmax_raw = _p(mip)
for ax, (title, img, cmap) in zip(axs.flat, panels):
    ax.set_facecolor(BG); ax.axis('off')
    for s in ax.spines.values(): s.set_color(FG)
    ax.set_title(title, color=FG, fontsize=9)
    if title.startswith('e:'):
        ax.imshow(l2rgb(labeled_raw, bg_label=0), aspect='auto', interpolation='none')
    elif title.startswith('f:'):
        # Show accepted nuclei boundaries on MIP
        bd = find_boundaries(labeled_final, mode='outer')
        ax.imshow(mip, cmap='gray', vmin=vmin_raw, vmax=vmax_raw, aspect='auto', interpolation='none')
        overlay = np.zeros((*mip.shape, 4), dtype=np.float32)
        overlay[bd] = [0, 1, 1, 0.9]  # cyan outlines
        ax.imshow(overlay, interpolation='none', aspect='auto')
    elif img is mip:
        ax.imshow(img, cmap=cmap, vmin=vmin_raw, vmax=vmax_raw, aspect='auto', interpolation='none')
    elif img is not None:
        vmin, vmax = (0, 1) if img.max() <= 1.0 else _p(img)
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')

# Add nucleus count annotation
axs[0,0].text(30, 60, f'{len(planes)} z-planes', color=FG, fontsize=8)
axs[1,1].text(30, 60, f'raw: {n_raw} blobs', color=FG, fontsize=8)
axs[1,2].text(30, 60, f'accepted: {n_final}', color='cyan', fontsize=8)

fig.savefig(FIGDIR / '01_segmentation_steps.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: 01_segmentation_steps.png")
"""),

    md("""\
## Step 6 — R1 vs R2 Segmentation Difference

| | Round 1 (R1) | Round 2 (R2) |
|---|---|---|
| Segmentation channel | ch01 = **BFP** (405 nm, cytosolic dye) | ch01 = **DAPI** (405 nm, nuclear stain) |
| Signal shape | Fills entire cell soma | Confined to nucleus |
| MIP appearance | Bright filled soma | Sharp nuclear ring |
| Nucleus detection | Works (soma fills nucleus region) | Works (direct nuclear signal) |
| N:C interpretation | **Not comparable to R2** — BFP is cytosolic, not nuclear | Primary readout: TDP-43 N:C ratio |

**Consequence:** R1 TDP-43 intensity values are valid, but the N:C ratio denominator
(cytoplasm = BFP soma) is not the same as R2 (cytoplasm = annular ring around DAPI nucleus).
R1 and R2 N:C values must NOT be compared directly without cross-round registration.

> WARD00034 R1 TIFF data is not on disk — see WARD00026/27 R1 data for comparison."""),

    md("## QC Summary — Image QC Parquet"),

    code("""\
qc = pd.read_parquet(QC_PAR)
print(f"Image QC parquet: {len(qc)} rows (= {len(qc)//9} wells × 9 fields)")
print(f"Columns: {list(qc.columns)}")
print()

# Filter status breakdown
print("Filter status:")
print(qc.filter_status.value_counts().to_string())
print()
print("Contrast check:")
print(qc.contrast_check.value_counts().to_string())

# Per-field nucleus count distribution
fig, axs = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG, dpi=DPI)
fig.suptitle('Image QC — nucleus count distribution', color=FG, y=1.01)
for ax in axs:
    ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    for s in ax.spines.values(): s.set_color(FG)

qc_pass = qc[qc.filter_status == 'pass']
axs[0].hist(qc_pass.n_nuclei, bins=40, color='steelblue', alpha=0.85, edgecolor='none')
axs[0].axvline(qc_pass.n_nuclei.median(), color='lime', ls='--', lw=2,
               label=f'median={qc_pass.n_nuclei.median():.0f}')
axs[0].set_xlabel('nuclei per field', fontsize=9); axs[0].set_ylabel('fields', fontsize=9)
axs[0].set_title('Nuclei per field (pass status)', color=FG, fontsize=9)
axs[0].legend(fontsize=8, labelcolor=FG, facecolor='#222')

# Well-level totals
well_n = qc_pass.groupby(['Row','Column'])['n_nuclei'].sum()
axs[1].hist(well_n, bins=40, color='mediumseagreen', alpha=0.85, edgecolor='none')
axs[1].axvline(well_n.median(), color='lime', ls='--', lw=2, label=f'median={well_n.median():.0f}')
low_wells = (well_n < 50).sum()
axs[1].set_xlabel('nuclei per well', fontsize=9); axs[1].set_ylabel('wells', fontsize=9)
axs[1].set_title(f'Nuclei per well  (≥9 fields)\\n{low_wells} wells <50 nuclei (potential seeding issues)',
                 color=FG, fontsize=9)
axs[1].legend(fontsize=8, labelcolor=FG, facecolor='#222')

fig.savefig(FIGDIR / '02_qc_nucleus_counts.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()

# Flag low-count wells
low = well_n[well_n < 50].reset_index()
if len(low) > 0:
    low['well'] = low.Row.apply(lambda r: chr(64+int(r))) + low.Column.apply(str)
    print(f"\\nWells with <50 nuclei total ({len(low)} wells):")
    print(low[['well','n_nuclei']].to_string(index=False))
else:
    print("\\nNo wells with <50 nuclei — all wells have good nuclear counts.")
"""),

    code("""\
# Verify figures
pngs = sorted(FIGDIR.glob('*.png'))
empty = [p for p in pngs if p.stat().st_size == 0]
assert pngs,  f"No figures written to {FIGDIR}"
assert not empty, f"0-byte figures: {empty}"
print(f"OK: {len(pngs)} figures in {FIGDIR.name}/")
for p in pngs:
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# PHASE C
# ─────────────────────────────────────────────────────────────────────────────

PHASE_C_CELLS = [
    md("""\
# Phase C — Building the Three TDP-43 Compartments

Three concentric zones are built around each nucleus centroid to measure TDP-43:
1. **Nucleus** (eroded disk, r≈4.5 µm) — nuclear TDP-43 signal
2. **Nuclear-adjacent ring** (annulus 4.5–9 µm) — peri-nuclear TDP-43
3. **Cytoplasm annulus** (annulus 9–18 µm, neighbour-excluded) — cytoplasmic TDP-43

The N:C ratio = nucleus_mean_bg / cytoplasm_mean_bg is the primary mislocalization readout."""),

    code("""\
%matplotlib inline
from pathlib import Path
import numpy as np, pandas as pd, tifffile
import matplotlib.pyplot as plt, matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap
from skimage import morphology, filters
from skimage.morphology import disk, erosion, dilation, label as sklabel
from skimage.measure import regionprops
from skimage.filters import laplace
from scipy.stats import norm

REPO    = Path('/Users/pmihack/claire/hs-array')
UUID    = '5c6bf125-6e0f-41be-a447-b03ff294c9cc'
RUN_DIR = REPO / 'outputs/run_20260806_141727_cyq9'
NUC_PAR = RUN_DIR / f'nuclei_features/{UUID}_nuclei_features.parquet'
TDP_PAR = REPO / f'outputs/tdp43_localization/{UUID}_tdp43_localization_20260806.parquet'
IMG_DIR = REPO / 'data/D28_AB_Rep1' / UUID / 'images'
FIGDIR  = Path('/Users/pmihack/claire/hs-array/resources/iNDI_microscopy_production/notebooks/figures/phase_C')
FIGDIR.mkdir(parents=True, exist_ok=True)

PX_UM = 0.2967; DPI = 150; BG = 'black'; FG = 'white'

# Compartment radii (from 3_tdp43_localization.py)
R_NUC_UM  = 4.5   # µm
R_ADJ_UM  = 9.0   # µm
R_CYTO_UM = 18.0  # µm
R_NUC  = R_NUC_UM  / PX_UM
R_ADJ  = R_ADJ_UM  / PX_UM
R_CYTO = R_CYTO_UM / PX_UM

print(f"Radii: nucleus={R_NUC:.1f}px ({R_NUC_UM}µm)  adj={R_ADJ:.1f}px ({R_ADJ_UM}µm)  cyto={R_CYTO:.1f}px ({R_CYTO_UM}µm)")
print(f"Ring width: {R_ADJ-R_NUC:.1f}px ({R_ADJ_UM-R_NUC_UM:.1f}µm)")
print(f"Cyto width: {R_CYTO-R_ADJ:.1f}px ({R_CYTO_UM-R_ADJ_UM:.1f}µm)")

def build_masks(cy, cx, others_yx, r_nuc, r_adj, r_cyto, y0, y1, x0, x1):
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy - cy)**2 + (xx - cx)**2
    nuc  = d2 <= r_nuc**2
    adj  = (d2 > r_nuc**2) & (d2 <= r_adj**2)
    cyto = (d2 > r_adj**2) & (d2 <= r_cyto**2)
    if len(others_yx):
        other = np.zeros_like(nuc)
        for oy, ox in others_yx:
            other |= (yy-oy)**2 + (xx-ox)**2 <= r_nuc**2
        adj  &= ~other
        cyto &= ~other
    return nuc, adj, cyto
"""),

    md("## Step 6 — Nucleus Mask (Eroded)"),

    code("""\
# Load nuclei features and pick one site
df_nuc = pd.read_parquet(NUC_PAR)
df_tdp = pd.read_parquet(TDP_PAR)

# Use well with good density, pick a frame
site_df = df_nuc[(df_nuc.Row==7) & (df_nuc.Column==10) & (df_nuc.Frame==5)].copy()
if len(site_df) < 5:
    # fallback: pick any dense site
    site_counts = df_nuc.groupby(['Row','Column','Frame']).size()
    rc = site_counts.idxmax()
    site_df = df_nuc[(df_nuc.Row==rc[0]) & (df_nuc.Column==rc[1]) & (df_nuc.Frame==rc[2])].copy()
print(f"Demo site: r{site_df.Row.iloc[0]:02d}c{site_df.Column.iloc[0]:02d} f{site_df.Frame.iloc[0]:02d}  ({len(site_df)} nuclei)")

# Load TDP-43 plane for this site
row, col, frame = int(site_df.Row.iloc[0]), int(site_df.Column.iloc[0]), int(site_df.Frame.iloc[0])
tdp_subdir = IMG_DIR / f'r{row:02d}c{col:02d}'
tdp_paths  = sorted(tdp_subdir.glob(f'r{row:02d}c{col:02d}f{frame:02d}p*-ch03t01.tiff'))

if tdp_paths:
    tdp_planes = [tifffile.imread(p).astype(np.float32) for p in tdp_paths]
    focus_scores = [np.var(laplace(p)) for p in tdp_planes]
    best_z = int(np.argmax(focus_scores))
    tdp_img = tdp_planes[best_z]
    print(f"TDP-43 best-focus plane: z={best_z+1}")
else:
    # Fallback: load DAPI z-stack best focus plane
    dapi_paths = sorted(tdp_subdir.glob(f'r{row:02d}c{col:02d}f{frame:02d}p*-ch01t01.tiff'))
    planes = [tifffile.imread(p).astype(np.float32) for p in dapi_paths]
    focus_scores = [np.var(laplace(p)) for p in planes]
    tdp_img = planes[int(np.argmax(focus_scores))]
    print("WARNING: using DAPI as fallback for TDP-43 channel")

# Pick one nucleus for the erosion demo
demo_nuc = site_df.sort_values('area', ascending=False).iloc[3]
cy, cx = float(demo_nuc['centroid-0']), float(demo_nuc['centroid-1'])
R = int(np.ceil(R_CYTO)) + 5

# Build actual nucleus mask from the parquet (re-segment a 128px crop)
CROP = 64  # px half-width for close-up
y0c, y1c = int(max(0, cy-CROP)), int(min(tdp_img.shape[0], cy+CROP))
x0c, x1c = int(max(0, cx-CROP)), int(min(tdp_img.shape[1], cx+CROP))

# For demo: build a disk mask at the nucleus centroid
R_NUC_INT = int(round(R_NUC))
nuc_mask_full = np.zeros(tdp_img.shape, bool)
nuc_mask_full[int(cy), int(cx)] = True
nuc_mask_full = dilation(nuc_mask_full, disk(R_NUC_INT))

# Erode by 2px
nuc_eroded = erosion(nuc_mask_full, disk(2))
area_orig   = nuc_mask_full.sum()
area_eroded = nuc_eroded.sum()

# Show close-up crop
fig, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor=BG, dpi=DPI)
fig.suptitle(f'Step 6: Nucleus mask (disk r={R_NUC:.0f}px = {R_NUC_UM}µm) and erosion by 2px',
             color=FG, y=1.01)
plt.subplots_adjust(wspace=0.05)
for ax in axs:
    ax.set_facecolor(BG); ax.axis('off')
    for s in ax.spines.values(): s.set_color(FG)

tdp_crop = tdp_img[y0c:y1c, x0c:x1c].astype(np.float32)
nuc_crop = nuc_mask_full[y0c:y1c, x0c:x1c]
ero_crop = nuc_eroded[y0c:y1c, x0c:x1c]
vmin, vmax = np.percentile(tdp_crop, [1, 99.5])

axs[0].imshow(tdp_crop, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
axs[0].set_title(f'TDP-43 best-focus crop (128px)', color=FG, fontsize=9)

axs[1].imshow(tdp_crop, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
ov1 = np.zeros((*tdp_crop.shape, 4), dtype=np.float32)
ov1[nuc_crop] = [0, 1, 1, 0.4]  # cyan fill
axs[1].imshow(ov1, aspect='auto', interpolation='none')
axs[1].set_title(f'Original nucleus disk\\n{area_orig} px² = {area_orig*PX_UM**2:.0f} µm²', color=FG, fontsize=9)

axs[2].imshow(tdp_crop, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
ov2 = np.zeros((*tdp_crop.shape, 4), dtype=np.float32)
ov2[ero_crop] = [0, 0.8, 1, 0.4]
ov2[nuc_crop & ~ero_crop] = [0.5, 0.5, 0, 0.5]  # eroded ring = orange
axs[2].imshow(ov2, aspect='auto', interpolation='none')
axs[2].set_title(f'Eroded by 2px\\n{area_eroded} px² = {area_eroded*PX_UM**2:.0f} µm²  '
                 f'(−{area_orig-area_eroded} px²)', color=FG, fontsize=9)

fig.savefig(FIGDIR / '01_nucleus_erosion.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print(f"Area loss from erosion: {area_orig-area_eroded} px² ({(1-area_eroded/area_orig)*100:.1f}%)")
"""),

    md("## Step 7 — Nuclear-Adjacent Ring  |  Step 8 — Cytoplasm Annulus"),

    code("""\
# Build all three compartments for the demo cell
y_c, x_c = int(round(cy)), int(round(cx))
R_full = int(np.ceil(R_CYTO)) + 5
y0, y1 = max(0, y_c-R_full), min(tdp_img.shape[0], y_c+R_full)
x0, x1 = max(0, x_c-R_full), min(tdp_img.shape[1], x_c+R_full)

# Get other nuclei centroids in this site (for neighbour exclusion)
others = site_df[site_df.label != demo_nuc.label][['centroid-0','centroid-1']].values.tolist()
others_yx = [(float(oy), float(ox)) for oy, ox in others]

nuc_m, adj_m, cyto_m = build_masks(cy, cx, others_yx, R_NUC, R_ADJ, R_CYTO, y0, y1, x0, x1)
tdp_tile = tdp_img[y0:y1, x0:x1]

# ── Three-zone figure
fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG, dpi=DPI)
fig.suptitle(f'Three TDP-43 compartments (well r{row:02d}c{col:02d} f{frame:02d})', color=FG, y=1.01)
plt.subplots_adjust(wspace=0.05)
for ax in axs: ax.set_facecolor(BG); ax.axis('off'); [s.set_color(FG) for s in ax.spines.values()]

vmin, vmax = np.percentile(tdp_tile, [1, 99.5])
K2G = LinearSegmentedColormap.from_list('k2g', ['black','lime'])

# Panel 1: nucleus zone
axs[0].imshow(tdp_tile, cmap=K2G, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
ov = np.zeros((*tdp_tile.shape, 4), dtype=np.float32)
ov[nuc_m] = [0, 1, 1, 0.5]   # cyan = nucleus
axs[0].imshow(ov, aspect='auto', interpolation='none')
axs[0].set_title(f'Step 6: Nucleus disk\\nr={R_NUC:.0f}px = {R_NUC_UM}µm', color='cyan', fontsize=9)
axs[0].text(5, 15, f'nucleus: r={R_NUC_UM}µm', color='cyan', fontsize=8)

# Panel 2: nucleus + ring
axs[1].imshow(tdp_tile, cmap=K2G, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
ov2 = np.zeros((*tdp_tile.shape, 4), dtype=np.float32)
ov2[nuc_m] = [0, 1, 1, 0.5]   # cyan
ov2[adj_m] = [1, 1, 0, 0.5]   # yellow = ring
axs[1].imshow(ov2, aspect='auto', interpolation='none')
axs[1].set_title(f'Step 7: Nuclear-adjacent ring\\n{R_NUC_UM}–{R_ADJ_UM}µm  = {R_ADJ_UM-R_NUC_UM}µm wide',
                 color='yellow', fontsize=9)

# Panel 3: all three zones
axs[2].imshow(tdp_tile, cmap=K2G, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
ov3 = np.zeros((*tdp_tile.shape, 4), dtype=np.float32)
ov3[nuc_m]  = [0, 1, 1, 0.5]   # cyan
ov3[adj_m]  = [1, 1, 0, 0.5]   # yellow
ov3[cyto_m] = [1, 0, 1, 0.35]  # magenta
axs[2].imshow(ov3, aspect='auto', interpolation='none')
axs[2].set_title(f'Step 8: All 3 zones\\ncyto: {R_ADJ_UM}–{R_CYTO_UM}µm annulus', color='magenta', fontsize=9)

# Legend for panel 3
import matplotlib.patches as mp
patches = [mp.Patch(color='cyan', label=f'nucleus ({R_NUC_UM}µm r)'),
           mp.Patch(color='yellow', label=f'ring ({R_NUC_UM}–{R_ADJ_UM}µm)'),
           mp.Patch(color='magenta', label=f'cytoplasm ({R_ADJ_UM}–{R_CYTO_UM}µm)')]
axs[2].legend(handles=patches, loc='lower right', fontsize=7, labelcolor=FG,
              facecolor='#111', framealpha=0.8)

fig.savefig(FIGDIR / '02_three_compartments.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()

# Print px dimensions
print(f"Compartment dimensions:")
print(f"  nucleus:    r={R_NUC:.1f}px ({R_NUC_UM}µm)  area={nuc_m.sum()} px²")
print(f"  adj ring:   {R_NUC:.1f}–{R_ADJ:.1f}px ({R_NUC_UM}–{R_ADJ_UM}µm)  area={adj_m.sum()} px²")
print(f"  cytoplasm:  {R_ADJ:.1f}–{R_CYTO:.1f}px ({R_ADJ_UM}–{R_CYTO_UM}µm)  area={cyto_m.sum()} px²")
"""),

    md("## Neighbour Nucleus Exclusion"),

    code("""\
# Show the problem and solution: two nearby cells, cytoplasm rings overlap

# Find two nearby nuclei
site_df_sorted = site_df.sort_values('area', ascending=False).reset_index(drop=True)
# Pick two nuclei close together
from scipy.spatial import distance_matrix
centroids = site_df_sorted[['centroid-0','centroid-1']].values
dm = distance_matrix(centroids, centroids)
np.fill_diagonal(dm, 999)
min_pair = np.unravel_index(dm.argmin(), dm.shape)
i1, i2 = min_pair
n1 = site_df_sorted.iloc[i1]
n2 = site_df_sorted.iloc[i2]
print(f"Pair: nucleus {i1} centroid=({n1['centroid-0']:.0f},{n1['centroid-1']:.0f})  "
      f"nucleus {i2} centroid=({n2['centroid-0']:.0f},{n2['centroid-1']:.0f})  "
      f"dist={dm[i1,i2]:.0f}px = {dm[i1,i2]*PX_UM:.1f}µm")

# Build crops around n1 with and without exclusion
cy1, cx1 = float(n1['centroid-0']), float(n1['centroid-1'])
cy2, cx2 = float(n2['centroid-0']), float(n2['centroid-1'])
R_full = int(np.ceil(R_CYTO)) + 5
y0, y1 = max(0, int(cy1)-R_full), min(tdp_img.shape[0], int(cy1)+R_full)
x0, x1 = max(0, int(cx1)-R_full), min(tdp_img.shape[1], int(cx1)+R_full)
tile = tdp_img[y0:y1, x0:x1]

# Without exclusion (pass empty others list)
nuc_no, adj_no, cyto_no = build_masks(cy1, cx1, [], R_NUC, R_ADJ, R_CYTO, y0, y1, x0, x1)

# With exclusion (n2 is a neighbour)
nuc_ex, adj_ex, cyto_ex = build_masks(cy1, cx1, [(cy2, cx2)], R_NUC, R_ADJ, R_CYTO, y0, y1, x0, x1)

vmin, vmax = np.percentile(tile, [1, 99.5])
K2G = LinearSegmentedColormap.from_list('k2g', ['black','lime'])

fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor=BG, dpi=DPI)
fig.suptitle('Neighbour nucleus exclusion from cytoplasm annulus', color=FG, y=1.01)
for ax in axs: ax.set_facecolor(BG); ax.axis('off'); [s.set_color(FG) for s in ax.spines.values()]

def _overlay(tile, nuc_m, adj_m, cyto_m):
    ov = np.zeros((*tile.shape, 4), dtype=np.float32)
    ov[nuc_m]  = [0, 1, 1, 0.5]
    ov[adj_m]  = [1, 1, 0, 0.4]
    ov[cyto_m] = [1, 0, 1, 0.3]
    return ov

axs[0].imshow(tile, cmap=K2G, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
axs[0].imshow(_overlay(tile, nuc_no, adj_no, cyto_no), aspect='auto', interpolation='none')
axs[0].set_title(f'WITHOUT exclusion\\ncyto={cyto_no.sum()}px² incl. neighbour nucleus footprint',
                 color=FG, fontsize=9)

axs[1].imshow(tile, cmap=K2G, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
axs[1].imshow(_overlay(tile, nuc_ex, adj_ex, cyto_ex), aspect='auto', interpolation='none')
axs[1].set_title(f'WITH exclusion (neighbour punched out)\\ncyto={cyto_ex.sum()}px²  '
                 f'(−{cyto_no.sum()-cyto_ex.sum()}px²)', color=FG, fontsize=9)

# Mark second nucleus centroid
lcy2, lcx2 = cy2-y0, cx2-x0
for ax in axs:
    ax.plot(lcx2, lcy2, 'r+', ms=12, mew=2, label='neighbour centroid')
    ax.legend(fontsize=7, labelcolor=FG, facecolor='#111')

fig.savefig(FIGDIR / '03_neighbour_exclusion.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print(f"Exclusion removes {cyto_no.sum()-cyto_ex.sum()} px² from cytoplasm annulus")
print(f"Without exclusion: neighbour nucleus {i2} leaks TDP-43 into cell {i1}'s cytoplasm → suppressed N:C")
"""),

    md("## Full Compartment Overlay on TDP-43 — Representative Cells"),

    code("""\
# Pick 6 cells with a range of NC_ratio (mix of high and low)
# Use TDP-43 parquet to get NC_ratio for this site
tdp_site = df_tdp[(df_tdp.Row==row) & (df_tdp.Column==col) & (df_tdp.Frame==frame)].copy()
tdp_site = tdp_site[(tdp_site.NC_ratio > 0) & (tdp_site.NC_ratio < 50)].sort_values('NC_ratio')
print(f"Site TDP-43 cells: {len(tdp_site)}  NC_ratio range: {tdp_site.NC_ratio.min():.2f}–{tdp_site.NC_ratio.max():.2f}")

if len(tdp_site) < 6:
    # fallback to any site with enough cells
    site_counts_tdp = df_tdp.groupby(['Row','Column','Frame']).size()
    rc_tdp = site_counts_tdp.idxmax()
    row, col, frame = rc_tdp
    tdp_subdir = IMG_DIR / f'r{row:02d}c{col:02d}'
    tdp_paths = sorted(tdp_subdir.glob(f'r{row:02d}c{col:02d}f{frame:02d}p*-ch03t01.tiff'))
    if tdp_paths:
        tdp_planes = [tifffile.imread(p).astype(np.float32) for p in tdp_paths]
        focus_scores = [np.var(laplace(p)) for p in tdp_planes]
        best_z = int(np.argmax(focus_scores))
        tdp_img = tdp_planes[best_z]
    tdp_site = df_tdp[(df_tdp.Row==row) & (df_tdp.Column==col) & (df_tdp.Frame==frame)]
    tdp_site = tdp_site[(tdp_site.NC_ratio>0)&(tdp_site.NC_ratio<50)].sort_values('NC_ratio')

# Pick 6 cells: 3 low, 3 high NC
n_pick = min(6, len(tdp_site))
picks = pd.concat([tdp_site.head(n_pick//2), tdp_site.tail(n_pick - n_pick//2)])
picks = picks.drop_duplicates().reset_index(drop=True)
print(f"Showing {len(picks)} cells: NC_ratio = {picks.NC_ratio.values}")

all_others = df_tdp[(df_tdp.Row==row)&(df_tdp.Column==col)&(df_tdp.Frame==frame)][['centroid-0','centroid-1']].values

CROP2 = 64  # px half-width
K2G   = LinearSegmentedColormap.from_list('k2g', ['black','lime'])
n_rows = len(picks)

fig, axs = plt.subplots(n_rows, 3, figsize=(9, n_rows*3+1), facecolor=BG, dpi=DPI)
if n_rows == 1: axs = axs[np.newaxis, :]
fig.suptitle(f'TDP-43 compartments: r{row:02d}c{col:02d} f{frame:02d}', color=FG, y=1.01)
plt.subplots_adjust(hspace=0.05, wspace=0.04)

col_titles = ['Raw TDP-43', 'Zones overlaid', 'Zone labels']

for ri, (_, cell_row) in enumerate(picks.iterrows()):
    cy_c = float(cell_row['centroid-0']); cx_c = float(cell_row['centroid-1'])
    nc   = float(cell_row['NC_ratio'])
    y0c, y1c = int(max(0, cy_c-CROP2)), int(min(tdp_img.shape[0], cy_c+CROP2))
    x0c, x1c = int(max(0, cx_c-CROP2)), int(min(tdp_img.shape[1], cx_c+CROP2))
    crop = tdp_img[y0c:y1c, x0c:x1c]

    others_yx = [(float(oy), float(ox)) for oy, ox in all_others
                 if abs(oy-cy_c) < (R_CYTO+R_NUC) and abs(ox-cx_c) < (R_CYTO+R_NUC)]

    nuc_m, adj_m, cyto_m = build_masks(cy_c, cx_c, others_yx, R_NUC, R_ADJ, R_CYTO, y0c, y1c, x0c, x1c)
    vmin, vmax = np.percentile(crop, [1, 99.5])

    for ci in range(3):
        ax = axs[ri, ci]
        ax.set_facecolor(BG); ax.axis('off')
        for s in ax.spines.values(): s.set_color(FG)
        if ri == 0: ax.set_title(col_titles[ci], color=FG, fontsize=9)

        ax.imshow(crop, cmap=K2G, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
        if ci == 1:
            ov = np.zeros((*crop.shape, 4), dtype=np.float32)
            ov[nuc_m] = [0,1,1,0.5]; ov[adj_m] = [1,1,0,0.4]; ov[cyto_m] = [1,0,1,0.3]
            ax.imshow(ov, aspect='auto', interpolation='none')
        elif ci == 2:
            zone_img = np.zeros((*crop.shape, 3), dtype=np.float32)
            zone_img[nuc_m]  = [0, 1, 1]    # cyan
            zone_img[adj_m]  = [1, 1, 0]    # yellow
            zone_img[cyto_m] = [1, 0, 1]    # magenta
            ax.imshow(zone_img, aspect='auto', interpolation='none')

    lbl = f"N:C={nc:.2f}"
    col_lbl = '#ff4444' if nc < 4 else '#88ff88'
    axs[ri, 0].set_ylabel(lbl, color=col_lbl, fontsize=9, rotation=0, labelpad=55, va='center')

fig.savefig(FIGDIR / '04_compartment_overlay_cells.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: 04_compartment_overlay_cells.png")
"""),

    code("""\
# Verify figures
pngs = sorted(FIGDIR.glob('*.png'))
empty = [p for p in pngs if p.stat().st_size == 0]
assert pngs,  f"No figures written to {FIGDIR}"
assert not empty, f"0-byte figures: {empty}"
print(f"OK: {len(pngs)} figures in {FIGDIR.name}/")
for p in pngs:
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# PHASE D
# ─────────────────────────────────────────────────────────────────────────────

PHASE_D_CELLS = [
    md("""\
# Phase D — TDP-43 Measurement (N:C Ratio)

**Dataset:** WARD00034-R2 (`5c6bf125`), 249,465 cells
**Readout:** Nuclear-to-cytoplasmic ratio = nucleus_mean_bg / cytoplasm_mean_bg
**Interpretation:** High N:C = healthy nuclear TDP-43; low N:C = cytoplasmic mislocalization

Sources: `3_tdp43_localization.py` — three concentric disk compartments (r=4.5, 9, 18 µm)"""),

    code("""\
%matplotlib inline
from pathlib import Path
import numpy as np, pandas as pd, tifffile
import matplotlib.pyplot as plt, matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap
from skimage.filters import laplace

REPO    = Path('/Users/pmihack/claire/hs-array')
UUID    = '5c6bf125-6e0f-41be-a447-b03ff294c9cc'
TDP_PAR = REPO / f'outputs/tdp43_localization/{UUID}_tdp43_localization_20260806.parquet'
PMAPCSV = REPO / 'resources/plate_maps/remap_d28_rep1.csv'
IMG_DIR = REPO / 'data/D28_AB_Rep1' / UUID / 'images'
FIGDIR  = Path('/Users/pmihack/claire/hs-array/resources/iNDI_microscopy_production/notebooks/figures/phase_D')
FIGDIR.mkdir(parents=True, exist_ok=True)

PX_UM = 0.2967; DPI = 150; BG = 'black'; FG = 'white'
R_NUC_UM = 4.5; R_ADJ_UM = 9.0; R_CYTO_UM = 18.0
R_NUC = R_NUC_UM / PX_UM; R_ADJ = R_ADJ_UM / PX_UM; R_CYTO = R_CYTO_UM / PX_UM
"""),

    md("## Setup — Load TDP-43 Localization Parquet"),

    code("""\
tdp = pd.read_parquet(TDP_PAR)
print(f"Columns:  {list(tdp.columns)}")
print(f"n_cells:  {len(tdp):,}")
print(f"n_wells:  {tdp[['Row','Column']].drop_duplicates().shape[0]}")
print()
print("NC_ratio stats (raw):")
print(tdp['NC_ratio'].describe().to_string())

# Join plate map
pmap = pd.read_csv(PMAPCSV)
pmap['Row']    = pmap.Well.str[0].apply(lambda x: ord(x)-64)
pmap['Column'] = pmap.Well.str[1:].astype(int)
pmap1 = pmap[pmap.Plate == 1][['Row','Column','Gene']]
tdp   = tdp.merge(pmap1, on=['Row','Column'], how='left')
print(f"\\nGenes in plate map: {tdp['Gene'].nunique()}")
print(f"NT wells: {tdp[tdp.Gene.str.startswith('NT',na=False)]['Gene'].nunique()} variants")

# Filter to reasonable NC range
tdp_filt = tdp[(tdp.NC_ratio > 0) & (tdp.NC_ratio < 50)].copy()
print(f"\\nAfter filtering NC_ratio 0–50: {len(tdp_filt):,} cells ({len(tdp_filt)/len(tdp)*100:.1f}%)")
"""),

    md("## Step 9 — Per-Compartment Intensity Measurement"),

    code("""\
# Show measurement logic using 3_tdp43_localization.py notation
print("Measurement logic (from 3_tdp43_localization.py):")
print()
print("  bg = np.percentile(tdp_plane, BG_PERCENTILE=5)  # 5th-percentile background")
print("  nuc_mean_bg  = nuc_pixels.mean()  - bg")
print("  cyto_mean_bg = cyto_pixels.mean() - bg")
print("  NC_ratio = nuc_mean_bg / cyto_mean_bg")
print()

# Pick one example cell with known NC_ratio
ex = tdp_filt[(tdp_filt.NC_ratio > 4) & (tdp_filt.NC_ratio < 7)].iloc[0]
print(f"Example cell: r{int(ex.Row):02d}c{int(ex.Column):02d} f{int(ex.Frame):02d}  "
      f"Nucleus_ID={int(ex.Nucleus_ID)}")
print(f"  bg_tdp:            {ex.bg_tdp:.1f}")
print(f"  tdp_nuc_mean:      {ex.tdp_nuc_mean:.1f}  (raw) → bg-subtracted: {ex.tdp_nuc_mean_bg:.1f}")
print(f"  tdp_adj_mean:      {ex.tdp_adj_mean:.1f}  → bg-subtracted: {ex.tdp_adj_mean_bg:.1f}")
print(f"  tdp_cyto_mean:     {ex.tdp_cyto_mean:.1f} → bg-subtracted: {ex.tdp_cyto_mean_bg:.1f}")
print(f"  NC_ratio:          {ex.NC_ratio:.3f}")
print(f"  adj_nuc_ratio:     {ex.adj_nuc_ratio:.3f}")

# Bar chart: per-compartment intensities for this cell
fig, axs = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG, dpi=DPI)
fig.suptitle(f'Step 9: Per-compartment TDP-43 intensity\\n'
             f'Cell r{int(ex.Row):02d}c{int(ex.Column):02d} f{int(ex.Frame):02d}  N:C={ex.NC_ratio:.2f}',
             color=FG, y=1.02)
for ax in axs:
    ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    for s in ax.spines.values(): s.set_color(FG)

compartments = ['Nucleus', 'Adj ring', 'Cytoplasm']
raw_means = [ex.tdp_nuc_mean, ex.tdp_adj_mean, ex.tdp_cyto_mean]
bg_means  = [ex.tdp_nuc_mean_bg, ex.tdp_adj_mean_bg, ex.tdp_cyto_mean_bg]
stds      = [ex.tdp_nuc_std, ex.tdp_adj_std, ex.tdp_cyto_std]
colors    = ['cyan', 'yellow', 'magenta']

axs[0].bar(compartments, raw_means, color=colors, alpha=0.85, edgecolor='none')
axs[0].axhline(ex.bg_tdp, color='white', ls='--', lw=1.5, label=f'bg={ex.bg_tdp:.0f}')
axs[0].set_title('Raw mean intensity per compartment', color=FG, fontsize=9)
axs[0].set_ylabel('intensity (ADU)', fontsize=9)
axs[0].legend(fontsize=8, labelcolor=FG, facecolor='#222')

axs[1].bar(compartments, bg_means, color=colors, alpha=0.85, edgecolor='none',
           yerr=[s/np.sqrt(n) for s, n in [(ex.tdp_nuc_std, ex.tdp_nuc_n),
                                             (ex.tdp_adj_std, ex.tdp_adj_n),
                                             (ex.tdp_cyto_std, ex.tdp_cyto_n)]],
           capsize=5, error_kw={'ecolor':'white', 'lw':1.5})
axs[1].set_title(f'Background-subtracted mean\\nNC_ratio = nuc/cyto = {ex.NC_ratio:.2f}', color=FG, fontsize=9)
axs[1].set_ylabel('bg-subtracted intensity (ADU)', fontsize=9)
for i, (comp, val) in enumerate(zip(compartments, bg_means)):
    axs[1].text(i, val+50, f'{val:.0f}', ha='center', va='bottom', color=FG, fontsize=9, fontweight='bold')

fig.savefig(FIGDIR / '01_per_compartment_intensity.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: 01_per_compartment_intensity.png")
"""),

    md("## Step 10 — Derived Readouts: N:C Ratio Distribution"),

    code("""\
# N:C ratio distribution, colored by NT vs other
nt_mask   = tdp_filt.Gene.str.startswith('NT', na=False)
nt_df     = tdp_filt[nt_mask]
gene_df   = tdp_filt[~nt_mask]
nt_median = nt_df.NC_ratio.median()

fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG, dpi=DPI)
fig.suptitle('Step 10: N:C Ratio Distribution — WARD00034 R2', color=FG, y=1.01)
for ax in axs:
    ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    for s in ax.spines.values(): s.set_color(FG)

bins = np.linspace(0, 30, 80)
axs[0].hist(gene_df.NC_ratio, bins=bins, color='steelblue', alpha=0.7, edgecolor='none', label=f'Gene wells (n={len(gene_df):,})')
axs[0].hist(nt_df.NC_ratio,   bins=bins, color='lime',      alpha=0.85, edgecolor='none', label=f'NT wells (n={len(nt_df):,})')
axs[0].axvline(nt_median, color='lime', ls='--', lw=2, label=f'NT median={nt_median:.2f}')
axs[0].set_xlabel('N:C ratio', fontsize=9); axs[0].set_ylabel('cells', fontsize=9)
axs[0].set_title('N:C ratio distribution (NC < 50)', color=FG, fontsize=9)
axs[0].legend(fontsize=8, labelcolor=FG, facecolor='#222')

# Cytoplasm mean intensity distribution
axs[1].hist(tdp_filt.tdp_cyto_mean_bg.clip(-500, 5000), bins=80, color='magenta', alpha=0.8, edgecolor='none')
axs[1].set_xlabel('cytoplasm mean bg-subtracted intensity (ADU)', fontsize=9)
axs[1].set_ylabel('cells', fontsize=9)
axs[1].set_title('Cytoplasm TDP-43 intensity (bg-subtracted)', color=FG, fontsize=9)

fig.savefig(FIGDIR / '02_nc_ratio_distribution.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print(f"NT median N:C: {nt_median:.2f}")
print(f"Overall median N:C: {tdp_filt.NC_ratio.median():.2f}")
"""),

    md("## Step 11 — Background Subtraction"),

    code("""\
# Demonstrate 5th-percentile background estimate on one TDP-43 field
# Use a well with both NT label
nt_well = nt_df.groupby(['Row','Column']).size().idxmax()
row_bg, col_bg = int(nt_well[0]), int(nt_well[1])
subdir_bg = IMG_DIR / f'r{row_bg:02d}c{col_bg:02d}'
tdp_paths_bg = sorted(subdir_bg.glob(f'r{row_bg:02d}c{col_bg:02d}f05p*-ch03t01.tiff'))

if not tdp_paths_bg:
    tdp_paths_bg = sorted(subdir_bg.glob(f'r{row_bg:02d}c{col_bg:02d}f01p*-ch03t01.tiff'))

if tdp_paths_bg:
    planes_bg = [tifffile.imread(p).astype(np.float32) for p in tdp_paths_bg]
    focus_bg  = [np.var(laplace(p)) for p in planes_bg]
    tdp_plane = planes_bg[int(np.argmax(focus_bg))]
else:
    tdp_plane = np.random.randint(100, 2000, (2160, 2160), dtype=np.uint16).astype(np.float32)

bg_5pct   = np.percentile(tdp_plane, 5)
bg_mean   = float(tdp_plane.mean())
bg_median = float(np.median(tdp_plane))

print(f"TDP-43 field r{row_bg:02d}c{col_bg:02d} f05:")
print(f"  5th-percentile background: {bg_5pct:.1f} ADU  ← used in script")
print(f"  Mean:                      {bg_mean:.1f} ADU  (biased high by bright cells)")
print(f"  Median:                    {bg_median:.1f} ADU  (less biased, but still affected)")
print(f"  Max (99th pct):            {np.percentile(tdp_plane, 99):.1f} ADU")

fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG, dpi=DPI)
fig.suptitle(f'Step 11: Background estimation — r{row_bg:02d}c{col_bg:02d} f05 TDP-43 plane', color=FG, y=1.01)
for ax in axs:
    ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    for s in ax.spines.values(): s.set_color(FG)
plt.subplots_adjust(wspace=0.25)

K2G = LinearSegmentedColormap.from_list('k2g', ['black','lime'])
vmin, vmax = np.percentile(tdp_plane, [0.5, 99.5])
axs[0].imshow(tdp_plane, cmap=K2G, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
axs[0].set_title(f'TDP-43 best-focus plane\\n5th-pct bg = {bg_5pct:.0f}', color=FG, fontsize=9)
axs[0].axis('off')
for s in axs[0].spines.values(): s.set_color(FG)

# Histogram with background lines
hist_vals = tdp_plane.ravel()
axs[1].hist(hist_vals[hist_vals < vmax], bins=100, color='steelblue', alpha=0.8, edgecolor='none',
            log=True)
axs[1].axvline(bg_5pct,   color='lime',   ls='-',  lw=2, label=f'5th-pct bg = {bg_5pct:.0f}')
axs[1].axvline(bg_mean,   color='orange', ls='--', lw=2, label=f'mean = {bg_mean:.0f}')
axs[1].axvline(bg_median, color='yellow', ls=':',  lw=2, label=f'median = {bg_median:.0f}')
axs[1].set_xlabel('intensity (ADU)', fontsize=9); axs[1].set_ylabel('pixels (log)', fontsize=9)
axs[1].set_title('Intensity histogram\\n5th-pct avoids bright-cell bias', color=FG, fontsize=9)
axs[1].legend(fontsize=7, labelcolor=FG, facecolor='#222')

# Bar: background-subtracted vs raw per compartment for the example cell
site_ex = tdp_filt[(tdp_filt.Row==row_bg) & (tdp_filt.Column==col_bg)].head(1).iloc[0]
comp_names = ['Nucleus', 'Adj ring', 'Cytoplasm']
raw_i  = [site_ex.tdp_nuc_mean, site_ex.tdp_adj_mean, site_ex.tdp_cyto_mean]
sub_i  = [site_ex.tdp_nuc_mean_bg, site_ex.tdp_adj_mean_bg, site_ex.tdp_cyto_mean_bg]
x = np.arange(3); w = 0.35
bars1 = axs[2].bar(x-w/2, raw_i, w, color='gray',  alpha=0.85, edgecolor='none', label='raw mean')
bars2 = axs[2].bar(x+w/2, sub_i, w, color='steelblue', alpha=0.85, edgecolor='none', label='bg-subtracted')
axs[2].axhline(site_ex.bg_tdp, color='red', ls='--', lw=1.5, label=f'bg={site_ex.bg_tdp:.0f}')
axs[2].set_xticks(x); axs[2].set_xticklabels(comp_names, fontsize=8)
axs[2].set_title(f'Raw vs bg-subtracted means\\nNC_ratio={site_ex.NC_ratio:.2f}', color=FG, fontsize=9)
axs[2].set_ylabel('intensity (ADU)', fontsize=9)
axs[2].legend(fontsize=7, labelcolor=FG, facecolor='#222')

fig.savefig(FIGDIR / '03_background_subtraction.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: 03_background_subtraction.png")
"""),

    md("## Per-Gene Summary"),

    code("""\
# Per-gene median N:C, sorted descending, NT highlighted
gene_stats = (tdp_filt
              .groupby('Gene')['NC_ratio']
              .agg(['median','count','std'])
              .query('count >= 50')
              .sort_values('median', ascending=False)
              .reset_index())

nt_genes   = gene_stats[gene_stats.Gene.str.startswith('NT')]
gene_stats2= gene_stats[~gene_stats.Gene.str.startswith('NT')].sort_values('median', ascending=False)

nt_median_all = nt_genes['median'].median()
print(f"Genes with ≥50 cells: {len(gene_stats)}")
print(f"NT variants: {len(nt_genes)}, pooled median NC = {nt_median_all:.2f}")
print()
print("Top-10 highest N:C (nuclear):")
print(gene_stats.head(10)[['Gene','median','count']].to_string(index=False))
print()
print("Top-10 lowest N:C (cytoplasmic):")
print(gene_stats.tail(10)[['Gene','median','count']].sort_values('median').to_string(index=False))

KNOWN_HITS = ['CSE1L','YPEL5','UBE2H','MAEA','ZC3H13','OGT']

fig, ax = plt.subplots(figsize=(18, 7), facecolor=BG, dpi=DPI)
ax.set_facecolor(BG); ax.tick_params(colors=FG, labelsize=7)
ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
for s in ax.spines.values(): s.set_color(FG)

# Plot gene bars (sorted by median)
gene_sorted = gene_stats2.sort_values('median', ascending=False)
colors_bar = []
for g in gene_sorted.Gene:
    if g in KNOWN_HITS: colors_bar.append('#ff6600')
    else:               colors_bar.append('#3366cc')

ax.bar(range(len(gene_sorted)), gene_sorted['median'], color=colors_bar, alpha=0.85, edgecolor='none', label='gene wells')
# Add error bars (SEM)
errs = gene_sorted['std'] / np.sqrt(gene_sorted['count'])
ax.errorbar(range(len(gene_sorted)), gene_sorted['median'], yerr=errs,
            fmt='none', ecolor='#ffffff55', elinewidth=0.5, capsize=0)

# NT median line
ax.axhline(nt_median_all, color='lime', ls='--', lw=2, label=f'NT median={nt_median_all:.2f}')

# NT bars on same plot
ax.bar(range(len(gene_sorted), len(gene_sorted)+len(nt_genes)),
       nt_genes.sort_values('median', ascending=False)['median'],
       color='lime', alpha=0.6, edgecolor='none', label='NT wells')

# Label known hits
for hit in KNOWN_HITS:
    hit_row = gene_sorted[gene_sorted.Gene == hit]
    if not hit_row.empty:
        idx = hit_row.index[0]
        pos = gene_sorted.index.get_loc(idx)
        ax.text(pos, hit_row['median'].values[0] - 0.3, hit, color='#ff6600',
                fontsize=7, ha='center', va='top', rotation=90)

ax.set_xlabel('Gene (sorted by N:C, descending)', fontsize=9)
ax.set_ylabel('Median N:C ratio', fontsize=9)
ax.set_title('Per-gene median TDP-43 N:C ratio  (WARD00034-R2)\\n'
             'orange = known hits from CLAUDE.md  |  green = NT controls', color=FG, fontsize=9)
ax.set_xticks([]); ax.legend(fontsize=8, labelcolor=FG, facecolor='#222')

fig.savefig(FIGDIR / '04_per_gene_nc_ratio.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: 04_per_gene_nc_ratio.png")
"""),

    md("## Representative Cell Montage — Top-3 Hit Genes vs NT"),

    code("""\
# Load TDP-43 TIFF for each gene's best well and show 3-4 cells per gene
def build_masks_local(cy, cx, others_yx, r_nuc, r_adj, r_cyto, y0, y1, x0, x1):
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy-cy)**2 + (xx-cx)**2
    nuc  = d2 <= r_nuc**2
    adj  = (d2 > r_nuc**2) & (d2 <= r_adj**2)
    cyto = (d2 > r_adj**2) & (d2 <= r_cyto**2)
    if len(others_yx):
        other = np.zeros_like(nuc)
        for oy, ox in others_yx:
            other |= (yy-oy)**2+(xx-ox)**2 <= r_nuc**2
        adj &= ~other; cyto &= ~other
    return nuc, adj, cyto

GENES_MONTAGE = ['CSE1L', 'YPEL5', 'UBE2H', 'NT18']
CELLS_PER_GENE = 4
CROP_MON = 64  # px half-width
K2G = LinearSegmentedColormap.from_list('k2g', ['black','lime'])

fig, axs = plt.subplots(len(GENES_MONTAGE), CELLS_PER_GENE,
                         figsize=(CELLS_PER_GENE*3, len(GENES_MONTAGE)*3.2),
                         facecolor=BG, dpi=DPI)
fig.suptitle('Representative cells: top hits vs NT control\\n(TDP-43 CF568 channel with zone outlines)',
             color=FG, y=1.01)
plt.subplots_adjust(hspace=0.05, wspace=0.04)

for gi, gene in enumerate(GENES_MONTAGE):
    gene_cells = tdp_filt[tdp_filt.Gene == gene].copy()
    # For low-NC genes, pick lowest NC; for NT pick median range
    if gene.startswith('NT'):
        med = gene_cells.NC_ratio.median()
        gene_cells = gene_cells[(gene_cells.NC_ratio > med*0.8) & (gene_cells.NC_ratio < med*1.2)]
    else:
        gene_cells = gene_cells.sort_values('NC_ratio').head(CELLS_PER_GENE * 3)
    gene_cells = gene_cells.drop_duplicates(subset=['Row','Column','Frame']).head(CELLS_PER_GENE)

    # Cache TDP-43 planes per site
    tdp_cache = {}

    for ci in range(CELLS_PER_GENE):
        ax = axs[gi, ci]
        ax.set_facecolor(BG); ax.axis('off')
        for s in ax.spines.values(): s.set_color(FG)

        if ci == 0:
            ax.set_ylabel(gene, color='lime' if gene.startswith('NT') else '#ff6600',
                          fontsize=10, fontweight='bold', rotation=0, labelpad=50, va='center')

        if ci >= len(gene_cells):
            continue

        cell = gene_cells.iloc[ci]
        row_c, col_c, frame_c = int(cell.Row), int(cell.Column), int(cell.Frame)
        nc_c = float(cell.NC_ratio)
        cy_c, cx_c = float(cell['centroid-0']), float(cell['centroid-1'])
        bf   = int(cell.best_focus_plane)

        site_key = (row_c, col_c, frame_c)
        if site_key not in tdp_cache:
            tdir = IMG_DIR / f'r{row_c:02d}c{col_c:02d}'
            tpaths = sorted(tdir.glob(f'r{row_c:02d}c{col_c:02d}f{frame_c:02d}p*-ch03t01.tiff'))
            if tpaths:
                all_planes = [tifffile.imread(p).astype(np.float32) for p in tpaths]
                focus_s = [np.var(laplace(p)) for p in all_planes]
                tdp_cache[site_key] = all_planes[int(np.argmax(focus_s))]
            else:
                tdp_cache[site_key] = None

        tdp_img_m = tdp_cache[site_key]
        if tdp_img_m is None:
            ax.text(0.5, 0.5, 'no TIFF', ha='center', va='center', color=FG,
                    transform=ax.transAxes, fontsize=8)
            continue

        y0c, y1c = int(max(0, cy_c-CROP_MON)), int(min(tdp_img_m.shape[0], cy_c+CROP_MON))
        x0c, x1c = int(max(0, cx_c-CROP_MON)), int(min(tdp_img_m.shape[1], cx_c+CROP_MON))
        crop = tdp_img_m[y0c:y1c, x0c:x1c]

        site_tdp = tdp_filt[(tdp_filt.Row==row_c)&(tdp_filt.Column==col_c)&(tdp_filt.Frame==frame_c)]
        others_yx = [(float(r['centroid-0']), float(r['centroid-1'])) for _, r in site_tdp.iterrows()
                     if abs(r['centroid-0']-cy_c)<(R_CYTO+R_NUC) and abs(r['centroid-1']-cx_c)<(R_CYTO+R_NUC)]

        nuc_m, adj_m, cyto_m = build_masks_local(cy_c, cx_c, others_yx, R_NUC, R_ADJ, R_CYTO, y0c, y1c, x0c, x1c)

        vmin, vmax = np.percentile(crop, [1, 99.5])
        ax.imshow(crop, cmap=K2G, vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')

        ov = np.zeros((*crop.shape, 4), dtype=np.float32)
        ov[nuc_m]  = [0, 1, 1, 0.4]
        ov[adj_m]  = [1, 1, 0, 0.3]
        ov[cyto_m] = [1, 0, 1, 0.25]
        ax.imshow(ov, aspect='auto', interpolation='none')

        nc_col = '#ff4444' if nc_c < nt_median_all * 0.8 else '#ffaa44' if nc_c < nt_median_all else '#88ff88'
        ax.set_title(f'N:C={nc_c:.2f}', color=nc_col, fontsize=8)

fig.savefig(FIGDIR / '05_representative_cells_montage.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: 05_representative_cells_montage.png")
print(f"NT baseline: {nt_median_all:.2f}")
"""),

    code("""\
# Verify figures
pngs = sorted(FIGDIR.glob('*.png'))
empty = [p for p in pngs if p.stat().st_size == 0]
assert pngs,  f"No figures written to {FIGDIR}"
assert not empty, f"0-byte figures: {empty}"
print(f"OK: {len(pngs)} figures in {FIGDIR.name}/")
for p in pngs:
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Write all notebooks
# ─────────────────────────────────────────────────────────────────────────────

notebooks = {
    'phase_A_preprocessing_corrections.ipynb': PHASE_A_CELLS,
    'phase_B_nucleus_segmentation.ipynb':      PHASE_B_CELLS,
    'phase_C_tdp43_compartments.ipynb':        PHASE_C_CELLS,
    'phase_D_tdp43_measurement.ipynb':         PHASE_D_CELLS,
}

for fname, cells in notebooks.items():
    path = NB_DIR / fname
    with open(path, 'w') as f:
        json.dump(nb(cells), f, indent=1)
    print(f"Written: {path}")

print("\\nAll notebooks written.")
