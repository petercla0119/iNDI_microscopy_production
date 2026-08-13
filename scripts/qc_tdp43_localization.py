"""QC + first-look figures for Step-3 TDP-43 localization output.

Joins the per-cell TDP-43 parquet to the plate map (NT vs gene), then writes:
  - nc_distribution.png     overall N:C histogram (log x) + median
  - nc_nt_vs_hits.png       N:C: NT controls vs named mislocalization hits
  - gene_summary.csv        per-gene median N:C, cyto/nuc means, n cells
  - montage_examples.png    3 example cells (low/median/high N:C) with the
                            nucleus / adjacent / cytoplasm compartment rings

Usage:
  python qc_tdp43_localization.py            # newest parquet in output dir
  python qc_tdp43_localization.py <parquet>
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import tifffile

# Must match 3_tdp43_localization.py
PIXEL_SIZE_UM = 0.2967
R_NUC = 4.5 / PIXEL_SIZE_UM
R_ADJ = 9.0 / PIXEL_SIZE_UM
R_CYTO = 18.0 / PIXEL_SIZE_UM

REPO = Path(__file__).resolve().parents[3]
TDP_DIR = REPO / "outputs/tdp43_localization"
META_DIR = REPO / "outputs/image_metadata"
PLATE_MAP = REPO / "resources/d28_rep1_plates_combined.csv"
NAMED_HITS = ["PAXBP1", "CPSF6", "ZC3H13", "YPEL5", "UBE2H", "MAEA",
              "CSE1L", "RAE1", "OGT", "NAA25"]


def well_of(row):
    return f"{chr(64 + int(row['Row']))}{int(row['Column']):02d}"


def load(parquet):
    res = pd.read_parquet(parquet)
    uuid = parquet.stem.split("_tdp43")[0]
    pm = pd.read_csv(PLATE_MAP)
    pmu = pm[pm["Round2_UUID"] == uuid][["Well", "Gene", "WARD_ID"]]
    res["Well"] = res.apply(well_of, axis=1)
    res = res.merge(pmu, on="Well", how="left")
    res["is_NT"] = res["Gene"].astype(str).str.startswith("NT")
    return res, uuid


def fig_distribution(res, out):
    nc = res["NC_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    nc = nc[(nc > 0.05) & (nc < 20)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log10(nc), bins=80, color="#4C72B0", alpha=0.85)
    med = res["NC_ratio"].median()
    ax.axvline(np.log10(med), color="crimson", ls="--", label=f"median = {med:.2f}")
    ax.axvline(0, color="k", lw=0.8, alpha=0.5, label="N:C = 1 (balanced)")
    ax.set_xlabel("log10(TDP-43 nuclear:cytoplasmic ratio)")
    ax.set_ylabel("cells")
    ax.set_title(f"TDP-43 N:C ratio  (n={len(nc):,} cells)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_nt_vs_hits(res, out):
    hits = [g for g in NAMED_HITS if g in set(res["Gene"].dropna())]
    groups = [("NT (all)", res[res["is_NT"]]["NC_ratio"])]
    for g in hits:
        groups.append((g, res[res["Gene"] == g]["NC_ratio"]))
    data = [v.replace([np.inf, -np.inf], np.nan).dropna().clip(0.05, 10) for _, v in groups]
    labels = [f"{n}\n(n={len(v)})" for (n, _), v in zip(groups, data)]

    fig, ax = plt.subplots(figsize=(1.2 * len(groups) + 2, 5))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, medianprops=dict(color="k"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#55A868" if i == 0 else "#C44E52"); patch.set_alpha(0.7)
    nt_med = data[0].median()
    ax.axhline(nt_med, color="#55A868", ls="--", lw=1, label=f"NT median = {nt_med:.2f}")
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("TDP-43 N:C ratio")
    ax.set_title("N:C: non-targeting vs named mislocalization hits\n(lower = more cytoplasmic mislocalization)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def gene_summary(res, out):
    g = (res.groupby(["Gene", "is_NT"])
         .agg(n_cells=("NC_ratio", "size"),
              median_NC=("NC_ratio", "median"),
              mean_tdp_nuc=("tdp_nuc_mean", "mean"),
              mean_tdp_cyto=("tdp_cyto_mean", "mean"))
         .reset_index().sort_values("median_NC"))
    g.to_csv(out, index=False)
    return g


def fig_montage(res, uuid, out):
    """3 cells (low / median / high N:C) from a well-populated NT well, with rings."""
    meta = pd.read_parquet(next(META_DIR.glob(f"{uuid}*metadata*.parquet")))
    tdp_meta = meta[meta["Channel_name"] == "CF568"]

    # pick an NT well site with many cells for a fair baseline montage
    cand = res[res["is_NT"] & res["NC_ratio"].notna()]
    site_counts = cand.groupby(["Row", "Column", "Frame"]).size()
    row, col, frame = site_counts.idxmax()
    site = cand[(cand.Row == row) & (cand.Column == col) & (cand.Frame == frame)].copy()
    site = site.sort_values("NC_ratio")
    picks = site.iloc[[0, len(site) // 2, -1]]  # low / median / high

    planes = (tdp_meta[(tdp_meta.Row == row) & (tdp_meta.Column == col) &
                       (tdp_meta.Frame == frame)].sort_values("Plane"))
    bf = int(picks["best_focus_plane"].iloc[0])
    tdp = tifffile.imread(planes["filepath"].iloc[bf]).astype(np.float32)
    vmax = np.percentile(tdp, 99.5)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    R = int(np.ceil(R_CYTO)) + 5
    for ax, (_, c) in zip(axes, picks.iterrows()):
        cy, cx = c["centroid-0"], c["centroid-1"]
        y0, y1 = int(cy) - R, int(cy) + R
        x0, x1 = int(cx) - R, int(cx) + R
        tile = tdp[max(y0, 0):y1, max(x0, 0):x1]
        ax.imshow(tile, cmap="magma", vmax=vmax)
        cyc, cxc = cy - max(y0, 0), cx - max(x0, 0)
        for rad, color, lab in [(R_NUC, "cyan", "nucleus"), (R_ADJ, "yellow", "adjacent"),
                                (R_CYTO, "white", "cytoplasm")]:
            ax.add_patch(Circle((cxc, cyc), rad, fill=False, ec=color, lw=1.4))
        ax.set_title(f"N:C = {c['NC_ratio']:.2f}\nnuc={c['tdp_nuc_mean']:.0f} "
                     f"cyto={c['tdp_cyto_mean']:.0f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"TDP-43 compartments — NT well {picks['Well'].iloc[0]} "
                 f"r{row}c{col}f{frame} (low / median / high N:C)", fontsize=11)
    handles = [plt.Line2D([], [], color=c, label=l) for c, l in
               [("cyan", "nucleus (4.5µm)"), ("yellow", "adjacent (9µm)"), ("white", "cytoplasm (18µm)")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    parquet = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        max(TDP_DIR.glob("*tdp43_localization*.parquet"), key=lambda p: p.stat().st_mtime)
    print(f"QC on {parquet.name}")
    qc = TDP_DIR / "qc"; qc.mkdir(parents=True, exist_ok=True)

    res, uuid = load(parquet)
    valid = res["NC_ratio"].notna().sum()
    print(f"  {len(res):,} cells | {valid:,} with valid N:C ({100*valid/len(res):.1f}%) | "
          f"joined genes: {res['Gene'].notna().sum():,}")

    fig_distribution(res, qc / "nc_distribution.png")
    fig_nt_vs_hits(res, qc / "nc_nt_vs_hits.png")
    g = gene_summary(res, qc / "gene_summary.csv")
    fig_montage(res, uuid, qc / "montage_examples.png")

    nt_med = res[res["is_NT"]]["NC_ratio"].median()
    print(f"  NT median N:C = {nt_med:.3f}")
    print("  lowest-N:C genes (most mislocalized):")
    print(g[~g["is_NT"]].head(8)[["Gene", "n_cells", "median_NC"]].to_string(index=False))
    print(f"  figures + gene_summary.csv → {qc}")


if __name__ == "__main__":
    main()
