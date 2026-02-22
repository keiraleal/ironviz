import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no window popups
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

grant_stats = pd.read_csv(os.path.join(OUTPUT_DIR, "grant_stats.csv"))
org_stats_reliable = pd.read_csv(os.path.join(OUTPUT_DIR, "org_stats_reliable.csv"))

# Grant-level: funding vs FCR
fcr_available = grant_stats[
    (grant_stats["fcr_status"] == "FCR available") & (grant_stats["funding_usd"].notna())
]
fcr_other = fcr_available[~fcr_available["RESEARCH_ORG_CLEAN"].str.contains("CARNEGIE MELLON", case=False, na=False)]
fcr_cmu = fcr_available[fcr_available["RESEARCH_ORG_CLEAN"].str.contains("CARNEGIE MELLON", case=False, na=False)]

plt.figure(figsize=(10, 6))
plt.scatter(
    fcr_other["funding_usd"],
    fcr_other["avg_field_citation_ratio"],
    s=fcr_other["num_publications"] * 0.1 + 5,
    alpha=0.3,
    edgecolors="none",
    label="Other",
)
plt.scatter(
    fcr_cmu["funding_usd"],
    fcr_cmu["avg_field_citation_ratio"],
    s=fcr_cmu["num_publications"] * 0.1 + 10,
    alpha=1.0,
    color="red",
    edgecolors="black",
    linewidths=0.5,
    zorder=5,
    label="Carnegie Mellon",
)
plt.xlabel("Grant Funding (USD)")
plt.ylabel("Avg Field Citation Ratio")
plt.title("Grant Funding vs. Avg FCR (dot size = number of publications)")
plt.xscale("log")
plt.yscale("log")
plt.xlim(left=1000)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "funding_vs_fcr.png"), dpi=150)
plt.close()
print("Saved funding_vs_fcr.png")

# Institution-level: total funding vs impact efficiency
org = org_stats_reliable[org_stats_reliable["impact_efficiency"].notna() & (org_stats_reliable["impact_efficiency"] > 0)]
org_other = org[~org["RESEARCH_ORG_CLEAN"].str.contains("CARNEGIE MELLON", case=False, na=False)]
org_cmu = org[org["RESEARCH_ORG_CLEAN"].str.contains("CARNEGIE MELLON", case=False, na=False)]

plt.figure(figsize=(12, 7))
plt.scatter(
    org_other["total_funding"],
    org_other["impact_efficiency"],
    s=org_other["num_grants"] * 0.5 + 5,
    alpha=0.4,
    edgecolors="none",
    label="Other",
)
plt.scatter(
    org_cmu["total_funding"],
    org_cmu["impact_efficiency"],
    s=org_cmu["num_grants"] * 0.5 + 5,
    alpha=1.0,
    color="red",
    edgecolors="black",
    linewidths=1.0,
    zorder=10,
    label="Carnegie Mellon",
)
for _, row in org_cmu.iterrows():
    plt.annotate("CMU", (row["total_funding"], row["impact_efficiency"]),
                 textcoords="offset points", xytext=(10, 8), fontsize=10, fontweight="bold", color="red",
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
plt.xlabel("Total Funding (USD)")
plt.ylabel("Impact Efficiency (FCR × Pubs / $)")
plt.title("Institution Impact Efficiency vs. Total Funding (dot size = # grants)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "org_efficiency.png"), dpi=150)
plt.close()
print("Saved org_efficiency.png")

# ============================================================
# MAIN CHART: Cost per Impact exploration
# ============================================================
org_cpi = org[org["cost_per_impact"].notna() & (org["cost_per_impact"] > 0)].copy()
cpi_other = org_cpi[~org_cpi["RESEARCH_ORG_CLEAN"].str.contains("CARNEGIE MELLON", case=False, na=False)]
cpi_cmu = org_cpi[org_cpi["RESEARCH_ORG_CLEAN"].str.contains("CARNEGIE MELLON", case=False, na=False)]

# --- Chart A: Size = total publications ---
fig, ax = plt.subplots(figsize=(14, 8))
sc = ax.scatter(
    cpi_other["total_funding"],
    cpi_other["cost_per_impact"],
    s=cpi_other["total_publications"] * 0.05 + 5,
    alpha=0.5,
    edgecolors="none",
    color="steelblue",
)
ax.scatter(
    cpi_cmu["total_funding"],
    cpi_cmu["cost_per_impact"],
    s=cpi_cmu["total_publications"] * 0.05 + 5,
    color="red",
    edgecolors="black",
    linewidths=1.0,
    zorder=10,
)
for _, row in cpi_cmu.iterrows():
    ax.annotate("CMU", (row["total_funding"], row["cost_per_impact"]),
                textcoords="offset points", xytext=(10, 8), fontsize=10, fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

# Trend line (log-log regression)
log_x = np.log10(org_cpi["total_funding"].values)
log_y = np.log10(org_cpi["cost_per_impact"].values)
slope, intercept = np.polyfit(log_x, log_y, 1)
x_fit = np.linspace(log_x.min(), log_x.max(), 100)
y_fit = slope * x_fit + intercept
ax.plot(10**x_fit, 10**y_fit, "r--", alpha=0.7, linewidth=2,
        label=f"Trend: slope={slope:.2f}")

ax.set_xlabel("Total Funding (USD)")
ax.set_ylabel("Cost per Impact ($ per FCR·Pub)")
ax.set_title("Cost per Impact vs. Total Funding\n(dot size = publications, lower = more efficient)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "org_cost_per_impact.png"), dpi=150)
plt.close(fig)
print("Saved org_cost_per_impact.png")

# --- Chart B: Funding bins → cost per impact distribution ---
org_cpi["funding_bin"] = pd.cut(
    org_cpi["total_funding"],
    bins=[0, 1e6, 1e7, 1e8, 1e9, np.inf],
    labels=["<$1M", "$1M–10M", "$10M–100M", "$100M–1B", ">$1B"],
)
cmu_val = cpi_cmu["cost_per_impact"].values[0] if len(cpi_cmu) > 0 else None
cmu_funding = cpi_cmu["total_funding"].values[0] if len(cpi_cmu) > 0 else None
cmu_bin = pd.cut([cmu_funding], bins=[0, 1e6, 1e7, 1e8, 1e9, np.inf],
                 labels=["<$1M", "$1M–10M", "$10M–100M", "$100M–1B", ">$1B"])[0] if cmu_funding else None

fig, ax = plt.subplots(figsize=(10, 6))
org_cpi.boxplot(column="cost_per_impact", by="funding_bin", ax=ax, showfliers=False)
ax.set_yscale("log")
if cmu_val:
    ax.axhline(cmu_val, color="red", linestyle="--", linewidth=2, label=f"CMU: ${cmu_val:,.0f} ({cmu_bin})")
    # Mark CMU's funding bracket with a red dot
    bin_labels = ["<$1M", "$1M–10M", "$10M–100M", "$100M–1B", ">$1B"]
    cmu_x = bin_labels.index(cmu_bin) + 1 if cmu_bin in bin_labels else None
    if cmu_x:
        ax.plot(cmu_x, cmu_val, "ro", markersize=12, zorder=10)
    ax.legend(fontsize=10)
ax.set_xlabel("Total Funding Bucket")
ax.set_ylabel("Cost per Impact ($ per FCR·Pub)")
ax.set_title("Cost per Impact by Funding Size")
plt.suptitle("")  # remove default pandas boxplot title
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cost_per_impact_by_funding.png"), dpi=150)
plt.close(fig)
print("Saved cost_per_impact_by_funding.png")

# --- Chart C: Publications vs cost per impact ---
fig, ax = plt.subplots(figsize=(12, 7))
sc = ax.scatter(
    cpi_other["total_publications"],
    cpi_other["cost_per_impact"],
    s=cpi_other["total_publications"] * 0.05 + 5,
    c=cpi_other["total_funding"],
    cmap="plasma",
    norm=LogNorm(),
    alpha=0.6,
    edgecolors="none",
)
ax.scatter(
    cpi_cmu["total_publications"],
    cpi_cmu["cost_per_impact"],
    s=cpi_cmu["total_publications"] * 0.05 + 5,
    color="red",
    edgecolors="black",
    linewidths=1.0,
    zorder=10,
)
for _, row in cpi_cmu.iterrows():
    ax.annotate("CMU", (row["total_publications"], row["cost_per_impact"]),
                textcoords="offset points", xytext=(10, 8), fontsize=10, fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

# Trend line
log_x2 = np.log10(org_cpi["total_publications"].values)
slope2, intercept2 = np.polyfit(log_x2, log_y, 1)
x_fit2 = np.linspace(log_x2.min(), log_x2.max(), 100)
y_fit2 = slope2 * x_fit2 + intercept2
ax.plot(10**x_fit2, 10**y_fit2, "r--", alpha=0.7, linewidth=2,
        label=f"Trend: slope={slope2:.2f}")

cbar = fig.colorbar(sc, ax=ax, label="Total Funding (USD)")
ax.set_xlabel("Total Publications")
ax.set_ylabel("Cost per Impact ($ per FCR·Pub)")
ax.set_title("Cost per Impact vs. Publication Volume\n(color = funding, size = publications)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cost_per_impact_vs_pubs.png"), dpi=150)
plt.close(fig)
print("Saved cost_per_impact_vs_pubs.png")

# --- Correlation stats ---
print("\n=== Correlations (log-space) ===")
log_funding = np.log10(org_cpi["total_funding"])
log_pubs = np.log10(org_cpi["total_publications"])
log_cpi = np.log10(org_cpi["cost_per_impact"])
print(f"  log(Funding) vs log(Cost/Impact):  r = {np.corrcoef(log_funding, log_cpi)[0,1]:.3f}")
print(f"  log(Pubs) vs log(Cost/Impact):     r = {np.corrcoef(log_pubs, log_cpi)[0,1]:.3f}")
print(f"  log(Funding) vs log(Pubs):         r = {np.corrcoef(log_funding, log_pubs)[0,1]:.3f}")
print(f"\n  Trend: Cost/Impact ~ Funding^{slope:.2f}")
print(f"  Trend: Cost/Impact ~ Pubs^{slope2:.2f}")
