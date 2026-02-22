import os
import re
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 30)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

publications = pd.read_csv(os.path.join(DATA_DIR, "DIMENSIONS_RESULTING_PUBLICATIONS.csv"),
    usecols=["Grant_ID", "Publication_ID", "field_citation_ratio", "relative_citation_ratio"])
organizations = pd.read_csv(os.path.join(DATA_DIR, "DIMENSIONS_RESEARCH_ORGANIZATIONS.csv"),
    usecols=["GRANT_ID", "RESEARCH_ORG_NAME"])
awards = pd.read_csv(os.path.join(DATA_DIR, "DIMENSIONS_CORE_AWARD_DETAILS.csv"),
    usecols=["GRANT_ID", "FUNDING_USD", "FUNDER_ORG_NAME"])

publications.rename(columns={"Grant_ID": "GRANT_ID"}, inplace=True)

def normalize_org(name):
    name = name.upper()
    name = re.sub(r"\(.*?\)", "", name)       # remove parentheses
    name = re.sub(r"DEPARTMENT.*", "", name)
    name = name.replace("/", " ")
    name = re.sub(r"[^A-Z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

organizations["RESEARCH_ORG_CLEAN"] = organizations["RESEARCH_ORG_NAME"].fillna("").apply(normalize_org)

# Count unique orgs per grant for splitting funding/publications
orgs_per_grant = organizations.groupby("GRANT_ID")["RESEARCH_ORG_CLEAN"].nunique().rename("n_orgs")

df = awards.merge(organizations, on="GRANT_ID", how="left") \
           .merge(publications, on="GRANT_ID", how="left") \
           .merge(orgs_per_grant, on="GRANT_ID", how="left")

df["n_orgs"] = df["n_orgs"].fillna(1)

grant_stats = df.groupby(["GRANT_ID", "RESEARCH_ORG_CLEAN"]).agg(
    funding_usd=("FUNDING_USD", "first"),
    n_orgs=("n_orgs", "first"),
    num_publications=("Publication_ID", "count"),
    avg_field_citation_ratio=("field_citation_ratio", "mean"),
    avg_relative_citation_ratio=("relative_citation_ratio", "mean"),
).reset_index()

# Split funding and publications evenly across orgs on the same grant
grant_stats["funding_usd"] = grant_stats["funding_usd"] / grant_stats["n_orgs"]
grant_stats["num_publications"] = grant_stats["num_publications"] / grant_stats["n_orgs"]

org_stats = grant_stats.groupby("RESEARCH_ORG_CLEAN").agg(
    total_funding=("funding_usd", "sum"),
    total_publications=("num_publications", "sum"),
    avg_field_citation_ratio=("avg_field_citation_ratio", "mean"),
    avg_relative_citation_ratio=("avg_relative_citation_ratio", "mean"),
    num_grants=("GRANT_ID", "count"),
).reset_index()

import numpy as np

# ── Classify funder → research field ──────────────────────────────────
def classify_funder(name):
    if pd.isna(name):
        return "Other"
    n = name.upper()
    if any(x in n for x in ["NATIONAL INSTITUTE OF", "NATIONAL CANCER", "NATIONAL HEART",
        "NATIONAL EYE", "ALCOHOL", "AGING", "ALLERGY", "ARTHRITIS", "DENTAL",
        "DIABETES", "DRUG ABUSE", "MENTAL HEALTH", "NEUROLOGICAL", "NURSING",
        "CHILD HEALTH", "GENOME", "BIOMEDICAL", "HEALTH SCIENCE",
        "CANADIAN INSTITUTES OF HEALTH", "PUBLIC HEALTH"]):
        return "Biomedical"
    if any(x in n for x in ["ENGINEERING", "COMPUTER", "INFORMATION SCIENCE"]):
        return "Engineering & CS"
    if any(x in n for x in ["MATHEMATICAL", "PHYSICAL", "NSERC",
        "NATURAL SCIENCES AND ENG", "NASA", "AERONAUTICS", "ENERGY",
        "NUCLEAR", "GEOSCIENCE", "PHYSICS"]):
        return "Physical Sciences"
    if any(x in n for x in ["SOCIAL", "HUMANITIES", "EDUCATION", "SSHRC"]):
        return "Social Sciences"
    if any(x in n for x in ["BIOLOGICAL", "BIOLOGY", "ECOLOGY", "ENVIRONMENTAL"]):
        return "Life Sciences"
    if any(x in n for x in ["NAVY", "ARMY", "AIR FORCE", "DEFENSE", "DARPA"]):
        return "Defense"
    if any(x in n for x in ["AGRICULTURE", "FOOD"]):
        return "Agriculture"
    return "Other"

grant_stats["field"] = df.groupby(["GRANT_ID", "RESEARCH_ORG_CLEAN"])["FUNDER_ORG_NAME"].first().reset_index()["FUNDER_ORG_NAME"].apply(classify_funder).values

# Dominant field per org (most common funder-field across their grants)
dominant_field = grant_stats.groupby("RESEARCH_ORG_CLEAN")["field"].agg(
    lambda x: x.value_counts().index[0]
).rename("dominant_field")
org_stats = org_stats.merge(dominant_field, on="RESEARCH_ORG_CLEAN", how="left")
org_stats["dominant_field"] = org_stats["dominant_field"].fillna("Other")

org_stats["impact_efficiency"] = (org_stats["avg_field_citation_ratio"] * org_stats["total_publications"]) / org_stats["total_funding"]
org_stats["cost_per_impact"] = 1 / org_stats["impact_efficiency"]  # $ per unit of citation impact

conditions = [
    grant_stats["avg_field_citation_ratio"].notna(),
    grant_stats["num_publications"] == 0,
    (grant_stats["num_publications"] > 0) & (grant_stats["avg_field_citation_ratio"].isna()),
]
labels = ["FCR available", "No publications", "Publications but no FCR"]

grant_stats["fcr_status"] = np.select(conditions, labels, default="Unknown")

grant_stats["impact_efficiency"] = (grant_stats["avg_field_citation_ratio"] * grant_stats["num_publications"]) / grant_stats["funding_usd"]

print(grant_stats["fcr_status"].value_counts())

no_funding = grant_stats["funding_usd"].isna() | (grant_stats["funding_usd"] == 0)
print(f"\nGrants with no funding data: {no_funding.sum()} / {len(grant_stats)}")

org_stats_funded = org_stats[org_stats["total_funding"].notna() & (org_stats["total_funding"] > 0)]

MIN_PUBS = 50
org_stats_reliable = org_stats_funded[org_stats_funded["total_publications"] >= MIN_PUBS]

print(f"\n=== Top 50 institutions by impact efficiency (funding > 0, pubs >= {MIN_PUBS}) ===")
print(f"({len(org_stats_reliable)} institutions meet criteria out of {len(org_stats_funded)} funded)\n")
top50 = org_stats_reliable.sort_values("cost_per_impact", ascending=True).head(50).copy()
top50["org_name"] = top50["RESEARCH_ORG_CLEAN"].str[:30]
print(top50[
    ["org_name", "num_grants", "total_funding", "total_publications", "avg_field_citation_ratio", "cost_per_impact"]
].to_string(index=False))

# How common is the multi-org-per-grant problem?
print(f"\n=== Multi-org grants ===")
print(f"Total grants in orgs table: {len(orgs_per_grant)}")
print(f"Grants with 1 org: {(orgs_per_grant == 1).sum()}")
print(f"Grants with 2+ orgs: {(orgs_per_grant > 1).sum()}")
print(f"Max orgs on a single grant: {orgs_per_grant.max()}")
print(f"\nDistribution of orgs per grant:")
print(orgs_per_grant.value_counts().sort_index().head(20))

ranked = org_stats_reliable.sort_values("impact_efficiency", ascending=False).reset_index(drop=True)
ranked["rank"] = ranked.index + 1
cmu_rank = ranked[ranked["RESEARCH_ORG_CLEAN"].str.contains("CARNEGIE MELLON", case=False, na=False)]

print("\n=== Carnegie Mellon ===")
for _, row in cmu_rank.iterrows():
    print(f"  Rank {int(row['rank'])} / {len(ranked)}: {row['RESEARCH_ORG_CLEAN']}")
    print(f"    Grants: {int(row['num_grants'])}  |  Funding: ${row['total_funding']:,.0f}  |  Pubs: {row['total_publications']:.0f}  |  Avg FCR: {row['avg_field_citation_ratio']:.2f}  |  Cost/Impact: ${row['cost_per_impact']:,.0f}")
    print()

# Save processed data for downstream scripts
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
grant_stats.to_csv(os.path.join(OUTPUT_DIR, "grant_stats.csv"), index=False)
org_stats_reliable.to_csv(os.path.join(OUTPUT_DIR, "org_stats_reliable.csv"), index=False)
print("Saved grant_stats.csv and org_stats_reliable.csv to output/")