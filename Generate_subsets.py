"""
generate_subsets_preserve_format.py

Generates stratified subsets and writes them in the exact original CSV format:
Package ID,Weight (kg),Type,Delay cost,Fragile?

Creates 3 subsets for each size in SIZES (defaults: 30,50,70).
"""

import os
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
PACKAGES_FILE = "Packages.csv"   # original packages file (keeps original formatting)
OUT_DIR = "subsets"
SIZES = [30, 50, 70]
SEEDS = [2000, 2002, 2004]       # 3 subsets per size
MIN_FRAGILE_PER_SUBSET = 2
MIN_HEAVY_PER_SUBSET = 2         # >200 kg considered heavy (kept for consistency)

# ---------- PREP ----------
os.makedirs(OUT_DIR, exist_ok=True)

# Read original packages file as strings so we preserve blank cells exactly
orig_df = pd.read_csv(PACKAGES_FILE, dtype=str).fillna('')

# Ensure expected columns exist (case-sensitive as requested)
expected_cols = ["Package ID", "Weight (kg)", "Type", "Delay cost", "Fragile?"]
for c in expected_cols:
    if c not in orig_df.columns:
        raise ValueError(f"Expected column '{c}' not found in {PACKAGES_FILE}. Found columns: {list(orig_df.columns)}")

# Create helper numeric/boolean columns used for stratified sampling
df = orig_df.copy()
# convert weight to float for heavy detection; non-numeric will raise if present
df["Weight_num"] = pd.to_numeric(df["Weight (kg)"], errors='coerce').fillna(0.0)
df["is_priority"] = df["Type"].str.strip().str.lower() == "priority"
df["is_fragile"] = df["Fragile?"].str.strip().str.lower() == "fragile"
df["is_heavy"] = df["Weight_num"] > 200.0

total_priority_ratio = df["is_priority"].mean()
total_fragile = int(df["is_fragile"].sum())
total_heavy = int(df["is_heavy"].sum())

print(f"Total packages: {len(df)}; priority ratio: {total_priority_ratio:.3f}; fragile total: {total_fragile}; heavy total: {total_heavy}")

# ---------- SAMPLING FUNCTION ----------
def stratified_sample_preserve(orig_df, helper_df, size, seed,
                               min_fragile=MIN_FRAGILE_PER_SUBSET,
                               min_heavy=MIN_HEAVY_PER_SUBSET):
    rnd = np.random.RandomState(seed)

    # Determine desired number of priorities based on overall ratio (rounded)
    desired_priority = int(round(size * total_priority_ratio))
    available_priority = int(helper_df["is_priority"].sum())
    desired_priority = min(desired_priority, available_priority)
    desired_economy = size - desired_priority

    pri_pool = helper_df[helper_df["is_priority"]].copy()
    eco_pool = helper_df[~helper_df["is_priority"]].copy()

    # Sample
    if len(pri_pool) <= desired_priority:
        chosen_pri = pri_pool.sample(n=len(pri_pool), random_state=rnd) if len(pri_pool)>0 else pri_pool.iloc[0:0]
    else:
        chosen_pri = pri_pool.sample(n=desired_priority, random_state=rnd)

    if len(eco_pool) <= desired_economy:
        chosen_eco = eco_pool.sample(n=len(eco_pool), random_state=rnd) if len(eco_pool)>0 else eco_pool.iloc[0:0]
    else:
        chosen_eco = eco_pool.sample(n=desired_economy, random_state=rnd)

    subset = pd.concat([chosen_pri, chosen_eco]).sample(frac=1, random_state=rnd).reset_index(drop=True)

    # Ensure fragile inclusion if possible
    fragile_in_subset = int(subset["is_fragile"].sum())
    if fragile_in_subset < min_fragile and total_fragile > 0:
        need = min_fragile - fragile_in_subset
        remaining_fragile = helper_df[helper_df["is_fragile"] & ~helper_df["Package ID"].isin(subset["Package ID"])]
        if len(remaining_fragile) > 0:
            to_add = remaining_fragile.sample(n=min(need, len(remaining_fragile)), random_state=rnd)
            # remove same number of non-fragile items (prefer low-delay economies if Delay present)
            non_fragile = subset[~subset["is_fragile"]]
            if len(non_fragile) >= len(to_add):
                drop_idx = non_fragile.sample(n=len(to_add), random_state=rnd).index
            else:
                drop_idx = non_fragile.index
            subset = subset.drop(drop_idx).append(to_add).reset_index(drop=True)

    # Ensure heavy inclusion (>200kg) if possible
    heavy_in_subset = int(subset["is_heavy"].sum())
    if heavy_in_subset < min_heavy and total_heavy > 0:
        need = min_heavy - heavy_in_subset
        remaining_heavy = helper_df[helper_df["is_heavy"] & ~helper_df["Package ID"].isin(subset["Package ID"])]
        if len(remaining_heavy) > 0:
            to_add = remaining_heavy.sample(n=min(need, len(remaining_heavy)), random_state=rnd)
            non_heavy = subset[~subset["is_heavy"]]
            if len(non_heavy) >= len(to_add):
                drop_idx = non_heavy.sample(n=len(to_add), random_state=rnd).index
            else:
                drop_idx = non_heavy.index
            subset = subset.drop(drop_idx).append(to_add).reset_index(drop=True)

    # Final size adjustment (trim or pad if needed)
    if len(subset) > size:
        subset = subset.sample(n=size, random_state=rnd).reset_index(drop=True)
    elif len(subset) < size:
        remaining = helper_df[~helper_df["Package ID"].isin(subset["Package ID"])]
        need = size - len(subset)
        if len(remaining) >= need:
            pad = remaining.sample(n=need, random_state=rnd)
            subset = pd.concat([subset, pad]).reset_index(drop=True)
        else:
            # not enough to pad; return as-is
            pass

    # Return rows from original dataframe (preserve original string formatting)
    chosen_ids = subset["Package ID"].tolist()
    out_df = orig_df[orig_df["Package ID"].isin(chosen_ids)].copy()
    # Keep the same order as in 'subset' (shuffle order from subset chosen)
    out_df = out_df.set_index("Package ID").loc[chosen_ids].reset_index()

    return out_df

# ---------- GENERATE & WRITE SUBSETS ----------
summary_rows = []
for size in SIZES:
    for seed in SEEDS:
        subset_df = stratified_sample_preserve(orig_df, df, size=size, seed=seed,
                                               min_fragile=min(MIN_FRAGILE_PER_SUBSET, total_fragile),
                                               min_heavy=min(MIN_HEAVY_PER_SUBSET, total_heavy))
        fname = f"pack_subset_{size}_{seed}.csv"
        out_path = os.path.join(OUT_DIR, fname)

        # Write in exact requested column order and keep blank cells as blanks
        subset_df.to_csv(out_path, columns=expected_cols, index=False, na_rep='')

        # gather summary info
        helper_subset = subset_df.copy()
        helper_subset["Weight_num"] = pd.to_numeric(helper_subset["Weight (kg)"], errors='coerce').fillna(0.0)
        n_priority = int((helper_subset["Type"].str.strip().str.lower() == "priority").sum())
        n_economy = int(len(helper_subset) - n_priority)
        n_fragile = int((helper_subset["Fragile?"].str.strip().str.lower() == "fragile").sum())
        n_heavy = int((helper_subset["Weight_num"] > 200.0).sum())

        summary_rows.append({
            "file": fname,
            "seed": seed,
            "size": len(subset_df),
            "priority_count": n_priority,
            "economy_count": n_economy,
            "fragile_count": n_fragile,
            "heavy_count": n_heavy
        })

print(f"Written subsets to folder: {OUT_DIR}")
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "summary.csv")
summary_df.to_csv(summary_csv, index=False)
print("Summary:")
print(summary_df)
