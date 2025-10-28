"""
milp_uld_gurobi.py

Usage:
  - put this script in the same folder as UIDs.csv and your subset file (or give full paths)
  - change PACKAGE_FILE to your subset (e.g. "subsets/pack_subset_30_2000.csv")
  - run: python milp_uld_gurobi.py

Notes:
  - Requires gurobipy (Gurobi Python API) and a valid license installed.
  - Implements capacity, fragility, priority constraints matching your heuristic.
  - Objective = K * (# priority-UIDs) + delay-cost (unassigned economies) + overload penalties.
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import math
import os

# ---------------- CONFIG ----------------
UID_FILE = "UIDs.csv"                         # expects columns "ULD ID","Weight limit (kg)"
PACKAGE_FILE = "Packages.csv"  # change to whichever subset you want
K = 40
FRAGILE_THRESHOLD = 200.0
OVERLOAD_RATIO = 0.9
TIME_LIMIT_SECONDS = 300     # set None to not set a time limit
M_BIG = 1e6                  # big-M (large enough given data). Adjust if needed.

# ---------------- READ INPUTS ----------------
ulds_df = pd.read_csv(UID_FILE)
pkgs_df = pd.read_csv(PACKAGE_FILE, dtype=str).fillna('')

# Normalize expected columns (adjust if your CSV uses slightly different names)
if 'ULD ID' in ulds_df.columns and 'Weight limit (kg)' in ulds_df.columns:
    ULD_ID_COL = 'ULD ID'
    ULD_CAP_COL = 'Weight limit (kg)'
elif 'UID' in ulds_df.columns and 'Capacity(kg)' in ulds_df.columns:
    ULD_ID_COL = 'UID'
    ULD_CAP_COL = 'Capacity(kg)'
else:
    raise ValueError("Unexpected columns in UIDs file. Expected 'ULD ID' and 'Weight limit (kg)' or 'UID' and 'Capacity(kg)'.")

# Required columns in packages
for c in ["Package ID", "Weight (kg)", "Type", "Delay cost", "Fragile?"]:
    if c not in pkgs_df.columns:
        raise ValueError(f"Missing column '{c}' in package file {PACKAGE_FILE}")

# Numeric conversions
pkgs_df['Weight_num'] = pd.to_numeric(pkgs_df['Weight (kg)'], errors='coerce').fillna(0.0)
pkgs_df['Delay_num'] = pd.to_numeric(pkgs_df['Delay cost'], errors='coerce').fillna(0.0)
pkgs_df['Fragile_bool'] = pkgs_df['Fragile?'].astype(str).str.strip().str.lower() == 'fragile'
pkgs_df['Priority_bool'] = pkgs_df['Type'].astype(str).str.strip().str.lower() == 'priority'
ulds_df[ULD_CAP_COL] = pd.to_numeric(ulds_df[ULD_CAP_COL], errors='coerce').fillna(0.0)

# Sets and mappings
P = list(pkgs_df.index)   # package indices
U = list(ulds_df.index)   # UID indices
pkg_id_map = pkgs_df['Package ID'].to_dict()
uld_id_map = ulds_df[ULD_ID_COL].to_dict()
pkg_weight = pkgs_df['Weight_num'].to_dict()
pkg_delay = pkgs_df['Delay_num'].to_dict()
pkg_fragile = pkgs_df['Fragile_bool'].to_dict()
pkg_priority = pkgs_df['Priority_bool'].to_dict()
uld_cap = ulds_df[ULD_CAP_COL].to_dict()

# Identify fragile packages and heavy (>FRAGILE_THRESHOLD) packages
fragile_pkgs = [p for p in P if pkg_fragile[p]]
heavy_pkgs = [p for p in P if pkg_weight[p] > FRAGILE_THRESHOLD + 1e-9]

# ---------------- BUILD MODEL ----------------
m = gp.Model("ULD_Packing_MILP")
if TIME_LIMIT_SECONDS:
    m.setParam('TimeLimit', TIME_LIMIT_SECONDS)
m.setParam('OutputFlag', 1)   # set to 0 to silence solver output

# Decision variables
# x[p,u] = 1 if package p assigned to uid u
x = m.addVars(P, U, vtype=GRB.BINARY, name="x")

# y[u] = 1 if uid u contains at least one priority package (for K cost)
y = m.addVars(U, vtype=GRB.BINARY, name="y")

# used[u] continuous = total weight used in u
used = m.addVars(U, lb=0.0, vtype=GRB.CONTINUOUS, name="used")

# o[u] binary indicator whether used > 0.9 * cap (for overload penalty)
o = m.addVars(U, vtype=GRB.BINARY, name="over_thresh")

# z[u] continuous >= overload penalty contribution for uid u
z = m.addVars(U, lb=0.0, vtype=GRB.CONTINUOUS, name="z_pen")

# ---------------- CONSTRAINTS ----------------

# 1) link used[u] with x: used[u] = sum_p weight_p * x[p,u]
for u in U:
    m.addConstr(used[u] == gp.quicksum(pkg_weight[p] * x[p, u] for p in P), name=f"used_def_{u}")

# 2) capacity strict: used[u] <= cap_u
for u in U:
    cap = float(uld_cap[u])
    m.addConstr(used[u] <= cap + 1e-9, name=f"cap_{u}")

# 3) priority guarantee: each priority package must be assigned to exactly one uid
for p in P:
    if pkg_priority[p]:
        m.addConstr(gp.quicksum(x[p, u] for u in U) == 1, name=f"prio_assign_{p}")
    else:
        # economy: can be assigned to at most one uid (or left unassigned)
        m.addConstr(gp.quicksum(x[p, u] for u in U) <= 1, name=f"econ_assign_{p}")

# 4) link y[u] with priority assignments: if any priority assigned to u, y[u]=1
#    sum_{p priority} x[p,u] <= bigM * y[u]; y[u] <= sum_{p priority} x[p,u]  not necessary but we can do:
for u in U:
    # If there are priority packages, enforce. Use bigM = number of priority packages.
    bigM = max(1, sum(1 for p in P if pkg_priority[p]))
    m.addConstr(gp.quicksum(x[p, u] for p in P if pkg_priority[p]) <= bigM * y[u], name=f"y_upper_{u}")
    # If y[u]==1 then at least one priority assigned: sum >= y[u]
    m.addConstr(gp.quicksum(x[p, u] for p in P if pkg_priority[p]) >= y[u], name=f"y_lower_{u}")

# 5) Fragility constraint:
# If a UID contains any fragile package, no package with individual weight > FRAGILE_THRESHOLD can be placed in that UID.
# Implement by forbidding assignment pair: for each fragile f and heavy h, x[f,u] + x[h,u] <= 1 for all u.
for u in U:
    for f in fragile_pkgs:
        for h in heavy_pkgs:
            if f == h:
                continue
            m.addConstr(x[f, u] + x[h, u] <= 1, name=f"frag_heavy_forbid_p{f}_h{h}_u{u}")

# 6) Overload penalty linearization
# o[u] is 1 if used[u] > 0.9 * cap_u
# enforce: used[u] <= 0.9*cap + M*o[u]
#          used[u] >= 0.9*cap + eps - M*(1 - o[u])   (optional to force o[u]=1 when used significantly above threshold)
# z[u] approximates (used[u]/cap_u)*100 when o[u]=1, otherwise 0.
for u in U:
    cap = float(uld_cap[u])
    thresh = OVERLOAD_RATIO * cap
    # if o[u]==0 -> used <= thresh
    m.addConstr(used[u] <= thresh + M_BIG * o[u], name=f"over_thresh_up_{u}")
    # optional: if used > thresh + tiny -> force o[u]=1 (helps numerical)
    eps = 1e-6
    m.addConstr(used[u] >= thresh + eps - M_BIG * (1 - o[u]), name=f"over_thresh_low_{u}")

    # now relate z[u] to used[u]/cap*100 when o[u]==1
    # Let val_u = (used[u]/cap)*100 (linear expression). We'll enforce:
    # z[u] >= val_u - M*(1 - o[u])
    # z[u] <= val_u
    # z[u] <= M * o[u]
    if cap <= 0:
        # avoid divide by zero
        m.addConstr(z[u] == 0.0, name=f"z_zerocap_{u}")
    else:
        val_factor = 100.0 / cap
        # z >= val - M*(1-o)
        m.addConstr(z[u] >= val_factor * used[u] - M_BIG * (1 - o[u]), name=f"z_lower_{u}")
        # z <= val
        m.addConstr(z[u] <= val_factor * used[u], name=f"z_upper1_{u}")
        # z <= M * o
        m.addConstr(z[u] <= M_BIG * o[u], name=f"z_upper2_{u}")

# ---------------- OBJECTIVE ----------------
# Objective = K * sum y[u] + sum_p (delay_p * (1 - sum_u x[p,u])) + sum_u z[u]
delay_term = gp.quicksum(pkg_delay[p] * (1 - gp.quicksum(x[p, u] for u in U)) for p in P if not pkg_priority[p])
uid_cost_term = K * gp.quicksum(y[u] for u in U)
overload_term = gp.quicksum(z[u] for u in U)

m.setObjective(uid_cost_term + delay_term + overload_term, GRB.MINIMIZE)

# ---------------- SOLVE ----------------
m.optimize()

# ---------------- EXTRACT SOLUTION ----------------
# Build assignment output (Package_ID -> Assigned_ULD or NONE)
assign_rows = []
for p in P:
    assigned_uid = None
    for u in U:
        if x[p, u].X is not None and x[p, u].X > 0.5:
            assigned_uid = uld_id_map[u]
            break
    if assigned_uid is None:
        assigned_uid = "NONE"
    assign_rows.append({
        "Package ID": pkg_id_map[p],
        "Assigned UID": assigned_uid
    })

assign_df = pd.DataFrame(assign_rows)
assign_df.to_csv("MILP_Assignment_Output.csv", index=False)

# UID utilization and details
uid_rows = []
for u in U:
    used_val = used[u].X if used[u].X is not None else 0.0
    cap = float(uld_cap[u])
    util_pct = (used_val / cap) * 100.0 if cap > 0 else 0.0
    contains_fragile = any(pkg_fragile[p] and (x[p,u].X > 0.5) for p in P)
    heavy_count = sum(1 for p in P if (pkg_weight[p] > FRAGILE_THRESHOLD + 1e-9) and (x[p,u].X > 0.5))
    priority_count = sum(1 for p in P if (pkg_priority[p]) and (x[p,u].X > 0.5))
    economy_count = sum(1 for p in P if (not pkg_priority[p]) and (x[p,u].X > 0.5))

    uid_rows.append({
        "UID": uld_id_map[u],
        "Capacity(kg)": cap,
        "Used(kg)": round(used_val, 4),
        "Utilization(%)": round(util_pct, 4),
        "Has_Fragile": contains_fragile,
        "Heavy_Packages_Count": int(heavy_count),
        "Contains_Priority": (y[u].X > 0.5),
        "Num_Packages": int(sum(1 for p in P if x[p,u].X > 0.5)),
        "Priority_Count": int(priority_count),
        "Economy_Count": int(economy_count),
        "Over_Threshold_Flag": int(o[u].X > 0.5),
        "Overload_Penalty_Contrib": round(z[u].X if z[u].X is not None else 0.0, 4)
    })

uid_df = pd.DataFrame(uid_rows).sort_values(by='UID')
uid_df.to_csv("MILP_UID_Utilization.csv", index=False)

# ---------------- PRINT COST BREAKDOWN ----------------
uid_cost_val = K * sum(1 for u in U if y[u].X > 0.5)
delay_cost_val = sum(pkg_delay[p] for p in P if (not pkg_priority[p]) and (all(x[p,u].X < 0.5 for u in U)))
overload_penalty_val = sum(z[u].X for u in U if z[u].X is not None)

print("\n--- MILP RESULT SUMMARY ---")
print("Objective (total):", m.objVal)
print("UID cost (K * #priority-UIDs):", uid_cost_val)
print("Delay cost:", delay_cost_val)
print("Overload penalty:", overload_penalty_val)
print("Total cost (sum components):", uid_cost_val + delay_cost_val + overload_penalty_val)
print("Assignment written to MILP_Assignment_Output.csv")
print("UID utilization written to MILP_UID_Utilization.csv")
