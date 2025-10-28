import pandas as pd
import random
import copy

# ---------- CONFIG ----------
SEEDS = [42, 43, 44]        # different seeds to explore multiple solutions
K = 40                      # UID cost per priority UID
FRAGILE_THRESHOLD = 200.0
OVERLOAD_RATIO = 0.9
MAX_ITER_LOCAL_SEARCH = 2000  # local search iterations

UID_FILE = "UIDs.csv"        # columns: "ULD ID", "Weight limit (kg)"
PACKAGE_FILE = "Packages.csv"  # change to subset path when running e.g. "subsets/pack_subset_30_2000.csv"

# ---------- READ INPUTS ----------
uids_df = pd.read_csv(UID_FILE)
pkgs_df = pd.read_csv(PACKAGE_FILE)

# ---------- PREPROCESS ----------
def build_structures():
    # Build UIDs dict
    UIDs = {}
    for _, r in uids_df.iterrows():
        uid = str(r['ULD ID']).strip()
        cap = float(r['Weight limit (kg)'])
        UIDs[uid] = {'capacity': cap, 'current': 0.0, 'packages': [], 'has_fragile': False, 'heavy_count': 0}

    # Build Packages dict
    Packages = {}
    for _, r in pkgs_df.iterrows():
        pid = str(r['Package ID']).strip()
        weight = float(r['Weight (kg)'])
        typ = str(r['Type']).strip()
        delay = 0.0
        if not pd.isna(r.get('Delay cost', None)) and str(r.get('Delay cost')).strip() != '':
            try:
                delay = float(r['Delay cost'])
            except:
                delay = 0.0
        fragile = False
        if not pd.isna(r.get('Fragile?', None)) and str(r.get('Fragile?')).strip() != '':
            fragile = str(r['Fragile?']).strip().lower() == 'fragile'
        Packages[pid] = {'weight': weight, 'type': typ, 'delay': delay, 'fragile': fragile, 'assigned': None}

    # Precompute quick maps
    package_weight_map = {pid: p['weight'] for pid,p in Packages.items()}
    package_fragile_map = {pid: p['fragile'] for pid,p in Packages.items()}
    package_type_map = {pid: p['type'] for pid,p in Packages.items()}
    package_delay_map = {pid: p['delay'] for pid,p in Packages.items()}

    return UIDs, Packages, package_weight_map, package_fragile_map, package_type_map, package_delay_map

# ---------- FRAGILITY CHECK ----------
def can_assign(pkg_id, uid, UIDs, Packages):
    pkg = Packages[pkg_id]
    u = UIDs[uid]
    new_total = u['current'] + pkg['weight']

    # Hard capacity limit (strict)
    if new_total > u['capacity'] + 1e-9:
        return False

    # Fragile rules: if adding fragile and UID already has heavy (>threshold), disallow
    if pkg['fragile'] and u['heavy_count'] > 0:
        return False
    # if UID has fragile, new package must be <= FRAGILE_THRESHOLD
    if u['has_fragile'] and pkg['weight'] > FRAGILE_THRESHOLD + 1e-9:
        return False

    return True

# ---------- PLACE / REMOVE ----------
def place(pkg_id, uid, UIDs, Packages, package_weight_map, package_fragile_map):
    w = package_weight_map[pkg_id]
    frag = package_fragile_map[pkg_id]
    UIDs[uid]['packages'].append(pkg_id)
    UIDs[uid]['current'] += w
    if frag:
        UIDs[uid]['has_fragile'] = True
    if w > FRAGILE_THRESHOLD:
        UIDs[uid]['heavy_count'] += 1
    Packages[pkg_id]['assigned'] = uid

def remove(pkg_id, uid, UIDs, Packages, package_weight_map, package_fragile_map):
    w = package_weight_map[pkg_id]
    frag = package_fragile_map[pkg_id]
    try:
        UIDs[uid]['packages'].remove(pkg_id)
    except ValueError:
        pass
    UIDs[uid]['current'] -= w
    if frag:
        UIDs[uid]['has_fragile'] = any(package_fragile_map[pid] for pid in UIDs[uid]['packages'])
    if w > FRAGILE_THRESHOLD:
        UIDs[uid]['heavy_count'] = max(0, UIDs[uid]['heavy_count'] - 1)
    Packages[pkg_id]['assigned'] = None

# ---------- PENALTY helper ----------
def overload_penalty_for(used, cap):
    if used > OVERLOAD_RATIO * cap:
        return (used / cap) * 100.0
    return 0.0

# ---------- COST CALCULATION ----------
def calculate_cost(UIDs, Packages):
    priority_uid_count = sum(
        1 for uid,u in UIDs.items() if any(Packages[pid]['type'].lower()=='priority' for pid in u['packages'])
    )
    uid_cost = K * priority_uid_count
    delay_cost = sum(Packages[pid]['delay'] for pid in Packages if Packages[pid]['type'].lower()=='economy' and Packages[pid]['assigned'] is None)
    overload_penalty = 0.0
    for uid,u in UIDs.items():
        used = u['current']
        cap = u['capacity']
        overload_penalty += overload_penalty_for(used, cap)
    total_cost = uid_cost + delay_cost + overload_penalty
    return total_cost, uid_cost, delay_cost, overload_penalty

# ---------- GREEDY WITH MARGINAL-COST INSERTION ----------
def assign_packages(seed):
    random.seed(seed)
    UIDs, Packages, package_weight_map, package_fragile_map, package_type_map, package_delay_map = build_structures()

    # Lists
    priority_list = [pid for pid,p in Packages.items() if p['type'].lower()=='priority']
    economy_list = [pid for pid,p in Packages.items() if p['type'].lower()=='economy']

    # Sort priority by weight descending (try to pack big ones first)
    priority_list.sort(key=lambda x: -Packages[x]['weight'])

    # Helper to detect if uid currently contains a priority
    def uid_has_priority(uid):
        return any(package_type_map[pid].lower()=='priority' for pid in UIDs[uid]['packages'])

    # Assign priorities: must be assigned, choose UID with minimal marginal cost
    for pid in priority_list:
        best_cost = float('inf')
        best_uid = None
        w = package_weight_map[pid]
        for uid in sorted(UIDs.keys(), key=lambda u: UIDs[u]['capacity'], reverse=True):
            if not can_assign(pid, uid, UIDs, Packages):
                continue
            # current components
            cur_used = UIDs[uid]['current']
            cap = UIDs[uid]['capacity']
            cur_over = overload_penalty_for(cur_used, cap)

            # after placing
            new_used = cur_used + w
            new_over = overload_penalty_for(new_used, cap)
            delta_over = new_over - cur_over

            # uid cost delta
            cur_has_prio = uid_has_priority(uid)
            delta_uid_cost = 0 if cur_has_prio else K

            # delay delta (priority has no delay)
            delta_delay = 0

            marginal = delta_uid_cost + delta_over + delta_delay

            if marginal < best_cost:
                best_cost = marginal
                best_uid = uid

        # If no feasible uid (extremely rare), leave None (should not happen by problem guarantee)
        if best_uid is None:
            Packages[pid]['assigned'] = None
        else:
            place(pid, best_uid, UIDs, Packages, package_weight_map, package_fragile_map)

    # Assign economies: evaluate marginal cost; only place if beneficial (or neutral if you want more load)
    # Sort economies by delay descending (higher delay should be prioritized)
    economy_list.sort(key=lambda x: -Packages[x]['delay'])

    for pid in economy_list:
        best_cost = float('inf')
        best_uid = None
        w = package_weight_map[pid]
        for uid in UIDs:
            if not can_assign(pid, uid, UIDs, Packages):
                continue
            cur_used = UIDs[uid]['current']
            cap = UIDs[uid]['capacity']
            cur_over = overload_penalty_for(cur_used, cap)
            new_used = cur_used + w
            new_over = overload_penalty_for(new_used, cap)
            delta_over = new_over - cur_over

            # uid cost delta: only if this economy is priority? no, it's economy so delta uid cost 0
            delta_uid_cost = 0

            # delay delta: assigning reduces delay cost by package's delay
            delta_delay = - package_delay_map[pid]

            marginal = delta_uid_cost + delta_over + delta_delay

            if marginal < best_cost:
                best_cost = marginal
                best_uid = uid

        # If best marginal is negative (reduces total cost) or zero (tie-breaker allow), place
        # Option: to push utilization while keeping cost same, accept small positive marginal up to threshold; but here we accept <=0
        if best_uid is not None and best_cost <= 0:
            place(pid, best_uid, UIDs, Packages, package_weight_map, package_fragile_map)
        else:
            # leave unassigned (NONE)
            Packages[pid]['assigned'] = None

    # ---------- LOCAL SEARCH: cost-reducing moves ----------
    # Try moving an assigned economy package to any other UID if it reduces total cost.
    econ_assigned = [pid for pid in economy_list if Packages[pid]['assigned'] is not None]

    for _ in range(MAX_ITER_LOCAL_SEARCH):
        if not econ_assigned:
            break
        pid = random.choice(econ_assigned)
        cur_uid = Packages[pid]['assigned']
        if cur_uid is None:
            econ_assigned = [p for p in econ_assigned if Packages[p]['assigned'] is not None]
            continue

        # compute current total cost contribution (recompute global for reliability)
        base_total, base_uid_cost, base_delay, base_overload = calculate_cost(UIDs, Packages)

        moved = False
        for uid in UIDs:
            if uid == cur_uid:
                continue
            if not can_assign(pid, uid, UIDs, Packages):
                continue

            # simulate move
            remove(pid, cur_uid, UIDs, Packages, package_weight_map, package_fragile_map)
            place(pid, uid, UIDs, Packages, package_weight_map, package_fragile_map)

            new_total, _, _, _ = calculate_cost(UIDs, Packages)

            if new_total < base_total - 1e-6:
                # keep move
                moved = True
                break
            else:
                # revert
                remove(pid, uid, UIDs, Packages, package_weight_map, package_fragile_map)
                place(pid, cur_uid, UIDs, Packages, package_weight_map, package_fragile_map)

        if moved:
            # refresh econ_assigned list and continue
            econ_assigned = [p for p in economy_list if Packages[p]['assigned'] is not None]
        # else continue to next random pick

    return UIDs, Packages

# ---------- OUTPUT ----------
def write_outputs(Packages, UIDs, pkgs_order, package_type_map):
    out_rows = []
    for pid in pkgs_order:
        assigned_uid = Packages[pid]['assigned']
        out_rows.append({'Package_ID': pid, 'Assigned_UID': assigned_uid if assigned_uid is not None else 'NONE'})
    pd.DataFrame(out_rows).to_csv("Package_Assignment_Output.csv", index=False)

    uid_rows = []
    for uid,u in UIDs.items():
        used = u['current']
        cap = u['capacity']
        util_percent = (used / cap) * 100.0 if cap > 0 else 0.0

        priority_count = sum(1 for pid in u['packages'] if package_type_map.get(pid,'').lower() == 'priority')
        economy_count = sum(1 for pid in u['packages'] if package_type_map.get(pid,'').lower() == 'economy')

        contains_priority = any(package_type_map.get(pid,'').lower()=='priority' for pid in u['packages'])
        uid_rows.append({
            'UID': uid,
            'Capacity(kg)': cap,
            'Used(kg)': round(used,2),
            'Utilization(%)': round(util_percent,2),
            'Has_Fragile': u['has_fragile'],
            'Heavy_Packages_Count': u['heavy_count'],
            'Contains_Priority': contains_priority,
            'Num_Packages': len(u['packages']),
            'Priority_Count': priority_count,
            'Economy_Count': economy_count
        })
    pd.DataFrame(uid_rows).sort_values(by='UID').to_csv("UID_Utilization.csv", index=False)

# ---------- MULTI-SEED SEARCH ----------
pkgs_order = list(pkgs_df['Package ID'])
best_total_cost = float('inf')
overall_best_assigned = None
overall_best_uids = None

# build map once for output
_, _, _, _, package_type_map, _ = build_structures()

for seed in SEEDS:
    UIDs, Packages = assign_packages(seed)
    total_cost, uid_cost, delay_cost, overload_penalty = calculate_cost(UIDs, Packages)
    print(f"[seed={seed}] total_cost={total_cost:.2f} (uid_cost={uid_cost:.2f}, delay={delay_cost:.2f}, overload={overload_penalty:.2f})")
    if total_cost < best_total_cost:
        best_total_cost = total_cost
        overall_best_assigned = copy.deepcopy(Packages)
        overall_best_uids = copy.deepcopy(UIDs)

# ---------- WRITE BEST OUTPUT ----------
write_outputs(overall_best_assigned, overall_best_uids, pkgs_order, package_type_map)
total_cost, uid_cost, delay_cost, overload_penalty = calculate_cost(overall_best_uids, overall_best_assigned)

print("\nBEST SOLUTION")
print(f"Total cost: {total_cost:.2f}")
print(f"UID cost: {uid_cost:.2f}")
print(f"Delay cost: {delay_cost:.2f}")
print(f"Overload penalty: {overload_penalty:.2f}")
print("Assignment written to Package_Assignment_Output.csv")
print("UID utilization written to UID_Utilization.csv")
