# ethics.py

def compute_ethics(conflicts):

    if not conflicts:
        return 0.0

    p_sum = 0.0
    pv_sum = 0.0

    for c in conflicts:

        TTC = c.get("TTC", 0.0) or 0.0
        Dmin = c.get("Dmin", 0.0) or 0.0
        DRAC = c.get("DRAC", 0.0)
        Ci = c.get("conflict_count", 0)
        vuln = c.get("vulnerability", 0.0)

        # --- normalized components ---
        r_ttc = min(1.0, TTC / 3.0)
        r_dist = min(1.0, Dmin / 2.0)
        r_conf = 1.0 / (1.0 + Ci)

        # --- collision criticality ---
        p = 1.0 - (r_ttc * r_dist * r_conf)
        
        if p < 0.05:
            continue


        # --- severity (DRAC-based) ---
        severity = 1.0 if DRAC >= 2.5 else 0.0

        # --- ethical valence ---
        v = - severity * vuln

        p_sum += p
        pv_sum += p * v

    if p_sum == 0:
        return 0.0

    E = 2 * (pv_sum / p_sum) - 1

    return E