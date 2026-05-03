"""Map a clinical target value to its risk class (0/1/2).

Mirrors the per-template thresholds from data/nhanes_benchmark_200.json.
This lives in utils/ instead of plots/ so the experiment runners don't
depend on the full plotting package for a 30-line lookup.
"""


def get_risk_class(template_name, val):
    try:
        val = float(val)
    except Exception:
        return 0

    if template_name == "ANEMIA":            # MCHC
        if val < 32: return 0
        elif val > 36: return 2
        else: return 1
    elif template_name == "AIP":             # log10(tg/hdl)
        if val < 0.11: return 0
        elif val <= 0.21: return 1
        else: return 2
    elif template_name == "CONICITY":        # conicity index
        return 1 if val > 1.25 else 0
    elif template_name == "FIB4":            # FIB-4
        if val < 1.30: return 0
        elif val <= 2.67: return 1
        else: return 2
    elif template_name == "VASCULAR":        # PPI
        return 1 if val > 0.60 else 0
    elif template_name == "TYG":             # log(tg*glu/2)
        return 1 if val > 8.5 else 0
    elif template_name == "HOMA":            # HOMA-IR
        if val < 1.0: return 0
        elif val < 2.5: return 1
        else: return 2
    elif template_name == "NLR":             # neu/lym
        return 1 if val >= 3.0 else 0
    return 0
