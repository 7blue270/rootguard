import numpy as np

# ==========================================
# 1. GRAPH STRUCTURES & DEFINITIONS
# ==========================================

# --- Edges (Dependencies) ---
# Defines the flow of data from Turn 1 -> Turn 5 -> Unique Utility Class
template_edges = [
    # 1. ANEMIA -> anemia_class
    ("hct", "mcv"), ("rbc", "mcv"),       
    ("hb", "mch"), ("rbc", "mch"),        
    ("hb", "mchc"), ("hct", "mchc"),      
    ("mchc", "anemia_class"),             

    # 2. FIB-4 -> fib4_risk
    ("age", "fib4_prod"), ("ast", "fib4_prod"),  
    ("plt", "fib4_denom"), ("alt", "fib4_denom"),
    ("fib4_prod", "fib4"), ("fib4_denom", "fib4"), 
    ("fib4", "fib4_risk"),                

    # 4. AIP -> aip_risk
    ("tc", "non_hdl"), ("hdl", "non_hdl"), 
    ("tc", "ldl"), ("hdl", "ldl"), ("tg", "ldl"), 
    ("tg", "aip"), ("hdl", "aip"),        
    ("aip", "aip_risk"),                  

    # 5. OBESITY -> ci_risk
    ("wt", "bmi"), ("ht", "bmi"),         
    ("waist", "wthr"), ("ht", "wthr"),    
    ("waist", "conicity"), ("wt", "conicity"), ("ht", "conicity"), 
    ("conicity", "ci_risk"),              

    # 6. VASCULAR -> ppi_risk
    ("sbp", "pp"), ("dbp", "pp"),         
    ("sbp", "map"), ("dbp", "map"),       
    ("sbp", "mbp"), ("dbp", "mbp"),       
    ("pp", "ppi"), ("sbp", "ppi"),        
    ("ppi", "ppi_risk"),                  

    # 7. TYG -> tyg_class
    ("tg", "tyg_prod"), ("glu", "tyg_prod"), 
    ("tyg_prod", "tyg_half"),                
    ("tyg_half", "tyg"),                     
    ("tyg", "tyg_class"),                 

    # 8. HOMA -> homa_class
    ("glu", "homa_prod"), ("ins", "homa_prod"), 
    ("homa_prod", "homa"),                   
    ("homa", "homa_class"),               

    # 9. NLR -> nlr_class
    ("neu", "nlr_sum"), ("lym", "nlr_sum"),   
    ("neu", "nlr_diff"), ("lym", "nlr_diff"), 
    ("neu", "nlr"), ("lym", "nlr"),            
    ("nlr", "nlr_class")                  
]

# --- Nodes (Dependency Lists) ---
# Explicitly lists inputs for every node. Roots are None.
template_nodes = {
    # ROOTS (Turn 1 & 2 Inputs)
    "hb": None, "hct": None, "rbc": None,
    "age": None,
    "ast": None, "alt": None, "plt": None,
    "tc": None, "hdl": None, "tg": None,
    "wt": None, "ht": None, "waist": None,
    "sbp": None, "dbp": None,
    "glu": None, "ins": None,
    "neu": None, "lym": None,

    # INTERMEDIATES & CALCULATED VALUES
    "mcv": ["hct", "rbc"],
    "mch": ["hb", "rbc"],
    "mchc": ["hb", "hct"],
    "fib4_prod": ["age", "ast"],
    "fib4_denom": ["plt", "alt"],
    "fib4": ["fib4_prod", "fib4_denom"], 
    "non_hdl": ["tc", "hdl"],
    "ldl": ["tc", "hdl", "tg"],
    "aip": ["tg", "hdl"],
    "bmi": ["wt", "ht"],
    "wthr": ["waist", "ht"],
    "conicity": ["waist", "wt", "ht"],
    "pp": ["sbp", "dbp"],
    "map": ["sbp", "dbp"],
    "mbp": ["sbp", "dbp"],
    "ppi": ["pp", "sbp"],
    "tyg_prod": ["tg", "glu"],
    "tyg_half": ["tyg_prod"],
    "tyg": ["tyg_half"], 
    "homa_prod": ["glu", "ins"],
    "homa": ["homa_prod"],
    "nlr_sum": ["neu", "lym"],
    "nlr_diff": ["neu", "lym"],
    "nlr": ["neu", "lym"],

    # UNIQUE UTILITY NODES
    "anemia_class": ["mchc"],
    "fib4_risk": ["fib4"],
    "aip_risk": ["aip"],
    "ci_risk": ["conicity"],
    "ppi_risk": ["ppi"],
    "tyg_class": ["tyg"],
    "homa_class": ["homa"],
    "nlr_class": ["nlr"]
}

# --- Layers ---
# Defines the depth in the graph.
template_layers = {
    # Layer 0: Roots
    **{k: 0 for k, v in template_nodes.items() if v is None},
    
    # Layer 1: First Calculation
    "mcv": 1, "mch": 1, "mchc": 1,
    "fib4_prod": 1, "fib4_denom": 1,
    "non_hdl": 1, "ldl": 1, "aip": 1,
    "bmi": 1, "wthr": 1, "conicity": 1,
    "pp": 1, "map": 1, "mbp": 1,
    "tyg_prod": 1,
    "homa_prod": 1,
    "nlr_sum": 1, "nlr_diff": 1, "nlr": 1,

    # Layer 2: Derived from Layer 1
    "fib4": 2,      
    "ppi": 2,       
    "tyg_half": 2,  
    "homa": 2,
    # Direct Utility Classifications (from L1)
    "anemia_class": 2, 
    "aip_risk": 2,     
    "ci_risk": 2,      
    "nlr_class": 2,    

    # Layer 3: Deep Derived
    "tyg": 3,
    # Utility Classifications (from L2)
    "fib4_risk": 3,    
    "ppi_risk": 3,     
    "homa_class": 3,   

    # Layer 4: Deepest
    "tyg_class": 4     
}

# --- Mathematical Expressions ---
# Used for string-based evaluation or debugging
template_expressions = {
    # Anemia
    "mcv": "(hct / rbc) * 10",
    "mch": "(hb / rbc) * 10",
    "mchc": "(hb / hct) * 100",
    "anemia_class": "1 if (mchc < 32 or mchc > 36) else 0", # 1=Abnormal
    
    # FIB-4
    "fib4_prod": "age * ast",
    "fib4_denom": "plt * np.sqrt(alt)",
    "fib4": "fib4_prod / fib4_denom",
    "fib4_risk": "2 if fib4 > 2.67 else (1 if fib4 > 1.30 else 0)", 
    
    # Lipids
    "non_hdl": "tc - hdl",
    "ldl": "tc - hdl - (tg / 5)",
    "aip": "np.log10(tg / hdl)",
    "aip_risk": "2 if aip > 0.21 else (1 if aip > 0.11 else 0)", 
    
    # Obesity
    "bmi": "wt / (ht**2)",
    "wthr": "waist / ht",
    "conicity": "waist / (0.109 * np.sqrt(wt / ht))",
    "ci_risk": "1 if conicity > 1.25 else 0", 
    
    # Vascular
    "pp": "sbp - dbp",
    "map": "dbp + (pp/3)",
    "mbp": "(sbp + dbp)/2",
    "ppi": "pp / sbp",
    "ppi_risk": "1 if ppi > 0.60 else 0", 
    
    # TyG
    "tyg_prod": "tg * glu",
    "tyg_half": "tyg_prod / 2",
    "tyg": "np.log(tyg_half)",
    "tyg_class": "1 if tyg > 8.5 else 0",
    
    # HOMA
    "homa_prod": "glu * ins",
    "homa": "homa_prod / 405",
    "homa_class": "1 if homa > 2.5 else 0", 
    
    # NLR
    "nlr_sum": "neu + lym",
    "nlr_diff": "abs(neu - lym)",
    "nlr": "neu / lym",
    "nlr_class": "1 if nlr >= 3.0 else 0" 
}

def get_all_bi_lipschitz_constants(samples=100000):
    """
    Calculates m and L for every computed node in the clinical templates.
    """
    
    # 1. Define Physiological Sampling Domain
    data = {
        "age": np.random.uniform(18, 90, samples),
        "ast": np.random.uniform(8, 300, samples),       
        "alt": np.random.uniform(7, 300, samples),       
        "plt": np.random.uniform(20, 700, samples),      
        "tc": np.random.uniform(80, 500, samples),       
        "hdl": np.random.uniform(15, 120, samples),      
        "tg": np.random.uniform(30, 800, samples),       
        "wt": np.random.uniform(35, 180, samples),       
        "ht": np.random.uniform(1.3, 2.1, samples),      
        "waist": np.random.uniform(0.5, 1.5, samples),   
        "sbp": np.random.uniform(70, 220, samples),      
        "dbp": np.random.uniform(30, 130, samples),      
        "glu": np.random.uniform(40, 500, samples),      
        "ins": np.random.uniform(1, 60, samples),        
        "neu": np.random.uniform(0.5, 20.0, samples),    
        "lym": np.random.uniform(0.2, 8.0, samples),     
        "hb": np.random.uniform(5, 20, samples),         
        "hct": np.random.uniform(15, 60, samples),       
        "rbc": np.random.uniform(2.0, 7.5, samples),     
        "stiff": np.random.uniform(1.5, 90.0, samples),  
        "cap": np.random.uniform(80, 450, samples),      
        
        # Ranges for Intermediate Nodes
        "fib4_prod": np.random.uniform(140, 27000, samples),
        "fib4_denom": np.random.uniform(50, 12000, samples),
        "pp": np.random.uniform(10, 120, samples),           
        "tyg_half": np.random.uniform(600, 200000, samples), 
        "homa_prod": np.random.uniform(40, 30000, samples)   
    }

    constants = {}
    
    # List of all calculation nodes
    nodes = [
        "mcv", "mch", "mchc",
        "fib4_prod", "fib4_denom", "fib4",
        "non_hdl", "ldl", "aip", "bmi", "wthr", "conicity",
        "pp", "map", "mbp", "ppi",
        "tyg_prod", "tyg_half", "tyg",
        "homa_prod", "homa",
        "nlr_sum", "nlr_diff", "nlr"
        # Utility nodes are handled as defaults or explicitly added below
    ]

    for func_name in nodes:
        grads_norm = []
        
        if func_name == "mcv":
            grads_norm = np.sqrt((10/data["rbc"])**2 + (-10*data["hct"]/(data["rbc"]**2))**2)
        elif func_name == "mch":
            grads_norm = np.sqrt((10/data["rbc"])**2 + (-10*data["hb"]/(data["rbc"]**2))**2)
        elif func_name == "mchc":
            grads_norm = np.sqrt((100/data["hct"])**2 + (-100*data["hb"]/(data["hct"]**2))**2)
        elif func_name == "fib4_prod":
            grads_norm = np.sqrt(data["ast"]**2 + data["age"]**2)
        elif func_name == "fib4_denom":
            d_plt = np.sqrt(data["alt"])
            d_alt = 0.5 * data["plt"] / np.sqrt(data["alt"])
            grads_norm = np.sqrt(d_plt**2 + d_alt**2)
        elif func_name == "fib4":
            d_prod = 1 / data["fib4_denom"]
            d_denom = -data["fib4_prod"] / (data["fib4_denom"]**2)
            grads_norm = np.sqrt(d_prod**2 + d_denom**2)
        elif func_name == "non_hdl": 
            grads_norm = np.full(samples, np.sqrt(1**2 + (-1)**2))
        elif func_name == "ldl": 
            grads_norm = np.full(samples, np.sqrt(1**2 + 1**2 + 0.2**2))
        elif func_name == "aip": 
            ln10 = np.log(10)
            grads_norm = np.sqrt((1/(data["tg"]*ln10))**2 + (-1/(data["hdl"]*ln10))**2)
        elif func_name == "bmi": 
            grads_norm = np.sqrt((1/data["ht"]**2)**2 + (-2*data["wt"]/(data["ht"]**3))**2)
        elif func_name == "wthr": 
            grads_norm = np.sqrt((1/data["ht"])**2 + (-data["waist"]/(data["ht"]**2))**2)
        elif func_name == "conicity":
            denom = 0.109 * np.sqrt(data["wt"] / data["ht"])
            d_waist = 1 / denom
            term = -0.5 * data["waist"] / (0.109 * np.sqrt(data["wt"]**3 / data["ht"])) 
            grads_norm = np.sqrt(d_waist**2 + term**2)
        elif func_name in ["pp", "map", "mbp"]:
            if func_name == "pp": val = np.sqrt(2)
            elif func_name == "map": val = np.sqrt((1/3)**2 + (2/3)**2)
            elif func_name == "mbp": val = np.sqrt(0.5**2 + 0.5**2)
            grads_norm = np.full(samples, val)
        elif func_name == "ppi": 
            grads_norm = np.sqrt((1/data["sbp"])**2 + (-data["pp"]/(data["sbp"]**2))**2)
        elif func_name == "tyg_prod": 
            grads_norm = np.sqrt(data["glu"]**2 + data["tg"]**2)
        elif func_name == "tyg_half": 
            grads_norm = np.full(samples, 0.5)
        elif func_name == "tyg": 
            grads_norm = 1 / data["tyg_half"]
        elif func_name == "homa_prod": 
            grads_norm = np.sqrt(data["ins"]**2 + data["glu"]**2)
        elif func_name == "homa": 
            grads_norm = np.full(samples, 1/405)
        elif func_name in ["nlr_sum", "nlr_diff"]:
            grads_norm = np.full(samples, np.sqrt(2))
        elif func_name == "nlr":
            grads_norm = np.sqrt((1/data["lym"])**2 + (-data["neu"]/(data["lym"]**2))**2)

        if len(grads_norm) > 0:
            m = float(np.percentile(grads_norm, 1))  
            L = float(np.percentile(grads_norm, 99)) 
            constants[func_name] = (round(m, 4), round(L, 4))
        else:
            constants[func_name] = (1.0, 1.0)
            
    # Add manual constants for Utility/Risk Nodes (Categorical = 1.0)
    utility_nodes = [
        "anemia_class", "fib4_risk", "aip_risk",
        "ci_risk", "ppi_risk", "tyg_class", "homa_class", 
        "nlr_class"
    ]
    for u in utility_nodes:
        constants[u] = (1.0, 1.0)

    return constants

# --- Execution ---
def analysis():
    lipschitz_map = get_all_bi_lipschitz_constants()
    return lipschitz_map

# Master constants dictionary with Unique Keys
bilipschitz_constants = {
    # Calculated Values
    'mcv': (3.81, 111.9106), 'mch': (1.8505, 37.3144), 'mchc': (1.7117, 9.0415),
    'fib4_prod': (34.6123, 303.2436), 'fib4_denom': (8.0504, 75.496), 'fib4': (0.0001, 0.4643), 
    'non_hdl': (1.4142, 1.4142), 'ldl': (1.4283, 1.4283), 'aip': (0.0038, 0.0272), 
    'bmi': (10.1571, 137.2895), 'wthr': (0.5053, 1.0796), 'conicity': (0.8191, 2.0561), 
    'pp': (1.4142, 1.4142), 'map': (0.7454, 0.7454), 'mbp': (0.7071, 0.7071), 'ppi': (0.0047, 0.0227), 
    'tyg_prod': (112.2769, 889.0305), 'tyg_half': (0.5, 0.5), 'tyg': (0.0, 0.0004), 
    'homa_prod': (52.0679, 496.2558), 'homa': (0.0025, 0.0025),
    'nlr_sum': (1.4142, 1.4142), 'nlr_diff': (1.4142, 1.4142), 'nlr': (0.1386, 136.3953),
    
    # Unique Utility / Risk Classes (Default 1.0)
    "anemia_class": (1.0, 1.0), "fib4_risk": (1.0, 1.0),
    "aip_risk": (1.0, 1.0), "ci_risk": (1.0, 1.0), "ppi_risk": (1.0, 1.0), 
    "tyg_class": (1.0, 1.0), "homa_class": (1.0, 1.0), 
    "nlr_class": (1.0, 1.0)
}

# --- Target Keys ---
# Maps template name (uppercase) to the target variable node for evaluation
template_target_keys = {
    "ANEMIA": "mchc", "AIP": "aip", "CONICITY": "conicity",
    "VASCULAR": "ppi", "FIB4": "fib4", "TYG": "tyg", "HOMA": "homa",
    "NLR": "nlr",
}

# --- CDC NHANES 2017-2018 Population Means ---
# Computed from the full adult male subpopulation (n=2840) of NHANES 2017-2018
# raw XPT files (data/CDC/*.xpt). These are public aggregate statistics.
# Used for population-mean sensitivity estimation in budget allocation.
CDC_POPULATION_MEANS = {
    "age": 50.2335, "alt": 26.1125, "ast": 23.7857,
    "dbp": 73.5658, "glu": 116.4223, "hb": 14.8091,
    "hct": 43.8850, "hdl": 48.2454, "ht": 1.7348,
    "ins": 14.6411, "lym": 2.1867, "neu": 4.1751,
    "plt": 226.4986, "rbc": 4.9536, "sbp": 127.3083,
    "tc": 183.0184, "tg": 123.2650, "waist": 1.0204, "wt": 88.2404,
}


def get_local_partials(node_name, values):
    """
    Return exact analytical partial derivatives of node_name w.r.t. all
    variables appearing in its expression, evaluated at the given values.

    These are LOCAL partials only (w.r.t. immediate expression variables).
    Use forward-mode AD through the DAG for end-to-end sensitivities.

    Args:
        node_name: Name of the computed node
        values: Dict of {variable_name: value} for all nodes

    Returns:
        Dict {variable: d(node)/d(variable)} or None for classification/unknown nodes
    """

    # --- Anemia ---
    if node_name == "mcv":
        # mcv = (hct / rbc) * 10
        return {"hct": 10.0 / values["rbc"],
                "rbc": -10.0 * values["hct"] / values["rbc"]**2}

    elif node_name == "mch":
        # mch = (hb / rbc) * 10
        return {"hb": 10.0 / values["rbc"],
                "rbc": -10.0 * values["hb"] / values["rbc"]**2}

    elif node_name == "mchc":
        # mchc = (hb / hct) * 100
        return {"hb": 100.0 / values["hct"],
                "hct": -100.0 * values["hb"] / values["hct"]**2}

    # --- FIB-4 ---
    elif node_name == "fib4_prod":
        # fib4_prod = age * ast
        return {"age": values["ast"], "ast": values["age"]}

    elif node_name == "fib4_denom":
        # fib4_denom = plt * sqrt(alt)
        alt = values["alt"]
        return {"plt": np.sqrt(alt),
                "alt": 0.5 * values["plt"] / np.sqrt(alt)}

    elif node_name == "fib4":
        # fib4 = fib4_prod / fib4_denom
        denom = values["fib4_denom"]
        return {"fib4_prod": 1.0 / denom,
                "fib4_denom": -values["fib4_prod"] / denom**2}

    # --- Lipids ---
    elif node_name == "non_hdl":
        # non_hdl = tc - hdl
        return {"tc": 1.0, "hdl": -1.0}

    elif node_name == "ldl":
        # ldl = tc - hdl - tg/5
        return {"tc": 1.0, "hdl": -1.0, "tg": -0.2}

    elif node_name == "aip":
        # aip = log10(tg / hdl)
        ln10 = np.log(10)
        return {"tg": 1.0 / (values["tg"] * ln10),
                "hdl": -1.0 / (values["hdl"] * ln10)}

    # --- Obesity ---
    elif node_name == "bmi":
        # bmi = wt / ht^2
        ht = values["ht"]
        return {"wt": 1.0 / ht**2,
                "ht": -2.0 * values["wt"] / ht**3}

    elif node_name == "wthr":
        # wthr = waist / ht
        ht = values["ht"]
        return {"waist": 1.0 / ht,
                "ht": -values["waist"] / ht**2}

    elif node_name == "conicity":
        # conicity = waist / (0.109 * sqrt(wt / ht))
        wt, ht, waist = values["wt"], values["ht"], values["waist"]
        sqrt_wt_over_ht = np.sqrt(wt / ht)
        return {
            "waist": 1.0 / (0.109 * sqrt_wt_over_ht),
            "wt": -waist / (2.0 * 0.109 * wt * sqrt_wt_over_ht),
            "ht": waist / (2.0 * 0.109 * ht * sqrt_wt_over_ht),
        }

    # --- Vascular ---
    elif node_name == "pp":
        # pp = sbp - dbp
        return {"sbp": 1.0, "dbp": -1.0}

    elif node_name == "map":
        # map = dbp + pp/3  (pp is an intermediate variable in scope)
        return {"dbp": 1.0, "pp": 1.0 / 3.0}

    elif node_name == "mbp":
        # mbp = (sbp + dbp) / 2
        return {"sbp": 0.5, "dbp": 0.5}

    elif node_name == "ppi":
        # ppi = pp / sbp
        sbp = values["sbp"]
        pp = values["pp"]
        return {"pp": 1.0 / sbp,
                "sbp": -pp / sbp**2}

    # --- TyG ---
    elif node_name == "tyg_prod":
        # tyg_prod = tg * glu
        return {"tg": values["glu"], "glu": values["tg"]}

    elif node_name == "tyg_half":
        # tyg_half = tyg_prod / 2
        return {"tyg_prod": 0.5}

    elif node_name == "tyg":
        # tyg = ln(tyg_half)
        return {"tyg_half": 1.0 / values["tyg_half"]}

    # --- HOMA ---
    elif node_name == "homa_prod":
        # homa_prod = glu * ins
        return {"glu": values["ins"], "ins": values["glu"]}

    elif node_name == "homa":
        # homa = homa_prod / 405
        return {"homa_prod": 1.0 / 405.0}

    # --- NLR ---
    elif node_name == "nlr_sum":
        # nlr_sum = neu + lym
        return {"neu": 1.0, "lym": 1.0}

    elif node_name == "nlr_diff":
        # nlr_diff = |neu - lym|
        sign = 1.0 if values["neu"] >= values["lym"] else -1.0
        return {"neu": sign, "lym": -sign}

    elif node_name == "nlr":
        # nlr = neu / lym
        lym = values["lym"]
        return {"neu": 1.0 / lym,
                "lym": -values["neu"] / lym**2}

    else:
        # Classification/risk nodes or unknown — not differentiable
        return None


def get_worst_case_partials(node_name, domains):
    """
    Return worst-case (supremum) absolute values of local partial derivatives
    of node_name w.r.t. each parent variable, over the given domain bounds.

    Data-independent: depends only on public domain bounds, not on any sample.
    Each derivative is monotonic in each variable over positive domains,
    so the supremum is attained at a domain boundary.

    Args:
        node_name: Name of the computed node
        domains: Dict of {variable_name: (lo, hi)} for all relevant nodes

    Returns:
        Dict {variable: sup|d(node)/d(variable)|} or None for classification/unknown nodes
    """

    # --- Anemia ---
    if node_name == "mcv":
        # mcv = (hct / rbc) * 10
        # |∂mcv/∂hct| = 10/|rbc|, maximized at rbc_lo
        # |∂mcv/∂rbc| = 10*|hct|/rbc², maximized at hct_hi, rbc_lo
        rbc_lo = domains["rbc"][0]
        hct_hi = domains["hct"][1]
        return {"hct": 10.0 / rbc_lo,
                "rbc": 10.0 * hct_hi / rbc_lo**2}

    elif node_name == "mch":
        # mch = (hb / rbc) * 10
        rbc_lo = domains["rbc"][0]
        hb_hi = domains["hb"][1]
        return {"hb": 10.0 / rbc_lo,
                "rbc": 10.0 * hb_hi / rbc_lo**2}

    elif node_name == "mchc":
        # mchc = (hb / hct) * 100
        hct_lo = domains["hct"][0]
        hb_hi = domains["hb"][1]
        return {"hb": 100.0 / hct_lo,
                "hct": 100.0 * hb_hi / hct_lo**2}

    # --- FIB-4 ---
    elif node_name == "fib4_prod":
        # fib4_prod = age * ast
        return {"age": domains["ast"][1],
                "ast": domains["age"][1]}

    elif node_name == "fib4_denom":
        # fib4_denom = plt * sqrt(alt)
        alt_hi = domains["alt"][1]
        alt_lo = domains["alt"][0]
        plt_hi = domains["plt"][1]
        return {"plt": np.sqrt(alt_hi),
                "alt": 0.5 * plt_hi / np.sqrt(alt_lo)}

    elif node_name == "fib4":
        # fib4 = fib4_prod / fib4_denom
        denom_lo = domains["fib4_denom"][0]
        prod_hi = domains["fib4_prod"][1]
        return {"fib4_prod": 1.0 / denom_lo,
                "fib4_denom": prod_hi / denom_lo**2}

    # --- Lipids ---
    elif node_name == "non_hdl":
        # non_hdl = tc - hdl  (constant partials)
        return {"tc": 1.0, "hdl": 1.0}

    elif node_name == "ldl":
        # ldl = tc - hdl - tg/5  (constant partials)
        return {"tc": 1.0, "hdl": 1.0, "tg": 0.2}

    elif node_name == "aip":
        # aip = log10(tg / hdl)
        # |∂aip/∂tg| = 1/(tg*ln10), maximized at tg_lo
        # |∂aip/∂hdl| = 1/(hdl*ln10), maximized at hdl_lo
        ln10 = np.log(10)
        return {"tg": 1.0 / (domains["tg"][0] * ln10),
                "hdl": 1.0 / (domains["hdl"][0] * ln10)}

    # --- Obesity ---
    elif node_name == "bmi":
        # bmi = wt / ht^2
        ht_lo = domains["ht"][0]
        wt_hi = domains["wt"][1]
        return {"wt": 1.0 / ht_lo**2,
                "ht": 2.0 * wt_hi / ht_lo**3}

    elif node_name == "wthr":
        # wthr = waist / ht
        ht_lo = domains["ht"][0]
        waist_hi = domains["waist"][1]
        return {"waist": 1.0 / ht_lo,
                "ht": waist_hi / ht_lo**2}

    elif node_name == "conicity":
        # conicity = waist / (0.109 * sqrt(wt / ht))
        # |∂/∂waist| = 1/(0.109*sqrt(wt/ht)), maximized at wt_lo, ht_hi
        # |∂/∂wt| = waist/(2*0.109*wt*sqrt(wt/ht)), maximized at waist_hi, wt_lo, ht_hi
        # |∂/∂ht| = waist/(2*0.109*ht*sqrt(wt/ht)), maximized at waist_hi, ht_lo, wt_lo (but also ht_lo in denom)
        wt_lo = domains["wt"][0]
        ht_lo = domains["ht"][0]
        ht_hi = domains["ht"][1]
        waist_hi = domains["waist"][1]
        sqrt_min = np.sqrt(wt_lo / ht_hi)  # smallest sqrt(wt/ht)
        return {
            "waist": 1.0 / (0.109 * sqrt_min),
            "wt": waist_hi / (2.0 * 0.109 * wt_lo * sqrt_min),
            "ht": waist_hi / (2.0 * 0.109 * ht_lo * np.sqrt(wt_lo / ht_lo)),
        }

    # --- Vascular ---
    elif node_name == "pp":
        # pp = sbp - dbp  (constant partials)
        return {"sbp": 1.0, "dbp": 1.0}

    elif node_name == "map":
        # map = dbp + pp/3  (constant partials)
        return {"dbp": 1.0, "pp": 1.0 / 3.0}

    elif node_name == "mbp":
        # mbp = (sbp + dbp) / 2  (constant partials)
        return {"sbp": 0.5, "dbp": 0.5}

    elif node_name == "ppi":
        # ppi = pp / sbp
        # |∂/∂pp| = 1/sbp, maximized at sbp_lo
        # |∂/∂sbp| = pp/sbp², maximized at pp_hi, sbp_lo
        sbp_lo = domains["sbp"][0]
        pp_hi = domains["pp"][1]
        return {"pp": 1.0 / sbp_lo,
                "sbp": pp_hi / sbp_lo**2}

    # --- TyG ---
    elif node_name == "tyg_prod":
        # tyg_prod = tg * glu
        return {"tg": domains["glu"][1],
                "glu": domains["tg"][1]}

    elif node_name == "tyg_half":
        # tyg_half = tyg_prod / 2  (constant partial)
        return {"tyg_prod": 0.5}

    elif node_name == "tyg":
        # tyg = ln(tyg_half)
        # |∂/∂tyg_half| = 1/tyg_half, maximized at tyg_half_lo
        return {"tyg_half": 1.0 / domains["tyg_half"][0]}

    # --- HOMA ---
    elif node_name == "homa_prod":
        # homa_prod = glu * ins
        return {"glu": domains["ins"][1],
                "ins": domains["glu"][1]}

    elif node_name == "homa":
        # homa = homa_prod / 405  (constant partial)
        return {"homa_prod": 1.0 / 405.0}

    # --- NLR ---
    elif node_name == "nlr_sum":
        # nlr_sum = neu + lym  (constant partials)
        return {"neu": 1.0, "lym": 1.0}

    elif node_name == "nlr_diff":
        # nlr_diff = |neu - lym|  (absolute value of sign = 1.0)
        return {"neu": 1.0, "lym": 1.0}

    elif node_name == "nlr":
        # nlr = neu / lym
        # |∂/∂neu| = 1/lym, maximized at lym_lo
        # |∂/∂lym| = neu/lym², maximized at neu_hi, lym_lo
        lym_lo = domains["lym"][0]
        neu_hi = domains["neu"][1]
        return {"neu": 1.0 / lym_lo,
                "lym": neu_hi / lym_lo**2}

    else:
        # Classification/risk nodes or unknown — not differentiable
        return None