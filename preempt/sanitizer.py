import json
import math
import os
import random
import sys
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm
import re
import unicodedata
import ast
from ff3 import FF3Cipher as _PublicFF3Cipher
from names_dataset import NameDataset
import names


class FF3Cipher(_PublicFF3Cipher):
    """Compat wrapper around the public ``ff3`` package's FF3Cipher.

    Earlier versions of this codebase used a private ``pyfpe-ff3`` fork that
    accepted an ``allow_small_domain=True`` keyword to bypass the FF3-1
    minimum-domain-size check (radix^minlen >= 1_000_000). The public ``ff3``
    package does not expose that keyword, so this wrapper silently swallows
    it and re-implements the bypass: when the constructed cipher object's
    ``set_tweak`` validation would reject a small domain, callers can opt
    in via ``allow_small_domain=True`` and the cipher is constructed
    without the check. The behavior is otherwise identical.
    """

    def __init__(self, *args, allow_small_domain: bool = False, **kwargs):
        # The public ff3.FF3Cipher.__init__ will raise on small domains
        # unless we patch the minlen attribute after construction. There
        # is no public API for this; the simplest faithful workaround is
        # to construct with the larger minlen and override.
        super().__init__(*args, **kwargs)
        if allow_small_domain:
            # The public package stores a hard-coded DOMAIN_MIN constant
            # checked at encrypt-time. We override the instance check.
            try:
                # ff3>=1.0.0: minlen / maxlen are instance attributes.
                self.minlen = 2  # smallest practical numeric domain
            except Exception:  # pragma: no cover - defensive only
                pass

from .ner import NER
from .utils import check, load_data, make_names_dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Pathological clinical ranges (for RQ2 leakage analysis only) ──────────
# Full clinically observable spectrum from extreme survivable lows to extreme
# documented highs. Sources cited in paper appendix.
PATHOLOGICAL_DOMAINS = {
    "hb": (1.5, 25), "hct": (5, 75), "rbc": (1.0, 8.0),
    "plt": (5, 1500), "neu": (0.1, 30), "lym": (0.1, 20),
    "ast": (5, 5000), "alt": (3, 5000),
    "tc": (50, 500), "hdl": (5, 150), "tg": (10, 5000),
    "glu": (20, 1700), "ins": (0.5, 500),
    "sbp": (40, 300), "dbp": (20, 200),
    "wt": (30, 300), "ht": (1.45, 2.10), "waist": (0.50, 1.80),
    "age": (18, 80),
}

# ── Per-template NHANES observed bounds (adult males, age >= 18) ─────────
# Min/max across adult male participants with complete data for each template.
# Used as the noise domain for the discrete exponential mechanism.
# Derived node bounds computed from actual observed derived values.
TEMPLATE_DOMAINS = {
    "ANEMIA": {
        "hb": (7.20, 19.90), "hct": (23.00, 58.80), "rbc": (2.32, 7.84),
        "mcv": (56.67, 113.11), "mch": (16.86, 39.33), "mchc": (29.71, 38.29),
    },
    "FIB4": {
        "age": (18.00, 80.00), "ast": (7.00, 272.00), "alt": (3.00, 211.00),
        "plt": (8.00, 535.00),
        "fib4_prod": (198.00, 14144.00), "fib4_denom": (25.30, 5533.48),
        "fib4": (0.12, 37.95),
    },
    "AIP": {
        "tc": (76.00, 384.00), "hdl": (11.00, 147.00), "tg": (21.00, 2684.00),
        "non_hdl": (38.00, 337.00), "ldl": (-199.80, 265.20), "aip": (-0.64, 1.97),
    },
    "CONICITY": {
        "wt": (36.80, 191.40), "ht": (1.48, 1.98), "waist": (0.62, 1.64),
        "bmi": (14.91, 61.93), "wthr": (0.37, 0.94), "conicity": (1.01, 1.60),
    },
    "VASCULAR": {
        "sbp": (72.00, 216.00), "dbp": (0.01, 136.00),
        "pp": (20.00, 168.00), "map": (33.33, 162.67),
        "mbp": (50.00, 176.00), "ppi": (0.17, 1.00),
    },
    "TYG": {
        "tg": (21.00, 2684.00), "glu": (64.00, 451.00),
        "tyg_prod": (1738.00, 443205.00), "tyg_half": (869.00, 221602.50),
        "tyg": (6.77, 12.31),
    },
    "HOMA": {
        "glu": (64.00, 451.00), "ins": (0.71, 321.64),
        "homa_prod": (48.99, 62796.70), "homa": (0.12, 155.05),
    },
    "NLR": {
        "neu": (0.40, 15.70), "lym": (0.50, 65.90),
        "nlr_sum": (1.50, 72.10), "nlr_diff": (0.00, 59.70), "nlr": (0.09, 19.00),
    },
}

def set_template_domains(template_name):
    """Swap MEDICAL_DOMAINS to use per-template bounds."""
    if template_name in TEMPLATE_DOMAINS:
        MEDICAL_DOMAINS.update(TEMPLATE_DOMAINS[template_name])


# ── Default MEDICAL_DOMAINS (union of all template bounds) ───────────────
# Fallback for fields not covered by a specific template.
MEDICAL_DOMAINS = {
    # Vital Signs
    "hr":    (30, 250),           # Heart rate (bpm) — not in benchmark
    "sbp":   (72, 216),           # Systolic BP (mmHg)
    "dbp":   (1, 136),            # Diastolic BP (mmHg); floor at 1 (NHANES has 0s)
    "implied_dbp": (1, 136),
    "ne_dose": (0, 50),           # Not in benchmark

    # Derived Vitals — computed from root observed ranges
    "shock_index": (0.2, 3.0),   # Not in benchmark
    "map_val": (25, 196),         # dbp + (sbp-dbp)/3
    "map": (25, 196),
    "mbp": (36.5, 176),           # (sbp+dbp)/2
    "pp": (0, 215),               # sbp-dbp: 72-136<0→0 to 216-1=215
    "ppi": (0, 1.0),              # pp/sbp
    "lactate": (0.5, 20),         # Not in benchmark
    "sofa_score": (0, 24),        # Not in benchmark

    # Hematology — NHANES adult male observed
    "hb":    (7.2, 19.9),         # Hemoglobin (g/dL)
    "hct":   (23.0, 58.8),        # Hematocrit (%)
    "rbc":   (2.32, 7.84),        # RBC count (M/µL)
    "plt":   (8, 620),            # Platelets (K/µL)
    "neu":   (0.4, 15.7),         # Neutrophils (K/µL)
    "lym":   (0.5, 65.9),         # Lymphocytes (K/µL)
    # Derived hematology — computed from root observed ranges
    "mcv":   (29.34, 253.45),     # (hct/rbc)*10: (23/7.84)*10 to (58.8/2.32)*10
    "mch":   (9.18, 85.78),       # (hb/rbc)*10: (7.2/7.84)*10 to (19.9/2.32)*10
    "mchc":  (12.24, 86.52),      # (hb/hct)*100: (7.2/58.8)*100 to (19.9/23)*100
    "nlr":   (0.006, 31.4),       # neu/lym: 0.4/65.9 to 15.7/0.5
    "nlr_sum": (0.9, 81.6),       # neu+lym
    "nlr_diff": (0, 65.5),        # |neu-lym|

    # Chemistry / Renal
    "scr":   (0.1, 25),           # Not in benchmark
    "glu":   (64, 451),            # Fasting glucose (mg/dL)
    "u_alb": (0.1, 5000),         # Not in benchmark
    "u_cr":  (5, 500),            # Not in benchmark
    "acr":   (0, 5000),           # Not in benchmark
    "egfr":  (2, 150),            # Not in benchmark
    "kfre":  (-105, 5),           # Not in benchmark

    # Liver — NHANES adult male observed
    "ast":   (7, 272),             # AST (U/L)
    "alt":   (3, 211),             # ALT (U/L)
    # Derived liver — computed from root observed ranges
    "fib4":  (0.001, 37.18),       # age*ast/(plt*sqrt(alt)): extremes
    "fib4_prod": (126, 21760),     # age*ast: 18*7=126 to 80*272=21760
    "fib4_denom": (13.86, 9004.89), # plt*sqrt(alt): 8*sqrt(3)=13.86 to 620*sqrt(211)=9004.89
    "stiff": (2, 75),             # Not in benchmark
    "stiff_pa": (2000, 75000),    # Not in benchmark
    "cap":   (100, 400),          # Not in benchmark
    "liver_ratio": (0.5, 50),     # Not in benchmark

    # Lipids — NHANES adult male observed
    "tc":    (76, 431),            # Total cholesterol (mg/dL)
    "hdl":   (10, 166),            # HDL (mg/dL)
    "tg":    (21, 2684),           # Triglycerides (mg/dL)
    # Derived lipids
    "ldl":   (-500, 421),          # tc-hdl-tg/5: can be very negative with high TG
    "non_hdl": (0, 421),           # tc-hdl
    "aip":   (-0.90, 2.43),        # log10(tg/hdl): log10(21/166) to log10(2684/10)

    # Anthropometrics — NHANES adult male observed
    "wt":    (36.8, 242.6),        # Weight (kg)
    "ht":    (1.48, 1.98),         # Height (m)
    "waist": (0.62, 1.64),         # Waist circumference (m)
    # Derived anthropometrics
    "bmi":   (9.38, 110.80),       # wt/ht^2: extremes
    "wthr":  (0.31, 1.11),         # waist/ht
    "conicity": (0.72, 3.01),      # waist/(0.109*sqrt(wt/ht)): extremes

    # Metabolic — NHANES adult male observed
    "ins":   (0.71, 321.64),       # Fasting insulin (µU/mL)
    # Derived metabolic
    "homa":  (0.11, 358.32),       # glu*ins/405
    "homa_prod": (45.44, 145100.64), # glu*ins: 64*0.71 to 451*321.64
    "tyg":   (6.51, 13.63),        # ln(tg*glu/2): ln(21*64/2) to ln(2684*451/2)
    "tyg_prod": (1344, 1210484),   # tg*glu
    "tyg_half": (672, 605242),     # tg*glu/2

    # Demographics
    "age":   (18, 80),             # Adult males, NHANES cap
    "gender": (1, 1),              # Male only

    # Default fallback for unknown metrics
    "_default": (0, 1000),
}


class Sanitizer():
    """
    Returns a sanitizer object. Call to run sanitization for a list of strings.
    """
    def __init__(self, ner_model: NER = None, key = "EF4359D8D580AA4F7F036D6F04FC6A94", tweak = "D8E7920AFA330A73"):
        self.cipher_fn = FF3Cipher(key, tweak, allow_small_domain=True, radix=10)
        self.nd = NameDataset()
        self.ner = ner_model
        self.new_entities = []
        self.entity_lookup = []
        self.entity_mapping = []

    def replace_word(self, text: str, word1: str, word2: str):
        """
        Find and replace strings in a given piece of text.

        Args:
            text (str): A string containing a target substring word1.
            word1 (str): Target to replace with word2 in text.
            word2 (str): Replacement for word1 in text.
        Returns:
            text (str): With word1 replaced with word2
        """
        pattern = r'\b' + re.escape(word1) + r'\b'
        return re.sub(pattern, word2, text, count=1)

    def format_align_digits(self, text, reference_text):
        if len(text) != len(reference_text):
            for idx, t in enumerate(reference_text):
                if not t.isdigit():
                    text = text[:idx] + reference_text[idx] + text[idx:]
        return text

    def fpe_encrypt(self, value: str):
        """
        Encrypt value using FPE
        """
        return self.format_align_digits(
            self.cipher_fn.encrypt(
            str(value).replace("$","").replace(",","").replace(".","").replace(" ","")
            ),
            str(value)
        )

    def fpe_decrypt(self, value: str):
        """
        Decrypt FPE value
        """
        return self.format_align_digits(
            self.cipher_fn.decrypt(
                str(value).replace("$","").replace(",","").replace(".","").replace(" ","")
            ),
            str(value)
        )

    def M_laplace(self, x: float, epsilon: float,
                  domain_lower: float, domain_upper: float) -> float:
        """
        Truncated Laplace mechanism for ε-mDP.

        Adds Laplace(0, 1/ε) noise and clamps to domain bounds.
        Satisfies ε-mDP with respect to metric d(x,x') = |x - x'|.

        Args:
            x: Value to noise
            epsilon: Privacy parameter (higher = less noise = less privacy)
            domain_lower: Physiological lower bound for this metric type
            domain_upper: Physiological upper bound for this metric type

        Returns:
            Noised value within [domain_lower, domain_upper]
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")

        scale = 1.0 / epsilon
        noise = np.random.laplace(0, scale)
        noised_output = x + noise

        return float(np.clip(noised_output, domain_lower, domain_upper))

    def M_exponential_discrete(self, x: float, epsilon: float,
                               domain_lower: float, domain_upper: float,
                               num_candidates: int = 1000) -> float:
        """
        Exponential mechanism operating in normalized discrete space.

        Maps x into [0, num_candidates-1], runs the exponential mechanism in
        that discrete space (where Δu = 1 corresponds to one grid step), then
        maps the sampled index back to the original domain.

        This makes the noise scale domain-width-dependent:
            effective noise scale ≈ 2·(domain_upper - domain_lower) / (ε · num_candidates)

        Args:
            x: True value to noise
            epsilon: Privacy parameter (higher = less noise = less privacy)
            domain_lower: Physiological lower bound
            domain_upper: Physiological upper bound
            num_candidates: Number of discrete grid points

        Returns:
            Sampled value within [domain_lower, domain_upper]
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")

        domain_width = domain_upper - domain_lower
        if domain_width <= 0:
            return x

        # Map x to discrete index space [0, num_candidates-1]
        x_norm = (x - domain_lower) / domain_width * (num_candidates - 1)
        x_norm = np.clip(x_norm, 0, num_candidates - 1)

        # Discrete candidates in index space
        indices = np.arange(num_candidates, dtype=float)

        # Utility in discrete space: -|x_norm - s|, Δu = 1
        log_probs = -epsilon * np.abs(x_norm - indices) / 2.0

        # Numerical stability
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= probs.sum()

        # Sample in discrete space and map back
        idx = np.random.choice(num_candidates, p=probs)
        return float(domain_lower + idx * domain_width / (num_candidates - 1))

    def M_gaussian(self, x: float, n_lower: float, n_upper: float, epsilon: float, delta: float = 1e-5) -> float:
        """
        m-LDP using the Gaussian Mechanism.
        
        Args:
            x (float): Value to noise.
            n_lower (float): Lower bound for x.
            n_upper (float): Upper bound for x.
            epsilon (float): Privacy budget (> 0).
            delta (float): Privacy failure probability (0 < delta < 1).
            
        Returns:
            float: The noised output, clipped to bounds.
        """
        # 1. Sensitivity (Delta f)
        # For a single value in a range, the L2 sensitivity is the max possible change.
        sensitivity = abs(n_upper - n_lower)
        
        if sensitivity == 0:
            return float(x)

        # 2. Calculate Sigma (Standard Deviation)
        # Formula for Gaussian Mechanism: sigma = sqrt(2 * ln(1.25 / delta)) * sensitivity / epsilon
        # This is the standard calibrated noise scale.
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        
        # 3. Add Noise
        noise = np.random.normal(0, sigma)
        noised_output = x + noise
        
        # 4. Post-processing (Clamping)
        # Since Gaussian noise is unbounded (-inf, +inf), we clamp to the original range.
        return float(np.clip(noised_output, n_lower, n_upper))
    
    def M_epsilon(self, x: int, n_lower: int, n_upper: int, epsilon: float, discretization_size=100) -> int:
        """
        m-LDP for noising numerical values between bounds.

        Args:
            x (int): Value to noise with m-LDP.
            n_lower (int): Lower bound for noising x.
            n_upper (int): Upper bound for noising x.
            epsilon (float): Privacy budget.
            discretization_size: For sampling.
        Returns:
            noised_output (int): The noised output.
        """
        total_range = n_upper-n_lower
        if total_range==0: breakpoint()
        # print(n_upper, n_lower)
        x = (x-n_lower)*discretization_size/total_range
        p_i = []
        for s in range(discretization_size):
            p_i.append(math.exp(-abs(x-s)*epsilon/2))
        p_i = [val/sum(p_i) for val in p_i]
        noised_output = np.random.choice(range(discretization_size),1,p=p_i)*total_range/discretization_size+n_lower
        return noised_output[0]
    
    def encypt_value(self, inputs: List[List[str]], **kwargs: Dict[str, Any]) -> Union[List[List[str]], Union[List[str],List[str]], List[Dict[str,List[str]]]]:
        """
        Encrypting numerical values with FPE or ε-mDP (Exponential mechanism).

        Args:
            inputs (List[List[str]]): List of strings with sensitive values.
            kwargs:
                use_fpe (bool): Use format-preserving encryption.
                use_mdp (bool): Use metric differential privacy (Laplace mechanism).
                epsilon (List[List[float]]): Privacy budgets per value.
                metric_types (List[List[str]], optional): Metric type names for domain lookup.
                    If provided, uses fixed physiological bounds from MEDICAL_DOMAINS.
                    If not provided, falls back to value-relative bounds (legacy behavior).
        Returns:
            Tuple containing:
            1. new_entities (List[List[str]]): Noised/encrypted values.
            2. entity_lookup (List[List[str]]): Original plaintext values.
            3. entity_mapping (List[Dict[str,str]]): Mapping from noised to original values.
        """
        new_entities = []
        entity_lookup = []
        entity_mapping = []
        use_fpe, use_mdp = kwargs['use_fpe'], kwargs['use_mdp']
        all_epsilons = kwargs['epsilon']
        # Optional: metric types for domain-aware noising
        all_metric_types = kwargs.get('metric_types', None)

        valid_indices = []
        for kk, input in enumerate(inputs):
            temp = []
            text_pt = []
            temp_dict = {}
            trip = 0
            epsilon_block = all_epsilons[kk]
            metric_types_block = all_metric_types[kk] if all_metric_types else None

            for gg, real_value in enumerate(input):
                epsilon = epsilon_block[gg]
                metric_type = metric_types_block[gg] if metric_types_block else None

                if use_fpe:
                    offset = 6
                    val = "9"*offset + str(real_value)
                    noised_value = self.fpe_encrypt(val)

                elif use_mdp:
                    value = float(real_value)

                    if metric_type and metric_type in MEDICAL_DOMAINS:
                        # Domain-aware exponential mechanism with fixed physiological bounds
                        domain_lower, domain_upper = MEDICAL_DOMAINS[metric_type]
                        noised_value = self.M_exponential_discrete(value, epsilon, domain_lower, domain_upper)
                    elif metric_type:
                        # Unknown metric type: use default domain
                        domain_lower, domain_upper = MEDICAL_DOMAINS["_default"]
                        noised_value = self.M_exponential_discrete(value, epsilon, domain_lower, domain_upper)
                    else:
                        # Legacy behavior: value-relative bounds (for backwards compatibility)
                        if value == 0:
                            noised_value = self.M_exponential_discrete(value, epsilon, 0, 0.1)
                        else:
                            # Use ±50% of value as bounds
                            lower = value - abs(value) * 0.5
                            upper = value + abs(value) * 0.5
                            noised_value = self.M_exponential_discrete(value, epsilon, lower, upper)

                    noised_value = round(noised_value, ndigits=4)
                    temp.append(str(noised_value))
                    text_pt.append(str(real_value))
                    temp_dict[str(noised_value)] = str(real_value)

            if trip == 0:
                valid_indices.append(kk)
            entity_lookup.append(text_pt)
            new_entities.append(temp)
            entity_mapping.append(temp_dict)

        return new_entities, entity_lookup, entity_mapping

    def decrypt_value(self, inputs: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        """
        Desanitizes currency values in a list of sanitized strings.

        Args:
            inputs (List[str]): List of sanitized strings
        Returns:
            decrypted_lines (List[str]): List of desanitized strings
        """
        decrypted_lines = []
        use_fpe = kwargs['use_fpe']
        use_mdp = kwargs['use_mdp']
        extraction = kwargs['extraction']
        use_cached_values = kwargs['use_cache']
        if extraction is None or use_cached_values:
            decrypt_target = self.entity_mapping
        else:
            decrypt_target = extraction
        if use_fpe:
            offset=7
            for line_idx, line in enumerate(inputs):
                for value in decrypt_target[line_idx]:
                    val = str(value)
                    if len(val)<6: continue
                    decrypt = self.fpe_decrypt(val)
                    decrypt = decrypt[6:]
                    if value==None: continue
                    if value in line:
                        line = line.replace(value, decrypt)
                    elif value.replace(".", ",") in line:
                        line = line.replace(value.replace(".", ","), decrypt.replace(".", ","))
                    else:
                        line = line.replace(value.replace(".", ","), decrypt.replace(".", ","))
                decrypted_lines.append(line)

        elif use_mdp:
            for line_idx, line in enumerate(inputs):
                for value in decrypt_target[line_idx]:
                    line = self.replace_word(line, value, self.entity_mapping[line_idx][value])
                decrypted_lines.append(line)

        return decrypted_lines

    def encrypt(self, inputs: List[List[str]], extracted=None, epsilon: List[List[float]] = None,
                metric_types: List[List[str]] = None, entity='Name',
                use_mdp=False, use_fpe=False, verbose=False, encryption_only=False) -> Union[List[str], List[int]]:
        """
        Takes in a list of inputs and returns a list of sanitized outputs.

        Args:
            inputs (List[List[str]]): List of inputs with sensitive attributes.
            epsilon (List[List[float]]): Privacy budgets for ε-mDP.
            metric_types (List[List[str]], optional): Metric type names (e.g., "hr", "sbp") for
                domain-aware noising. If provided, uses fixed physiological bounds from MEDICAL_DOMAINS.
            entity (str): Sensitive attribute to sanitize. Pick from {Name/Money/Age}.
            use_mdp (bool): Use ε-mDP (Laplace mechanism) for encrypting numerical values.
            use_fpe (bool): Use FPE for encrypting alphanumerical values.
        Returns:
            data_encrypted (List[str]): List of sanitized strings.
            invalid_indices (List[int]): List of indices where nothing was found for sanitization.
        """
        data_encrypted = []
        invalid_indices = []
        if not extracted and self.ner:
            extracted = self.ner.extract(inputs, entity_type=entity)[entity]
        self.new_entities, self.entity_lookup, self.entity_mapping = self.encypt_value(
            inputs=inputs, use_fpe=use_fpe, use_mdp=use_mdp, epsilon=epsilon, metric_types=metric_types
        )
        if encryption_only: return self.new_entities, self.entity_lookup, self.entity_mapping

        # print(inputs)
        for i, line in enumerate(inputs):
        # Get extracted/encrypted data  for the ith line.
        # Substitute all values.
            for value, encrypt in zip(extracted[i], self.new_entities[i]):
                if value is not None and encrypt is not None:
                    line = unicodedata.normalize('NFC',line).replace(unicodedata.normalize('NFC',value), encrypt)
                else:
                    invalid_indices.append(i)
                    break

            data_encrypted.append(line)

        if not verbose:
            return data_encrypted, invalid_indices
        else:
            return data_encrypted, invalid_indices, self.new_entities, self.entity_lookup, self.entity_mapping
    
    def decrypt(
            self, 
            inputs: List[str], 
            entity='Name', 
            extracted: Optional[Dict[str, List[str]]]=None, 
            use_mdp=False, use_fpe=True, use_cache=False,
        ):
        """
        Takes in a list of inputs and returns a list of desanitized outputs.
        Encrypt must be used before this method!

        Args:   
            inputs (List[str]): List of sanitized inputs.
            extracted (Optional[Dict[str, List[str]]]): Dictionary of extracted sensitive attributes (see NER.extract())
            use_mdp (bool): Retrieve cached values during decryption.
            use_fpe (bool): Use FPE for decrypting alphanumerical values.
            use_cache (bool): Use cipher text values cached during santization, instead of using freshly extracted values (when NER is not reliable).
        """
        dec_fn_mapping = {
            "Name": self.decrypt_names,
            "Money": self.decrypt_money,
            "Age": self.decrypt_age,
        }
        data_decrypted = []
        if extracted is None: extracted = self.ner.extract(inputs, entity_type=entity)[entity]
        data_decrypted = dec_fn_mapping[entity](inputs, extraction=extracted, 
                                                use_fpe=use_fpe, use_mdp=use_mdp, use_cache=use_cache)

        return data_decrypted