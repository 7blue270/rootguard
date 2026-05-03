"""
Staircase Mechanism Baseline — information-theoretically optimal for ε-DP.

Geng & Viswanath (IEEE Trans. IT, 2015) proved that the staircase mechanism
achieves the minimum variance for single real-valued queries under ε-DP.

Two variants:
  Staircase-All (sanitize_staircase_all): Noise every node independently.
  Staircase-Roots (sanitize_staircase_roots): Noise only roots, post-process.

Sensitivity is normalized to W/(m-1) to match the discrete exponential
mechanism's index-space metric. Output clamped to domain bounds.
"""

import math
import random
import numpy as np
from diffprivlib.mechanisms import Staircase

from preempt.sanitizer import MEDICAL_DOMAINS
from utils.utils import get_topological_order


def staircase_noise(value, epsilon, domain_lower, domain_upper,
                    num_candidates=1000):
    """
    Staircase mechanism on a discrete grid in index space.

    Uses diffprivlib's Staircase (Geng & Viswanath, 2015), provably optimal
    for a single scalar query under pure ε-DP. Sensitivity Δ = 1 in index space.

    1. Map to grid index: t = (x - ℓ) / δ
    2. Add staircase noise: t̃ = Staircase(ε, Δ=1).randomise(t)
    3. Clamp to [0, m-1], round to nearest integer
    4. Map back: x' = s · δ + ℓ

    Args:
        value: True value to noise
        epsilon: Privacy parameter
        domain_lower: Lower bound of domain
        domain_upper: Upper bound of domain
        num_candidates: Grid resolution (matches PREEMPT's m=1000)

    Returns:
        Noised value on the discrete grid within [domain_lower, domain_upper]
    """
    domain_width = domain_upper - domain_lower
    if domain_width <= 0 or epsilon <= 0:
        return value

    delta = domain_width / (num_candidates - 1)
    t = (value - domain_lower) / delta
    mech = Staircase(epsilon=epsilon, sensitivity=1)
    t_noisy = mech.randomise(t)
    s = int(np.clip(round(t_noisy), 0, num_candidates - 1))
    return float(s * delta + domain_lower)


def sanitize_staircase_all(samples, nodes, edges, epsilon,
                           num_candidates=1000):
    """
    Staircase-All: noise every non-classification node independently with
    the staircase mechanism. No DAG awareness.

    Args:
        samples: List of sample dicts
        nodes: Dict mapping node names to parent dependencies
        edges: List of (source, target) edges
        epsilon: Privacy budget per node
        num_candidates: Grid resolution

    Returns:
        (all_values, epsilon_blocks)
    """
    epsilon_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        sanitized = {}
        budgets = []

        nodes_to_release = [n for n in topo_order if n not in skip_nodes]
        for node in nodes_to_release:
            lo, hi = MEDICAL_DOMAINS.get(node, MEDICAL_DOMAINS["_default"])
            noised = staircase_noise(
                float(gt_values[node]), epsilon, lo, hi, num_candidates
            )
            sanitized[node] = round(noised, ndigits=4)
            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

    return all_values, epsilon_blocks


def sanitize_staircase_roots(samples, nodes, edges, expressions, epsilon,
                             num_candidates=1000):
    """
    Staircase-Roots: noise only root nodes with the staircase mechanism,
    then post-process derived nodes via DAG expressions.

    Total budget B = n_nodes * epsilon, distributed uniformly across k roots.
    Each root gets B/k. Derived nodes computed from noised roots (free).

    Args:
        samples: List of sample dicts
        nodes: Dict mapping node names to parent dependencies
        edges: List of (source, target) edges
        expressions: Dict mapping node names to computation expressions
        epsilon: Base privacy budget per node
        num_candidates: Grid resolution

    Returns:
        (all_values, epsilon_blocks)
    """
    epsilon_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}

        root_nodes = []
        derived_nodes = []
        for node in topo_order:
            if node not in nodes:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue
            prevs = nodes[node]
            if prevs:
                prevs = [p for p in prevs if p in ground_truth]
            if not prevs:
                root_nodes.append(node)
            else:
                derived_nodes.append(node)

        # Total budget matches VP/P++
        n_nodes = len([n for n in topo_order if n not in skip_nodes])
        total_budget = n_nodes * epsilon

        # Uniform allocation across roots
        k = len(root_nodes)
        root_eps = total_budget / k if k > 0 else epsilon

        # Noise roots
        for root in root_nodes:
            lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
            noised = staircase_noise(
                ground_truth[root], root_eps, lo, hi, num_candidates
            )
            sanitized_values[root] = round(noised, ndigits=4)
            actual_budgets[root] = root_eps

        # Post-process derived nodes
        for node in derived_nodes:
            if node in skip_nodes:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue
            if node in expressions:
                f_x_prime = eval(
                    expressions[node],
                    {"random": random, "round": round, "np": np, "math": math},
                    sanitized_values
                )
                if not np.isfinite(f_x_prime):
                    prevs = nodes.get(node, [])
                    if prevs:
                        prevs = [p for p in prevs if p in ground_truth]
                    parent_vals = {p: sanitized_values.get(p) for p in prevs}
                    raise ValueError(
                        f"Node '{node}' produced non-finite value: {f_x_prime}. "
                        f"Expression: {expressions[node]}, Parents: {parent_vals}"
                    )
                sanitized_values[node] = f_x_prime
                actual_budgets[node] = 0
            else:
                raise ValueError(
                    f"Derived node '{node}' has no expression and is not skip"
                )

        all_values.append(sanitized_values)
        epsilon_blocks.append(
            [actual_budgets.get(node, 0) for node in topo_order]
        )

    return all_values, epsilon_blocks


def sanitize_staircase_roots_opt(samples, nodes, edges, expressions, epsilon,
                                 target_keys, population_means,
                                 num_candidates=1000, epsilon_min=0.005):
    """
    Staircase-Roots with P++'s optimal budget allocation.

    Same as sanitize_staircase_roots but replaces uniform B/k allocation with
    sensitivity-weighted allocation using exact staircase mechanism moments.
    """
    from utils.utils import (optimal_budget_allocation_staircase,
                             compute_population_mean_sensitivities)
    from preempt import sanitizer

    epsilon_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]
    _sens_cache = {}

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}

        root_nodes = []
        derived_nodes = []
        for node in topo_order:
            if node not in nodes:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue
            prevs = nodes[node]
            if prevs:
                prevs = [p for p in prevs if p in ground_truth]
            if not prevs:
                root_nodes.append(node)
            else:
                derived_nodes.append(node)

        # Find target node
        target_node = None
        for tname, tnode in target_keys.items():
            if tnode in ground_truth and tnode not in skip_nodes:
                target_node = tnode
                break

        n_nodes = len([n for n in topo_order if n not in skip_nodes])
        total_budget = n_nodes * epsilon

        if target_node and len(root_nodes) > 0:
            cache_key = (target_node, frozenset(root_nodes))
            if cache_key not in _sens_cache:
                _sens_cache[cache_key] = compute_population_mean_sensitivities(
                    target_node, root_nodes, expressions, skip_nodes,
                    topo_order, nodes, population_means)
            sensitivities = _sens_cache[cache_key]

            dw = {}
            for root in root_nodes:
                lo, hi = sanitizer.MEDICAL_DOMAINS.get(
                    root, sanitizer.MEDICAL_DOMAINS["_default"])
                dw[root] = hi - lo

            allocation = optimal_budget_allocation_staircase(
                sensitivities, total_budget, epsilon_min,
                domain_widths=dw, num_candidates=num_candidates)
        else:
            allocation = {r: total_budget / len(root_nodes) for r in root_nodes}

        # Noise roots with allocated budgets
        for root in root_nodes:
            root_eps = allocation.get(root, epsilon)
            lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
            noised = staircase_noise(
                ground_truth[root], root_eps, lo, hi, num_candidates)
            sanitized_values[root] = round(noised, ndigits=4)
            actual_budgets[root] = root_eps

        # Post-process derived nodes
        for node in derived_nodes:
            if node in skip_nodes:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue
            if node in expressions:
                f_x_prime = eval(
                    expressions[node],
                    {"random": random, "round": round, "np": np,
                     "math": math},
                    sanitized_values)
                if not np.isfinite(f_x_prime):
                    prevs = nodes.get(node, [])
                    if prevs:
                        prevs = [p for p in prevs if p in ground_truth]
                    parent_vals = {p: sanitized_values.get(p) for p in prevs}
                    raise ValueError(
                        f"Node '{node}' produced non-finite value: {f_x_prime}. "
                        f"Expression: {expressions[node]}, Parents: {parent_vals}")
                sanitized_values[node] = f_x_prime
                actual_budgets[node] = 0
            else:
                raise ValueError(
                    f"Derived node '{node}' has no expression and is not skip")

        all_values.append(sanitized_values)
        epsilon_blocks.append(
            [actual_budgets.get(node, 0) for node in topo_order])

    return all_values, epsilon_blocks


def sanitize_staircase_roots_opt_weighted(
    samples, nodes, edges, expressions, epsilon,
    target_keys, population_means, s_dom,
    num_candidates=1000, epsilon_min=0.005
):
    """
    Staircase-Roots with optimal allocation under domain-space weighted constraint.

    Constraint: sum(eps_i * D_i) = epsilon * s_dom
    instead of sum(eps_i) = n * epsilon.

    Args:
        s_dom: Domain-space S_max for this template.
        (all other args same as sanitize_staircase_roots_opt)
    """
    from utils.utils import (optimal_budget_allocation_staircase_weighted,
                             compute_population_mean_sensitivities)
    from preempt import sanitizer

    epsilon_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]
    _sens_cache = {}

    constraint_target = epsilon * s_dom

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}

        root_nodes = []
        derived_nodes = []
        for node in topo_order:
            if node not in nodes:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue
            prevs = nodes[node]
            if prevs:
                prevs = [p for p in prevs if p in ground_truth]
            if not prevs:
                root_nodes.append(node)
            else:
                derived_nodes.append(node)

        target_node = None
        for tname, tnode in target_keys.items():
            if tnode in ground_truth and tnode not in skip_nodes:
                target_node = tnode
                break

        if target_node and len(root_nodes) > 0:
            cache_key = (target_node, frozenset(root_nodes))
            if cache_key not in _sens_cache:
                _sens_cache[cache_key] = compute_population_mean_sensitivities(
                    target_node, root_nodes, expressions, skip_nodes,
                    topo_order, nodes, population_means)
            sensitivities = _sens_cache[cache_key]

            dw = {}
            for root in root_nodes:
                lo, hi = sanitizer.MEDICAL_DOMAINS.get(
                    root, sanitizer.MEDICAL_DOMAINS["_default"])
                dw[root] = hi - lo

            allocation = optimal_budget_allocation_staircase_weighted(
                sensitivities, constraint_target, epsilon_min,
                domain_widths=dw, constraint_weights=dw,
                num_candidates=num_candidates)
        else:
            dw = {}
            for root in root_nodes:
                lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
                dw[root] = hi - lo
            sum_D = sum(dw.values())
            allocation = {r: constraint_target / sum_D for r in root_nodes}

        for root in root_nodes:
            root_eps = allocation.get(root, epsilon)
            lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
            noised = staircase_noise(
                ground_truth[root], root_eps, lo, hi, num_candidates)
            sanitized_values[root] = round(noised, ndigits=4)
            actual_budgets[root] = root_eps

        for node in derived_nodes:
            if node in skip_nodes:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue
            if node in expressions:
                f_x_prime = eval(
                    expressions[node],
                    {"random": random, "round": round, "np": np,
                     "math": math},
                    sanitized_values)
                if not np.isfinite(f_x_prime):
                    prevs = nodes.get(node, [])
                    if prevs:
                        prevs = [p for p in prevs if p in ground_truth]
                    parent_vals = {p: sanitized_values.get(p) for p in prevs}
                    raise ValueError(
                        f"Node '{node}' produced non-finite value: {f_x_prime}. "
                        f"Expression: {expressions[node]}, Parents: {parent_vals}")
                sanitized_values[node] = f_x_prime
                actual_budgets[node] = 0
            else:
                raise ValueError(
                    f"Derived node '{node}' has no expression and is not skip")

        all_values.append(sanitized_values)
        epsilon_blocks.append(
            [actual_budgets.get(node, 0) for node in topo_order])

    return all_values, epsilon_blocks
