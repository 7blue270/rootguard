import math
import numpy as np
from . import med_domain
from preempt import sanitizer

from collections import defaultdict
import random

import networkx as nx


def get_topological_order(sample, edges):
    """
    Computes the linear logical sequence of nodes, using only edges 
    relevant to the variables present in the sample.
    """
    # Define the edges (The dependencies in your code)
    """
    e.g. for sepsis. We can hard-code these.
    edges = [
        ("hr", "shock_index"), 
        ("sbp", "shock_index"),
        ("sbp", "map_val"), 
        ("implied_dbp", "map_val"),
        ("shock_index", "lactate"),
        ("map_val", "sofa_score"), 
        ("ne_dose", "sofa_score")
    ]
    """
    # 1. Map values for context and identify ALL present nodes
    node_values = {}
    for turn in sample["turns"]:
        # Update ensures we collect all unique variables across turns
        node_values.update(turn["private_value"])
        
    present_nodes = set(node_values.keys())

    # 2. Filter edges: Keep only if BOTH source (u) and target (v) are in the sample
    relevant_edges = [
        (u, v) for u, v in edges 
        if u in present_nodes and v in present_nodes
    ]
    
    # 3. Build Graph
    G = nx.DiGraph()
    # Explicitly add all present nodes (handles isolated roots/leaves correctly)
    G.add_nodes_from(present_nodes)
    G.add_edges_from(relevant_edges)

    # 4. Check if it's a valid DAG
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The dependency graph is not a DAG!")

    # 5. Perform Topological Sort
    topo_order = list(nx.topological_sort(G))
    
    return topo_order, node_values


def calculate_budgets(topo_order, task_nodes, 
                      bilipschitz_constants, epsilon=1):
    """
    I need to know the privacy budget saved per turn.
    Effective epsilons carry as is if less than epsilon_target.
    
    """

    budgets = defaultdict()
    effective_epsilons = defaultdict()

    for node in topo_order:
        prevs = task_nodes[node]
        if not prevs: 
            budget = epsilon
            effective_epsilon = epsilon
        else:
            prev_budget = max([budgets[prev] for prev in prevs])
            prev_effective_epsilon = max([effective_epsilons[prev] for prev in prevs])
            effective_epsilon = epsilon/bilipschitz_constants[node][0]
            
            if prev_effective_epsilon/bilipschitz_constants[node][0] <= epsilon:
                budget = prev_effective_epsilon/bilipschitz_constants[node][0]
            elif effective_epsilon <= epsilon:
                budget = effective_epsilon
            else:
                var_target = 1.0 / (epsilon**2)
                var_inherited = 1.0 / (effective_epsilon**2)
                
                var_needed = var_target - var_inherited
                
                budget = math.sqrt(1.0 / var_needed)

        budgets[node] = budget
        effective_epsilons[node]  = effective_epsilon
    
    return budgets, effective_epsilons


def calculate_budgets_v2(topo_order, task_nodes, 
                      bilipschitz_constants, epsilon=1):
    """
    I need to know the privacy budget saved per turn.
    Effective epsilons carry as is if less than epsilon_target.
    
    """

    budgets = defaultdict()
    effective_epsilons = defaultdict()

    for node in topo_order:
        if node not in task_nodes:
            task_nodes[node] = None
        
        prevs = task_nodes[node]
        if not prevs: 
            budget = epsilon
            effective_epsilon = epsilon
        else:
            budget = 0
            effective_epsilon = 0
        budgets[node] = budget
        effective_epsilons[node]  = effective_epsilon
    
    return budgets, effective_epsilons


def calculate_budgets_v3(topo_order, task_nodes,
                      bilipschitz_constants, epsilon=1):
    """
    Compute potential privacy budgets for threshold-based sanitization.

    Based on the Implied Privacy Theorem for bi-Lipschitz functions:
    If f: X → Y has bi-Lipschitz constants (m, L), and we noise x with ε-mDP,
    then f(x') is (ε/m)-mDP in the output space Y.

    Equivalently, noising f(x) directly with ε/m gives the same privacy guarantee.

    This function computes:
      - budget: The cost if we choose to reset (noise f(x) directly) = ε/m
      - effective_epsilon: The privacy level achieved in the output space

    Note: The actual budget used depends on whether the error threshold is
    exceeded during sanitization. See sanitize_dep_aware_v3.

    Args:
        topo_order: List of nodes in topological order
        task_nodes: Dict mapping each node to its parent dependencies
        bilipschitz_constants: Dict of {node: (m_lower, L_upper)}
        epsilon: Base privacy budget for root nodes

    Returns:
        budgets: Dict of potential budget cost per node (if reset is chosen)
        effective_epsilons: Dict of effective privacy level per node
    """
    budgets = defaultdict()
    effective_epsilons = defaultdict()

    for node in topo_order:
        if node not in task_nodes:
            task_nodes[node] = None

        prevs = task_nodes[node]
        if not prevs:
            # Root nodes: noised with full epsilon
            budget = epsilon
            effective_epsilon = epsilon
        else:
            # Derived nodes y = f(x):
            # By the theorem, φ = f ∘ M is (ε/m)-mDP in Y-space
            # Budget to achieve this via direct noising: ε/m
            m = bilipschitz_constants.get(node, (1.0, 1.0))[0]
            budget = epsilon / m if m > 0 else epsilon
            # Effective privacy level in output space
            effective_epsilon = budget

        budgets[node] = budget
        effective_epsilons[node] = effective_epsilon

    return budgets, effective_epsilons

def sanitize_vanilla(samples, nodes, edges, bi_lipschitz_constants, epsilon):
    """
    Vanilla sanitization: independently noise each value with epsilon.

    Each value is noised with the full epsilon budget. This is appropriate for
    online/streaming settings where conversation length is unknown beforehand.
    Total privacy cost scales with number of releases (n * epsilon under basic composition).
    """
    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    metric_type_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

    for sample in samples:
        topo_order, values = get_topological_order(sample, edges)
        nodes_to_release = [node for node in topo_order if node not in skip_nodes]

        all_values.append([values[node] for node in nodes_to_release])
        epsilon_blocks.append([epsilon for _ in nodes_to_release])
        metric_type_blocks.append(nodes_to_release)

    new_entities, _, _ = san.encrypt(inputs=all_values,
                                        epsilon=epsilon_blocks,
                                        metric_types=metric_type_blocks,
                                        use_mdp=True, encryption_only=True)

    all_new_vals = []
    for i in range(len(samples)):
        topo_order, _ = get_topological_order(samples[i], edges)
        nodes_to_release = [node for node in topo_order if node not in skip_nodes]
        new_vals = dict()
        for j, node in enumerate(nodes_to_release):
            new_vals[node] = float(new_entities[i][j])

        all_new_vals.append(new_vals)

    return all_new_vals, epsilon_blocks


def sanitize_vanilla_discrete(samples, nodes, edges, bi_lipschitz_constants, epsilon):
    """
    Vanilla sanitization using the discrete exponential mechanism.

    Same as sanitize_vanilla but uses M_exponential_discrete so that noise
    is domain-width-dependent, matching V6 for fair comparison.
    """
    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

    for sample in samples:
        topo_order, values = get_topological_order(sample, edges)
        nodes_to_release = [node for node in topo_order if node not in skip_nodes]
        sanitized = {}
        budgets = []

        for node in nodes_to_release:
            domain_lower, domain_upper = sanitizer.MEDICAL_DOMAINS.get(
                node, sanitizer.MEDICAL_DOMAINS["_default"]
            )
            noised = san.M_exponential_discrete(
                float(values[node]), epsilon, domain_lower, domain_upper
            )
            sanitized[node] = round(float(noised), ndigits=4)
            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

    return all_values, epsilon_blocks


def sanitize_dep_aware(samples, nodes, edges, expressions,
                       bi_lipschitz_constants, epsilon):
    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk", 
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]
    for sample in samples:
        topo_order, values = get_topological_order(sample, edges)
        epsilon_budgets, epsilon_effectives = calculate_budgets(topo_order, nodes,bi_lipschitz_constants, epsilon)
        # all_values.append([values[node] for node in topo_order])
        epsilon_blocks.append([epsilon_budgets[node] for node in topo_order])
        # We iterate through the expressions and update the vitals dict
        for node in topo_order:
            print(values)
            # use sanitized root node values
            if node in expressions and node not in skip_nodes:
                values[node] = eval(expressions[node], {"random": random, "round": round}, values)
            # sanitize values, 
            if epsilon_budgets[node] >= epsilon:
                encrypted_val, _, _ = san.encrypt(inputs=[[values[node]]], 
                                                    epsilon=[[epsilon_budgets[node]]], 
                                                    use_mdp=True, encryption_only=True)
                
                values[node] = float(encrypted_val[0][0])

        all_values.append(values)

def sanitize_dep_aware_v2(samples, nodes, edges, expressions,
                       bi_lipschitz_constants, epsilon):
    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]
    for sample in samples:
        topo_order, values = get_topological_order(sample, edges)
        epsilon_budgets, epsilon_effectives = calculate_budgets_v2(topo_order, nodes, bi_lipschitz_constants, epsilon)
        epsilon_blocks.append([epsilon_budgets[node] for node in topo_order])
        # We iterate through the expressions and update the vitals dict
        for node in topo_order:
            # use sanitized root node values
            if node in expressions and node not in skip_nodes:
                values[node] = eval(expressions[node], {"random": random, "round": round, "np": np}, values)
            # sanitize values
            if epsilon_budgets[node] >= epsilon:
                encrypted_val, _, _ = san.encrypt(
                    inputs=[[values[node]]],
                    epsilon=[[epsilon_budgets[node]]],
                    metric_types=[[node]],
                    use_mdp=True,
                    encryption_only=True
                )
                values[node] = float(encrypted_val[0][0])

        all_values.append(values)

    return all_values, epsilon_blocks, epsilon_effectives


def sanitize_dep_aware_v3(samples, nodes, edges, expressions,
                          bi_lipschitz_constants, epsilon,
                          threshold=0.1, threshold_type="relative",
                          min_m=1e-6):
    """
    Dependency-aware sanitization with threshold-based error control using the
    implied privacy theorem for bi-Lipschitz functions.

    THEOREM (Implied Privacy for Bi-Lipschitz Functions):
    Let M: X → X be an ε-mDP mechanism, and f: X → Y be bi-Lipschitz with
    constants (m, L) satisfying:
        m · d_X(x, x') ≤ d_Y(f(x), f(x')) ≤ L · d_X(x, x')

    Then φ = f ∘ M is (ε/m)-mDP with respect to d_Y(f(x), f(x')).

    IMPLICATION:
    Adding mDP noise directly to y = f(x) with privacy parameter ε/m provides
    at least as strong privacy as noising x with ε and then applying f.

    APPROACH:
    For derived nodes y = f(x), we have two equivalent-privacy options:
      1. Post-process: Use f(x') where x' = M(x). Privacy: ε_parent/m in Y-space. Cost: 0.
      2. Direct noise: Compute f(x), add noise with ε_parent/m. Privacy: ε_parent/m. Cost: ε_parent/m.

    Both achieve (ε_parent/m)-mDP in the output space. We choose based on error:
      - If |f(x') - f(x)| ≤ threshold: use post-processing (better budget)
      - If |f(x') - f(x)| > threshold: use direct noising (better utility)

    Args:
        samples: List of sample dictionaries with conversation turns
        nodes: Dictionary mapping node names to their parent dependencies
        edges: List of (source, target) dependency edges
        expressions: Dictionary mapping node names to their computation expressions
        bi_lipschitz_constants: Dict of {node: (m_lower, L_upper)} constants
        epsilon: Privacy budget for root nodes
        threshold: Error threshold for deciding when to reset
        threshold_type: "relative" (|f(x')-f(x)|/|f(x)|) or "absolute" (|f(x')-f(x)|)
        min_m: Minimum bi-Lipschitz constant to avoid numerical issues (default 1e-6)

    Returns:
        all_values: List of sanitized value dictionaries
        epsilon_blocks: List of actual epsilon budgets used per node
        all_reset_info: List of dicts tracking reset decisions, errors, and effective privacy
    """
    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    all_values = []
    all_reset_info = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)

        # Store ground truth values before any modification
        ground_truth = {k: v for k, v in gt_values.items()}

        # Working copy for sanitized values
        sanitized_values = {}

        # Track budgets actually used and reset decisions
        actual_budgets = {}
        effective_privacy = {}  # ε/m effective privacy level at each node
        reset_info = {}

        for node in topo_order:
            if node not in nodes:
                # Constant/auxiliary node in sample data but not in template definition — skip
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                effective_privacy[node] = epsilon
                reset_info[node] = {"is_root": False, "reset": False, "constant": True}
                continue

            prevs = nodes[node]
            # Filter to parents present in this sample
            if prevs:
                prevs = [p for p in prevs if p in ground_truth]

            if not prevs:
                # Root node: noise with full epsilon
                # Effective privacy in input space: ε
                encrypted_val, _, _ = san.encrypt(
                    inputs=[[ground_truth[node]]],
                    epsilon=[[epsilon]],
                    metric_types=[[node]],
                    use_mdp=True,
                    encryption_only=True
                )
                sanitized_values[node] = float(encrypted_val[0][0])
                actual_budgets[node] = epsilon
                effective_privacy[node] = epsilon
                reset_info[node] = {
                    "is_root": True,
                    "reset": False,
                    "budget_used": epsilon,
                    "effective_privacy": epsilon
                }

            elif node in skip_nodes:
                # Skip nodes: classification outputs, copy ground truth
                # Inherit effective privacy from parents for propagation purposes
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                # Use parent's effective privacy for downstream propagation
                parent_effective = max(
                    (effective_privacy.get(p, epsilon) for p in prevs if p not in skip_nodes),
                    default=epsilon
                )
                effective_privacy[node] = parent_effective
                reset_info[node] = {"is_root": False, "reset": False, "skipped": True}

            elif node in expressions:
                # Derived node y = f(x)
                # Get bi-Lipschitz constant m for this function
                if node not in bi_lipschitz_constants:
                    raise ValueError(f"Node '{node}' missing from bi_lipschitz_constants")
                m_raw = bi_lipschitz_constants[node][0]
                # Clamp m to avoid numerical issues with very small values
                m = max(m_raw, min_m)

                # Compute f(x') using sanitized parent values (post-processing)
                f_x_prime = eval(
                    expressions[node],
                    {"random": random, "round": round, "np": np, "math": math},
                    sanitized_values
                )
                if not np.isfinite(f_x_prime):
                    parent_vals = {p: sanitized_values.get(p) for p in prevs}
                    raise ValueError(
                        f"Node '{node}' expression produced non-finite value: {f_x_prime}. "
                        f"Expression: {expressions[node]}, "
                        f"Parent values: {parent_vals}"
                    )

                # Get ground truth f(x)
                f_x = ground_truth[node]

                # Calculate error d_Y(f(x'), f(x))
                if threshold_type == "relative":
                    # Relative error: |f(x') - f(x)| / |f(x)|
                    if abs(f_x) < 1e-10:
                        # Near-zero ground truth: fall back to absolute error
                        error = abs(f_x_prime - f_x)
                    else:
                        error = abs(f_x_prime - f_x) / abs(f_x)
                else:
                    # Absolute error: |f(x') - f(x)|
                    error = abs(f_x_prime - f_x)

                # Compute effective privacy from parent nodes
                parent_privacies = [effective_privacy[p] for p in prevs]
                if any(ep == float('inf') for ep in parent_privacies):
                    parent_details = {p: effective_privacy[p] for p in prevs}
                    raise ValueError(
                        f"Node '{node}' has parent(s) with undefined (inf) effective privacy: "
                        f"{parent_details}"
                    )
                parent_effective = max(parent_privacies)

                # Reset budget:
                #   m < 1 (amplifying): use ε/m > ε — more budget, less noise, better utility
                #   m >= 1 (contracting): use ε — no need to add more noise than a root node
                reset_budget = epsilon / m if m < 1 else epsilon

                # Decision: use f(x') (post-processing) or f(x)' (direct noise)?
                if error > threshold:
                    # Error too high: reset by noising ground truth directly with ε/m
                    encrypted_val, _, _ = san.encrypt(
                        inputs=[[f_x]],
                        epsilon=[[reset_budget]],
                        metric_types=[[node]],
                        use_mdp=True,
                        encryption_only=True
                    )
                    sanitized_values[node] = float(encrypted_val[0][0])
                    actual_budgets[node] = reset_budget
                    effective_privacy[node] = reset_budget
                    reset_info[node] = {
                        "is_root": False,
                        "reset": True,
                        "reason": "error_exceeded_threshold",
                        "error": error,
                        "threshold": threshold,
                        "f_x_prime": f_x_prime,
                        "f_x": f_x,
                        "budget_used": reset_budget,
                        "effective_privacy": reset_budget,
                        "parent_effective": parent_effective,
                        "bi_lipschitz_m": m,
                        "bi_lipschitz_m_raw": m_raw
                    }
                else:
                    # Error acceptable: use f(x') with no additional budget
                    # Privacy comes "for free" via post-processing theorem
                    sanitized_values[node] = f_x_prime
                    actual_budgets[node] = 0
                    effective_privacy[node] = reset_budget
                    reset_info[node] = {
                        "is_root": False,
                        "reset": False,
                        "reason": "error_within_threshold",
                        "error": error,
                        "threshold": threshold,
                        "f_x_prime": f_x_prime,
                        "f_x": f_x,
                        "budget_used": 0,
                        "effective_privacy": reset_budget,
                        "parent_effective": parent_effective,
                        "bi_lipschitz_m": m,
                        "bi_lipschitz_m_raw": m_raw
                    }
            else:
                raise ValueError(
                    f"Node '{node}' has parents {prevs} but no expression and is not a skip node"
                )

        all_values.append(sanitized_values)
        epsilon_blocks.append([actual_budgets[node] for node in topo_order])
        all_reset_info.append(reset_info)

    return all_values, epsilon_blocks, all_reset_info


def sanitize_dep_aware_v4(samples, nodes, edges, expressions,
                          bi_lipschitz_constants, epsilon,
                          threshold=0.1, threshold_type="relative",
                          min_m=1e-6):
    """
    Dependency-aware sanitization with greedy budget banking.

    Builds on v3's threshold-based error control but adds a budget bank:
    - Root nodes spend ε directly (no bank interaction).
    - Each derived node allocates ε to the bank.
    - If error ≤ threshold: use post-processed value for free, ε stays banked.
    - If error > threshold: spend the ENTIRE bank greedily on a single reset.

    This ensures total privacy spend ≤ vanilla (n_root·ε + n_derived·ε) while
    concentrating budget at reset points for lower noise (higher utility).

    When bank = B and we reset, noise is calibrated to budget B.
    Since B ≥ ε (at least the current node's contribution), the noise is
    always ≤ vanilla noise (1/B ≤ 1/ε).

    Args:
        samples: List of sample dictionaries with conversation turns
        nodes: Dictionary mapping node names to their parent dependencies
        edges: List of (source, target) dependency edges
        expressions: Dictionary mapping node names to their computation expressions
        bi_lipschitz_constants: Dict of {node: (m_lower, L_upper)} constants
        epsilon: Privacy budget for root nodes
        threshold: Error threshold for deciding when to reset
        threshold_type: "relative" (|f(x')-f(x)|/|f(x)|) or "absolute" (|f(x')-f(x)|)
        min_m: Minimum bi-Lipschitz constant to avoid numerical issues (default 1e-6)

    Returns:
        all_values: List of sanitized value dictionaries
        epsilon_blocks: List of actual epsilon budgets used per node
        all_reset_info: List of dicts tracking reset decisions, errors, and effective privacy
    """
    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    all_values = []
    all_reset_info = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)

        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}

        actual_budgets = {}
        effective_privacy = {}
        reset_info = {}
        bank = 0.0

        for node in topo_order:
            if node not in nodes:
                # Constant/auxiliary node — pass through
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                effective_privacy[node] = epsilon
                reset_info[node] = {"is_root": False, "reset": False, "constant": True}
                continue

            prevs = nodes[node]
            if prevs:
                prevs = [p for p in prevs if p in ground_truth]

            if not prevs:
                # Root node: spend ε, no bank interaction
                encrypted_val, _, _ = san.encrypt(
                    inputs=[[ground_truth[node]]],
                    epsilon=[[epsilon]],
                    metric_types=[[node]],
                    use_mdp=True,
                    encryption_only=True
                )
                sanitized_values[node] = float(encrypted_val[0][0])
                actual_budgets[node] = epsilon
                effective_privacy[node] = epsilon
                reset_info[node] = {
                    "is_root": True,
                    "reset": False,
                    "budget_used": epsilon,
                    "effective_privacy": epsilon,
                    "bank_after": bank
                }

            elif node in skip_nodes:
                # Classification outputs — pass through, no bank interaction
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                parent_effective = max(
                    (effective_privacy.get(p, epsilon) for p in prevs if p not in skip_nodes),
                    default=epsilon
                )
                effective_privacy[node] = parent_effective
                reset_info[node] = {"is_root": False, "reset": False, "skipped": True,
                                    "bank_after": bank}

            elif node in expressions:
                # Derived node: allocate ε to bank
                bank += epsilon

                if node not in bi_lipschitz_constants:
                    raise ValueError(f"Node '{node}' missing from bi_lipschitz_constants")
                m_raw = bi_lipschitz_constants[node][0]
                m = max(m_raw, min_m)

                # Compute f(x') from sanitized parents
                f_x_prime = eval(
                    expressions[node],
                    {"random": random, "round": round, "np": np, "math": math},
                    sanitized_values
                )
                if not np.isfinite(f_x_prime):
                    parent_vals = {p: sanitized_values.get(p) for p in prevs}
                    raise ValueError(
                        f"Node '{node}' expression produced non-finite value: {f_x_prime}. "
                        f"Expression: {expressions[node]}, "
                        f"Parent values: {parent_vals}"
                    )

                f_x = ground_truth[node]

                # Calculate error
                if threshold_type == "relative":
                    if abs(f_x) < 1e-10:
                        error = abs(f_x_prime - f_x)
                    else:
                        error = abs(f_x_prime - f_x) / abs(f_x)
                else:
                    error = abs(f_x_prime - f_x)

                if error > threshold:
                    # Spend entire bank greedily
                    reset_budget = bank
                    bank = 0.0

                    encrypted_val, _, _ = san.encrypt(
                        inputs=[[f_x]],
                        epsilon=[[reset_budget]],
                        metric_types=[[node]],
                        use_mdp=True,
                        encryption_only=True
                    )
                    sanitized_values[node] = float(encrypted_val[0][0])
                    actual_budgets[node] = reset_budget
                    effective_privacy[node] = reset_budget
                    reset_info[node] = {
                        "is_root": False,
                        "reset": True,
                        "reason": "error_exceeded_threshold",
                        "error": error,
                        "threshold": threshold,
                        "f_x_prime": f_x_prime,
                        "f_x": f_x,
                        "budget_used": reset_budget,
                        "effective_privacy": reset_budget,
                        "bi_lipschitz_m": m,
                        "bi_lipschitz_m_raw": m_raw,
                        "bank_before": reset_budget,
                        "bank_after": 0.0
                    }
                else:
                    # Error acceptable: use f(x'), ε stays banked
                    sanitized_values[node] = f_x_prime
                    actual_budgets[node] = 0
                    effective_privacy[node] = epsilon
                    reset_info[node] = {
                        "is_root": False,
                        "reset": False,
                        "reason": "error_within_threshold",
                        "error": error,
                        "threshold": threshold,
                        "f_x_prime": f_x_prime,
                        "f_x": f_x,
                        "budget_used": 0,
                        "effective_privacy": epsilon,
                        "bi_lipschitz_m": m,
                        "bi_lipschitz_m_raw": m_raw,
                        "bank_before": bank,
                        "bank_after": bank
                    }
            else:
                raise ValueError(
                    f"Node '{node}' has parents {prevs} but no expression and is not a skip node"
                )

        all_values.append(sanitized_values)
        epsilon_blocks.append([actual_budgets[node] for node in topo_order])
        all_reset_info.append(reset_info)

    return all_values, epsilon_blocks, all_reset_info


def compute_target_sensitivities(target_node, root_nodes, expressions, ground_truth,
                                 skip_nodes, topo_order, nodes):
    """
    Compute exact partial derivatives ∂(target)/∂(root_i) via forward-mode
    automatic differentiation using exact analytical partial derivatives.

    Propagates gradients through the DAG in topological order using the
    chain rule with closed-form local derivatives from get_local_partials().

    Args:
        target_node: Name of the target node (e.g., "homa", "nlr")
        root_nodes: List of root node names for this sample
        expressions: Dict of node -> expression string
        ground_truth: Dict of node -> ground truth value for this sample
        skip_nodes: List of classification nodes to skip
        topo_order: Topological ordering of nodes
        nodes: Dict of node -> parent list (template_nodes)

    Returns:
        sensitivities: Dict {root_name: ∂target/∂root}
    """
    from utils.med_domain.all_templates import get_local_partials

    # Ensure all intermediate values are computed from ground truth
    values = dict(ground_truth)
    for node in topo_order:
        if node in values:
            continue
        if node in skip_nodes:
            continue
        if node in expressions:
            try:
                values[node] = eval(
                    expressions[node],
                    {"random": random, "round": round, "np": np, "math": math},
                    values
                )
            except Exception:
                values[node] = ground_truth.get(node, 0)

    # Forward-mode AD: grads[node] = {root: ∂node/∂root}
    grads = {}
    for root in root_nodes:
        grads[root] = {r: (1.0 if r == root else 0.0) for r in root_nodes}

    for node in topo_order:
        if node in grads:
            continue
        if node in skip_nodes:
            continue

        local_partials = get_local_partials(node, values)
        if local_partials is None:
            continue

        # Chain rule: grads[node][root] = Σ_var (∂node/∂var) * grads[var][root]
        node_grads = {r: 0.0 for r in root_nodes}
        for var, d_node_d_var in local_partials.items():
            if var in grads:
                for root in root_nodes:
                    node_grads[root] += d_node_d_var * grads[var].get(root, 0.0)
        grads[node] = node_grads

    if target_node in grads:
        return {root: grads[target_node][root] for root in root_nodes}
    else:
        return {root: 0.0 for root in root_nodes}


def compute_worst_case_sensitivities(target_node, root_nodes, expressions,
                                      skip_nodes, topo_order, nodes, domains):
    """
    Compute worst-case (supremum) absolute partial derivatives
    |∂(target)/∂(root_i)| over the entire domain via corner evaluation.

    Data-independent: depends only on public domain bounds, not on any sample.
    Evaluates exact end-to-end sensitivities (forward-mode AD) at all 2^k
    domain corners and takes the max absolute value for each root. Exact
    when partial derivatives are monotonic in each root variable (true for
    all current templates).

    Args:
        target_node: Name of the target node
        root_nodes: List of root node names
        expressions: Dict mapping node names to computation expressions
        skip_nodes: List of classification nodes to skip
        topo_order: Topological ordering of nodes
        nodes: Dict of node -> parent list
        domains: Dict of {node_name: (lo, hi)} for domain bounds

    Returns:
        sensitivities: Dict {root_name: sup|∂target/∂root|}
    """
    import itertools

    # Get (lo, hi) for each root
    bounds = []
    for r in root_nodes:
        lo, hi = domains.get(r, domains.get("_default", (0, 1)))
        bounds.append((lo, hi))

    # Evaluate exact end-to-end sensitivities at each of the 2^k corners
    best = {r: 0.0 for r in root_nodes}
    for corner in itertools.product(*bounds):
        ground_truth = {r: v for r, v in zip(root_nodes, corner)}
        sens = compute_target_sensitivities(
            target_node, root_nodes, expressions, ground_truth,
            skip_nodes, topo_order, nodes
        )
        for r in root_nodes:
            abs_val = abs(sens[r])
            if abs_val > best[r]:
                best[r] = abs_val

    return best


def compute_expected_sensitivities(target_node, root_nodes, expressions,
                                    skip_nodes, topo_order, nodes, domains,
                                    n_samples=10000):
    """
    Compute expected absolute partial derivatives E[|∂(target)/∂(root_i)|]
    under a uniform prior over the domain.

    Data-independent: samples uniformly from the public domain bounds.
    Less pessimistic than worst-case corner evaluation; gives sensitivities
    that reflect the average behavior across the domain rather than the
    extreme corner.

    Args:
        target_node, root_nodes, expressions, skip_nodes, topo_order, nodes:
            Same as compute_worst_case_sensitivities.
        domains: Dict of {node_name: (lo, hi)} for domain bounds
        n_samples: Number of Monte Carlo samples (default 10000)

    Returns:
        sensitivities: Dict {root_name: E[|∂target/∂root|]}
    """
    accum = {r: 0.0 for r in root_nodes}
    rng = np.random.RandomState(42)

    for _ in range(n_samples):
        point = {}
        for r in root_nodes:
            lo, hi = domains.get(r, domains.get("_default", (0, 1)))
            point[r] = rng.uniform(lo, hi)

        sens = compute_target_sensitivities(
            target_node, root_nodes, expressions, point,
            skip_nodes, topo_order, nodes
        )
        for r in root_nodes:
            accum[r] += abs(sens[r])

    return {r: accum[r] / n_samples for r in root_nodes}


def compute_population_mean_sensitivities(target_node, root_nodes, expressions,
                                           skip_nodes, topo_order, nodes,
                                           population_means):
    """
    Compute absolute partial derivatives |∂(target)/∂(root_i)| evaluated
    at the population mean of each root.

    Data-independent: uses published CDC NHANES population means (aggregate
    statistics from public data, not individual records).

    Args:
        target_node, root_nodes, expressions, skip_nodes, topo_order, nodes:
            Same as compute_worst_case_sensitivities.
        population_means: Dict {root_name: population_mean_value}

    Returns:
        sensitivities: Dict {root_name: |∂target/∂root| at population mean}
    """
    point = {r: population_means[r] for r in root_nodes}
    sens = compute_target_sensitivities(
        target_node, root_nodes, expressions, point,
        skip_nodes, topo_order, nodes
    )
    return {r: abs(sens[r]) for r in root_nodes}


def optimal_budget_allocation(sensitivities, total_budget, epsilon_min,
                              epsilon_max=None, domain_widths=None):
    """
    Compute optimal privacy budget allocation across root nodes to minimize
    variance of noise at the target node.

    Solves:  Minimize  Σᵢ cᵢ / εᵢ²
             s.t.      Σᵢ εᵢ = B
                       ε_min ≤ εᵢ ≤ ε_max

    When domain_widths is None (V5 continuous mechanism):
        cᵢ = sᵢ²                    → εᵢ ∝ |sᵢ|^(2/3)

    When domain_widths is provided (V6 discrete exponential mechanism):
        cᵢ = (sᵢ · Wᵢ)²            → εᵢ ∝ |sᵢ · Wᵢ|^(2/3)

    This accounts for the discrete mechanism's noise being proportional to
    domain width Wᵢ, giving wider-domain roots more budget.

    Closed-form (unconstrained):  εᵢ = B · cᵢ^(1/3) / Σⱼ cⱼ^(1/3)
    This is exact under the continuous Laplace approximation E[|ηᵢ|] ≈ 2δᵢ/εᵢ,
    which upper-bounds the discrete mechanism's expected noise for all grid
    positions tᵢ.

    With bounds: iteratively clamp and redistribute.

    Args:
        sensitivities: Dict {root_name: ∂target/∂root}
        total_budget: Total budget B to allocate
        epsilon_min: Minimum budget per root (privacy floor)
        epsilon_max: Maximum budget per root (None = no cap)
        domain_widths: Dict {root_name: domain width Wᵢ} for discrete
                       mechanism. None for continuous mechanism (V5).

    Returns:
        allocation: Dict {root_name: εᵢ}
    """
    roots = list(sensitivities.keys())
    k = len(roots)

    if k == 0:
        return {}

    # cᵢ = (sᵢ · Wᵢ)² or sᵢ²,  weight = cᵢ^(1/3)
    weights = {}
    for r in roots:
        s_i = sensitivities[r]
        if domain_widths is not None:
            c_i = (s_i * domain_widths.get(r, 1.0)) ** 2
        else:
            c_i = s_i ** 2
        weights[r] = c_i ** (1.0 / 3.0) if c_i > 0 else 0.0

    allocation = {}
    remaining_budget = total_budget
    active_roots = list(roots)

    # Iteratively allocate with clamping
    for _ in range(k):
        total_weight = sum(weights[r] for r in active_roots)

        if total_weight == 0:
            # All remaining roots have zero sensitivity — split evenly
            per_root = remaining_budget / len(active_roots)
            for r in active_roots:
                allocation[r] = max(per_root, epsilon_min)
            break

        newly_clamped = False
        to_remove = []
        for r in active_roots:
            eps_r = remaining_budget * weights[r] / total_weight

            if eps_r < epsilon_min:
                allocation[r] = epsilon_min
                remaining_budget -= epsilon_min
                to_remove.append(r)
                newly_clamped = True
            elif epsilon_max is not None and eps_r > epsilon_max:
                allocation[r] = epsilon_max
                remaining_budget -= epsilon_max
                to_remove.append(r)
                newly_clamped = True

        for r in to_remove:
            active_roots.remove(r)

        if not newly_clamped:
            for r in active_roots:
                allocation[r] = remaining_budget * weights[r] / total_weight
            break

    return allocation


def sanitize_dep_aware_v5(samples, nodes, edges, expressions,
                          bi_lipschitz_constants, epsilon,
                          target_keys, epsilon_min=0.05,
                          epsilon_max=None):
    """
    Dependency-aware sanitization with optimal budget allocation.

    Allocates privacy budget across root nodes to minimize variance of the
    propagated noise at the target node. Uses exact partial derivatives
    (chain rule through DAG expressions) to compute sensitivities, then
    solves the constrained allocation problem.

    Root nodes with higher influence on the target get more budget (less noise).
    All derived nodes are computed via post-processing from noised roots.

    Total budget = n·ε where n is the number of non-class nodes in the
    conversation (same total as vanilla).

    Args:
        samples: List of sample dictionaries with conversation turns
        nodes: Dictionary mapping node names to their parent dependencies
        edges: List of (source, target) dependency edges
        expressions: Dictionary mapping node names to their computation expressions
        bi_lipschitz_constants: Unused, kept for interface consistency
        epsilon: Base privacy budget (vanilla uses ε per root)
        target_keys: Dict mapping template name (uppercase) to target node name
                     e.g., {"HOMA": "homa", "NLR": "nlr"}
        epsilon_min: Minimum budget per root node (privacy floor)
        epsilon_max: Maximum budget per root node (None = no cap)

    Returns:
        all_values: List of sanitized value dictionaries
        epsilon_blocks: List of actual epsilon budgets used per node
        all_allocation_info: List of dicts with allocation details per sample
    """
    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    all_values = []
    all_allocation_info = []
    _wc_sensitivity_cache = {}
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}

        # Identify root and derived nodes for this sample
        root_nodes = []
        derived_nodes = []
        for node in topo_order:
            if node not in nodes:
                # Constant/auxiliary — pass through
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

        # Find the target node present in this sample
        target_node = None
        for tname, tnode in target_keys.items():
            if tnode in ground_truth and tnode not in skip_nodes:
                target_node = tnode
                break

        # B = count of all nodes excluding skip/class nodes, times ε
        n_nodes = len([n for n in topo_order if n not in skip_nodes])
        total_budget = n_nodes * epsilon

        if target_node and len(root_nodes) > 0:
            # Worst-case sensitivities: data-independent (same for all patients)
            cache_key = (target_node, frozenset(root_nodes))
            if cache_key not in _wc_sensitivity_cache:
                _wc_sensitivity_cache[cache_key] = compute_worst_case_sensitivities(
                    target_node, root_nodes, expressions, skip_nodes,
                    topo_order, nodes, sanitizer.MEDICAL_DOMAINS
                )
            sensitivities = _wc_sensitivity_cache[cache_key]
            allocation = optimal_budget_allocation(
                sensitivities, total_budget, epsilon_min, epsilon_max
            )
        else:
            # Fallback: uniform allocation
            allocation = {r: epsilon for r in root_nodes}
            sensitivities = {r: 0.0 for r in root_nodes}

        # Noise root nodes with allocated budgets
        for root in root_nodes:
            root_eps = allocation.get(root, epsilon)
            encrypted_val, _, _ = san.encrypt(
                inputs=[[ground_truth[root]]],
                epsilon=[[root_eps]],
                metric_types=[[root]],
                use_mdp=True,
                encryption_only=True
            )
            sanitized_values[root] = float(encrypted_val[0][0])
            actual_budgets[root] = root_eps

        # Post-process all derived nodes from noised roots
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
                        f"Node '{node}' expression produced non-finite value: {f_x_prime}. "
                        f"Expression: {expressions[node]}, "
                        f"Parent values: {parent_vals}"
                    )
                sanitized_values[node] = f_x_prime
                actual_budgets[node] = 0
            else:
                raise ValueError(
                    f"Derived node '{node}' has no expression and is not a skip node"
                )

        all_values.append(sanitized_values)
        epsilon_blocks.append([actual_budgets.get(node, 0) for node in topo_order])
        all_allocation_info.append({
            "root_nodes": root_nodes,
            "target_node": target_node,
            "sensitivities": {r: float(s) for r, s in sensitivities.items()},
            "allocation": {r: float(e) for r, e in allocation.items()},
            "total_budget": total_budget,
        })

    return all_values, epsilon_blocks, all_allocation_info


def sanitize_dep_aware_v6(samples, nodes, edges, expressions,
                          bi_lipschitz_constants, epsilon,
                          target_keys, epsilon_min=0.05,
                          epsilon_max=None):
    """
    Dependency-aware sanitization with optimal budget allocation (V6).

    Same as V5 but uses the discrete-space exponential mechanism
    (M_exponential_discrete) instead of M_exponential. This makes the noise
    scale domain-width-dependent: wider domains get proportionally more noise
    in original units.

    Args:
        samples: List of sample dictionaries with conversation turns
        nodes: Dictionary mapping node names to their parent dependencies
        edges: List of (source, target) dependency edges
        expressions: Dictionary mapping node names to their computation expressions
        bi_lipschitz_constants: Unused, kept for interface consistency
        epsilon: Base privacy budget (vanilla uses ε per root)
        target_keys: Dict mapping template name (uppercase) to target node name
        epsilon_min: Minimum budget per root node (privacy floor)
        epsilon_max: Maximum budget per root node (None = no cap)

    Returns:
        all_values: List of sanitized value dictionaries
        epsilon_blocks: List of actual epsilon budgets used per node
        all_allocation_info: List of dicts with allocation details per sample
    """
    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    all_values = []
    all_allocation_info = []
    _wc_sensitivity_cache = {}
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}

        # Identify root and derived nodes for this sample
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

        # Find the target node present in this sample
        target_node = None
        for tname, tnode in target_keys.items():
            if tnode in ground_truth and tnode not in skip_nodes:
                target_node = tnode
                break

        # B = count of all nodes excluding skip/class nodes, times ε
        n_nodes = len([n for n in topo_order if n not in skip_nodes])
        total_budget = n_nodes * epsilon

        if target_node and len(root_nodes) > 0:
            # Worst-case sensitivities: data-independent (same for all patients)
            cache_key = (target_node, frozenset(root_nodes))
            if cache_key not in _wc_sensitivity_cache:
                _wc_sensitivity_cache[cache_key] = compute_worst_case_sensitivities(
                    target_node, root_nodes, expressions, skip_nodes,
                    topo_order, nodes, sanitizer.MEDICAL_DOMAINS
                )
            sensitivities = _wc_sensitivity_cache[cache_key]
            # Build domain-width dict for discrete exponential mechanism
            dw = {}
            for root in root_nodes:
                lo, hi = sanitizer.MEDICAL_DOMAINS.get(
                    root, sanitizer.MEDICAL_DOMAINS["_default"]
                )
                dw[root] = hi - lo
            allocation = optimal_budget_allocation(
                sensitivities, total_budget, epsilon_min, epsilon_max,
                domain_widths=dw
            )
        else:
            allocation = {r: epsilon for r in root_nodes}
            sensitivities = {r: 0.0 for r in root_nodes}

        # Noise root nodes with allocated budgets using discrete exponential mechanism
        for root in root_nodes:
            root_eps = allocation.get(root, epsilon)
            domain_lower, domain_upper = sanitizer.MEDICAL_DOMAINS.get(
                root, sanitizer.MEDICAL_DOMAINS["_default"]
            )
            noised_value = san.M_exponential_discrete(
                ground_truth[root], root_eps, domain_lower, domain_upper
            )
            sanitized_values[root] = round(float(noised_value), ndigits=4)
            actual_budgets[root] = root_eps

        # Post-process all derived nodes from noised roots
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
                        f"Node '{node}' expression produced non-finite value: {f_x_prime}. "
                        f"Expression: {expressions[node]}, "
                        f"Parent values: {parent_vals}"
                    )
                sanitized_values[node] = f_x_prime
                actual_budgets[node] = 0
            else:
                raise ValueError(
                    f"Derived node '{node}' has no expression and is not a skip node"
                )

        all_values.append(sanitized_values)
        epsilon_blocks.append([actual_budgets.get(node, 0) for node in topo_order])
        all_allocation_info.append({
            "root_nodes": root_nodes,
            "target_node": target_node,
            "sensitivities": {r: float(s) for r, s in sensitivities.items()},
            "allocation": {r: float(e) for r, e in allocation.items()},
            "total_budget": total_budget,
        })

    return all_values, epsilon_blocks, all_allocation_info


# =============================================================================
# V7: Exact Discrete Budget Allocation
# =============================================================================

def compute_sums(alpha, t_i, m):
    """
    Compute all 6 discrete sums for the exponential mechanism in one O(m) pass.

    Args:
        alpha: exp(-epsilon/2), in (0, 1)
        t_i: Real-valued index of the true value in [0, m-1]
        m: Number of discrete candidates

    Returns:
        Z, A, B, D, P, C  —  normalizing constant and moment sums
    """
    s_arr = np.arange(m, dtype=float)
    d = np.abs(t_i - s_arr)
    w = alpha ** d
    Z = np.sum(w)
    A = np.sum(d * w)
    B = np.sum(d ** 2 * w)
    D = np.sum(d ** 3 * w)
    signed = s_arr - t_i
    P = np.sum(signed * w)
    C = np.sum(signed * d * w)
    return Z, A, B, D, P, C


def objective_absolute(epsilons, s_vec, delta_vec, t_vec, m):
    """Objective for mode='abs': sum_i |s_i| * delta_i * E[|S_i - t_i|]."""
    total = 0.0
    for i in range(len(epsilons)):
        alpha = np.exp(-epsilons[i] / 2.0)
        Z, A, B, D, P, C = compute_sums(alpha, t_vec[i], m)
        total += np.abs(s_vec[i]) * delta_vec[i] * A / Z
    return total


def gradient_absolute(epsilons, s_vec, delta_vec, t_vec, m):
    """Gradient for mode='abs': d/d(eps_i) = -|s_i|*delta_i/2 * Var(|S_i-t_i|)."""
    grad = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        alpha = np.exp(-epsilons[i] / 2.0)
        Z, A, B, D, P, C = compute_sums(alpha, t_vec[i], m)
        var_abs = B / Z - (A / Z) ** 2
        grad[i] = -np.abs(s_vec[i]) * delta_vec[i] / 2.0 * var_abs
    return grad


def objective_variance(epsilons, s_vec, delta_vec, t_vec, m):
    """Objective for mode='var': sum_i s_i^2 * delta_i^2 * Var(S_i)."""
    total = 0.0
    for i in range(len(epsilons)):
        alpha = np.exp(-epsilons[i] / 2.0)
        Z, A, B, D, P, C = compute_sums(alpha, t_vec[i], m)
        var_s = B / Z - (P / Z) ** 2
        total += s_vec[i] ** 2 * delta_vec[i] ** 2 * var_s
    return total


def gradient_variance(epsilons, s_vec, delta_vec, t_vec, m):
    """Gradient for mode='var' using covariance expressions with all 6 sums."""
    grad = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        alpha = np.exp(-epsilons[i] / 2.0)
        Z, A, B, D, P, C = compute_sums(alpha, t_vec[i], m)
        cov1 = D / Z - B * A / Z ** 2
        cov2 = C / Z - P * A / Z ** 2
        dVar = -0.5 * (cov1 - 2.0 * (P / Z) * cov2)
        grad[i] = s_vec[i] ** 2 * delta_vec[i] ** 2 * dVar
    return grad


def _compute_phi(epsilon_i, s_i, delta_i, t_i, m, mode):
    """Compute phi_i(epsilon_i) — the marginal cost that equals lambda at optimum."""
    alpha = np.exp(-epsilon_i / 2.0)
    Z, A, B, D, P, C = compute_sums(alpha, t_i, m)
    if mode == "abs":
        var_abs = B / Z - (A / Z) ** 2
        return -np.abs(s_i) * delta_i / 2.0 * var_abs
    else:
        cov1 = D / Z - B * A / Z ** 2
        cov2 = C / Z - P * A / Z ** 2
        dVar = -0.5 * (cov1 - 2.0 * (P / Z) * cov2)
        return s_i ** 2 * delta_i ** 2 * dVar


def _solve_eps_for_lambda(lambda_val, s_i, delta_i, t_i, m,
                           eps_lo, eps_hi, mode, tol):
    """Find epsilon_i such that phi_i(epsilon_i) = lambda_val via bisection."""
    phi_lo = _compute_phi(eps_lo, s_i, delta_i, t_i, m, mode)
    if phi_lo >= lambda_val:
        return eps_lo
    phi_hi = _compute_phi(eps_hi, s_i, delta_i, t_i, m, mode)
    if phi_hi <= lambda_val:
        return eps_hi
    for _ in range(200):
        if eps_hi - eps_lo < tol:
            break
        eps_mid = (eps_lo + eps_hi) / 2.0
        if _compute_phi(eps_mid, s_i, delta_i, t_i, m, mode) < lambda_val:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid
    return (eps_lo + eps_hi) / 2.0


def optimal_budget_allocation_exact(
    sensitivities, total_budget, epsilon_min, epsilon_max=None,
    domain_widths=None, true_indices=None, num_candidates=1000,
    mode="abs", tol=1e-8
):
    """
    Compute optimal privacy budget allocation using exact discrete exponential
    mechanism moments (not the continuous Laplace approximation).

    Minimizes either expected absolute error (mode='abs') or variance of
    propagated noise (mode='var') at the target node.

    Args:
        sensitivities: Dict {root_name: d(target)/d(root)}
        total_budget: Total epsilon budget B
        epsilon_min: Minimum budget per root
        epsilon_max: Maximum budget per root (None = total_budget)
        domain_widths: Dict {root_name: hi - lo}
        true_indices: Dict {root_name: real-valued index in [0, m-1]}
        num_candidates: Number of discrete grid points (m)
        mode: "abs" or "var"
        tol: Convergence tolerance

    Returns:
        allocation: Dict {root_name: epsilon_i}
    """
    roots = list(sensitivities.keys())
    k = len(roots)

    if k == 0:
        return {}

    if epsilon_max is None:
        epsilon_max = total_budget

    # --- Edge cases ---
    # All sensitivities zero → uniform
    all_zero = all(sensitivities[r] == 0 for r in roots)
    if all_zero:
        eps_each = max(total_budget / k, epsilon_min)
        return {r: eps_each for r in roots}

    # Single root → full budget (clamped)
    if k == 1:
        eps = np.clip(total_budget, epsilon_min, epsilon_max)
        return {roots[0]: eps}

    # Insufficient budget for minimums → uniform
    if k * epsilon_min > total_budget:
        eps_each = total_budget / k
        return {r: eps_each for r in roots}

    # --- Build arrays ---
    s_vec = np.array([sensitivities[r] for r in roots], dtype=float)
    delta_vec = np.ones(k, dtype=float)
    t_vec = np.full(k, (num_candidates - 1) / 2.0)  # default: center

    if domain_widths is not None:
        for i, r in enumerate(roots):
            dw = domain_widths.get(r, 1.0)
            delta_vec[i] = dw / (num_candidates - 1)

    if true_indices is not None:
        for i, r in enumerate(roots):
            if r in true_indices:
                t_vec[i] = true_indices[r]

    # --- Laplace closed-form initial guess (approximate for discrete mechanism) ---
    if mode == "abs":
        raw_weights = np.array([
            (np.abs(s_vec[i]) * delta_vec[i]) ** 0.5 for i in range(k)
        ])
    else:
        raw_weights = np.array([
            (np.abs(s_vec[i]) * delta_vec[i]) ** (2.0 / 3.0) for i in range(k)
        ])

    w_sum = np.sum(raw_weights)
    if w_sum > 0:
        x0 = total_budget * raw_weights / w_sum
    else:
        x0 = np.full(k, total_budget / k)
    x0 = np.clip(x0, epsilon_min, epsilon_max)
    # Rescale to satisfy budget constraint after clamping
    x0 = x0 * (total_budget / np.sum(x0))

    # --- Primary solver: scipy SLSQP ---
    try:
        from scipy.optimize import minimize as sp_minimize

        bounds = [(epsilon_min, epsilon_max)] * k
        constraint = {'type': 'eq', 'fun': lambda e: np.sum(e) - total_budget}

        if mode == "abs":
            obj_fn = lambda e: objective_absolute(e, s_vec, delta_vec, t_vec, num_candidates)
            jac_fn = lambda e: gradient_absolute(e, s_vec, delta_vec, t_vec, num_candidates)
        else:
            obj_fn = lambda e: objective_variance(e, s_vec, delta_vec, t_vec, num_candidates)
            jac_fn = lambda e: gradient_variance(e, s_vec, delta_vec, t_vec, num_candidates)

        result = sp_minimize(
            obj_fn, x0, jac=jac_fn, method='SLSQP',
            bounds=bounds, constraints=constraint,
            options={'ftol': tol, 'maxiter': 500, 'disp': False}
        )
        eps_opt = result.x
    except ImportError:
        # --- Fallback: nested bisection ---
        s_abs = np.abs(s_vec)
        # Compute lambda bounds
        lambda_lo = min(
            _compute_phi(epsilon_min, s_abs[i], delta_vec[i], t_vec[i],
                         num_candidates, mode)
            for i in range(k)
        )
        lambda_hi = max(
            _compute_phi(epsilon_max, s_abs[i], delta_vec[i], t_vec[i],
                         num_candidates, mode)
            for i in range(k)
        )

        for _ in range(200):
            if lambda_hi - lambda_lo < tol:
                break
            lambda_mid = (lambda_lo + lambda_hi) / 2.0
            eps_trial = np.array([
                _solve_eps_for_lambda(
                    lambda_mid, s_abs[i], delta_vec[i], t_vec[i],
                    num_candidates, epsilon_min, epsilon_max, mode, tol
                )
                for i in range(k)
            ])
            budget_used = np.sum(eps_trial)
            if budget_used > total_budget:
                # More negative lambda → smaller eps → less budget
                lambda_hi = lambda_mid
            else:
                lambda_lo = lambda_mid

        eps_opt = np.array([
            _solve_eps_for_lambda(
                (lambda_lo + lambda_hi) / 2.0, s_abs[i], delta_vec[i],
                t_vec[i], num_candidates, epsilon_min, epsilon_max, mode, tol
            )
            for i in range(k)
        ])

    # Final normalization to exactly satisfy budget
    eps_opt = np.clip(eps_opt, epsilon_min, epsilon_max)
    eps_opt = eps_opt * (total_budget / np.sum(eps_opt))

    return {roots[i]: float(eps_opt[i]) for i in range(k)}


def optimal_budget_allocation_exact_weighted(
    sensitivities, constraint_target, epsilon_min, epsilon_max=None,
    domain_widths=None, constraint_weights=None,
    true_indices=None, num_candidates=1000,
    mode="abs", tol=1e-8
):
    """
    Optimal budget allocation under a WEIGHTED constraint for the discrete
    exponential mechanism.

    Instead of  sum(eps_i) = B,  the constraint is:
        sum(eps_i * D_i) = constraint_target
    where D_i = constraint_weights[root_i] (typically domain widths).

    The objective is unchanged: min sum_i |h_i| * delta_i * E[|S_i - t_i|].
    The stationarity condition becomes |h_i| * eta'(eps_i) = lambda * D_i.

    Args:
        sensitivities: Dict {root_name: d(target)/d(root)}
        constraint_target: RHS of weighted constraint (e.g. eps * S_dom)
        epsilon_min: Minimum budget per root
        epsilon_max: Maximum budget per root (None = no cap)
        domain_widths: Dict {root_name: hi - lo} for delta_i computation
        constraint_weights: Dict {root_name: D_i} for the weighted constraint
        true_indices: Dict {root_name: real-valued index in [0, m-1]}
        num_candidates: Number of discrete grid points (m)
        mode: "abs" or "var"
        tol: Convergence tolerance

    Returns:
        allocation: Dict {root_name: epsilon_i}
    """
    roots = list(sensitivities.keys())
    k = len(roots)

    if k == 0:
        return {}

    # Build weight vector
    D_vec = np.array([constraint_weights[r] for r in roots], dtype=float)

    if epsilon_max is None:
        epsilon_max = constraint_target / np.min(D_vec)  # upper bound per root

    # --- Edge cases ---
    all_zero = all(sensitivities[r] == 0 for r in roots)
    if all_zero:
        eps_each = constraint_target / np.sum(D_vec)
        eps_each = max(eps_each, epsilon_min)
        return {r: eps_each for r in roots}

    if k == 1:
        eps = constraint_target / D_vec[0]
        eps = np.clip(eps, epsilon_min, epsilon_max)
        return {roots[0]: float(eps)}

    # Insufficient budget for minimums
    if np.dot(np.full(k, epsilon_min), D_vec) > constraint_target:
        eps_each = constraint_target / np.sum(D_vec)
        return {r: eps_each for r in roots}

    # --- Build arrays ---
    s_vec = np.array([sensitivities[r] for r in roots], dtype=float)
    delta_vec = np.ones(k, dtype=float)
    t_vec = np.full(k, (num_candidates - 1) / 2.0)

    if domain_widths is not None:
        for i, r in enumerate(roots):
            dw = domain_widths.get(r, 1.0)
            delta_vec[i] = dw / (num_candidates - 1)

    if true_indices is not None:
        for i, r in enumerate(roots):
            if r in true_indices:
                t_vec[i] = true_indices[r]

    # --- Warm start ---
    # Stationarity: |h_i| * eta'(eps_i) = lambda * D_i
    # Laplace approx: eps_i proportional to sqrt(|s_i| * delta_i / D_i)
    if mode == "abs":
        raw_weights = np.array([
            (np.abs(s_vec[i]) * delta_vec[i] / D_vec[i]) ** 0.5
            for i in range(k)
        ])
    else:
        raw_weights = np.array([
            (np.abs(s_vec[i]) * delta_vec[i] / D_vec[i]) ** (2.0 / 3.0)
            for i in range(k)
        ])

    w_sum = np.dot(raw_weights, D_vec)
    if w_sum > 0:
        x0 = constraint_target * raw_weights / w_sum
    else:
        x0 = np.full(k, constraint_target / np.sum(D_vec))
    x0 = np.clip(x0, epsilon_min, epsilon_max)
    # Rescale to satisfy weighted constraint after clamping
    x0 = x0 * (constraint_target / np.dot(x0, D_vec))

    # --- Primary solver: scipy SLSQP ---
    try:
        from scipy.optimize import minimize as sp_minimize

        bounds = [(epsilon_min, epsilon_max)] * k
        constraint = {
            'type': 'eq',
            'fun': lambda e: np.dot(e, D_vec) - constraint_target
        }

        if mode == "abs":
            obj_fn = lambda e: objective_absolute(
                e, s_vec, delta_vec, t_vec, num_candidates)
            jac_fn = lambda e: gradient_absolute(
                e, s_vec, delta_vec, t_vec, num_candidates)
        else:
            obj_fn = lambda e: objective_variance(
                e, s_vec, delta_vec, t_vec, num_candidates)
            jac_fn = lambda e: gradient_variance(
                e, s_vec, delta_vec, t_vec, num_candidates)

        result = sp_minimize(
            obj_fn, x0, jac=jac_fn, method='SLSQP',
            bounds=bounds, constraints=constraint,
            options={'ftol': tol, 'maxiter': 500, 'disp': False}
        )
        eps_opt = result.x
    except ImportError:
        # --- Fallback: nested bisection with weighted stationarity ---
        # Stationarity: |h_i| * eta'(eps_i) = lambda * D_i
        # So for root i, solve phi_i(eps_i) = lambda * D_i
        s_abs = np.abs(s_vec)
        lambda_lo = min(
            _compute_phi(epsilon_min, s_abs[i], delta_vec[i], t_vec[i],
                         num_candidates, mode) / D_vec[i]
            for i in range(k)
        )
        lambda_hi = max(
            _compute_phi(epsilon_max, s_abs[i], delta_vec[i], t_vec[i],
                         num_candidates, mode) / D_vec[i]
            for i in range(k)
        )

        for _ in range(200):
            if lambda_hi - lambda_lo < tol:
                break
            lambda_mid = (lambda_lo + lambda_hi) / 2.0
            eps_trial = np.array([
                _solve_eps_for_lambda(
                    lambda_mid * D_vec[i], s_abs[i], delta_vec[i], t_vec[i],
                    num_candidates, epsilon_min, epsilon_max, mode, tol
                )
                for i in range(k)
            ])
            budget_used = np.dot(eps_trial, D_vec)
            if budget_used > constraint_target:
                lambda_hi = lambda_mid
            else:
                lambda_lo = lambda_mid

        eps_opt = np.array([
            _solve_eps_for_lambda(
                (lambda_lo + lambda_hi) / 2.0 * D_vec[i],
                s_abs[i], delta_vec[i], t_vec[i],
                num_candidates, epsilon_min, epsilon_max, mode, tol
            )
            for i in range(k)
        ])

    # Final normalization to exactly satisfy weighted constraint
    eps_opt = np.clip(eps_opt, epsilon_min, epsilon_max)
    eps_opt = eps_opt * (constraint_target / np.dot(eps_opt, D_vec))

    return {roots[i]: float(eps_opt[i]) for i in range(k)}


# =============================================================================
# Bounded Laplace: Exact Discrete Budget Allocation
# =============================================================================

def blap_cdf(x, epsilon):
    """Laplace CDF with scale b = 1/epsilon.

    F(x) = 0.5 + 0.5 * sign(x) * (1 - exp(-|x| * epsilon))
    """
    x = np.asarray(x, dtype=float)
    return 0.5 + 0.5 * np.sign(x) * (1.0 - np.exp(-np.abs(x) * epsilon))


def blap_expected_abs(epsilon, t, m):
    """E[|S - t|] for bounded Laplace on discrete grid {0, ..., m-1}.

    Computes exact PMF P(S=s | t, epsilon) via Laplace CDF differences,
    then sums |s - t| * P(S=s).
    """
    s_arr = np.arange(m, dtype=float)
    upper = blap_cdf(s_arr + 0.5 - t, epsilon)
    lower = blap_cdf(s_arr - 0.5 - t, epsilon)
    probs = upper - lower
    probs[0] = blap_cdf(0.5 - t, epsilon)
    probs[-1] = 1.0 - blap_cdf(m - 1.5 - t, epsilon)
    return np.sum(np.abs(s_arr - t) * probs)


def blap_grad_expected_abs(epsilon, t, m):
    """d/d(epsilon) E[|S - t|] for bounded Laplace.

    Uses dF(x; 1/eps)/d(eps) = 0.5 * x * exp(-|x| * eps).
    """
    s_arr = np.arange(m, dtype=float)

    def dcdf(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * x * np.exp(-np.abs(x) * epsilon)

    dupper = dcdf(s_arr + 0.5 - t)
    dlower = dcdf(s_arr - 0.5 - t)
    dprobs = dupper - dlower
    dprobs[0] = dcdf(0.5 - t)
    dprobs[-1] = -dcdf(m - 1.5 - t)
    return np.sum(np.abs(s_arr - t) * dprobs)


def objective_absolute_blap(epsilons, s_vec, delta_vec, t_vec, m):
    """Objective for BLap mode='abs': sum_i |s_i| * delta_i * E_BLap[|S_i - t_i|]."""
    total = 0.0
    for i in range(len(epsilons)):
        total += np.abs(s_vec[i]) * delta_vec[i] * blap_expected_abs(
            epsilons[i], t_vec[i], m)
    return total


def gradient_absolute_blap(epsilons, s_vec, delta_vec, t_vec, m):
    """Gradient for BLap mode='abs'."""
    grad = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        grad[i] = np.abs(s_vec[i]) * delta_vec[i] * blap_grad_expected_abs(
            epsilons[i], t_vec[i], m)
    return grad


def optimal_budget_allocation_blap(
    sensitivities, total_budget, epsilon_min, epsilon_max=None,
    domain_widths=None, num_candidates=1000, tol=1e-8
):
    """Optimal budget allocation using exact bounded Laplace discrete moments.

    Same interface and structure as optimal_budget_allocation_exact(), but
    uses the Laplace CDF to compute E[|S-t|] on the discrete grid instead
    of the exponential mechanism's moments.

    Uses integer center t = m//2 for the sup over grid positions. (The
    Laplace and staircase CDFs coincide at all integers, so half-integer
    t would conflate the two mechanisms; integer t correctly evaluates
    CDF differences at half-integer offsets where the distributions differ.)
    """
    roots = list(sensitivities.keys())
    k = len(roots)

    if k == 0:
        return {}
    if epsilon_max is None:
        epsilon_max = total_budget

    # Edge cases
    all_zero = all(sensitivities[r] == 0 for r in roots)
    if all_zero:
        eps_each = max(total_budget / k, epsilon_min)
        return {r: eps_each for r in roots}
    if k == 1:
        eps = np.clip(total_budget, epsilon_min, epsilon_max)
        return {roots[0]: eps}
    if k * epsilon_min > total_budget:
        eps_each = total_budget / k
        return {r: eps_each for r in roots}

    # Build arrays
    s_vec = np.array([sensitivities[r] for r in roots], dtype=float)
    delta_vec = np.ones(k, dtype=float)
    t_vec = np.full(k, float(num_candidates // 2))

    if domain_widths is not None:
        for i, r in enumerate(roots):
            dw = domain_widths.get(r, 1.0)
            delta_vec[i] = dw / (num_candidates - 1)

    # Warm start: eps_i proportional to sqrt(|s_i| * delta_i)
    raw_weights = np.array([
        (np.abs(s_vec[i]) * delta_vec[i]) ** 0.5 for i in range(k)
    ])
    w_sum = np.sum(raw_weights)
    if w_sum > 0:
        x0 = total_budget * raw_weights / w_sum
    else:
        x0 = np.full(k, total_budget / k)
    x0 = np.clip(x0, epsilon_min, epsilon_max)
    x0 = x0 * (total_budget / np.sum(x0))

    # SLSQP solver
    try:
        from scipy.optimize import minimize as sp_minimize

        bounds = [(epsilon_min, epsilon_max)] * k
        constraint = {'type': 'eq', 'fun': lambda e: np.sum(e) - total_budget}

        obj_fn = lambda e: objective_absolute_blap(e, s_vec, delta_vec, t_vec, num_candidates)
        jac_fn = lambda e: gradient_absolute_blap(e, s_vec, delta_vec, t_vec, num_candidates)

        result = sp_minimize(
            obj_fn, x0, jac=jac_fn, method='SLSQP',
            bounds=bounds, constraints=constraint,
            options={'ftol': tol, 'maxiter': 500, 'disp': False}
        )
        eps_opt = result.x
    except ImportError:
        eps_opt = x0  # fallback to warm start (exact for BLap)

    eps_opt = np.clip(eps_opt, epsilon_min, epsilon_max)
    eps_opt = eps_opt * (total_budget / np.sum(eps_opt))

    return {roots[i]: float(eps_opt[i]) for i in range(k)}


def optimal_budget_allocation_blap_weighted(
    sensitivities, constraint_target, epsilon_min, epsilon_max=None,
    domain_widths=None, constraint_weights=None,
    num_candidates=1000, tol=1e-8
):
    """Optimal budget allocation under weighted constraint for bounded Laplace.

    Constraint: sum(eps_i * D_i) = constraint_target
    where D_i = constraint_weights[root_i].

    Same interface as optimal_budget_allocation_blap but with weighted constraint.
    Uses integer center t = m//2.
    """
    roots = list(sensitivities.keys())
    k = len(roots)

    if k == 0:
        return {}

    D_vec = np.array([constraint_weights[r] for r in roots], dtype=float)

    if epsilon_max is None:
        epsilon_max = constraint_target / np.min(D_vec)

    # Edge cases
    all_zero = all(sensitivities[r] == 0 for r in roots)
    if all_zero:
        eps_each = max(constraint_target / np.sum(D_vec), epsilon_min)
        return {r: eps_each for r in roots}
    if k == 1:
        eps = np.clip(constraint_target / D_vec[0], epsilon_min, epsilon_max)
        return {roots[0]: float(eps)}
    if np.dot(np.full(k, epsilon_min), D_vec) > constraint_target:
        eps_each = constraint_target / np.sum(D_vec)
        return {r: eps_each for r in roots}

    # Build arrays
    s_vec = np.array([sensitivities[r] for r in roots], dtype=float)
    delta_vec = np.ones(k, dtype=float)
    t_vec = np.full(k, float(num_candidates // 2))

    if domain_widths is not None:
        for i, r in enumerate(roots):
            dw = domain_widths.get(r, 1.0)
            delta_vec[i] = dw / (num_candidates - 1)

    # Warm start: eps_i proportional to sqrt(|s_i| * delta_i / D_i)
    raw_weights = np.array([
        (np.abs(s_vec[i]) * delta_vec[i] / D_vec[i]) ** 0.5 for i in range(k)
    ])
    w_sum = np.dot(raw_weights, D_vec)
    if w_sum > 0:
        x0 = constraint_target * raw_weights / w_sum
    else:
        x0 = np.full(k, constraint_target / np.sum(D_vec))
    x0 = np.clip(x0, epsilon_min, epsilon_max)
    x0 = x0 * (constraint_target / np.dot(x0, D_vec))

    # SLSQP solver
    try:
        from scipy.optimize import minimize as sp_minimize

        bounds = [(epsilon_min, epsilon_max)] * k
        constraint = {
            'type': 'eq',
            'fun': lambda e: np.dot(e, D_vec) - constraint_target
        }

        obj_fn = lambda e: objective_absolute_blap(
            e, s_vec, delta_vec, t_vec, num_candidates)
        jac_fn = lambda e: gradient_absolute_blap(
            e, s_vec, delta_vec, t_vec, num_candidates)

        result = sp_minimize(
            obj_fn, x0, jac=jac_fn, method='SLSQP',
            bounds=bounds, constraints=constraint,
            options={'ftol': tol, 'maxiter': 500, 'disp': False}
        )
        eps_opt = result.x
    except ImportError:
        eps_opt = x0

    eps_opt = np.clip(eps_opt, epsilon_min, epsilon_max)
    eps_opt = eps_opt * (constraint_target / np.dot(eps_opt, D_vec))

    return {roots[i]: float(eps_opt[i]) for i in range(k)}


# =============================================================================
# Staircase: Exact Discrete Budget Allocation
# =============================================================================

def staircase_cdf(x, epsilon):
    """CDF of the Geng-Viswanath staircase noise with sensitivity=1.

    Derived from diffprivlib's Staircase.randomise() sampling procedure.
    The noise eta is symmetric with |eta| density:
      f(x) = (1-p) * p^{g-1/2}  on [g, g+gamma)
      f(x) = (1-p) * p^{g+1/2}  on [g+gamma, g+1)
    where p = e^{-eps}, gamma = sqrt(p)/(sqrt(p)+1), g = floor(x).

    CDF of |eta|:
      F(x) = 1 - p^g + (1-p)*p^{g-1/2}*(x-g)        for x in [g, g+gamma)
      F(x) = 1 - p^{g+1/2} + (1-p)*p^{g+1/2}*(x-g-gamma)  for x in [g+gamma, g+1)
    """
    x = np.asarray(x, dtype=float)
    p = np.exp(-epsilon)
    sqrt_p = np.sqrt(p)
    gamma = sqrt_p / (sqrt_p + 1.0)

    ax = np.abs(x)
    g = np.floor(ax)
    frac = ax - g

    in_lower = frac < gamma

    # Lower part: F_{|eta|}(x) = 1 - p^g + (1-p)*p^{g-0.5}*frac
    F_abs_lower = 1.0 - p**g + (1.0 - p) * p**(g - 0.5) * frac
    # Upper part: F_{|eta|}(x) = 1 - p^{g+0.5} + (1-p)*p^{g+0.5}*(frac - gamma)
    F_abs_upper = 1.0 - p**(g + 0.5) + (1.0 - p) * p**(g + 0.5) * (frac - gamma)

    F_abs = np.where(in_lower, F_abs_lower, F_abs_upper)

    # Full symmetric CDF
    return np.where(x >= 0, 0.5 + 0.5 * F_abs, 0.5 * (1.0 - F_abs))


def staircase_expected_abs(epsilon, t, m):
    """E[|S - t|] for staircase mechanism on discrete grid {0, ..., m-1}.

    Computes exact PMF P(S=s | t, epsilon) via staircase CDF differences,
    then sums |s - t| * P(S=s).
    """
    s_arr = np.arange(m, dtype=float)
    upper = staircase_cdf(s_arr + 0.5 - t, epsilon)
    lower = staircase_cdf(s_arr - 0.5 - t, epsilon)
    probs = upper - lower
    probs[0] = staircase_cdf(0.5 - t, epsilon)
    probs[-1] = 1.0 - staircase_cdf(m - 1.5 - t, epsilon)
    return np.sum(np.abs(s_arr - t) * probs)


def staircase_grad_expected_abs(epsilon, t, m, h=1e-7):
    """d/d(epsilon) E[|S - t|] for staircase via central finite differences."""
    return (staircase_expected_abs(epsilon + h, t, m)
            - staircase_expected_abs(epsilon - h, t, m)) / (2.0 * h)


def objective_absolute_staircase(epsilons, s_vec, delta_vec, t_vec, m):
    """Objective for Staircase mode='abs': sum_i |s_i| * delta_i * E_Stair[|S_i - t_i|]."""
    total = 0.0
    for i in range(len(epsilons)):
        total += np.abs(s_vec[i]) * delta_vec[i] * staircase_expected_abs(
            epsilons[i], t_vec[i], m)
    return total


def gradient_absolute_staircase(epsilons, s_vec, delta_vec, t_vec, m):
    """Gradient for Staircase mode='abs'."""
    grad = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        grad[i] = np.abs(s_vec[i]) * delta_vec[i] * staircase_grad_expected_abs(
            epsilons[i], t_vec[i], m)
    return grad


def optimal_budget_allocation_staircase(
    sensitivities, total_budget, epsilon_min, epsilon_max=None,
    domain_widths=None, num_candidates=1000, tol=1e-8
):
    """Optimal budget allocation using exact staircase discrete moments.

    Same interface and structure as optimal_budget_allocation_exact(), but
    uses the Geng-Viswanath staircase CDF to compute E[|S-t|] on the
    discrete grid instead of the exponential mechanism's moments.

    Uses integer center t = m//2 for the sup over grid positions.
    """
    roots = list(sensitivities.keys())
    k = len(roots)

    if k == 0:
        return {}
    if epsilon_max is None:
        epsilon_max = total_budget

    # Edge cases
    all_zero = all(sensitivities[r] == 0 for r in roots)
    if all_zero:
        eps_each = max(total_budget / k, epsilon_min)
        return {r: eps_each for r in roots}
    if k == 1:
        eps = np.clip(total_budget, epsilon_min, epsilon_max)
        return {roots[0]: eps}
    if k * epsilon_min > total_budget:
        eps_each = total_budget / k
        return {r: eps_each for r in roots}

    # Build arrays
    s_vec = np.array([sensitivities[r] for r in roots], dtype=float)
    delta_vec = np.ones(k, dtype=float)
    t_vec = np.full(k, float(num_candidates // 2))

    if domain_widths is not None:
        for i, r in enumerate(roots):
            dw = domain_widths.get(r, 1.0)
            delta_vec[i] = dw / (num_candidates - 1)

    # Warm start: eps_i proportional to sqrt(|s_i| * delta_i)
    raw_weights = np.array([
        (np.abs(s_vec[i]) * delta_vec[i]) ** 0.5 for i in range(k)
    ])
    w_sum = np.sum(raw_weights)
    if w_sum > 0:
        x0 = total_budget * raw_weights / w_sum
    else:
        x0 = np.full(k, total_budget / k)
    x0 = np.clip(x0, epsilon_min, epsilon_max)
    x0 = x0 * (total_budget / np.sum(x0))

    # SLSQP solver
    try:
        from scipy.optimize import minimize as sp_minimize

        bounds = [(epsilon_min, epsilon_max)] * k
        constraint = {'type': 'eq', 'fun': lambda e: np.sum(e) - total_budget}

        obj_fn = lambda e: objective_absolute_staircase(e, s_vec, delta_vec, t_vec, num_candidates)
        jac_fn = lambda e: gradient_absolute_staircase(e, s_vec, delta_vec, t_vec, num_candidates)

        result = sp_minimize(
            obj_fn, x0, jac=jac_fn, method='SLSQP',
            bounds=bounds, constraints=constraint,
            options={'ftol': tol, 'maxiter': 500, 'disp': False}
        )
        eps_opt = result.x
    except ImportError:
        eps_opt = x0  # fallback to warm start

    eps_opt = np.clip(eps_opt, epsilon_min, epsilon_max)
    eps_opt = eps_opt * (total_budget / np.sum(eps_opt))

    return {roots[i]: float(eps_opt[i]) for i in range(k)}


def optimal_budget_allocation_staircase_weighted(
    sensitivities, constraint_target, epsilon_min, epsilon_max=None,
    domain_widths=None, constraint_weights=None,
    num_candidates=1000, tol=1e-8
):
    """Optimal budget allocation under weighted constraint for staircase mechanism.

    Constraint: sum(eps_i * D_i) = constraint_target
    where D_i = constraint_weights[root_i].

    Same interface as optimal_budget_allocation_staircase but with weighted
    constraint. Uses integer center t = m//2.
    """
    roots = list(sensitivities.keys())
    k = len(roots)

    if k == 0:
        return {}

    D_vec = np.array([constraint_weights[r] for r in roots], dtype=float)

    if epsilon_max is None:
        epsilon_max = constraint_target / np.min(D_vec)

    # Edge cases
    all_zero = all(sensitivities[r] == 0 for r in roots)
    if all_zero:
        eps_each = max(constraint_target / np.sum(D_vec), epsilon_min)
        return {r: eps_each for r in roots}
    if k == 1:
        eps = np.clip(constraint_target / D_vec[0], epsilon_min, epsilon_max)
        return {roots[0]: float(eps)}
    if np.dot(np.full(k, epsilon_min), D_vec) > constraint_target:
        eps_each = constraint_target / np.sum(D_vec)
        return {r: eps_each for r in roots}

    # Build arrays
    s_vec = np.array([sensitivities[r] for r in roots], dtype=float)
    delta_vec = np.ones(k, dtype=float)
    t_vec = np.full(k, float(num_candidates // 2))

    if domain_widths is not None:
        for i, r in enumerate(roots):
            dw = domain_widths.get(r, 1.0)
            delta_vec[i] = dw / (num_candidates - 1)

    # Warm start: eps_i proportional to sqrt(|s_i| * delta_i / D_i)
    raw_weights = np.array([
        (np.abs(s_vec[i]) * delta_vec[i] / D_vec[i]) ** 0.5 for i in range(k)
    ])
    w_sum = np.dot(raw_weights, D_vec)
    if w_sum > 0:
        x0 = constraint_target * raw_weights / w_sum
    else:
        x0 = np.full(k, constraint_target / np.sum(D_vec))
    x0 = np.clip(x0, epsilon_min, epsilon_max)
    x0 = x0 * (constraint_target / np.dot(x0, D_vec))

    # SLSQP solver
    try:
        from scipy.optimize import minimize as sp_minimize

        bounds = [(epsilon_min, epsilon_max)] * k
        constraint = {
            'type': 'eq',
            'fun': lambda e: np.dot(e, D_vec) - constraint_target
        }

        obj_fn = lambda e: objective_absolute_staircase(
            e, s_vec, delta_vec, t_vec, num_candidates)
        jac_fn = lambda e: gradient_absolute_staircase(
            e, s_vec, delta_vec, t_vec, num_candidates)

        result = sp_minimize(
            obj_fn, x0, jac=jac_fn, method='SLSQP',
            bounds=bounds, constraints=constraint,
            options={'ftol': tol, 'maxiter': 500, 'disp': False}
        )
        eps_opt = result.x
    except ImportError:
        eps_opt = x0

    eps_opt = np.clip(eps_opt, epsilon_min, epsilon_max)
    eps_opt = eps_opt * (constraint_target / np.dot(eps_opt, D_vec))

    return {roots[i]: float(eps_opt[i]) for i in range(k)}


def sanitize_dep_aware_v8(samples, nodes, edges, expressions,
                          bi_lipschitz_constants, epsilon,
                          target_keys, epsilon_min=0.05,
                          epsilon_max=None, mode="abs",
                          num_candidates=1000,
                          sensitivity_method="worst_case",
                          population_means=None,
                          use_pop_mean_index=False):
    """
    Dependency-aware sanitization with data-independent budget allocation (V8).

    Preempt++ V8: exact discrete exponential mechanism moments with
    data-independent sensitivity and noise estimation.
    Eliminates two side-channel leaks present in earlier versions:

      1. Sensitivities are computed as worst-case over the entire domain
         (via compute_worst_case_sensitivities at all 2^k domain corners),
         not from the patient's actual values.
      2. Expected noise E[|S_i - t_i|] uses the center of the grid
         (true_indices=None), not the patient's true position.

    Result: every patient with the same template gets an identical
    budget allocation — the allocation reveals nothing about the data.

    Supports two optimization modes:
      - 'abs': minimize expected absolute error at target
      - 'var': minimize variance of propagated noise at target

    Args:
        samples: List of sample dictionaries with conversation turns
        nodes: Dictionary mapping node names to their parent dependencies
        edges: List of (source, target) dependency edges
        expressions: Dict mapping node names to computation expressions
        bi_lipschitz_constants: Unused, kept for interface consistency
        epsilon: Base privacy budget
        target_keys: Dict mapping template name to target node name
        epsilon_min: Minimum budget per root node
        epsilon_max: Maximum budget per root node (None = no cap)
        mode: 'abs' (expected absolute error) or 'var' (variance)
        num_candidates: Number of discrete grid points (m)

    Returns:
        all_values: List of sanitized value dictionaries
        epsilon_blocks: List of actual epsilon budgets used per node
        all_allocation_info: List of dicts with allocation details per sample
    """
    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    all_values = []
    all_allocation_info = []
    _wc_sensitivity_cache = {}
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}

        # Identify root and derived nodes for this sample
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

        # Find the target node present in this sample
        target_node = None
        for tname, tnode in target_keys.items():
            if tnode in ground_truth and tnode not in skip_nodes:
                target_node = tnode
                break

        # B = count of all nodes excluding skip/class nodes, times ε
        n_nodes = len([n for n in topo_order if n not in skip_nodes])
        total_budget = n_nodes * epsilon

        if target_node and len(root_nodes) > 0:
            # Compute data-independent sensitivities (same for all patients)
            cache_key = (target_node, frozenset(root_nodes), sensitivity_method)
            if cache_key not in _wc_sensitivity_cache:
                if sensitivity_method == "expected":
                    _wc_sensitivity_cache[cache_key] = compute_expected_sensitivities(
                        target_node, root_nodes, expressions, skip_nodes,
                        topo_order, nodes, sanitizer.MEDICAL_DOMAINS
                    )
                elif sensitivity_method == "population_mean":
                    _wc_sensitivity_cache[cache_key] = compute_population_mean_sensitivities(
                        target_node, root_nodes, expressions, skip_nodes,
                        topo_order, nodes, population_means
                    )
                else:  # "worst_case" (default)
                    _wc_sensitivity_cache[cache_key] = compute_worst_case_sensitivities(
                        target_node, root_nodes, expressions, skip_nodes,
                        topo_order, nodes, sanitizer.MEDICAL_DOMAINS
                    )
            sensitivities = _wc_sensitivity_cache[cache_key]
            # Build domain-width dict for discrete mechanism
            dw = {}
            for root in root_nodes:
                lo, hi = sanitizer.MEDICAL_DOMAINS.get(
                    root, sanitizer.MEDICAL_DOMAINS["_default"]
                )
                dw[root] = hi - lo
            # Compute true_indices from population means if requested
            true_indices = None
            if use_pop_mean_index and population_means is not None:
                true_indices = {}
                for root in root_nodes:
                    lo, hi = sanitizer.MEDICAL_DOMAINS.get(
                        root, sanitizer.MEDICAL_DOMAINS["_default"]
                    )
                    mu = population_means[root]
                    true_indices[root] = (num_candidates - 1) * (mu - lo) / (hi - lo)

            allocation = optimal_budget_allocation_exact(
                sensitivities, total_budget, epsilon_min, epsilon_max,
                domain_widths=dw, true_indices=true_indices,
                num_candidates=num_candidates, mode=mode
            )
        else:
            allocation = {r: epsilon for r in root_nodes}
            sensitivities = {r: 0.0 for r in root_nodes}

        # Noise root nodes with allocated budgets using discrete exponential mechanism
        for root in root_nodes:
            root_eps = allocation.get(root, epsilon)
            domain_lower, domain_upper = sanitizer.MEDICAL_DOMAINS.get(
                root, sanitizer.MEDICAL_DOMAINS["_default"]
            )
            noised_value = san.M_exponential_discrete(
                ground_truth[root], root_eps, domain_lower, domain_upper,
                num_candidates=num_candidates
            )
            sanitized_values[root] = round(float(noised_value), ndigits=4)
            actual_budgets[root] = root_eps

        # Post-process all derived nodes from noised roots
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
                        f"Node '{node}' expression produced non-finite value: {f_x_prime}. "
                        f"Expression: {expressions[node]}, "
                        f"Parent values: {parent_vals}"
                    )
                sanitized_values[node] = f_x_prime
                actual_budgets[node] = 0
            else:
                raise ValueError(
                    f"Derived node '{node}' has no expression and is not a skip node"
                )

        all_values.append(sanitized_values)
        epsilon_blocks.append([actual_budgets.get(node, 0) for node in topo_order])
        all_allocation_info.append({
            "root_nodes": root_nodes,
            "target_node": target_node,
            "sensitivities": {r: float(s) for r, s in sensitivities.items()},
            "allocation": {r: float(e) for r, e in allocation.items()},
            "total_budget": total_budget,
            "mode": mode,
        })

    return all_values, epsilon_blocks, all_allocation_info


# Backward-compatible alias: v7 code already contains the data-independent fixes
sanitize_dep_aware_v7 = sanitize_dep_aware_v8


def sanitize_dep_aware_v8_weighted(
    samples, nodes, edges, expressions,
    bi_lipschitz_constants, epsilon,
    target_keys, s_dom,
    epsilon_min=0.05, epsilon_max=None, mode="abs",
    num_candidates=1000,
    sensitivity_method="worst_case",
    population_means=None,
    use_pop_mean_index=False
):
    """
    Dependency-aware sanitization under domain-space weighted constraint.

    Like sanitize_dep_aware_v8, but the allocation uses the weighted constraint:
        sum(eps_i * D_i) = epsilon * s_dom
    instead of sum(eps_i) = n * epsilon.

    Args:
        s_dom: Domain-space S_max for this template. The constraint target
               is epsilon * s_dom.
        (all other args same as sanitize_dep_aware_v8)
    """
    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    all_values = []
    all_allocation_info = []
    _wc_sensitivity_cache = {}
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

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
            cache_key = (target_node, frozenset(root_nodes), sensitivity_method)
            if cache_key not in _wc_sensitivity_cache:
                if sensitivity_method == "expected":
                    _wc_sensitivity_cache[cache_key] = compute_expected_sensitivities(
                        target_node, root_nodes, expressions, skip_nodes,
                        topo_order, nodes, sanitizer.MEDICAL_DOMAINS
                    )
                elif sensitivity_method == "population_mean":
                    _wc_sensitivity_cache[cache_key] = compute_population_mean_sensitivities(
                        target_node, root_nodes, expressions, skip_nodes,
                        topo_order, nodes, population_means
                    )
                else:
                    _wc_sensitivity_cache[cache_key] = compute_worst_case_sensitivities(
                        target_node, root_nodes, expressions, skip_nodes,
                        topo_order, nodes, sanitizer.MEDICAL_DOMAINS
                    )
            sensitivities = _wc_sensitivity_cache[cache_key]

            dw = {}
            for root in root_nodes:
                lo, hi = sanitizer.MEDICAL_DOMAINS.get(
                    root, sanitizer.MEDICAL_DOMAINS["_default"]
                )
                dw[root] = hi - lo

            true_indices = None
            if use_pop_mean_index and population_means is not None:
                true_indices = {}
                for root in root_nodes:
                    lo, hi = sanitizer.MEDICAL_DOMAINS.get(
                        root, sanitizer.MEDICAL_DOMAINS["_default"]
                    )
                    mu = population_means[root]
                    true_indices[root] = (num_candidates - 1) * (mu - lo) / (hi - lo)

            allocation = optimal_budget_allocation_exact_weighted(
                sensitivities, constraint_target, epsilon_min, epsilon_max,
                domain_widths=dw, constraint_weights=dw,
                true_indices=true_indices,
                num_candidates=num_candidates, mode=mode
            )
        else:
            # Fallback: uniform weighted allocation
            dw = {}
            for root in root_nodes:
                lo, hi = sanitizer.MEDICAL_DOMAINS.get(
                    root, sanitizer.MEDICAL_DOMAINS["_default"]
                )
                dw[root] = hi - lo
            sum_D = sum(dw.values())
            allocation = {r: constraint_target / sum_D for r in root_nodes}
            sensitivities = {r: 0.0 for r in root_nodes}

        for root in root_nodes:
            root_eps = allocation.get(root, epsilon)
            domain_lower, domain_upper = sanitizer.MEDICAL_DOMAINS.get(
                root, sanitizer.MEDICAL_DOMAINS["_default"]
            )
            noised_value = san.M_exponential_discrete(
                ground_truth[root], root_eps, domain_lower, domain_upper,
                num_candidates=num_candidates
            )
            sanitized_values[root] = round(float(noised_value), ndigits=4)
            actual_budgets[root] = root_eps

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
                        f"Node '{node}' expression produced non-finite value: "
                        f"{f_x_prime}. Expression: {expressions[node]}, "
                        f"Parent values: {parent_vals}"
                    )
                sanitized_values[node] = f_x_prime
                actual_budgets[node] = 0
            else:
                raise ValueError(
                    f"Derived node '{node}' has no expression and is not a skip node"
                )

        all_values.append(sanitized_values)
        epsilon_blocks.append([actual_budgets.get(node, 0) for node in topo_order])
        all_allocation_info.append({
            "root_nodes": root_nodes,
            "target_node": target_node,
            "sensitivities": {r: float(s) for r, s in sensitivities.items()},
            "allocation": {r: float(e) for r, e in allocation.items()},
            "constraint_target": constraint_target,
            "constraint_check": sum(
                allocation.get(r, 0) * (dw.get(r, 1.0))
                for r in root_nodes
            ),
            "mode": mode,
        })

    return all_values, epsilon_blocks, all_allocation_info


def sanitize_vanilla_roots_only(samples, nodes, edges, expressions,
                                epsilon, num_candidates=1000):
    """
    Vanilla Preempt Roots-Only: noise only root nodes with uniform budget,
    then post-process all derived nodes from the noised roots.

    Same total budget B = n_nodes * epsilon as VP and P++, but distributed
    uniformly across the k root nodes (each gets B/k). Derived nodes are
    computed via post-processing (free by the post-processing theorem).

    This isolates the effect of post-processing from optimal allocation:
      VP:             all nodes noised, uniform epsilon
      VP Roots-Only:  roots-only noised, uniform B/k
      P++:            roots-only noised, optimal allocation

    Args:
        samples: List of sample dictionaries with conversation turns
        nodes: Dictionary mapping node names to their parent dependencies
        edges: List of (source, target) dependency edges
        expressions: Dict mapping node names to computation expressions
        epsilon: Base privacy budget per node
        num_candidates: Number of discrete grid points (m)

    Returns:
        all_values: List of sanitized value dictionaries
        epsilon_blocks: List of actual epsilon budgets used per node
    """
    san = sanitizer.Sanitizer()
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

        # Total budget matches VP and P++
        n_nodes = len([n for n in topo_order if n not in skip_nodes])
        total_budget = n_nodes * epsilon

        # Uniform allocation across roots
        k = len(root_nodes)
        root_eps = total_budget / k if k > 0 else epsilon

        # Noise roots with uniform budget
        for root in root_nodes:
            domain_lower, domain_upper = sanitizer.MEDICAL_DOMAINS.get(
                root, sanitizer.MEDICAL_DOMAINS["_default"]
            )
            noised = san.M_exponential_discrete(
                ground_truth[root], root_eps, domain_lower, domain_upper,
                num_candidates=num_candidates
            )
            sanitized_values[root] = round(float(noised), ndigits=4)
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
                    f"Derived node '{node}' has no expression and is not a skip node"
                )

        all_values.append(sanitized_values)
        epsilon_blocks.append([actual_budgets.get(node, 0) for node in topo_order])

    return all_values, epsilon_blocks
