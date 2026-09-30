"""
fault_collapsing.py
===================

Fault collapsing utilities for stuck-at ATPG.

This module provides:

    1. Exhaustive fault equivalence detection
    2. Fault equivalence classes
    3. Representative fault selection
    4. Fault list reduction
    5. Fault collapsing statistics

For small combinational circuits, exhaustive simulation gives a
safe way to determine whether two stuck-at faults have identical
observable behavior.

A fault is considered equivalent to another fault when every
possible primary-input vector produces the same primary outputs
for both faults.

Example:

    A/SA0
    N1/SA0

may or may not be equivalent depending on the circuit.

The module deliberately does NOT assume arbitrary dominance
relationships. Such relationships depend on gate type and
fault-model conventions.
"""

from itertools import product

from fault import (
    StuckAtFault,
    generate_stuck_at_faults
)

from simulator import simulate


# ============================================================
# PRIMARY INPUT VECTORS
# ============================================================

def generate_all_input_vectors(circuit):
    """
    Generate every possible binary primary-input vector.

    Args:
        circuit: Circuit object.

    Returns:
        list of dictionaries.

    Example:

        {
            "A": 0,
            "B": 1,
            "C": 0
        }
    """

    inputs = circuit.inputs

    vectors = []

    for bits in product(
        [0, 1],
        repeat=len(inputs)
    ):

        vector = dict(
            zip(
                inputs,
                bits
            )
        )

        vectors.append(vector)

    return vectors


# ============================================================
# FAULT OUTPUT SIGNATURE
# ============================================================

def fault_output_signature(
    circuit,
    fault,
    input_vectors=None
):
    """
    Calculate the observable output signature of a fault.

    For every possible primary-input vector, the primary-output
    values of the faulty circuit are recorded.

    Args:
        circuit: Circuit object.
        fault: StuckAtFault object.
        input_vectors: Optional precomputed vectors.

    Returns:
        tuple containing the complete output signature.
    """

    if input_vectors is None:

        input_vectors = (
            generate_all_input_vectors(
                circuit
            )
        )

    signature = []

    for vector in input_vectors:

        outputs, _ = simulate(
            circuit,
            vector,
            fault
        )

        output_tuple = tuple(
            outputs[name]
            for name in circuit.outputs
        )

        signature.append(
            output_tuple
        )

    return tuple(signature)


# ============================================================
# FAULT EQUIVALENCE
# ============================================================

def are_faults_equivalent(
    circuit,
    fault1,
    fault2,
    input_vectors=None
):
    """
    Determine whether two faults are functionally equivalent.

    Two faults are equivalent if their faulty circuits produce
    identical primary outputs for every possible primary-input
    combination.

    Args:
        circuit: Circuit object.
        fault1: First StuckAtFault.
        fault2: Second StuckAtFault.
        input_vectors: Optional precomputed vectors.

    Returns:
        True or False.
    """

    if not isinstance(
        fault1,
        StuckAtFault
    ):
        raise TypeError(
            "fault1 must be a StuckAtFault"
        )

    if not isinstance(
        fault2,
        StuckAtFault
    ):
        raise TypeError(
            "fault2 must be a StuckAtFault"
        )

    signature1 = fault_output_signature(
        circuit,
        fault1,
        input_vectors
    )

    signature2 = fault_output_signature(
        circuit,
        fault2,
        input_vectors
    )

    return signature1 == signature2


# ============================================================
# BUILD EQUIVALENCE CLASSES
# ============================================================

def find_equivalence_classes(
    circuit,
    faults=None
):
    """
    Group faults into functional equivalence classes.

    Args:
        circuit: Circuit object.
        faults: Optional list of faults.

    Returns:
        list of lists.

    Example:

        [
            [A/SA0, N1/SA0],
            [A/SA1],
            [B/SA0],
            ...
        ]
    """

    if faults is None:

        faults = generate_stuck_at_faults(
            circuit
        )

    input_vectors = (
        generate_all_input_vectors(
            circuit
        )
    )

    # --------------------------------------------------------
    # Calculate signatures once
    # --------------------------------------------------------

    signatures = {}

    for fault in faults:

        signatures[fault] = (
            fault_output_signature(
                circuit,
                fault,
                input_vectors
            )
        )

    # --------------------------------------------------------
    # Group identical signatures
    # --------------------------------------------------------

    groups = {}

    for fault, signature in signatures.items():

        if signature not in groups:

            groups[signature] = []

        groups[signature].append(
            fault
        )

    return list(
        groups.values()
    )


# ============================================================
# SELECT REPRESENTATIVE
# ============================================================

def select_representative(
    fault_class
):
    """
    Select one representative from an equivalence class.

    The first fault is used to preserve deterministic ordering.

    Args:
        fault_class: List of equivalent faults.

    Returns:
        StuckAtFault.
    """

    if not fault_class:

        raise ValueError(
            "Cannot select representative "
            "from an empty fault class"
        )

    return fault_class[0]


# ============================================================
# COLLAPSE FAULT LIST
# ============================================================

def collapse_faults(
    circuit,
    faults=None
):
    """
    Collapse functionally equivalent faults.

    Args:
        circuit: Circuit object.
        faults: Optional fault list.

    Returns:
        collapsed_faults:
            List containing one representative per
            equivalence class.

        equivalence_classes:
            Complete equivalence classes.
    """

    if faults is None:

        faults = generate_stuck_at_faults(
            circuit
        )

    classes = find_equivalence_classes(
        circuit,
        faults
    )

    collapsed_faults = []

    for fault_class in classes:

        representative = (
            select_representative(
                fault_class
            )
        )

        collapsed_faults.append(
            representative
        )

    return (
        collapsed_faults,
        classes
    )


# ============================================================
# COLLAPSING STATISTICS
# ============================================================

def collapsing_statistics(
    original_faults,
    collapsed_faults
):
    """
    Calculate fault-collapsing statistics.

    Returns:

        {
            "original_faults": ...,
            "collapsed_faults": ...,
            "removed_faults": ...,
            "reduction_percent": ...
        }
    """

    original_count = len(
        original_faults
    )

    collapsed_count = len(
        collapsed_faults
    )

    removed_count = (
        original_count -
        collapsed_count
    )

    if original_count == 0:

        reduction = 0.0

    else:

        reduction = (
            removed_count /
            original_count
        ) * 100.0

    return {
        "original_faults": original_count,
        "collapsed_faults": collapsed_count,
        "removed_faults": removed_count,
        "reduction_percent": reduction
    }


# ============================================================
# PRINT EQUIVALENCE CLASSES
# ============================================================

def print_equivalence_classes(
    equivalence_classes
):
    """
    Print fault equivalence classes.
    """

    print()
    print("=" * 70)
    print("FAULT EQUIVALENCE CLASSES")
    print("=" * 70)

    for index, fault_class in enumerate(
        equivalence_classes,
        start=1
    ):

        representative = (
            select_representative(
                fault_class
            )
        )

        print(
            f"Class {index}:"
        )

        print(
            f"  Representative : "
            f"{representative}"
        )

        print(
            "  Faults         : "
            + ", ".join(
                str(fault)
                for fault in fault_class
            )
        )

        print()


# ============================================================
# PRINT COLLAPSING SUMMARY
# ============================================================

def print_collapsing_summary(
    original_faults,
    collapsed_faults
):
    """
    Print fault collapsing statistics.
    """

    stats = collapsing_statistics(
        original_faults,
        collapsed_faults
    )

    print()
    print("=" * 70)
    print("FAULT COLLAPSING SUMMARY")
    print("=" * 70)

    print(
        f"Original faults   : "
        f"{stats['original_faults']}"
    )

    print(
        f"Collapsed faults  : "
        f"{stats['collapsed_faults']}"
    )

    print(
        f"Removed faults    : "
        f"{stats['removed_faults']}"
    )

    print(
        f"Reduction         : "
        f"{stats['reduction_percent']:.2f}%"
    )

    print("=" * 70)


# ============================================================
# COMPLETE FAULT COLLAPSING FLOW
# ============================================================

def run_fault_collapsing(circuit):
    """
    Execute the complete fault-collapsing flow.

    Steps:

        1. Generate all stuck-at faults
        2. Find equivalence classes
        3. Select representatives
        4. Produce collapsed fault list
        5. Print statistics

    Returns:

        collapsed_faults,
        equivalence_classes
    """

    print()
    print("=" * 70)
    print("FAULT COLLAPSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Generate original fault list
    # --------------------------------------------------------

    original_faults = (
        generate_stuck_at_faults(
            circuit
        )
    )

    print(
        f"Original fault count : "
        f"{len(original_faults)}"
    )

    # --------------------------------------------------------
    # Collapse
    # --------------------------------------------------------

    (
        collapsed_faults,
        equivalence_classes
    ) = collapse_faults(
        circuit,
        original_faults
    )

    # --------------------------------------------------------
    # Print equivalence classes
    # --------------------------------------------------------

    print_equivalence_classes(
        equivalence_classes
    )

    # --------------------------------------------------------
    # Print collapsed list
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COLLAPSED FAULT LIST")
    print("=" * 70)

    for index, fault in enumerate(
        collapsed_faults,
        start=1
    ):

        print(
            f"{index:3d}. {fault}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_collapsing_summary(
        original_faults,
        collapsed_faults
    )

    return (
        collapsed_faults,
        equivalence_classes
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "fault_collapsing.py loaded successfully."
    )

    print(
        "Use run_fault_collapsing(circuit) "
        "from main.py."
    )