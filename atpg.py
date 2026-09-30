"""
atpg.py
=======

Automatic Test Pattern Generation (ATPG) utilities.

This module provides:
    1. ATPG for a single stuck-at fault
    2. ATPG for all faults
    3. Fault coverage calculation
    4. Test-vector compaction
    5. Detection of redundant/untestable faults

The actual D-algorithm search is performed by d_algorithm.py.
"""

from d_algorithm import d_algorithm_atpg
from fault import StuckAtFault, generate_stuck_at_faults
from simulator import simulate


# ============================================================
# SINGLE FAULT ATPG
# ============================================================

def generate_test_vector(circuit, fault):
    """
    Generate a test vector for one stuck-at fault.

    Args:
        circuit: Circuit object.
        fault: StuckAtFault object.

    Returns:
        dict: Test vector if fault is testable.
        None: If fault is untestable.
    """

    if not isinstance(fault, StuckAtFault):
        raise TypeError(
            "fault must be a StuckAtFault object"
        )

    try:
        vector = d_algorithm_atpg(
            circuit,
            fault
        )
    except Exception as exc:
        print(
            f"[ATPG ERROR] {fault}: {exc}"
        )
        return None

    return vector


# ============================================================
# VERIFY TEST VECTOR
# ============================================================

def verify_test_vector(circuit, fault, test_vector):
    """
    Verify whether a generated test vector actually detects
    the selected stuck-at fault.

    A fault is detected when at least one primary output differs
    between the good and faulty circuits.

    Args:
        circuit: Circuit object.
        fault: StuckAtFault object.
        test_vector: Dictionary of primary-input values.

    Returns:
        True  -> fault detected
        False -> fault not detected
    """

    if test_vector is None:
        return False

    # --------------------------------------------------------
    # Simulate good circuit
    # --------------------------------------------------------

    good_outputs, good_values = simulate(
        circuit,
        test_vector
    )

    # --------------------------------------------------------
    # Simulate faulty circuit
    # --------------------------------------------------------

    faulty_outputs, faulty_values = simulate(
        circuit,
        test_vector,
        fault
    )

    # --------------------------------------------------------
    # Compare primary outputs
    # --------------------------------------------------------

    for output in circuit.outputs:

        if good_outputs[output] != faulty_outputs[output]:
            return True

    return False


# ============================================================
# ATPG FOR ALL FAULTS
# ============================================================

def generate_all_test_vectors(circuit):
    """
    Generate ATPG vectors for every stuck-at fault.

    Args:
        circuit: Circuit object.

    Returns:
        dict:
            {
                fault: test_vector,
                ...
            }

        Untestable faults are stored with value None.
    """

    faults = generate_stuck_at_faults(circuit)

    results = {}

    for fault in faults:

        vector = generate_test_vector(
            circuit,
            fault
        )

        # ----------------------------------------------------
        # Verify generated vector
        # ----------------------------------------------------

        if vector is not None:

            detected = verify_test_vector(
                circuit,
                fault,
                vector
            )

            if not detected:
                vector = None

        results[fault] = vector

    return results


# ============================================================
# FAULT COVERAGE
# ============================================================

def calculate_fault_coverage(circuit, results):
    """
    Calculate fault coverage.

    Coverage is calculated as:

        Coverage =
            detected faults / total faults * 100

    Args:
        circuit: Circuit object.
        results: Dictionary returned by
                 generate_all_test_vectors().

    Returns:
        float: Fault coverage percentage.
    """

    total_faults = len(results)

    if total_faults == 0:
        return 0.0

    detected_faults = 0

    for fault, vector in results.items():

        if vector is not None:
            detected_faults += 1

    coverage = (
        detected_faults /
        total_faults
    ) * 100.0

    return coverage


# ============================================================
# TEST VECTOR COMPACTION
# ============================================================

def compact_test_vectors(results):
    """
    Remove duplicate test vectors.

    Multiple faults may be detected by the same input vector.

    Args:
        results:
            Dictionary:
                fault -> test vector

    Returns:
        list:
            Unique test vectors.
    """

    unique_vectors = []

    for fault, vector in results.items():

        if vector is None:
            continue

        # ----------------------------------------------------
        # Convert dictionary into canonical representation
        # ----------------------------------------------------

        already_present = False

        for existing in unique_vectors:

            if existing == vector:
                already_present = True
                break

        if not already_present:
            unique_vectors.append(vector)

    return unique_vectors


# ============================================================
# FIND FAULTS DETECTED BY EACH VECTOR
# ============================================================

def faults_detected_by_vector(circuit, vector, faults):
    """
    Determine which faults are detected by a given test vector.

    Args:
        circuit: Circuit object.
        vector: Input vector.
        faults: List of StuckAtFault objects.

    Returns:
        list: Faults detected by the vector.
    """

    detected = []

    for fault in faults:

        if verify_test_vector(
            circuit,
            fault,
            vector
        ):
            detected.append(fault)

    return detected


# ============================================================
# CREATE COMPACT ATPG SET
# ============================================================

def create_compact_atpg_set(circuit, results):
    """
    Create a compact test set while retaining the association
    between vectors and detected faults.

    Returns:

        [
            {
                "vector": {...},
                "faults": [...]
            },
            ...
        ]
    """

    faults = list(results.keys())

    unique_vectors = compact_test_vectors(
        results
    )

    compact_set = []

    for vector in unique_vectors:

        detected = faults_detected_by_vector(
            circuit,
            vector,
            faults
        )

        compact_set.append(
            {
                "vector": vector,
                "faults": detected
            }
        )

    return compact_set


# ============================================================
# PRINT ATPG RESULTS
# ============================================================

def print_atpg_results(circuit, results):
    """
    Print a formatted ATPG result table.
    """

    print()
    print("=" * 70)
    print("ATPG RESULTS")
    print("=" * 70)

    print(
        f"{'FAULT':<15}"
        f"{'STATUS':<15}"
        f"TEST VECTOR"
    )

    print("-" * 70)

    for fault, vector in results.items():

        if vector is None:

            status = "UNTESTABLE"
            vector_text = "None"

        else:

            status = "DETECTED"
            vector_text = str(vector)

        print(
            f"{str(fault):<15}"
            f"{status:<15}"
            f"{vector_text}"
        )

    print("=" * 70)


# ============================================================
# PRINT COVERAGE
# ============================================================

def print_fault_coverage(results):
    """
    Print fault coverage information.
    """

    total_faults = len(results)

    detected_faults = sum(
        1
        for vector in results.values()
        if vector is not None
    )

    undetected_faults = (
        total_faults -
        detected_faults
    )

    if total_faults > 0:

        coverage = (
            detected_faults /
            total_faults
        ) * 100.0

    else:

        coverage = 0.0

    print()
    print("=" * 70)
    print("FAULT COVERAGE")
    print("=" * 70)

    print(
        f"Total faults       : {total_faults}"
    )

    print(
        f"Detected faults    : {detected_faults}"
    )

    print(
        f"Undetected faults  : {undetected_faults}"
    )

    print(
        f"Fault coverage     : {coverage:.2f}%"
    )

    print("=" * 70)


# ============================================================
# COMPLETE ATPG FLOW
# ============================================================

def run_atpg(circuit):
    """
    Execute the complete ATPG flow.

    Steps:

        1. Generate stuck-at faults
        2. Run D-algorithm
        3. Verify every generated vector
        4. Calculate fault coverage
        5. Compact test vectors
        6. Return all results
    """

    print()
    print("=" * 70)
    print("AUTOMATIC TEST PATTERN GENERATION")
    print("=" * 70)

    print(
        f"Primary inputs  : {circuit.inputs}"
    )

    print(
        f"Primary outputs : {circuit.outputs}"
    )

    # --------------------------------------------------------
    # Generate faults
    # --------------------------------------------------------

    faults = generate_stuck_at_faults(
        circuit
    )

    print(
        f"Total faults    : {len(faults)}"
    )

    print()
    print("Running D-algorithm...")
    print()

    # --------------------------------------------------------
    # Generate vectors
    # --------------------------------------------------------

    results = generate_all_test_vectors(
        circuit
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_atpg_results(
        circuit,
        results
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    print_fault_coverage(
        results
    )

    # --------------------------------------------------------
    # Compact vectors
    # --------------------------------------------------------

    compact_set = create_compact_atpg_set(
        circuit,
        results
    )

    print()
    print("=" * 70)
    print("TEST VECTOR COMPACTION")
    print("=" * 70)

    print(
        f"Original fault count : {len(faults)}"
    )

    print(
        f"Unique test vectors  : {len(compact_set)}"
    )

    print()

    for index, entry in enumerate(
        compact_set,
        start=1
    ):

        print(
            f"Vector {index}: "
            f"{entry['vector']}"
        )

        print(
            "  Detects:"
        )

        for fault in entry["faults"]:

            print(
                f"    - {fault}"
            )

    print("=" * 70)

    return results, compact_set


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "atpg.py loaded successfully."
    )

    print(
        "Use run_atpg(circuit) "
        "from main.py to execute ATPG."
    )