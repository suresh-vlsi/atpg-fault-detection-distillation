from parser import parse_netlist
from fault import generate_stuck_at_faults
from simulator import simulate
from d_algorithm import d_algorithm_atpg
from fault_collapsing import (
    collapse_faults,
    print_equivalence_classes,
    print_collapsing_summary
)


# ============================================================
# VERIFY A TEST VECTOR
# ============================================================

def verify_fault(circuit, fault, test_vector):
    """
    Verify whether a test vector detects a fault.
    """

    if test_vector is None:
        return False

    # Good circuit
    good_outputs, _ = simulate(
        circuit,
        test_vector
    )

    # Faulty circuit
    faulty_outputs, _ = simulate(
        circuit,
        test_vector,
        fault=fault
    )

    # Compare primary outputs
    for output in circuit.outputs:

        if good_outputs[output] != faulty_outputs[output]:
            return True

    return False


# ============================================================
# RUN ATPG FOR A FAULT LIST
# ============================================================

def run_atpg(circuit, faults):
    """
    Run D-algorithm ATPG on the supplied fault list.
    """

    results = []

    for fault in faults:

        print()
        print("-" * 60)
        print(f"Fault : {fault}")
        print("-" * 60)

        # ----------------------------------------------------
        # Generate test vector
        # ----------------------------------------------------

        print("Running D-algorithm...")

        try:

            vector = d_algorithm_atpg(
                circuit,
                fault
            )

        except Exception as error:

            print(
                f"D-algorithm error: {error}"
            )

            vector = None

        # ----------------------------------------------------
        # No vector
        # ----------------------------------------------------

        if vector is None:

            print(
                "Test vector : NONE"
            )

            print(
                "Status      : UNTESTABLE"
            )

            results.append({
                "fault": fault,
                "vector": None,
                "detected": False
            })

            continue

        # ----------------------------------------------------
        # Display vector
        # ----------------------------------------------------

        print(
            f"Test vector : {vector}"
        )

        # ----------------------------------------------------
        # Verify vector
        # ----------------------------------------------------

        print(
            "Verifying test vector..."
        )

        try:

            detected = verify_fault(
                circuit,
                fault,
                vector
            )

        except Exception as error:

            print(
                f"Verification error: {error}"
            )

            detected = False

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if detected:

            print(
                "RESULT      : FAULT DETECTED"
            )

        else:

            print(
                "RESULT      : NOT DETECTED"
            )

        results.append({
            "fault": fault,
            "vector": vector,
            "detected": detected
        })

    return results


# ============================================================
# CALCULATE COVERAGE
# ============================================================

def calculate_coverage(results):

    total = len(results)

    detected = sum(
        1
        for result in results
        if result["detected"]
    )

    if total == 0:
        coverage = 0.0
    else:
        coverage = (
            detected /
            total
        ) * 100.0

    return (
        total,
        detected,
        total - detected,
        coverage
    )


# ============================================================
# GET UNIQUE TEST VECTORS
# ============================================================

def get_unique_vectors(results):

    vectors = []

    for result in results:

        vector = result["vector"]

        if vector is None:
            continue

        if vector not in vectors:
            vectors.append(vector)

    return vectors


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(title, results):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for result in results:

        fault = result["fault"]
        vector = result["vector"]
        detected = result["detected"]

        if vector is None:

            status = "UNTESTABLE"

        elif detected:

            status = "DETECTED"

        else:

            status = "NOT DETECTED"

        print(
            f"{str(fault):12} | "
            f"{status:12} | "
            f"{vector}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # STEP 1 — READ CIRCUIT
    # ========================================================

    print()
    print("=" * 70)
    print("STUCK-AT FAULT ATPG")
    print("=" * 70)

    circuit = parse_netlist(
        "examples/simple.bench"
    )

    print()
    print(
        f"Primary inputs : {circuit.inputs}"
    )

    print(
        f"Primary outputs: {circuit.outputs}"
    )

    # ========================================================
    # STEP 2 — GENERATE ORIGINAL FAULT LIST
    # ========================================================

    original_faults = (
        generate_stuck_at_faults(
            circuit
        )
    )

    print()
    print("=" * 70)
    print("ORIGINAL FAULT LIST")
    print("=" * 70)

    for fault in original_faults:
        print(
            f"  {fault}"
        )

    print()
    print(
        f"Total faults : "
        f"{len(original_faults)}"
    )

    # ========================================================
    # STEP 3 — BASELINE ATPG
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE ATPG — BEFORE COLLAPSING")
    print("=" * 70)

    baseline_results = run_atpg(
        circuit,
        original_faults
    )

    print_results(
        "BASELINE RESULTS",
        baseline_results
    )

    (
        baseline_total,
        baseline_detected,
        baseline_undetected,
        baseline_coverage
    ) = calculate_coverage(
        baseline_results
    )

    baseline_vectors = (
        get_unique_vectors(
            baseline_results
        )
    )

    print()
    print("=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    print(
        f"Faults           : "
        f"{baseline_total}"
    )

    print(
        f"Detected         : "
        f"{baseline_detected}"
    )

    print(
        f"Undetected       : "
        f"{baseline_undetected}"
    )

    print(
        f"Coverage         : "
        f"{baseline_coverage:.2f}%"
    )

    print(
        f"Unique vectors   : "
        f"{len(baseline_vectors)}"
    )

    # ========================================================
    # STEP 4 — FAULT COLLAPSING
    # ========================================================

    print()
    print("=" * 70)
    print("FAULT COLLAPSING")
    print("=" * 70)

    (
        collapsed_faults,
        equivalence_classes
    ) = collapse_faults(
        circuit,
        original_faults
    )

    # --------------------------------------------------------
    # Equivalence classes
    # --------------------------------------------------------

    print_equivalence_classes(
        equivalence_classes
    )

    # --------------------------------------------------------
    # Collapsing statistics
    # --------------------------------------------------------

    print_collapsing_summary(
        original_faults,
        collapsed_faults
    )

    # --------------------------------------------------------
    # Collapsed fault list
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COLLAPSED FAULT LIST")
    print("=" * 70)

    for fault in collapsed_faults:

        print(
            f"  {fault}"
        )

    # ========================================================
    # STEP 5 — ATPG AFTER COLLAPSING
    # ========================================================

    print()
    print("=" * 70)
    print("ATPG — AFTER FAULT COLLAPSING")
    print("=" * 70)

    collapsed_results = run_atpg(
        circuit,
        collapsed_faults
    )

    print_results(
        "COLLAPSED ATPG RESULTS",
        collapsed_results
    )

    (
        collapsed_total,
        collapsed_detected,
        collapsed_undetected,
        collapsed_coverage
    ) = calculate_coverage(
        collapsed_results
    )

    collapsed_vectors = (
        get_unique_vectors(
            collapsed_results
        )
    )

    # ========================================================
    # STEP 6 — FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Metric':30}"
        f"{'Before':15}"
        f"{'After':15}"
    )

    print("-" * 60)

    print(
        f"{'Fault count':30}"
        f"{baseline_total:<15}"
        f"{collapsed_total:<15}"
    )

    print(
        f"{'Detected faults':30}"
        f"{baseline_detected:<15}"
        f"{collapsed_detected:<15}"
    )

    print(
        f"{'Undetected faults':30}"
        f"{baseline_undetected:<15}"
        f"{collapsed_undetected:<15}"
    )

    print(
        f"{'Fault coverage':30}"
        f"{baseline_coverage:.2f}%{'':<9}"
        f"{collapsed_coverage:.2f}%"
    )

    print(
        f"{'Unique test vectors':30}"
        f"{len(baseline_vectors):<15}"
        f"{len(collapsed_vectors):<15}"
    )

    # ========================================================
    # STEP 7 — COLLAPSED TEST VECTORS
    # ========================================================

    print()
    print("=" * 70)
    print("TEST VECTORS AFTER COLLAPSING")
    print("=" * 70)

    for index, vector in enumerate(
        collapsed_vectors,
        start=1
    ):

        print(
            f"Vector {index}: {vector}"
        )

    # ========================================================
    # STEP 8 — FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL ATPG REPORT")
    print("=" * 70)

    reduction = (
        baseline_total -
        collapsed_total
    )

    if baseline_total > 0:

        reduction_percent = (
            reduction /
            baseline_total
        ) * 100.0

    else:

        reduction_percent = 0.0

    print()
    print(
        f"Original faults       : "
        f"{baseline_total}"
    )

    print(
        f"Collapsed faults      : "
        f"{collapsed_total}"
    )

    print(
        f"Fault reduction       : "
        f"{reduction}"
    )

    print(
        f"Reduction percentage  : "
        f"{reduction_percent:.2f}%"
    )

    print(
        f"Original coverage     : "
        f"{baseline_coverage:.2f}%"
    )

    print(
        f"Collapsed coverage    : "
        f"{collapsed_coverage:.2f}%"
    )

    print(
        f"Original test vectors : "
        f"{len(baseline_vectors)}"
    )

    print(
        f"Collapsed vectors     : "
        f"{len(collapsed_vectors)}"
    )

    print()
    print("=" * 70)
    print("ATPG FLOW COMPLETE")
    print("=" * 70)
    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()