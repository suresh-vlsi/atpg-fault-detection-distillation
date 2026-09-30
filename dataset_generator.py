from itertools import product
import csv
import os

from fault import generate_stuck_at_faults
from simulator import simulate


# ============================================================
# Generate all primary-input combinations
# ============================================================

def generate_input_vectors(circuit):

    input_names = circuit.inputs

    vectors = []

    for bits in product([0, 1], repeat=len(input_names)):

        vector = {
            name: bit
            for name, bit in zip(input_names, bits)
        }

        vectors.append(vector)

    return vectors


# ============================================================
# Generate fault-detection dataset
# ============================================================

def generate_fault_dataset(circuit):

    input_vectors = generate_input_vectors(circuit)

    faults = generate_stuck_at_faults(circuit)

    dataset = []

    for vector in input_vectors:

        # ----------------------------------------------------
        # Simulate fault-free circuit
        # ----------------------------------------------------

        good_outputs, _ = simulate(
            circuit,
            vector
        )

        for fault in faults:

            # ------------------------------------------------
            # Simulate faulty circuit
            # ------------------------------------------------

            faulty_outputs, _ = simulate(
                circuit,
                vector,
                fault
            )

            # ------------------------------------------------
            # Determine whether fault is detected
            # ------------------------------------------------

            detected = int(
                good_outputs != faulty_outputs
            )

            # ------------------------------------------------
            # Extract fault information
            # ------------------------------------------------

            fault_net = fault.net
            stuck_value = fault.stuck_value

            # ------------------------------------------------
            # Extract output value
            #
            # Current circuit has one output: N2
            # ------------------------------------------------

            good_output = list(
                good_outputs.values()
            )[0]

            faulty_output = list(
                faulty_outputs.values()
            )[0]

            # ------------------------------------------------
            # Create ML sample
            # ------------------------------------------------

            sample = {}

            # Primary inputs
            for name in circuit.inputs:
                sample[name] = vector[name]

            # Fault information
            sample["fault_net"] = fault_net
            sample["stuck_value"] = stuck_value

            # Circuit response
            sample["good_output"] = good_output
            sample["faulty_output"] = faulty_output

            # Target label
            sample["detected"] = detected

            dataset.append(sample)

    return dataset


# ============================================================
# Save dataset to CSV
# ============================================================

def save_dataset_csv(dataset, filename):

    # Create directory if necessary
    directory = os.path.dirname(filename)

    if directory:
        os.makedirs(directory, exist_ok=True)

    # Get column names
    fieldnames = list(dataset[0].keys())

    with open(
        filename,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(dataset)


# ============================================================
# Print dataset statistics
# ============================================================

def print_dataset_statistics(dataset):

    total = len(dataset)

    detected = sum(
        sample["detected"]
        for sample in dataset
    )

    undetected = total - detected

    print()
    print("=" * 70)
    print("ATPG FAULT-DETECTION DATASET")
    print("=" * 70)

    print(f"Total samples      : {total}")
    print(f"Detected samples   : {detected}")
    print(f"Undetected samples : {undetected}")

    if total > 0:

        coverage = (
            detected / total
        ) * 100

        print(
            f"Detection ratio    : {coverage:.2f}%"
        )


# ============================================================
# Display first few samples
# ============================================================

def print_sample_data(dataset):

    print()
    print("=" * 70)
    print("SAMPLE DATA")
    print("=" * 70)

    for sample in dataset[:10]:

        print(
            f"Inputs=("
            f"A={sample['A']}, "
            f"B={sample['B']}, "
            f"C={sample['C']}"
            f") | "
            f"Fault={sample['fault_net']}/"
            f"SA{sample['stuck_value']} | "
            f"Good={sample['good_output']} | "
            f"Faulty={sample['faulty_output']} | "
            f"Detected={sample['detected']}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    from parser import parse_netlist

    # --------------------------------------------------------
    # Read circuit
    # --------------------------------------------------------

    circuit = parse_netlist(
        "examples/simple.bench"
    )

    print()
    print("Generating fault-detection dataset...")

    # --------------------------------------------------------
    # Generate dataset
    # --------------------------------------------------------

    dataset = generate_fault_dataset(
        circuit
    )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print_dataset_statistics(
        dataset
    )

    # --------------------------------------------------------
    # Display samples
    # --------------------------------------------------------

    print_sample_data(
        dataset
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_file = (
        "dataset/"
        "fault_detection_dataset.csv"
    )

    save_dataset_csv(
        dataset,
        output_file
    )

    print()
    print("=" * 70)
    print("DATASET SAVED")
    print("=" * 70)

    print(
        f"File : {output_file}"
    )

    print()
    print("Dataset generation complete.")