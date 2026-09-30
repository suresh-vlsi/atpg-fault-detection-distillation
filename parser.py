import re

from circuit import Circuit, Gate


# ============================================================
# PARSE BENCH / NETLIST FILE
# ============================================================

def parse_bench(filename):
    """
    Parse an ISCAS-style .bench netlist.

    Supported examples:

        INPUT(A)
        INPUT(B)
        INPUT(C)

        OUTPUT(N2)

        N1 = AND(A, B)
        N2 = OR(N1, C)

    Returns:
        Circuit object
    """

    circuit = Circuit()

    with open(filename, "r") as f:

        for raw_line in f:

            line = raw_line.strip()

            # ------------------------------------------------
            # Ignore empty lines
            # ------------------------------------------------
            if not line:
                continue

            # ------------------------------------------------
            # Ignore comments
            # ------------------------------------------------
            if line.startswith("#"):
                continue

            # Remove inline comments
            if "#" in line:
                line = line.split("#", 1)[0].strip()

            if not line:
                continue

            # ------------------------------------------------
            # INPUT(...)
            # ------------------------------------------------
            input_match = re.match(
                r"^\s*INPUT\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$",
                line,
                re.IGNORECASE
            )

            if input_match:

                name = input_match.group(1)

                circuit.add_input(name)

                continue

            # ------------------------------------------------
            # OUTPUT(...)
            # ------------------------------------------------
            output_match = re.match(
                r"^\s*OUTPUT\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$",
                line,
                re.IGNORECASE
            )

            if output_match:

                name = output_match.group(1)

                circuit.add_output(name)

                continue

            # ------------------------------------------------
            # Gate assignment
            #
            # Example:
            #
            # N1 = AND(A, B)
            # N2 = OR(N1, C)
            # ------------------------------------------------
            gate_match = re.match(
                r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
                r"([A-Za-z]+)\s*\((.*?)\)\s*$",
                line
            )

            if gate_match:

                output = gate_match.group(1)
                gate_type = gate_match.group(2).upper()
                input_string = gate_match.group(3)

                # ------------------------------------------------
                # Parse gate inputs
                # ------------------------------------------------
                inputs = [
                    x.strip()
                    for x in input_string.split(",")
                    if x.strip()
                ]

                if not inputs:
                    raise ValueError(
                        f"Gate {output} has no inputs"
                    )

                # ------------------------------------------------
                # Validate supported gate
                # ------------------------------------------------
                supported_gates = {
                    "AND",
                    "OR",
                    "NOT",
                    "NAND",
                    "NOR",
                    "XOR",
                    "XNOR",
                    "BUF"
                }

                if gate_type not in supported_gates:

                    raise ValueError(
                        f"Unsupported gate type '{gate_type}' "
                        f"in line: {line}"
                    )

                # ------------------------------------------------
                # Basic input-count validation
                # ------------------------------------------------
                if gate_type == "NOT" and len(inputs) != 1:

                    raise ValueError(
                        f"NOT gate '{output}' must have exactly "
                        f"one input"
                    )

                if gate_type == "BUF" and len(inputs) != 1:

                    raise ValueError(
                        f"BUF gate '{output}' must have exactly "
                        f"one input"
                    )

                if gate_type in {"XOR", "XNOR"} and len(inputs) != 2:

                    raise ValueError(
                        f"{gate_type} gate '{output}' must have "
                        f"exactly two inputs"
                    )

                # ------------------------------------------------
                # Create gate
                # ------------------------------------------------
                gate = Gate(
                    gate_type=gate_type,
                    inputs=inputs,
                    output=output
                )

                circuit.add_gate(gate)

                continue

            # ------------------------------------------------
            # Unknown line
            # ------------------------------------------------
            raise ValueError(
                f"Cannot parse line:\n{line}"
            )

    return circuit


# ============================================================
# ALIAS
# ============================================================

def parse_netlist(filename):
    """
    Compatibility wrapper.

    main.py currently imports parse_netlist(), while the
    actual parser implementation is parse_bench().
    """

    return parse_bench(filename)


# ============================================================
# CIRCUIT INFORMATION
# ============================================================

def print_circuit(circuit):
    """
    Print parsed circuit information.
    """

    print("\n" + "=" * 60)
    print("CIRCUIT INFORMATION")
    print("=" * 60)

    print(f"Primary inputs  : {circuit.inputs}")
    print(f"Primary outputs : {circuit.outputs}")

    print("\nGates:")
    print("-" * 60)

    for i, gate in enumerate(circuit.gates, start=1):

        input_string = ", ".join(gate.inputs)

        print(
            f"{i:3d}. "
            f"{gate.output} = "
            f"{gate.gate_type}({input_string})"
        )

    print("=" * 60)


# ============================================================
# OPTIONAL TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "    python parser.py <bench_file>"
        )

        sys.exit(1)

    filename = sys.argv[1]

    circuit = parse_netlist(filename)

    print_circuit(circuit)