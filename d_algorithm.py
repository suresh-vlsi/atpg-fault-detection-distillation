from itertools import product


# ============================================================
# FIVE-VALUED LOGIC
# ============================================================

VALUES = ["0", "1", "X", "D", "D'"]


def good_fault_pair(value):
    """
    Convert five-valued logic into:

        (good circuit value, faulty circuit value)

    0  -> (0, 0)
    1  -> (1, 1)
    X  -> (None, None)
    D  -> (1, 0)
    D' -> (0, 1)
    """

    table = {
        "0": (0, 0),
        "1": (1, 1),
        "X": (None, None),
        "D": (1, 0),
        "D'": (0, 1),
    }

    return table[value]


def pair_to_value(good, faulty):
    """
    Convert good/faulty binary values back to
    five-valued notation.
    """

    if good is None or faulty is None:
        return "X"

    if good == 0 and faulty == 0:
        return "0"

    if good == 1 and faulty == 1:
        return "1"

    if good == 1 and faulty == 0:
        return "D"

    if good == 0 and faulty == 1:
        return "D'"

    return "X"


# ============================================================
# BOOLEAN GATE EVALUATION
# ============================================================

def boolean_gate(gate_type, inputs):
    """
    Evaluate a gate using ordinary binary logic.

    Returns:
        0 or 1
    """

    gate_type = gate_type.lower()

    if gate_type == "and":
        return int(all(inputs))

    elif gate_type == "or":
        return int(any(inputs))

    elif gate_type == "nand":
        return int(not all(inputs))

    elif gate_type == "nor":
        return int(not any(inputs))

    elif gate_type == "xor":
        result = 0

        for value in inputs:
            result ^= value

        return result

    elif gate_type == "xnor":
        result = 0

        for value in inputs:
            result ^= value

        return int(not result)

    elif gate_type == "not":
        return int(not inputs[0])

    elif gate_type == "buf":
        return inputs[0]

    else:
        raise ValueError(
            f"Unsupported gate type: {gate_type}"
        )


# ============================================================
# FIVE-VALUED GATE EVALUATION
# ============================================================

def evaluate_gate_5value(gate_type, values):
    """
    Evaluate a gate using five-valued D-algorithm logic.

    D  = good 1, faulty 0
    D' = good 0, faulty 1
    X  = unknown
    """

    pairs = [
        good_fault_pair(value)
        for value in values
    ]

    good_inputs = [
        pair[0]
        for pair in pairs
    ]

    faulty_inputs = [
        pair[1]
        for pair in pairs
    ]

    # --------------------------------------------------------
    # Evaluate good circuit
    # --------------------------------------------------------

    if any(value is None for value in good_inputs):

        good_result = None

    else:

        good_result = boolean_gate(
            gate_type,
            good_inputs
        )

    # --------------------------------------------------------
    # Evaluate faulty circuit
    # --------------------------------------------------------

    if any(value is None for value in faulty_inputs):

        faulty_result = None

    else:

        faulty_result = boolean_gate(
            gate_type,
            faulty_inputs
        )

    # --------------------------------------------------------
    # Convert back to D notation
    # --------------------------------------------------------

    return pair_to_value(
        good_result,
        faulty_result
    )


# ============================================================
# D-ALGORITHM
# ============================================================

class DAlgorithm:

    def __init__(self, circuit, fault):

        self.circuit = circuit
        self.fault = fault

        # ----------------------------------------------------
        # Five-valued signal state
        # ----------------------------------------------------

        self.values = {}

        # Primary inputs
        for signal in circuit.inputs:

            self.values[signal] = "X"

        # Gate outputs
        #
        # IMPORTANT:
        # Do NOT use circuit.wires.
        #
        # The current Circuit class does not have a
        # 'wires' attribute. Internal signals can be
        # obtained directly from gate outputs.
        #
        for gate in circuit.gates:

            self.values[gate.output] = "X"

        # Primary outputs
        for signal in circuit.outputs:

            self.values[signal] = "X"

        # ----------------------------------------------------
        # Fault information
        # ----------------------------------------------------
        self.fault_signal = fault.net
        self.fault_value = fault.stuck_value

        # ----------------------------------------------------
        # Search control
        # ----------------------------------------------------

        self.depth = 0

        self.max_depth = 100

    # ========================================================
    # STATE MANAGEMENT
    # ========================================================

    def copy_state(self):

        return dict(self.values)


    def restore_state(self, state):

        self.values = dict(state)


    # ========================================================
    # FAULT ACTIVATION
    # ========================================================

    def activate_fault(self):
        """
        Activate the stuck-at fault.

        SA0:
            Good circuit must have 1
            Faulty circuit has 0

            Therefore:
                D = (1,0)

        SA1:
            Good circuit must have 0
            Faulty circuit has 1

            Therefore:
                D' = (0,1)
        """

        if self.fault_value == 0:

            required = "D"

        else:

            required = "D'"

        current = self.values.get(
            self.fault_signal,
            "X"
        )

        # Unknown fault site
        if current == "X":

            self.values[
                self.fault_signal
            ] = required

            return True

        # Already correctly activated
        if current == required:

            return True

        return False


    # ========================================================
    # FIND GATES DRIVING A SIGNAL
    # ========================================================

    def gates_driving(self, signal):

        result = []

        for gate in self.circuit.gates:

            if gate.output == signal:

                result.append(gate)

        return result


    # ========================================================
    # FIND GATES USING A SIGNAL
    # ========================================================

    def gates_using(self, signal):

        result = []

        for gate in self.circuit.gates:

            if signal in gate.inputs:

                result.append(gate)

        return result


    # ========================================================
    # D-FRONTIER
    # ========================================================

    def find_d_frontier(self):
        """
        D-frontier:

        A gate whose output is X and whose input contains
        D or D'.
        """

        frontier = []

        for gate in self.circuit.gates:

            output_value = self.values.get(
                gate.output,
                "X"
            )

            # Output must still be unknown
            if output_value != "X":

                continue

            input_values = []

            for signal in gate.inputs:

                input_values.append(
                    self.values.get(
                        signal,
                        "X"
                    )
                )

            # At least one D or D'
            has_d = any(
                value in ("D", "D'")
                for value in input_values
            )

            if has_d:

                frontier.append(gate)

        return frontier


    # ========================================================
    # J-FRONTIER
    # ========================================================

    def find_j_frontier(self):
        """
        J-frontier:

        A gate whose output is known but one or more
        of its inputs are still unknown.
        """

        frontier = []

        for gate in self.circuit.gates:

            output_value = self.values.get(
                gate.output,
                "X"
            )

            # Output must be known
            if output_value == "X":

                continue

            input_values = []

            for signal in gate.inputs:

                input_values.append(
                    self.values.get(
                        signal,
                        "X"
                    )
                )

            # At least one unknown input
            if any(
                value == "X"
                for value in input_values
            ):

                frontier.append(gate)

        return frontier


    # ========================================================
    # D PROPAGATION
    # ========================================================

    def propagate_gate(self, gate):
        """
        Try to propagate D or D' through a gate.

        Unknown inputs are temporarily assigned 0/1
        combinations.

        A successful assignment must produce D or D'.
        """

        current_inputs = []

        for signal in gate.inputs:

            current_inputs.append(
                self.values.get(
                    signal,
                    "X"
                )
            )

        # ----------------------------------------------------
        # Find unknown inputs
        # ----------------------------------------------------

        unknown_positions = []

        for index, value in enumerate(current_inputs):

            if value == "X":

                unknown_positions.append(index)

        # ----------------------------------------------------
        # No unknown inputs
        # ----------------------------------------------------

        if not unknown_positions:

            output = evaluate_gate_5value(
                gate.gate_type,
                current_inputs
            )

            if output in ("D", "D'"):

                self.values[
                    gate.output
                ] = output

                return True

            return False

        # ----------------------------------------------------
        # Try all assignments
        # ----------------------------------------------------

        for replacement in product(
            ["0", "1"],
            repeat=len(unknown_positions)
        ):

            trial = list(current_inputs)

            for index, value in zip(
                unknown_positions,
                replacement
            ):

                trial[index] = value

            output = evaluate_gate_5value(
                gate.gate_type,
                trial
            )

            if output in ("D", "D'"):

                # Apply selected assignments
                for index, value in zip(
                    unknown_positions,
                    replacement
                ):

                    self.values[
                        gate.inputs[index]
                    ] = value

                # Apply propagated D
                self.values[
                    gate.output
                ] = output

                return True

        return False


    # ========================================================
    # JUSTIFICATION
    # ========================================================

    def justify_signal(self, signal, desired):
        """
        Justify an internal signal.

        Example:

            N1 = AND(A,B)

        To justify:

            N1 = 1

        the algorithm may choose:

            A = 1
            B = 1
        """

        # ----------------------------------------------------
        # Primary input
        # ----------------------------------------------------

        if signal in self.circuit.inputs:

            current = self.values.get(
                signal,
                "X"
            )

            if current == "X":

                self.values[
                    signal
                ] = desired

                return True

            return current == desired

        # ----------------------------------------------------
        # Find gate driving the signal
        # ----------------------------------------------------

        drivers = self.gates_driving(signal)

        if not drivers:

            return False

        gate = drivers[0]

        input_count = len(gate.inputs)

        # ----------------------------------------------------
        # Try binary input combinations
        # ----------------------------------------------------

        for combination in product(
            ["0", "1"],
            repeat=input_count
        ):

            output = evaluate_gate_5value(
                gate.gate_type,
                list(combination)
            )

            if output != desired:

                continue

            # ------------------------------------------------
            # Check consistency with current state
            # ------------------------------------------------

            valid = True

            for input_signal, value in zip(
                gate.inputs,
                combination
            ):

                current = self.values.get(
                    input_signal,
                    "X"
                )

                if current != "X" and current != value:

                    valid = False

                    break

            if not valid:

                continue

            # ------------------------------------------------
            # Apply assignment
            # ------------------------------------------------

            for input_signal, value in zip(
                gate.inputs,
                combination
            ):

                self.values[
                    input_signal
                ] = value

            self.values[
                signal
            ] = desired

            return True

        return False


    # ========================================================
    # CHECK D AT PRIMARY OUTPUT
    # ========================================================

    def has_d_at_output(self):

        for output in self.circuit.outputs:

            value = self.values.get(
                output,
                "X"
            )

            if value in ("D", "D'"):

                return True

        return False


    # ========================================================
    # EXTRACT PRIMARY INPUT VECTOR
    # ========================================================

    def get_test_vector(self):
        """
        Convert the five-valued primary-input assignment
        into an ordinary binary test vector.
        """

        vector = {}

        for signal in self.circuit.inputs:

            value = self.values.get(
                signal,
                "X"
            )

            # D means good circuit = 1
            if value == "D":

                vector[signal] = 1

            # D' means good circuit = 0
            elif value == "D'":

                vector[signal] = 0

            elif value == "0":

                vector[signal] = 0

            elif value == "1":

                vector[signal] = 1

            else:

                # Test vector is incomplete
                return None

        return vector


    # ========================================================
    # PRINT CURRENT STATE
    # ========================================================

    def print_state(self):

        print(
            "\nCurrent D-algorithm state:"
        )

        for signal, value in self.values.items():

            print(
                f"  {signal:<8} = {value}"
            )


    # ========================================================
    # D-ALGORITHM SEARCH
    # ========================================================

    def search(self):

        self.depth += 1

        # ----------------------------------------------------
        # Prevent infinite recursion
        # ----------------------------------------------------

        if self.depth > self.max_depth:

            self.depth -= 1

            return None

        # ----------------------------------------------------
        # Activate fault
        # ----------------------------------------------------

        required = (
            "D"
            if self.fault_value == 0
            else "D'"
        )

        current = self.values.get(
            self.fault_signal,
            "X"
        )

        if current == "X":

            if not self.justify_signal(
                self.fault_signal,
                required
            ):

                self.depth -= 1

                return None

        elif current != required:

            self.depth -= 1

            return None

        # ----------------------------------------------------
        # Check whether D has reached an output
        # ----------------------------------------------------

        if self.has_d_at_output():

            vector = self.get_test_vector()

            if vector is not None:

                self.depth -= 1

                return vector

        # ----------------------------------------------------
        # Find D-frontier
        # ----------------------------------------------------

        d_frontier = self.find_d_frontier()

        # ----------------------------------------------------
        # Propagate D
        # ----------------------------------------------------

        for gate in d_frontier:

            old_state = self.copy_state()

            if self.propagate_gate(gate):

                result = self.search()

                if result is not None:

                    self.depth -= 1

                    return result

            self.restore_state(old_state)

        # ----------------------------------------------------
        # Find J-frontier
        # ----------------------------------------------------

        j_frontier = self.find_j_frontier()

        # ----------------------------------------------------
        # Justify unknown inputs
        # ----------------------------------------------------

        for gate in j_frontier:

            for signal in gate.inputs:

                if self.values.get(
                    signal,
                    "X"
                ) != "X":

                    continue

                # --------------------------------------------
                # Try 0
                # --------------------------------------------

                old_state = self.copy_state()

                if self.justify_signal(
                    signal,
                    "0"
                ):

                    result = self.search()

                    if result is not None:

                        self.depth -= 1

                        return result

                self.restore_state(old_state)

                # --------------------------------------------
                # Try 1
                # --------------------------------------------

                if self.justify_signal(
                    signal,
                    "1"
                ):

                    result = self.search()

                    if result is not None:

                        self.depth -= 1

                        return result

                self.restore_state(old_state)

        # ----------------------------------------------------
        # No solution at this branch
        # ----------------------------------------------------

        self.depth -= 1

        return None


# ============================================================
# PUBLIC API
# ============================================================

def d_algorithm_atpg(circuit, fault):
    """
    Run D-algorithm ATPG for one stuck-at fault.

    Parameters
    ----------
    circuit:
        Parsed Circuit object.

    fault:
        StuckAtFault object.

    Returns
    -------
    dict or None:
        Primary-input test vector if found,
        otherwise None.
    """

    engine = DAlgorithm(
        circuit,
        fault
    )

    return engine.search()