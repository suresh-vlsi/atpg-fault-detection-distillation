class Gate:
    def __init__(self, gate_type, inputs, output):
        self.gate_type = gate_type.upper()
        self.inputs = inputs
        self.output = output

    def evaluate(self, values):
        inputs = [values[x] for x in self.inputs]

        if self.gate_type == "AND":
            return int(all(inputs))

        elif self.gate_type == "OR":
            return int(any(inputs))

        elif self.gate_type == "NOT":
            return int(not inputs[0])

        elif self.gate_type == "NAND":
            return int(not all(inputs))

        elif self.gate_type == "NOR":
            return int(not any(inputs))

        elif self.gate_type == "XOR":
            return inputs[0] ^ inputs[1]

        elif self.gate_type == "XNOR":
            return int(inputs[0] == inputs[1])

        elif self.gate_type == "BUF":
            return inputs[0]

        else:
            raise ValueError(f"Unsupported gate: {self.gate_type}")


class Circuit:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.gates = []

    def add_input(self, name):
        self.inputs.append(name)

    def add_output(self, name):
        self.outputs.append(name)

    def add_gate(self, gate):
        self.gates.append(gate)