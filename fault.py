class StuckAtFault:
    """
    Represents a single stuck-at fault.

    Examples:
        A/SA0
        A/SA1
        N1/SA0
        N1/SA1
    """

    def __init__(self, net, stuck_value):
        if stuck_value not in (0, 1):
            raise ValueError("Stuck value must be 0 or 1")

        self.net = net
        self.stuck_value = stuck_value

    def __str__(self):
        return f"{self.net}/SA{self.stuck_value}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            isinstance(other, StuckAtFault)
            and self.net == other.net
            and self.stuck_value == other.stuck_value
        )

    def __hash__(self):
        return hash((self.net, self.stuck_value))


def generate_stuck_at_faults(circuit):
    """
    Generate SA0 and SA1 faults for every circuit net.
    """

    faults = []

    # Primary inputs
    nets = list(circuit.inputs)

    # Gate output nets
    for gate in circuit.gates:
        nets.append(gate.output)

    # Remove duplicates while preserving order
    nets = list(dict.fromkeys(nets))

    # Generate SA0 and SA1 for every net
    for net in nets:
        faults.append(StuckAtFault(net, 0))
        faults.append(StuckAtFault(net, 1))

    return faults