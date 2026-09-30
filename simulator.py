def simulate(circuit, input_values, fault=None):
    """
    Simulate the circuit.

    Args:
        circuit: Circuit object.
        input_values: Dictionary containing primary input values.
        fault: Optional StuckAtFault object.

    Returns:
        outputs: Primary output values.
        values: All internal signal values.
    """

    values = {}

    # -------------------------------------------------
    # 1. Apply primary inputs
    # -------------------------------------------------
    for name in circuit.inputs:

        if name not in input_values:
            raise ValueError(
                f"Missing input value for {name}"
            )

        value = input_values[name]

        # If the fault is on a primary input,
        # force the input to the stuck value.
        if fault is not None and fault.net == name:
            value = fault.stuck_value

        values[name] = value

    # -------------------------------------------------
    # 2. Evaluate gates
    # -------------------------------------------------
    for gate in circuit.gates:

        # Check that all inputs are available
        for input_net in gate.inputs:
            if input_net not in values:
                raise ValueError(
                    f"Signal {input_net} has not been evaluated"
                )

        output_value = gate.evaluate(values)

        # -------------------------------------------------
        # Inject fault at gate output
        # -------------------------------------------------
        if fault is not None and fault.net == gate.output:
            output_value = fault.stuck_value

        values[gate.output] = output_value

    # -------------------------------------------------
    # 3. Collect primary outputs
    # -------------------------------------------------
    outputs = {}

    for name in circuit.outputs:

        if name not in values:
            raise ValueError(
                f"Output {name} has not been evaluated"
            )

        outputs[name] = values[name]

    return outputs, values