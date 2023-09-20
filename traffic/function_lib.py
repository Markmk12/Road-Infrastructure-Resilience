def velocity_change(PCI, maxSpeed):
    """
    Calculates the modified maxSpeed based on the given PCI value.

    The function takes the PCI value and computes `delta_v`. The new maxSpeed is then determined
    by subtracting `delta_v` from the maximum maxSpeed (which is set to 100).

    Args:
        PCI (float): A value used to compute delta_v.
        maxSpeed (float): A given maxSpeed. (Note: This parameter is not used in the current code.)

    Returns:
        float: The newly calculated maxSpeed.

    Note:
        The current code does not take into account the `maxSpeed` parameter. The maximum maxSpeed
        is hardcoded to 100.
    """
    if PCI <= 0:
        PCI = 1

    delta_v = 1/PCI
    # delta_v = (1385.406/PCI) -15.985
    maxSpeed = 100                          # deltaV always refers to maxSpeed
    new_velocity = maxSpeed - delta_v * maxSpeed
    return new_velocity


def travel_time(velocity, length):
    if velocity <= 0:
        velocity = 1

    edge_travel_time = length/velocity*60
    return edge_travel_time
