def velocity_change(pci, velocity, max_speed):
    """
    Calculates the modified max_speed based on the given PCI value.

    The function takes the PCI value and computes `delta_v`. The new max_speed is then determined
    by subtracting `delta_v` from the maximum max_speed (which is set to 100).

    Args:
        pci (float): A value used to compute delta_v.
        max_speed (float): A given max_speed. (Note: This parameter is not used in the current code.)

    Returns:
        float: The newly calculated max_speed.

    Note:
        The current code does not take into account the `max_speed` parameter. The maximum max_speed
        is hardcoded to 100.
    """
    if pci <= 0:
        pci = 1

    delta_v = 1/pci
    new_velocity = velocity - delta_v * max_speed
    return new_velocity


def travel_time(velocity, length):
    if velocity <= 0:
        velocity = 1

    edge_travel_time = length/velocity*60
    return edge_travel_time


def traffic_redistribution(AAT):
    return AAT
