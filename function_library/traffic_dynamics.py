def velocity_change(pci, velocity, max_speed):
    """
    Calculates the modified maxspeed based on the given PCI value.

    The function takes the PCI value and computes `delta_v`. The new maxspeed is then determined
    by subtracting `delta_v` from the maximum maxspeed (which is set to 100).

    Args:
        pci (float): A value used to compute delta_v.
        max_speed (float): A given maxspeed. (Note: This parameter is not used in the current code.)

    Returns:
        float: The newly calculated maxspeed.

    Note:
        The current code does not take into account the `maxspeed` parameter. The maximum maxspeed
        is hardcoded to 100.
    """
    if pci <= 0:
        pci = 1

    delta_v = 1/pci
    # new_velocity = velocity - delta_v * max_speed
    new_velocity = max_speed - delta_v * max_speed

    if new_velocity <= 0:
        new_velocity = 5      # Schrittgeschwindigkeit

    return new_velocity


def velocity_change_linear(pci, velocity, max_speed):

    if pci < 0:
        pci = 0

    new_velocity = max_speed - (0.5*(100 - pci))

    if new_velocity <= 0:
        new_velocity = 5      # Schrittgeschwindigkeit

    return new_velocity


def travel_time(velocity, length):

    if velocity <= 0:
        velocity = 5

    edge_travel_time = (length/1000)/velocity*60            # length from [m] in [km]
    return edge_travel_time
