def velocity_change(PCI, velocity):
    """
    Calculates the modified velocity based on the given PCI value.

    The function takes the PCI value and computes `delta_v`. The new velocity is then determined
    by subtracting `delta_v` from the maximum velocity (which is set to 100).

    Args:
        PCI (float): A value used to compute delta_v.
        velocity (float): A given velocity. (Note: This parameter is not used in the current code.)

    Returns:
        float: The newly calculated velocity.

    Note:
        The current code does not take into account the `velocity` parameter. The maximum velocity
        is hardcoded to 100.
    """
    delta_v = 1/PCI
    velocity = 100                      # deltaV always refers to maxSpeed
    new_velocity = velocity - delta_v*100
    return new_velocity
