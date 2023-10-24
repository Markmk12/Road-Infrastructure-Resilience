

def velocity_change_linear(pci, velocity, max_speed):
    """
    Calculate the adjusted velocity based on the Pavement Condition Index (PCI) using a linear relationship.

    The function uses a linear relationship between PCI and velocity. As the PCI decreases, the adjusted velocity
    decreases linearly from the maximum speed. If the resulting velocity goes below a certain threshold, it defaults
    to a base value which is representative of walking speed.

    Parameters:
    -----------
    pci : float or int
        The Pavement Condition Index, indicating the condition of the pavement, expected to be between 0 and 100.
        Values below 0 are adjusted to 0 within the function.

    velocity : float or int
        The initial velocity of the vehicle. Note: This parameter is unused in the function.

    max_speed : float or int
        The maximum possible speed for the vehicle on optimal road conditions.

    Returns:
    --------
    float or int
        The adjusted velocity based on the PCI, with a minimum value representative of walking speed.

    """

    if pci < 0:
        pci = 0

    new_velocity = max_speed - (0.5*(100 - pci))

    if new_velocity < 0:
        new_velocity = 2                                    # very slow walking speed

    return new_velocity


def travel_time(velocity, length):
    """
    Calculate the travel time based on the given velocity and road length.

    The function computes the time required to travel a specified road length at a given velocity.
    If the provided velocity is non-positive, the velocity defaults to a base value representative of walking speed.

    Parameters:
    -----------
    velocity : float or int
        The velocity of the vehicle. Non-positive values will be adjusted to a default walking speed within the function.

    length : float or int
        The length of the road segment in meters.

    Returns:
    --------
    float
        The travel time (in minutes) required to traverse the given road length at the specified velocity.

    """

    if velocity <= 0:
        velocity = 2                                        # very slow walking speed

    edge_travel_time = (length/1000)/velocity*60            # length from [m] in [km]
    return edge_travel_time
