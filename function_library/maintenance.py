import numpy as np


def inspection(pci, maintenance_status):
    """
    Determine the maintenance status based on the Pavement Condition Index (PCI).

    This function uses an imperfect inspection mechanism considering factors such as budget and manpower. Depending on the PCI,
    the maintenance planning can be either for corrective measures or preventive measures. These statuses are determined
    through a random choice mechanism with predefined probabilities.

    Parameters:
    -----------
    pci : float or int
        The Pavement Condition Index. A value indicating the condition of the pavement, expected to be between 0 and 100.

    maintenance_status : str
        The initial maintenance status. This value gets updated based on the provided PCI value and can be one of
        the following: 'no', 'corrective_measures_planning', or 'preventive_measures_planning'.

    Returns:
    --------
    str
        Updated maintenance status. The possible returned statuses are 'no', 'corrective_measures_planning',
        or 'preventive_measures_planning'.

    """

    if pci < 30:                                                                                            # 25
        maintenance_status = np.random.choice(['no', 'corrective_measures_planning'], p=[0.3, 0.7])       # 0.25, 0.75

    elif pci < 90:
        maintenance_status = np.random.choice(['no', 'preventive_measures_planning'], p=[0.5, 0.5])         # 0.3, 0.7

    return maintenance_status


def preventive_maintenance(quality_level, pci, length, lanes):
    """
    Calculate the effects of preventive maintenance on road infrastructure based on the quality of maintenance.

    Depending on the quality level of the maintenance ('sparse', 'moderate', or 'extensive'), this function determines the
    impact on various parameters such as the Pavement Condition Index (PCI), travel time, duration of maintenance, and costs.
    The function incorporates random fluctuations in the PCI improvements based on the quality of the maintenance measures
    taken. Each quality level corresponds to different types of maintenance actions.

    Important: The quality level 'none' is not in use.

    Parameters:
    -----------
    quality_level : str
        The quality of the preventive maintenance. Can be one of the following: 'none', 'moderate', or 'extensive'.

    pci : float or int
        The initial Pavement Condition Index, indicating the current condition of the pavement, expected to be between 0 and 100.

    length : float or int
        The length of the road segment being considered for maintenance (in some unit, e.g., meters or kilometers).

    lanes : int
        The number of lanes on the road segment being considered for maintenance.

    Returns:
    --------
    tuple
        A tuple containing values for:
        - travel_time_impact (float): The multiplier on travel time due to the maintenance.
        - duration (int): The duration of the maintenance.
        - pci (float or int): The updated Pavement Condition Index after maintenance.
        - age_reset (int): The number of years after which the road's age is considered reset due to the maintenance.
        - costs (float or int): The estimated cost of the maintenance.
    """

    # No measures at all
    # if quality_level == 'none':
    #     pci = pci
    #     travel_time_impact = 1
    #     duration = 0
    #     age_reset = 0
    #     costs = length*0
    #     maintenance_status = 'no'

    # Sparse measures
    if quality_level == 'sparse':
        pci = pci + np.random.normal(2, 2)
        travel_time_impact = 1.5
        duration = 0
        age_reset = 1
        costs = length*5
        maintenance_status = 'no'

    # Patching, crack sealing, repair of small potholes
    elif quality_level == 'moderate':

        # Consider variance in PCI improvement
        pci = pci + np.random.normal(25, 5)
        travel_time_impact = 1.25
        duration = 1
        age_reset = 6
        costs = length*lanes*12.5
        maintenance_status = 'no'

    # Resurfacing (Repaving)
    elif quality_level == 'extensive':

        # Consider variance in PCI improvement
        pci = pci + np.random.normal(40, 5)
        travel_time_impact = 1.5
        duration = 2
        age_reset = 10
        costs = length*lanes*25
        maintenance_status = 'no'

    else:
        raise ValueError(f"Invalid value for 'quality_level: {quality_level}")

    return travel_time_impact, duration, pci, age_reset, costs


def corrective_maintenance(quality_level, pci, length, age, lanes):
    """
    Calculate the effects of corrective maintenance on road infrastructure based on the quality of maintenance.

    Depending on the quality level of the maintenance ('sparse', 'moderate', or 'extensive'), this function determines the
    impact on various parameters such as the Pavement Condition Index (PCI), travel time, duration of maintenance, and costs.
    The function incorporates random fluctuations in the PCI improvements based on the quality of the maintenance measures
    taken. Each quality level corresponds to specific corrective actions on the road.

    Important: The quality level 'none' is not in use.

    Parameters:
    -----------
    quality_level : str
        The quality of the corrective maintenance. Can be one of the following: 'none', 'moderate', or 'extensive'.

    pci : float or int
        The initial Pavement Condition Index, indicating the current condition of the pavement, expected to be between 0 and 100.

    length : float or int
        The length of the road segment being considered for maintenance (in some unit, e.g., meters or kilometers).

    age : int
        The current age of the road segment in years.

    lanes : int
        The number of lanes on the road segment being considered for maintenance.

    Returns:
    --------
    tuple
        A tuple containing values for:
        - travel_time_impact (float): The multiplier on travel time due to the maintenance.
        - duration (int): The duration of the maintenance.
        - pci (float or int): The updated Pavement Condition Index after maintenance.
        - age_reset (int): The number of years after which the road's age is considered reset due to the maintenance.
        - costs (float or int): The estimated cost of the maintenance.
        - duration (int): The duration of the maintenance (Note: seems redundant as it's mentioned twice in the function).
    """

    # No measures at all
    # if quality_level == 'none':
    #
    #     pci = pci
    #     travel_time_impact = 1
    #     duration = 0
    #     age_reset = 0
    #     costs = length*lanes*0
    #     maintenance_status = 'no'

    # Sparse measures (temporary provisional repair)
    if quality_level == 'sparse':

        pci = pci + np.random.normal(20, 5)
        travel_time_impact = 2
        duration = 1
        age_reset = 3
        costs = length*lanes*25
        maintenance_status = 'no'

    # Road rehabilitation/renovation
    elif quality_level == 'moderate':

        # Consider variance in PCI improvement
        pci = pci + np.random.normal(60, 5)
        travel_time_impact = 2
        duration = 2
        age_reset = 15                      # 10
        costs = length*lanes*50             # 50 EUR per m
        maintenance_status = 'no'

    # Reconstruction
    elif quality_level == 'extensive':

        # PCI as good as new
        pci = 100
        travel_time_impact = float('inf')
        duration = 4
        age_reset = age
        costs = length*lanes*100            # 100 EUR per m
        maintenance_status = 'no'

    else:
        raise ValueError(f"Invalid value for 'quality_level: {quality_level}")

    return travel_time_impact, duration, pci, age_reset, costs, duration
