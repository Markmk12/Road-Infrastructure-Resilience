import numpy as np


def inspection(pci, maintenance_status):

    if pci < 50:
        maintenance_status = 'scheduled'

    elif pci < 25:
        maintenance_status = 'corrective_measures_planned'

    elif pci < 70:
        maintenance_status = 'preventive_measures_planned'

    return maintenance_status


def preventive_maintenance(quality_level, pci, length):

    # No measures at all
    if quality_level == 'none':
        return

    # Patching, crack sealing, repair of small potholes
    elif quality_level == 'moderate':
        pci = pci + 10
        travel_time_impact = 1.25
        duration = 1
        age_reset = 1
        costs = length*12.5

    # Resurfacing (Repaving)
    elif quality_level == 'extensive':
        pci = pci + 20
        travel_time_impact = 1.5
        duration = 1
        age_reset = 1
        costs = length*25

    return pci, travel_time_impact, duration, age_reset, costs


def corrective_maintenance(quality_level, pci, length, age):

    # No measures at all
    if quality_level == 'none':
        return

    # Road rehabilitation/renovation
    elif quality_level == 'moderate':
        pci = pci + 50
        travel_time_impact = 2
        duration = 1
        age_reset = 1
        costs = length*50

    # Reconstruction
    elif quality_level == 'extensive':
        pci = 100
        travel_time_impact = 3              # road should be closed !!!!!!!! infinite travel time ???
        duration = 2
        age_reset = age
        costs = length*100

    return pci, travel_time_impact, duration, age_reset, costs


def maintenance_duration(maintenance_status):
    maintenance_status = 1
    return maintenance_status


def simple_maintenance(pci):
    new_pci = pci + np.random.normal(30, 5)
    # new_pci = 100
    return new_pci
