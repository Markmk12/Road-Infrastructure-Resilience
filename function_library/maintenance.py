import numpy as np
import math


def inspection(pci, maintenance_status):

    # Imperfect inspection considers (budget and manpower)
    if pci < 25:
        maintenance_status = np.random.choice(['no', 'corrective_measures_planning_and_realization'], p=[0.3, 0.7])

    elif pci < 90:
        maintenance_status = np.random.choice(['no', 'preventive_measures_planning_and_realization'], p=[0.3, 0.7])

    return maintenance_status


def preventive_maintenance(quality_level, pci, length):

    # No measures at all
    if quality_level == 'none':
        pci = pci
        travel_time_impact = 1
        duration = 0
        age_reset = 0
        costs = length*0
        maintenance_status = 'no'

    # Patching, crack sealing, repair of small potholes
    elif quality_level == 'moderate':

        # Consider variance in PCI improvement
        pci = pci + np.random.normal(15, 2)
        travel_time_impact = 1.25
        duration = 1
        age_reset = 3
        costs = length*12.5
        maintenance_status = 'no'

    # Resurfacing (Repaving)
    elif quality_level == 'extensive':

        # Consider variance in PCI improvement
        pci = pci + np.random.normal(30, 4)
        travel_time_impact = 1.5
        duration = 1
        age_reset = 5
        costs = length*25
        maintenance_status = 'no'

    return travel_time_impact, duration, pci, maintenance_status, age_reset, costs


def corrective_maintenance(quality_level, pci, length, age):

    # No measures at all
    if quality_level == 'none':

        pci = pci
        travel_time_impact = 0
        duration = 0
        age_reset = 0
        costs = length * 0
        maintenance_status = 'no'


    # Road rehabilitation/renovation
    elif quality_level == 'moderate':

        # Consider variance in PCI improvement
        pci = pci + np.random.normal(55, 7)
        travel_time_impact = 2
        duration = 1
        age_reset = 10
        costs = length*50
        maintenance_status = 'no'

    # Reconstruction
    elif quality_level == 'extensive':

        # PCI as good as new
        pci = 100
        travel_time_impact = float('inf')
        duration = 2
        age_reset = age
        costs = length*100
        maintenance_status = 'no'

    return travel_time_impact, duration, pci, maintenance_status, age_reset, costs


def maintenance_duration(maintenance_status):
    maintenance_status = 1
    return maintenance_status


def simple_maintenance(pci):
    new_pci = pci + np.random.normal(30, 5)
    # new_pci = 100
    return new_pci
