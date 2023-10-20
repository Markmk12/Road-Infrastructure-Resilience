import numpy as np
import math


def inspection(pci, maintenance_status):

    # Imperfect inspection considers (budget and manpower)
    if pci < 25:
        maintenance_status = np.random.choice(['no', 'corrective_measures_planning'], p=[0.3, 0.7])

    elif pci < 90:
        maintenance_status = np.random.choice(['no', 'preventive_measures_planning'], p=[0.3, 0.7])

    return maintenance_status


def preventive_maintenance(quality_level, pci, length, lanes):

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
        pci = pci + np.random.normal(25, 5)                 # 15 2
        travel_time_impact = 1.25
        duration = 1
        age_reset = 6                                       # 3
        costs = length*lanes*12.5
        maintenance_status = 'no'

    # Resurfacing (Repaving)
    elif quality_level == 'extensive':

        # Consider variance in PCI improvement
        pci = pci + np.random.normal(40, 5)                 # 30 4
        travel_time_impact = 1.5
        duration = 2
        age_reset = 10
        costs = length*lanes*25
        maintenance_status = 'no'

    return travel_time_impact, duration, pci, age_reset, costs


def corrective_maintenance(quality_level, pci, length, age, lanes):

    # No measures at all
    if quality_level == 'none':

        pci = pci
        travel_time_impact = 1
        duration = 0
        age_reset = 0
        costs = length*lanes*0
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
        duration = 4                        #length / (60 * 365 * 2)              #((length/60)/365)/2           # länge/60 m pro Tag / 365 Standartjahr / 2 Halbjahr
        age_reset = age
        costs = length*lanes*100            # 100 EUR per m
        maintenance_status = 'no'

    return travel_time_impact, duration, pci, age_reset, costs, duration
