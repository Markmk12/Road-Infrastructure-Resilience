import numpy as np


def inspection(pci, maintenance_status):

    if pci < 50:
        maintenance_status = 'scheduled'

    return maintenance_status


def maintenance_duration(maintenance_status):
    maintenance_status = 1
    return maintenance_status


def simple_maintenance(pci):
    new_pci = pci + np.random.normal(30, 5)
    # new_pci = 100
    return new_pci

# Preventive Maintenance
# Clock based
def clock_based_maintenance():
    return
