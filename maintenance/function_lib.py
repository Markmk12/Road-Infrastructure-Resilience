import numpy as np


def maintenance_duration(maintenance_status):
    maintenance_status = 1
    return maintenance_status


def simple_maintenance(pci, maintenance_status):
    new_pci = pci + np.random.normal(30, 5)
    maintenance_status = 0
    return new_pci
