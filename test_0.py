import numpy as np
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
sections = np.array_split(my_list, 3)
print(sections)
