import numpy as np
# my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# sections = np.array_split(my_list, 3)
# print(sections)
#
# [(1, 2, -275.87914433907014), (2, 3, -300.150634997869), (2, 3, -320.8835872200407), (1, 2, -282.87104894897345)]


# Parameter environment (natural deterioration???)
alpha_environment = 1  # shape minimum 2 so that it starts by 0
beta_environment = 0.5  # rate

# Parameter traffic load
alpha_traffic = 1
beta_traffic = 0.7

time_range = range(0, 51)

for t in time_range:
    deterioration_delta = np.random.gamma(alpha_environment * t, beta_environment) + np.random.gamma(alpha_traffic * t,
                                                                                                 beta_traffic)
    print(deterioration_delta)
