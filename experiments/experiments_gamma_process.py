import matplotlib.pyplot as plt
import numpy as np
import os

# Environmental deterioration
alpha_environment = 0.1           # 1
beta_environment = 1            # 1

# Deterioration through traffic
alpha_traffic = 0.05         # 0.5
beta_traffic = 0.3           # 0.3

# Weight of the degradation
weight_environment = 0.35
weight_traffic = 0.65

X_1_values = []
X_2_values = []
Y_values = []
Y_weighted_values = []

interval = np.arange(0, 13, 0.25)
for t in interval:

    X_1 = np.random.gamma(alpha_environment * t, beta_environment)
    X_2 = np.random.gamma(alpha_traffic * t, beta_traffic)
    Y = (np.random.gamma(alpha_environment * t, beta_environment) + np.random.gamma(alpha_traffic * t, beta_traffic))
    Y_weighted = (weight_environment * np.random.gamma(alpha_environment * t, beta_environment) + weight_traffic * np.random.gamma(alpha_traffic * t, beta_traffic))

    X_1_values.append(X_1)
    X_2_values.append(X_2)
    Y_values.append(Y)
    Y_weighted_values.append(Y_weighted)


# Calculate cumulative sums
cum_X_1 = np.cumsum(X_1_values)
cum_X_2 = np.cumsum(X_2_values)
cum_Y = np.cumsum(Y_values)
cum_Y_weighted = np.cumsum(Y_weighted_values)

# Plots
plt.step(interval, cum_X_1, label='Cumulated X_1')
plt.step(interval, cum_X_2, label='Cumulated X_2')
plt.step(interval, cum_Y, label='Cumulated Y')
plt.step(interval, cum_Y_weighted, label='Cumulated Y_weighted')

plt.title('Gamma Processes')
plt.xlabel('t')
plt.ylabel('Cumulated Sum')
plt.legend()

# Saving the plots in different file formats
plt.savefig("gamma.png")
plt.savefig("gamma.svg")
plt.savefig("gamma.eps")  # no transparency

# Plot show
plt.tight_layout()
plt.show()
