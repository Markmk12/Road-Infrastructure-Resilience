import matplotlib.pyplot as plt
import numpy as np
import os

# Disable LaTeX in plots (use the built-in Matplotlib renderer)
plt.rcParams['text.usetex'] = False

# Environmental deterioration           # traffic ???
alpha_environment = 0.5           # 1
beta_environment = 0.3            # 1

# Deterioration through traffic         # environment ???
alpha_traffic = 1         # 0.5
beta_traffic = 0.6          # 0.3

# Weight of the degradation
weight_environment = 0.35
weight_traffic = 0.65

X_1_values = []
X_2_values = []
Y_values = []
Y_weighted_values = []

interval = np.arange(0, 13, 0.25)
for t in interval:

    X_2 = np.random.gamma(alpha_environment * t, beta_environment)
    X_1 = np.random.gamma(alpha_traffic * t, beta_traffic)
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
plt.step(interval, cum_X_1, label=r'$X_{traffic}(t)$')
plt.step(interval, cum_X_2, label=r'$X_{environment}(t)$')
plt.step(interval, cum_Y, label=r'$Y_{unweighted}(t)$')
plt.step(interval, cum_Y_weighted, label=r'$Y_{weighted}(t)$')

# plt.title('Gamma Processes')
plt.xlabel('t')
plt.ylabel('Pavement degradation')  # Using built-in Matplotlib renderer for y label
plt.legend()

# Saving the plots in different file formats
plt.savefig("gamma_4.png")
plt.savefig("gamma_4.svg")
plt.savefig("gamma_4.eps")  # no transparency

# Plot show
plt.tight_layout()
plt.show()
