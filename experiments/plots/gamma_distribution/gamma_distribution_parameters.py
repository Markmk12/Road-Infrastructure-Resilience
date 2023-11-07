import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Disable LaTeX in plots (use the built-in Matplotlib renderer)
plt.rcParams['text.usetex'] = False

# Parameters
shape_parameters = [1, 2, 3, 5, 7, 9]  # Different shape parameters for gamma distribution
scale_parameter = 1  # Scale parameter is kept constant
x = np.linspace(0, 20, 1000)  # Range of x values

# Plot
plt.figure(figsize=(10, 6))

for k in shape_parameters:
    dist = gamma(a=k, scale=scale_parameter)
    plt.plot(x, dist.pdf(x), label=rf'$\alpha={k}, \beta={scale_parameter}$')  # Greek alpha for shape parameter

# plt.title('Gamma Distributions with Different Parameters')
plt.xlabel('X')
plt.ylabel(r'$f_X(x; \alpha, \beta)$')  # Using built-in Matplotlib renderer for y label
plt.legend()
# plt.grid(True)

# Save the plot as PNG and SVG files
plt.savefig('gamma_distributions.png', format='png', dpi=300)
plt.savefig('gamma_distributions.svg', format='svg')
plt.savefig('gamma_distributions.eps', format='eps')

# Show the plot in a window
plt.show()

# # Parameters
# shape_parameter = 4  # Fixed shape parameter
# scale_parameters = [0.2, 0.5, 1, 2, 3]  # Various scale parameters
# x = np.linspace(0, 20, 1000)  # Range of x values
#
# # Plot
# plt.figure(figsize=(10, 6))
#
# for theta in scale_parameters:
#     dist = gamma(a=shape_parameter, scale=theta)  # Use theta as the scale parameter
#     plt.plot(x, dist.pdf(x), label=rf'$\alpha={shape_parameter}, \beta={theta}$')  # Corrected label
#
# plt.xlabel('X')
# plt.ylabel(r'$f_X(x; \alpha, \beta)$')
# plt.legend()
#
# # Save the plot as PNG and SVG files
# plt.savefig('gamma_distributions2.png', format='png', dpi=300)
# plt.savefig('gamma_distributions2.svg', format='svg')
# plt.savefig('gamma_distributions2.eps', format='eps')
#
# # Show the plot in a window
# plt.show()