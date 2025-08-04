import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exponential_growth(t, a, tau):
    return a * (1 - np.exp(-tau * t))

def logarithmic_growth(t, a, b):
    return a + b * np.log(t)

# Hardcoded data from the Excel sheet
times = np.array([0, 5, 15, 45, 90, 1260])
population_avg = np.array([0, 27.01108518, 33.84773289, 35.34506354, 39.07825382, 66.05648638])
std_dev = np.array([0, 5.956057453, 7.883647628, 9.876533621, 7.524919142, 20.30349078])

# Compute the derivative of the average population
dp_dt = np.gradient(population_avg, times)

# Plot the data with error bars
plt.errorbar(times, population_avg, yerr=std_dev, fmt='o', color='black', label='Avg population (data)')

# Fit exponential curve
popt, pcov = curve_fit(exponential_growth, times, population_avg, p0=[70, 0.001])
a_fitted, tau_fitted = popt
fitted_pop = exponential_growth(times, a_fitted, tau_fitted)
plt.plot(times, fitted_pop, color='red', label='Fitted exponential')
print(f"Fitted exponential parameters: a = {a_fitted:.2f}, tau = {tau_fitted:.2f}")

# Fit logarithmic curve, skipping t=0
times_fit = times[1:]
pop_fit = population_avg[1:]
a_log0 = np.max(pop_fit)
b_log0 = (pop_fit[-1] - pop_fit[0]) / np.log(times_fit[-1] / times_fit[0])
log_popt, log_pcov = curve_fit(logarithmic_growth, times_fit, pop_fit, p0=[a_log0, b_log0])
log_a, log_b = log_popt
log_fitted_pop = np.zeros_like(times)
log_fitted_pop[1:] = logarithmic_growth(times_fit, log_a, log_b)
log_fitted_pop[0] = log_a  # Set t=0 to a, though log not defined
plt.plot(times, log_fitted_pop, color='blue', label='Fitted logarithmic')
print(f"Fitted logarithmic parameters: a = {log_a:.2f}, b = {log_b:.2f}")

# Plot derivative on secondary y-axis
ax = plt.gca()
# ax2 = ax.twinx()
# ax2.plot(times, dp_dt, color='green', label='Derivative of avg population')
# ax2.set_ylabel('Rate of change (proteins/min)')

# Combine legends from both axes
lines1, labels1 = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax.legend(lines1, labels1, loc='upper left')

# Customize plot
plt.xlabel('Time (min)')
ax.set_ylabel('Population Count')
plt.title('Population Count and Derivative of Proteins on DNA')
plt.grid(True)
plt.show()
