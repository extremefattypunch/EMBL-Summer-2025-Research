import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit
from protein_dna_utils_real import *

def exponential_growth(t, a, tau):
    return a * (1 - np.exp(-tau * t))

def logarithmic_growth(t, a, b):
    return a + b * np.log(t)

num_sim =1 
population_counts = [[] for _ in range(num_sim)]  # List of population counts for each sim
finish_times = []  # Store finishing time for each sim
tau = 10
dt = 1
dx = 1
k_on = 1
num_protein = 60
length_dna = dx * num_protein

# Run simulations
for counter in range(num_sim):
    tick = 0
    global_time = 0
    protein_count = 0
    DNA = dna(length_dna, dt, dx, tau, 0, counter)
    while not DNA.all_filled(protein_count, num_protein):
        print("counter: ",counter," protein_count: ",protein_count," global_time: ",global_time)
        if protein_count < num_protein and tick % k_on == 0:
            new_protein = protein(protein_count, "done", 0)
            if DNA.add_protein_end(new_protein):
                protein_count += 1
        DNA.set_dir_next_all()
        DNA.resolve_collisions()
        global_time += dt
        tick += 1
        population_counts[counter].append(protein_count)
    finish_times.append(global_time)

# Find maximum finishing time
T_max = max(finish_times)
max_steps = int(T_max / dt) + 1

# Extend each simulation's population counts to max_steps
for pop_counts in population_counts:
    while len(pop_counts) < max_steps:
        pop_counts.append(num_protein)

# Compute average population using numpy for efficiency
population_array = np.array(population_counts)  # Shape: (num_sim, max_steps)
population_avg = np.mean(population_array, axis=0)  # Shape: (max_steps,)
times_avg = np.arange(0, max_steps * dt, dt)

# Compute the derivative of the average population
dp_dt = np.gradient(population_avg, dt)

# Plot individual simulations without labels to avoid legend clutter
colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_sim))
for i in range(num_sim):
    times = np.arange(0, len(population_counts[i]) * dt, dt)
    plt.plot(times, population_counts[i], color=colors[i])

# Plot average population
plt.plot(times_avg, population_avg, color="black", label="Avg population")

# Fit exponential curve
popt, pcov = curve_fit(exponential_growth, times_avg, population_avg, p0=[num_protein, 1])
a_fitted, tau_fitted = popt
fitted_pop = exponential_growth(times_avg, a_fitted, tau_fitted)
plt.plot(times_avg, fitted_pop, color='red', label='Fitted exponential')
print(f"Fitted exponential parameters: a = {a_fitted:.2f}, tau = {tau_fitted:.2f}")

# Fit logarithmic curve
a_log0 = np.max(population_avg)
t_min, t_max = np.min(times_avg), np.max(times_avg)
b_log0 = (population_avg[-1] - population_avg[0]) / (np.log(t_max) - np.log(t_min))
log_popt, log_pcov = curve_fit(logarithmic_growth, times_avg, population_avg, p0=[a_log0, b_log0])
log_a, log_b = log_popt
log_fitted_pop = logarithmic_growth(times_avg, log_a, log_b)
plt.plot(times_avg, log_fitted_pop, color='blue', label='Fitted logarithmic')
print(f"Fitted logarithmic parameters: a = {log_a:.2f}, b = {log_b:.2f}")

# Plot derivative on secondary y-axis
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(times_avg, dp_dt, color='green', label='Derivative of avg population')
ax2.set_ylabel('Rate of change (proteins/s)')

# Combine legends from both axes
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Customize plot
plt.xlabel('Time (s)')
ax.set_ylabel('Population Count')
plt.title(f'Population Count and Derivative of {num_protein} Proteins on DNA with Repulsion (Length {length_dna}) and tau:{tau}')
plt.grid(True)
plt.show()
