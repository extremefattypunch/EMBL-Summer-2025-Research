import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import random
from scipy.optimize import curve_fit
from protein_dna_utilsOOP import *

# Define the exponential function for fitting
def exponential_growth(t, a, tau):
    return a * (1 - np.exp(-tau * t))

# Define the logarithmic function for fitting
def logarithmic_growth(t, a, b):
    return a + b * np.log(t)

num_sim = 20
population = [[] for i in range(num_sim)]
population_avg = defaultdict(float)

for counter in range(num_sim):
    tick = 0
    global_time = 0
    protein_count = 0
    length_dna = 20
    tau = 500  # s^-1
    dt = 0.001  # s
    dx = 1
    k_on = 3
    lower_bound_probability = 1e-5
    total_time_on_dna = -np.log(lower_bound_probability) / tau
    num_protein = 20 
    DNA = dna(length_dna, dt, dx, tau)
    
    while not DNA.all_filled(protein_count, num_protein):
        if protein_count < num_protein and tick % k_on == 0:
            new_protein = protein(protein_count, "done", total_time_on_dna)
            if DNA.add_protein_end(new_protein):
                protein_count += 1
        DNA.set_dir_next_all()
        DNA.resolve_collisions()
        global_time += dt
        tick += 1
        population[counter].append((global_time, protein_count))
        population_avg[global_time] += protein_count / num_sim
    print(f"sim: {counter}, filled max at {global_time}")
    global_time += dt
    while global_time<10:
        population_avg[global_time] += num_protein / num_sim
        global_time += dt

# Plot individual simulations
colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_sim))
for i in range(num_sim):
    times, pop_count = zip(*population[i])
    plt.plot(times, pop_count, color=colors[i], label=f'Sim {i+1}')

# Plot average population
times_avg = np.array(list(population_avg.keys()))
pop_avg = np.array(list(population_avg.values()))
plt.plot(times_avg, pop_avg, color="black", label="avg population")

# Fit the exponential function to the average population
a0 = num_protein  # Initial guess for a (maximum population)
tau0 = 0.1  # Initial guess for tau (rate constant)
popt, pcov = curve_fit(exponential_growth, times_avg, pop_avg, p0=[a0, tau0])
a_fitted, tau_fitted = popt
print(f"Fitted exponential parameters: a = {a_fitted:.2f}, tau = {tau_fitted:.2f}")

# Generate and plot fitted exponential curve
fitted_pop = exponential_growth(times_avg, a_fitted, tau_fitted)
plt.plot(times_avg, fitted_pop, color='red', label='Fitted exponential')

# Fit the logarithmic function to the average population
# Initial guesses: a ≈ max population, b estimated from early and late points
a_log0 = np.max(pop_avg)  # Should be close to 20
t_min = np.min(times_avg)  # Should be 0.001
t_max = np.max(times_avg)  # Varies, but includes extended period
b_log0 = (pop_avg[-1] - pop_avg[0]) / (np.log(t_max) - np.log(t_min))  # Slope estimate
log_popt, log_pcov = curve_fit(logarithmic_growth, times_avg, pop_avg, p0=[a_log0, b_log0])
log_a, log_b = log_popt
print(f"Fitted logarithmic parameters: a = {log_a:.2f}, b = {log_b:.2f}")

# Generate and plot fitted logarithmic curve
log_fitted_pop = logarithmic_growth(times_avg, log_a, log_b)
plt.plot(times_avg, log_fitted_pop, color='black', label='Fitted logarithmic')

# Customize plot
plt.xlabel('Time (s)')
plt.ylabel('Population Count')
plt.title(f'Population Count of {num_protein} Proteins on DNA with Repulsion (Length {length_dna})')
plt.grid(True)
plt.legend()
plt.show()
