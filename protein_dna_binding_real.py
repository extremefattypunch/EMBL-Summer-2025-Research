import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit
import torch

def exponential_growth(t, a, tau):
    return a * (1 - np.exp(-tau * t))

def logarithmic_growth(t, a, b):
    return a + b * np.log(t)

num_sim =1 
population_counts = [[] for _ in range(num_sim)]
tau = 10.0
dt = 1.0
dx = 1
k_on = 1
num_protein =60 
length_dna = dx * num_protein
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
occupancy = torch.zeros((num_sim, length_dna), dtype=torch.bool, device=device)
time_on_dna = torch.zeros((num_sim, length_dna), dtype=torch.float, device=device)
dir_next = torch.zeros((num_sim, length_dna), dtype=torch.long, device=device)
finished = torch.zeros(num_sim, dtype=torch.bool, device=device)
finish_time_tensor = torch.zeros(num_sim, dtype=torch.float, device=device)
tick = 0
global_time = 0.0
while not torch.all(finished):
    #torch.all : returns if all elements are true
    current_num = torch.sum(occupancy, dim=1)
    #current_num has shape (m,)
    if tick % k_on == 0:
        can_add = (current_num < num_protein) & ~occupancy[:,0]
        #~ is bitwise not which performs NOT operation on pytorch tensors
        #although num_protein has shape (1,) it is broadcasted to (m,) to match current_num
        #can_add is (m,) indicating whether each sim can add next(boolean vector)
        occupancy[can_add, 0] = True
        #hence boolean indexing can_add select rows while 1st column(position) is selected
        time_on_dna[can_add, 0] = 0.0
        #reset times
        dir_next[can_add, 0] = 0
        #reset next positions
    # Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1), 1st param is desired shape of output
    occ = occupancy
    time_on_dna[occ] += dt
    p_t = 1 - torch.exp(-time_on_dna / tau)
    rand1 = torch.rand((num_sim, length_dna), device=device)
    change = (rand1 < p_t) & occ
    rand2 = torch.rand((num_sim, length_dna), device=device)
    new_dir = torch.where(rand2 > 0.5, 1, 2)
    dir_next[change] = new_dir[change]
    time_on_dna[change] = 0.0
    for i in range(length_dna):
        gap = ~occupancy[:,i]
        left_moving = torch.zeros(num_sim, dtype=torch.bool, device=device)
        if i > 0:
            left_moving = occupancy[:,i-1] & (dir_next[:,i-1] == 2)
        right_moving = torch.zeros(num_sim, dtype=torch.bool, device=device)
        if i < length_dna - 1:
            right_moving = occupancy[:,i+1] & (dir_next[:,i+1] == 1)
        both = gap & left_moving & right_moving
        #handle collisions
        if i > 0:
            dir_next[both, i-1] = 0
        if i < length_dna - 1:
            dir_next[both, i+1] = 0
        only_left = gap & left_moving & ~right_moving
        if i > 0:
            occupancy[only_left, i] = True
            occupancy[only_left, i-1] = False
            time_on_dna[only_left, i] = time_on_dna[only_left, i-1]
            dir_next[only_left, i] = 0
            time_on_dna[only_left, i-1] = 0.0
            dir_next[only_left, i-1] = 0
        only_right = gap & ~left_moving & right_moving
        if i < length_dna - 1:
            occupancy[only_right, i] = True
            occupancy[only_right, i+1] = False
            time_on_dna[only_right, i] = time_on_dna[only_right, i+1]
            dir_next[only_right, i] = 0
            time_on_dna[only_right, i+1] = 0.0
            dir_next[only_right, i+1] = 0
        if i == 0:
            out_left = occupancy[:,i] & (dir_next[:,i] == 1)
            dir_next[out_left, i] = 0
        if i == length_dna - 1:
            out_right = occupancy[:,i] & (dir_next[:,i] == 2)
            dir_next[out_right, i] = 0
    global_time += dt
    tick += 1
    current_num = torch.sum(occupancy, dim=1)
    print("all sim protein_count: ",current_num.shape," global_time: ",global_time)
    pops = current_num.cpu().numpy()
    for i in range(num_sim):
        population_counts[i].append(int(pops[i]))
    newly_finished = (current_num == length_dna) & ~finished
    finish_time_tensor[newly_finished] = global_time
    finished = finished | newly_finished
finish_times = finish_time_tensor.cpu().numpy().tolist()
T_max = max(finish_times)
max_steps = int(T_max / dt) + 1
for pop_counts in population_counts:
    while len(pop_counts) < max_steps:
        pop_counts.append(num_protein)
population_array = np.array(population_counts)
population_avg = np.mean(population_array, axis=0)
times_avg = np.arange(0, max_steps * dt, dt)
dp_dt = np.gradient(population_avg, dt)
colors = plt.cm.tab20(np.linspace(0, 1, num_sim))
for i in range(num_sim):
    times = np.arange(0, len(population_counts[i]) * dt, dt)
    plt.plot(times, population_counts[i], color=colors[i])
plt.plot(times_avg, population_avg, color="black", label="Avg population")
popt, pcov = curve_fit(exponential_growth, times_avg, population_avg, p0=[num_protein, 1])
a_fitted, tau_fitted = popt
fitted_pop = exponential_growth(times_avg, a_fitted, tau_fitted)
plt.plot(times_avg, fitted_pop, color='red', label='Fitted exponential')
print(f"Fitted exponential parameters: a = {a_fitted:.2f}, tau = {tau_fitted:.2f}")
times_fit = times_avg[1:]
pop_fit = population_avg[1:]
a_log0 = np.max(pop_fit)
b_log0 = (pop_fit[-1] - pop_fit[0]) / np.log(times_fit[-1] / times_fit[0])
log_popt, log_pcov = curve_fit(logarithmic_growth, times_fit, pop_fit, p0=[a_log0, b_log0])
log_a, log_b = log_popt
log_fitted_pop = np.zeros_like(times_avg)
log_fitted_pop[1:] = logarithmic_growth(times_fit, log_a, log_b)
log_fitted_pop[0] = log_a
plt.plot(times_avg, log_fitted_pop, color='blue', label='Fitted logarithmic')
print(f"Fitted logarithmic parameters: a = {log_a:.2f}, b = {log_b:.2f}")
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(times_avg, dp_dt, color='green', label='Derivative of avg population')
ax2.set_ylabel('Rate of change (proteins/s)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.xlabel('Time (s)')
ax.set_ylabel('Population Count')
plt.title(f'Population Count and Derivative of {num_protein} Proteins on DNA with Repulsion (Length {length_dna}) and tau:{tau}')
plt.grid(True)
plt.show()
