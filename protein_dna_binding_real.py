import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit
import torch
import os
import pickle
from matplotlib.widgets import CheckButtons

def exponential_growth(t, a, tau):
    return a * (1 - np.exp(-tau * t))

def logarithmic_growth(t, a, b):
    return a + b * np.log(t)

num_sim =100
tau = 100.00
dt = 1.0
dx = 1
k_on = 1
num_protein =60 
length_dna = dx * num_protein
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cache_file = f"simulation_cache_tau{tau:.2f}_num{num_protein}_nsim{num_sim}.pkl"
load_cache = False
population_counts = [[] for _ in range(num_sim)]
finish_times = []

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    if 'population_counts' in data and len(data['population_counts']) == num_sim:
        population_counts = data['population_counts']
        finish_times = data['finish_times']
        load_cache = True

if not load_cache:
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
    data = {'population_counts': population_counts, 'finish_times': finish_times}
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

T_max = max(finish_times)
max_steps = int(T_max / dt) + 1
for pop_counts in population_counts:
    while len(pop_counts) < max_steps:
        pop_counts.append(num_protein)
population_array = np.array(population_counts)
population_avg = np.mean(population_array, axis=0)
population_std = np.std(population_array, axis=0)
times_avg = np.arange(0, max_steps * dt, dt)
dp_dt = np.gradient(population_avg, dt)
colors = plt.cm.tab20(np.linspace(0, 1, num_sim))
ind_lines = []
for i in range(num_sim):
    line, = plt.plot(times_avg, population_counts[i], color=colors[i])
    ind_lines.append(line)
avg_line, = plt.plot(times_avg, population_avg, color="black", label="Avg population")
interval = max(1, len(times_avg) // 20)  # Approximately 20 error bars
err = plt.errorbar(times_avg[::interval], population_avg[::interval], yerr=population_std[::interval], fmt='none', ecolor='gray', capsize=3)
popt, pcov = curve_fit(exponential_growth, times_avg, population_avg, p0=[num_protein, 1])
a_fitted, tau_fitted = popt
fitted_pop = exponential_growth(times_avg, a_fitted, tau_fitted)
exp_line, = plt.plot(times_avg, fitted_pop, color='red', label='Fitted exponential')
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
log_line, = plt.plot(times_avg, log_fitted_pop, color='blue', label='Fitted logarithmic')
print(f"Fitted logarithmic parameters: a = {log_a:.2f}, b = {log_b:.2f}")

# Add real data
real_times_min = [0, 5, 15, 45, 90, 1260]
real_times_sec = [t * 60 for t in real_times_min]
real_means = [0, 27.01108508518, 33.84773289, 35.34506354, 39.07825382, 66.05648638]
real_stds = [0, 5.956057453, 7.883647628, 9.876533621, 7.524919142, 20.30349078]
real_err = plt.errorbar(real_times_sec, real_means, yerr=real_stds, fmt='o-', color='magenta', label='Real experimental data', capsize=5)
real_line = real_err[0]

# Fit exponential to real data
real_times_sec_arr = np.array(real_times_sec)
real_means_arr = np.array(real_means)
popt_real_exp, pcov_real_exp = curve_fit(exponential_growth, real_times_sec_arr, real_means_arr, p0=[70, 0.001])
a_real_exp, tau_real_exp = popt_real_exp
print(f"Real fitted exponential parameters: a = {a_real_exp:.2f}, tau = {tau_real_exp:.2f}")
times_plot_real = np.linspace(0, max(real_times_sec), 100)
real_exp_fit = exponential_growth(times_plot_real, *popt_real_exp)
real_exp_line, = plt.plot(times_plot_real, real_exp_fit, color='orange', label='Real fitted exponential')

# Fit logarithmic to real data
times_fit_real = real_times_sec_arr[1:]
pop_fit_real = real_means_arr[1:]
a_log0_real = np.max(pop_fit_real)
b_log0_real = (pop_fit_real[-1] - pop_fit_real[0]) / np.log(times_fit_real[-1] / times_fit_real[0])
log_popt_real, log_pcov_real = curve_fit(logarithmic_growth, times_fit_real, pop_fit_real, p0=[a_log0_real, b_log0_real])
log_a_real, log_b_real = log_popt_real
print(f"Real fitted logarithmic parameters: a = {log_a_real:.2f}, b = {log_b_real:.2f}")
times_plot_real_log = times_plot_real[times_plot_real > 0]
real_log_fit = logarithmic_growth(times_plot_real_log, *log_popt_real)
real_log_line, = plt.plot(times_plot_real_log, real_log_fit, color='purple', label='Real fitted logarithmic')

ax = plt.gca()
ax2 = ax.twinx()
der_line, = ax2.plot(times_avg, dp_dt, color='green', label='Derivative of avg population')
ax2.set_ylabel('Rate of change (proteins/s)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.subplots_adjust(left=0.2)
rax = plt.axes([0.05, 0.4, 0.1, 0.2])
check = CheckButtons(rax, ('Individuals', 'Average', 'Exponential', 'Logarithmic', 'Derivative', 'Real', 'Real Exponential', 'Real Logarithmic'), (True, True, True, True, True, True, True, True))
def toggle(label):
    if label == 'Individuals':
        vis = not ind_lines[0].get_visible()
        for line in ind_lines:
            line.set_visible(vis)
    elif label == 'Average':
        vis = not avg_line.get_visible()
        avg_line.set_visible(vis)
        for cap in err[1]:
            cap.set_visible(vis)
        for bar in err[2]:
            bar.set_visible(vis)
    elif label == 'Exponential':
        exp_line.set_visible(not exp_line.get_visible())
    elif label == 'Logarithmic':
        log_line.set_visible(not log_line.get_visible())
    elif label == 'Derivative':
        der_line.set_visible(not der_line.get_visible())
    elif label == 'Real':
        vis = not real_line.get_visible()
        real_line.set_visible(vis)
        for cap in real_err[1]:
            cap.set_visible(vis)
        for bar in real_err[2]:
            bar.set_visible(vis)
    elif label == 'Real Exponential':
        real_exp_line.set_visible(not real_exp_line.get_visible())
    elif label == 'Real Logarithmic':
        real_log_line.set_visible(not real_log_line.get_visible())
    plt.draw()
check.on_clicked(toggle)
plt.xlabel('Time (s)')
ax.set_ylabel('Population Count')
plt.suptitle(f'Population Count and Derivative of {num_protein} Proteins on DNA with Repulsion (Length {length_dna}) and tau:{tau} vs Real Data and Fits')
plt.grid(True)
plt.show()
