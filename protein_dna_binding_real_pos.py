import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import torch
import os
import pickle
from matplotlib.widgets import CheckButtons, Slider
import csv
import collections
from matplotlib.animation import FuncAnimation,FFMpegWriter

def exponential_growth(t, a, tau):
    return a * (1 - np.exp(-tau * t))

def logarithmic_growth(t, a, b):
    return a + b * np.log(t)

dt = 1.0
dx = 1
k_on = 1
num_protein = 60
length_dna = dx * num_protein
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tau_values = np.array([10.0])
num_tau = len(tau_values)
num_sim_per_tau = 100
total_sim = num_tau * num_sim_per_tau

tau_tensor = torch.tensor(tau_values, dtype=torch.float, device=device).repeat_interleave(num_sim_per_tau).view(total_sim, 1)

data_per_tau = {}
all_t_max = []

# Check if all caches exist and have histogram_history
load_cache_all = True
for tau in tau_values:
    cache_file = f"simulation_cache_tau{tau:.2f}_num{num_protein}_nsim{num_sim_per_tau}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        if 'histogram_history' not in data:
            load_cache_all = False
            break
    else:
        load_cache_all = False
        break

if not load_cache_all:
    occupancy = torch.zeros((total_sim, length_dna), dtype=torch.bool, device=device)
    time_on_dna = torch.zeros((total_sim, length_dna), dtype=torch.float, device=device)
    dir_next = torch.zeros((total_sim, length_dna), dtype=torch.long, device=device)
    finished = torch.zeros(total_sim, dtype=torch.bool, device=device)
    finish_time_tensor = torch.zeros(total_sim, dtype=torch.float, device=device)
    tick = 0
    global_time = 0.0
    population_history = []
    histogram_per_tau = collections.defaultdict(list)
    # Initial state at t=0
    population_history.append(torch.zeros(total_sim, dtype=torch.float, device=device).cpu().numpy())
    for itau, tau in enumerate(tau_values):
        start = itau * num_sim_per_tau
        end = start + num_sim_per_tau
        hist = torch.sum(occupancy[start:end, :], dim=0).cpu().numpy()
        histogram_per_tau[tau].append(hist.copy())
    while not torch.all(finished):
        current_num = torch.sum(occupancy, dim=1)
        if tick % k_on == 0:
            can_add = (current_num < num_protein) & ~occupancy[:,0]
            occupancy[can_add, 0] = True
            time_on_dna[can_add, 0] = 0.0
            dir_next[can_add, 0] = 0
        occ = occupancy
        time_on_dna[occ] += dt
        p_t = 1 - torch.exp(-time_on_dna / tau_tensor)
        rand1 = torch.rand((total_sim, length_dna), device=device)
        change = (rand1 < p_t) & occ
        rand2 = torch.rand((total_sim, length_dna), device=device)
        new_dir = torch.where(rand2 > 0.5, 1, 2)
        dir_next[change] = new_dir[change]
        time_on_dna[change] = 0.0
        for i in range(length_dna):
            gap = ~occupancy[:,i]
            left_moving = torch.zeros(total_sim, dtype=torch.bool, device=device)
            if i > 0:
                left_moving = occupancy[:,i-1] & (dir_next[:,i-1] == 2)
            right_moving = torch.zeros(total_sim, dtype=torch.bool, device=device)
            if i < length_dna - 1:
                right_moving = occupancy[:,i+1] & (dir_next[:,i+1] == 1)
            both = gap & left_moving & right_moving
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
        print(f"all sim protein_count: {current_num.shape}, global_time: {global_time}")
        population_history.append(current_num.cpu().numpy())
        # Append histograms after updates
        for itau, tau in enumerate(tau_values):
            start = itau * num_sim_per_tau
            end = start + num_sim_per_tau
            hist = torch.sum(occupancy[start:end, :], dim=0).cpu().numpy()
            histogram_per_tau[tau].append(hist.copy())
        newly_finished = (current_num == length_dna) & ~finished
        finish_time_tensor[newly_finished] = global_time
        finished = finished | newly_finished
    population_history = np.stack(population_history, axis=1)  # (total_sim, ticks+1)
    finish_times_all = finish_time_tensor.cpu().numpy()

    # Split and save per tau
    for itau, tau in enumerate(tau_values):
        start = itau * num_sim_per_tau
        end = start + num_sim_per_tau
        population_counts = [[] for _ in range(num_sim_per_tau)]
        for j in range(num_sim_per_tau):
            population_counts[j] = population_history[start + j].tolist()
        finish_times = finish_times_all[start:end].tolist()
        cache_file = f"simulation_cache_tau{tau:.2f}_num{num_protein}_nsim{num_sim_per_tau}.pkl"
        data = {'population_counts': population_counts, 'finish_times': finish_times, 'histogram_history': histogram_per_tau[tau]}
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

# Load all cache and process as before
for tau in tau_values:
    cache_file = f"simulation_cache_tau{tau:.2f}_num{num_protein}_nsim{num_sim_per_tau}.pkl"
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    population_counts = data['population_counts']
    finish_times = data['finish_times']
    histogram_history = data['histogram_history']
    all_t_max.append(max(finish_times))
    population_array = np.array(population_counts)
    pop_avg = np.mean(population_array, axis=0)
    pop_std = np.std(population_array, axis=0)
    times_tau = np.arange(0, len(pop_avg) * dt, dt)
    data_per_tau[tau] = {
        'population_array': population_array,
        'pop_avg': pop_avg,
        'pop_std': pop_std,
        'times': times_tau,
        'population_counts': population_counts,
        'finish_times': finish_times,
        'histogram_history': np.stack(histogram_history)
    }

global_max_steps = max(len(d['pop_avg']) for d in data_per_tau.values())
times_global = np.arange(0, global_max_steps * dt, dt)

for tau in tau_values:
    d = data_per_tau[tau]
    current_steps = len(d['pop_avg'])
    pad_len = global_max_steps - current_steps
    if pad_len > 0:
        pad_array = np.full((num_sim_per_tau, pad_len), num_protein)
        d['population_array'] = np.hstack((d['population_array'], pad_array))
        d['pop_avg'] = np.mean(d['population_array'], axis=0)
        d['pop_std'] = np.std(d['population_array'], axis=0)
        pad_hist = np.full((pad_len, length_dna), num_sim_per_tau)
        d['histogram_history'] = np.vstack((d['histogram_history'], pad_hist))
    d['times'] = times_global
    dp_dt = np.gradient(d['pop_avg'], dt)
    d['dp_dt'] = dp_dt
    popt, pcov = curve_fit(exponential_growth, times_global, d['pop_avg'], p0=[num_protein, 1/tau])
    d['popt_exp'] = popt
    fitted_pop = exponential_growth(times_global, *popt)
    d['fitted_exp'] = fitted_pop
    times_fit = times_global[1:]
    pop_fit = d['pop_avg'][1:]
    a_log0 = np.max(pop_fit)
    b_log0 = (pop_fit[-1] - pop_fit[0]) / np.log(times_fit[-1] / times_fit[0]) if times_fit[-1] > times_fit[0] else 1
    log_popt, log_pcov = curve_fit(logarithmic_growth, times_fit, pop_fit, p0=[a_log0, b_log0])
    d['popt_log'] = log_popt
    log_fitted_pop = np.zeros_like(times_global)
    log_fitted_pop[1:] = logarithmic_growth(times_fit, *log_popt)
    log_fitted_pop[0] = log_popt[0]
    d['fitted_log'] = log_fitted_pop
    print(f"tau={tau:.2f} Fitted exponential parameters: a = {popt[0]:.2f}, tau = {popt[1]:.2f}")
    print(f"tau={tau:.2f} Fitted logarithmic parameters: a = {log_popt[0]:.2f}, b = {log_popt[1]:.2f}")

# Real data
real_times_min = [0, 5, 15, 45, 90, 1260]
real_times_sec = np.array([t * 60 for t in real_times_min])
real_means = np.array([0, 27.01108508518, 33.84773289, 35.34506354, 39.07825382, 66.05648638])
real_stds = np.array([0, 5.956057453, 7.883647628, 9.876533621, 7.524919142, 20.30349078])

# Fit exponential to real data
popt_real_exp, pcov_real_exp = curve_fit(exponential_growth, real_times_sec, real_means, p0=[70, 0.001])
print(f"Real fitted exponential parameters: a = {popt_real_exp[0]:.2f}, tau = {popt_real_exp[1]:.2f}")
real_exp_fit_at_real = exponential_growth(real_times_sec, *popt_real_exp)

# Fit logarithmic to real data
times_fit_real = real_times_sec[1:]
pop_fit_real = real_means[1:]
a_log0_real = np.max(pop_fit_real)
b_log0_real = (pop_fit_real[-1] - pop_fit_real[0]) / np.log(times_fit_real[-1] / times_fit_real[0]) if times_fit_real[-1] > times_fit_real[0] else 1
log_popt_real, log_pcov_real = curve_fit(logarithmic_growth, times_fit_real, pop_fit_real, p0=[a_log0_real, b_log0_real])
print(f"Real fitted logarithmic parameters: a = {log_popt_real[0]:.2f}, b = {log_popt_real[1]:.2f}")
real_log_fit_at_real = np.zeros_like(real_times_sec)
real_log_fit_at_real[1:] = logarithmic_growth(times_fit_real, *log_popt_real)
real_log_fit_at_real[0] = log_popt_real[0]

# Compute SSD at real points
ssd_dict = {}
for tau in tau_values:
    d = data_per_tau[tau]
    pop_interp = interp1d(times_global, d['pop_avg'], kind='linear', fill_value='extrapolate')
    pop_at_real = pop_interp(real_times_sec)
    ssd = np.sum((pop_at_real - real_means)**2)
    ssd_dict[f"tau={tau:.2f}"] = ssd

ssd_real_exp = np.sum((real_exp_fit_at_real - real_means)**2)
ssd_real_log = np.sum((real_log_fit_at_real - real_means)**2)
ssd_dict["Real Exponential"] = ssd_real_exp
ssd_dict["Real Logarithmic"] = ssd_real_log

min_name = min(ssd_dict, key=ssd_dict.get)

# Write CSV
with open('comparison.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Sum Squared Difference'])
    for name, ssd in ssd_dict.items():
        writer.writerow([name, ssd])
    writer.writerow(['Lowest', min_name])

# Plotting
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25, right=0.8)

# Real data fixed
real_err = ax.errorbar(real_times_sec, real_means, yerr=real_stds, fmt='o-', color='magenta', label='Real experimental data', capsize=5)
real_line = real_err[0]
times_plot_real = np.linspace(0, max(real_times_sec), 100)
real_exp_fit = exponential_growth(times_plot_real, *popt_real_exp)
real_exp_line, = ax.plot(times_plot_real, real_exp_fit, color='orange', label='Real fitted exponential')
times_plot_real_log = times_plot_real[times_plot_real > 0]
real_log_fit = logarithmic_growth(times_plot_real_log, *log_popt_real)
real_log_line, = ax.plot(times_plot_real_log, real_log_fit, color='purple', label='Real fitted logarithmic')

# Initial tau
init_idx = 0
init_tau = tau_values[init_idx]
d_init = data_per_tau[init_tau]
colors = plt.cm.tab20(np.linspace(0, 1, num_sim_per_tau))
ind_lines = []
for i in range(num_sim_per_tau):
    line, = ax.plot(times_global, d_init['population_array'][i], color=colors[i])
    ind_lines.append(line)
avg_line, = ax.plot(times_global, d_init['pop_avg'], color="black", label="Avg population")
interval = max(1, len(times_global) // 20)
err = ax.errorbar(times_global[::interval], d_init['pop_avg'][::interval], yerr=d_init['pop_std'][::interval], fmt='none', ecolor='gray', capsize=3)
exp_line, = ax.plot(times_global, d_init['fitted_exp'], color='red', label='Fitted exponential')
log_line, = ax.plot(times_global, d_init['fitted_log'], color='blue', label='Fitted logarithmic')

# Squared diff
real_interp_func = interp1d(real_times_sec, real_means, kind='linear', fill_value='extrapolate')
mask = (times_global >= min(real_times_sec)) & (times_global <= max(real_times_sec))
times_for_diff = times_global[mask]
real_interp = real_interp_func(times_for_diff)
sq_diff_init = (real_interp - d_init['pop_avg'][mask]) ** 2

ax2 = ax.twinx()
der_line, = ax2.plot(times_global, d_init['dp_dt'], color='green', label='Derivative of avg population')
ax2.set_ylabel('Rate of change (proteins/s)', color='green')

ax3 = ax.twinx()
ax3.spines.right.set_position(("axes", 1.15))
sq_line, = ax3.plot(times_for_diff, sq_diff_init, color='brown', label='Squared difference')
ax3.set_ylabel('Squared Difference', color='brown')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3 = [sq_line]
labels3 = [sq_line.get_label()]
ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

rax = plt.axes([0.05, 0.4, 0.1, 0.2])
check = CheckButtons(rax, ('Individuals', 'Average', 'Exponential', 'Logarithmic', 'Derivative', 'Real', 'Real Exponential', 'Real Logarithmic', 'Squared Diff'), (True, True, True, True, True, True, True, True, True))
def toggle(label):
    if label == 'Individuals':
        vis = not ind_lines[0].get_visible()
        for line in ind_lines:
            line.set_visible(vis)
    elif label == 'Average':
        vis = not avg_line.get_visible()
        avg_line.set_visible(vis)
        if 'err' in globals():
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
    elif label == 'Squared Diff':
        sq_line.set_visible(not sq_line.get_visible())
    plt.draw()
check.on_clicked(toggle)

ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Tau Index', 0, len(tau_values) - 1, valinit=init_idx, valstep=1)

def update(val):
    global err
    idx = int(val)
    tau = tau_values[idx]
    d = data_per_tau[tau]
    for i, line in enumerate(ind_lines):
        line.set_ydata(d['population_array'][i])
    avg_line.set_ydata(d['pop_avg'])
    # Remove old err
    for cap in err[1]:
        cap.remove()
    for bar in err[2]:
        bar.remove()
    # New err
    err = ax.errorbar(times_global[::interval], d['pop_avg'][::interval], yerr=d['pop_std'][::interval], fmt='none', ecolor='gray', capsize=3)
    exp_line.set_ydata(d['fitted_exp'])
    log_line.set_ydata(d['fitted_log'])
    der_line.set_ydata(d['dp_dt'])
    sq_diff = (real_interp_func(times_for_diff) - d['pop_avg'][mask]) ** 2
    sq_line.set_ydata(sq_diff)
    fig.suptitle(f'Population Count and Derivative of {num_protein} Proteins on DNA with Repulsion (Length {length_dna}) and tau:{tau:.2f} vs Real Data and Fits')
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.xlabel('Time (s)')
ax.set_ylabel('Population Count')
fig.suptitle(f'Population Count and Derivative of {num_protein} Proteins on DNA with Repulsion (Length {length_dna}) and tau:{init_tau:.2f} vs Real Data and Fits')
plt.grid(True)
plt.show()

for tau_anim in tau_values:
    if tau_anim in data_per_tau:
        print(f'video saving for tau{tau_anim}')
        d = data_per_tau[tau_anim]
        histogram_history = d['histogram_history']
        fig_anim, ax_anim = plt.subplots()
        
        # Initialize the bar plot once (outside the animate function)
        bars = ax_anim.bar(np.arange(length_dna), histogram_history[0], width=1)
        ax_anim.set_xlim(0, length_dna)
        ax_anim.set_ylim(0, num_sim_per_tau + 5)
        ax_anim.set_xlabel('Position on DNA')
        ax_anim.set_ylabel('Sum of protein particles over simulations')
        ax_anim.set_title(f'Time: {d["times"][0]:.0f} s')
        
        def animate(frame):
            # Update bar heights instead of clearing/redrawing
            for rect, h in zip(bars, histogram_history[frame]):
                rect.set_height(h)
            # Update title (which changes per frame)
            ax_anim.set_title(f'Time: {d["times"][frame]:.0f} s')
            return bars
        
        ani = FuncAnimation(fig_anim, animate, frames=len(histogram_history), interval=100)
        
        # Use FFMpegWriter with NVENC for GPU-accelerated encoding
        writer = FFMpegWriter(fps=120, extra_args=['-vcodec', 'h264_nvenc', '-preset', 'fast'])
        ani.save(f'protein_dna_binding_animation_tau{tau_anim:.2f}.mp4', writer=writer,dpi=50)
