import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from protein_dna_utilsOOP import *
import matplotlib.animation as animation

# Initialize probabilities as a defaultdict to avoid KeyError
probabilities = defaultdict(float)
num_simulations = 20 
length_dna = 200  # Defined here for consistency

def simulation():
    tick = 0
    global_time = 0
    protein_count = 0
    tau = 50  # s^-1
    dt = 0.001  # s
    dx = 1
    k_on = 3
    lower_bound_probability = 1e-5
    total_time_on_dna = -np.log(lower_bound_probability) / tau
    num_protein = 100
    DNA = dna(length_dna, dt, dx, tau)
    trajectories = [[] for _ in range(num_protein)]
    while not DNA.all_left(protein_count, num_protein):
        if protein_count < num_protein and tick % k_on == 0:
            new_protein = protein(protein_count, "done", total_time_on_dna)
            if DNA.add_protein_end(new_protein):
                protein_count += 1
        DNA.set_dir_next_all()
        DNA.resolve_collisions()
        global_time += dt
        tick += 1
        for x in range(DNA.length):
            if DNA.array[x] != '':
                # Use tuple (x, global_time) as key
                probabilities[(x, global_time)] += 1 / num_simulations
                trajectories[DNA.array[x].ID_num].append((global_time, x))

# Run simulations
for _ in range(num_simulations):
    simulation()

# Prepare data for animation
global_times = sorted(set(global_time for x, global_time in probabilities.keys()))
x_values = np.arange(length_dna)
#print(probabilities)

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(x_values, [0] * length_dna, 'o-', label='Probability')
ax.set_ylim(0, 1)  # Probabilities are between 0 and 1
ax.set_xlabel('Position x')
ax.set_ylabel('Probability')
ax.set_title('Probability Distribution at t=0.000')
ax.legend()
ax.grid(True)

# Animation update function
def update(frame):
    global_time = global_times[frame]
    probs = [probabilities.get((x, global_time), 0) for x in range(length_dna)]
    line.set_ydata(probs)
    ax.set_title(f'Probability Distribution at t={global_time}')
    return line,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(global_times), interval=100, blit=True,repeat=False)

ani.save('probability.mp4', writer='ffmpeg')
plt.show()
