import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from protein_dna_utils import *
def main():
# Parameters
    length_dna=200
    tau = 500  # s^-1
    dt = 0.001  # s
    dx = 1
    lower_bound_probability = 1e-5
    time_on_dna = -np.log(lower_bound_probability) / tau
    num_particles = 100 
# Initialize
    trajectories=[[] for i in range(num_particles)]
    distributed_dna=distribute_particles(num_particles,length_dna)
    print("at t=0 distributed_dna: ",distributed_dna)
    t = 0
    while t < time_on_dna:
        # Decide intended moves
        resolved_dna=resolve_collisions(distributed_dna);
        for i in range(length_dna):
            if resolved_dna[i]!='':
                nth_particle=resolved_dna[i][0]
                x_particle=i
                trajectories[nth_particle].append((t,x_particle))
        distributed_dna=resolved_dna.copy()
        distributed_dna=reset_directions(distributed_dna)
        print("at t=",t," distributed_dna: ",distributed_dna)

        t += dt
# Plot trajectories
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, num_particles ))
    for i in range(num_particles):
        times, positions = zip(*trajectories[i])
        plt.plot(times, positions, color=colors[i], label=f'Particle {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (length units)')
    plt.title(f'Trajectories of {num_particles} Proteins on DNA with Repulsion')
    plt.grid(True)
    plt.legend()
    plt.show()
    calculate_msd(trajectories)
main()
