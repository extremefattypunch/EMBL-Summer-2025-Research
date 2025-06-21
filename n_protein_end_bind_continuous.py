import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from protein_dna_utilsOOP import *
def main():
    tick=0
    global_time=0
    protein_count=0
    length_dna=200
    tau = 500  # s^-1
    dt = 0.001  # s
    dx = 1
    k_on=3
    lower_bound_probability = 1e-5
    total_time_on_dna = -np.log(lower_bound_probability) / tau
    num_protein = 100 
    DNA=dna(length_dna,dt,dx,tau)
    trajectories = [[] for i in range(num_protein)]
    while not DNA.all_left(protein_count,num_protein):
        #print(DNA.all_left(protein_count,num_protein)," protein_count ",protein_count," num_protein ",num_protein)
        print("protein_count: ",protein_count," tick%k_on: ",tick%k_on)
        if (protein_count<num_protein and tick%k_on==0):
            new_protein=protein(protein_count,"done",total_time_on_dna)
            if DNA.add_protein_end(new_protein):
                protein_count+=1
        DNA.set_dir_next_all()
        DNA.resolve_collisions()
        global_time+=dt
        tick+=1
        for x in range(DNA.length):
            if DNA.array[x]!='':
                #print(x," ",DNA.length," DNA.array[x].ID_num ",DNA.array[x].ID_num)
                trajectories[DNA.array[x].ID_num].append((global_time,x))
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_protein))
    for i in range(num_protein):
        times, positions = zip(*trajectories[i])
        plt.plot(times, positions, color=colors[i], label=f'Particle {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (length units)')
    plt.title(f'Trajectories of {num_protein} Proteins on DNA with Repulsion')
    plt.grid(True)
    plt.legend()
    plt.show()
    calculate_msd(trajectories)
main()
