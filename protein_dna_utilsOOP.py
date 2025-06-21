import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
class dna:
    def __init__(self,length,dt,dx,tau,dna_protein_num=0):
        self.length=length
        self.array = ["" for _ in range(self.length)] 
        self.dt=dt
        self.dx=dx
        self.tau=tau
        self.dna_protein_num=dna_protein_num
    def set_dir_next_all(self):
        dt=self.dt
        for i in range(self.length):
            if self.array[i] != "":
                self.array[i].time_on_dna_left-=dt
                if self.array[i].time_on_dna_left<=0:
                    self.array[i]=""
                    self.dna_protein_num-=1
                else:
                    r = random.uniform(0, 1)
                    new_dir = "left" if r > 0.5 else "right"
                    self.array[i].dir_next=new_dir
    def add_protein_end(self,new_protein):
        if self.array[0]!="":
            return False
        self.array[0]=new_protein
        self.dna_protein_num+=1
        return True
    def resolve_collisions(self):
        L = self.length 
        dx=self.dx
        changes_made = True
        while changes_made==True:
            changes_made=False
            # Create a copy to apply changes after scanning
            new_array = self.array.copy() 
            #print(new_array)
            # Scan each position in the array
            for i in range(L):
                # Case 1: Handle gaps (empty spots)
                if self.array[i] == '':
                    # Check if particle on left is moving right towards the gap
                    left_moving = (i > 0 and self.array[i-dx] != '' and self.array[i-dx].dir_next == 'right')
                    # Check if particle on right is moving left towards the gap
                    right_moving = (i < L-dx and self.array[i+dx] != '' and self.array[i+dx].dir_next== 'left')
                    
                    if left_moving and right_moving:
                        # Collision: Two particles moving towards the gap
                        new_array[i-dx].dir_next =  'done'
                        new_array[i+dx].dir_next =  'done'                        
                        changes_made = True
                    elif left_moving:
                        # Fill: Only left particle moves into the gap
                        temp = self.array[i-dx]
                        new_array[i] = temp 
                        new_array[i].dir_next='done'
                        new_array[i-dx] = ''
                        changes_made = True
                    elif right_moving:
                        # Fill: Only right particle moves into the gap
                        temp = self.array[i+dx]
                        new_array[i] = temp 
                        new_array[i].dir_next='done'
                        new_array[i+dx] = ''
                        changes_made = True
                
                # Case 2: Handle edge particles that cannot move
                elif i == 0 and self.array[i] != '' and self.array[i].dir_next == 'left':
                    # Particle at left edge wants to move left (out of bounds)
                    new_array[i].dir_next = 'done'
                    changes_made = True
                elif i == L-dx and self.array[i] != '' and self.array[i].dir_next == 'right':
                    # Particle at right edge wants to move right (out of bounds)
                    new_array[i].dir_next = 'done'
                    changes_made = True
            # Update the array with the new state
            self.array = new_array
            
            # If no changes were made, all gaps are resolved
        return
    def all_left(self,protein_count,protein_num):
        if(self.dna_protein_num==0 and protein_count==protein_num):
            return True
        return False

class protein:
    def __init__(self,ID_num,dir_next,time_on_dna_left):
        self.ID_num = ID_num
        self.dir_next = dir_next
        self.time_on_dna_left=time_on_dna_left
def calculate_msd(trajectories):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Write MSD data to file
    with open("dataMSD.txt", "w") as f:
        f.write("MSD for individual trajectories:\n")
    totalmsd = 0
    for i in range(len(trajectories)):
        msd_temp = 0
        for j in range(1, len(trajectories[i])):
            msd_temp += (trajectories[i][j][1] - trajectories[i][0][1])**2
        msd = msd_temp / len(trajectories[i])
        totalmsd += msd
        with open("dataMSD.txt", "a") as f:
            f.write("Particle " + str(i) + " mean square displacment: " + str(msd) + "\n")
    
    average_msd = totalmsd / len(trajectories)
    with open("dataMSD.txt", "a") as f:
        f.write("Average MSD: " + str(average_msd) + "\n")
    
    # Plotting
    plt.figure()
    for i in range(len(trajectories)):
        times = [trajectories[i][j][0] for j in range(1, len(trajectories[i]))]
        squared_displacements = [(trajectories[i][j][1] - trajectories[i][0][1])**2 for j in range(1, len(trajectories[i]))]
        plt.scatter(times, squared_displacements, label=f'Particle {i}')
        if len(times) > 1:
            coefficients = np.polyfit(times, squared_displacements, 1)
            fitted_line = np.poly1d(coefficients)
            plt.plot(times, fitted_line(times), label=f'Fit for Particle {i}')
    
    # Add average MSD label
    plt.text(0.05, 0.95, f'Average MSD: {average_msd:.2f}', transform=plt.gca().transAxes, verticalalignment='top',horizontalalignment='right')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Squared Displacement')
    plt.title('MSD Trajectories with Best Fit Lines')
    plt.show()
    
    return

