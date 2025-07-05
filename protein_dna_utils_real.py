import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import *
import random
class dna:
    def __init__(self,length,dt,dx,tau,dna_protein_num,sim_id):
        self.length=length
        self.array = ["" for _ in range(self.length)] 
        self.dt=dt
        self.dx=dx
        self.tau=tau
        self.dna_protein_num=dna_protein_num
        self.sim_id=sim_id
    def pick_value_exp(self,t):
        p_t = expon.cdf(t,self.tau)
        b = 1 if random.random() < p_t else 0 
        print("self.dna_protein_num: ",self.dna_protein_num,"t:",t," p_t: ",p_t,"b: ",b,"sim_id: ",self.sim_id)
        return b 
    def set_dir_next_all(self):
        dt=self.dt
        for i in range(self.length):
            if self.array[i] != "":
                self.array[i].time_on_dna+=dt 
                t=self.array[i].time_on_dna
                if self.pick_value_exp(t)==1:
                    r = random.uniform(0, 1)
                    new_dir = "left" if r > 0.5 else "right"
                    self.array[i].dir_next=new_dir
                    self.array[i].time_on_dna=0
    def add_protein_end(self,new_protein):
        if self.array[0]!="":
            return False
        self.array[0]=new_protein
        self.dna_protein_num+=1
        return True
    def resolve_collisions(self):
        L = self.length 
        dx=self.dx
        #print(self.array)
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
                    self.array[i-dx].dir_next =  'done'
                    self.array[i+dx].dir_next =  'done'                        
                elif left_moving:
                    # Fill: Only left particle moves into the gap
                    temp = self.array[i-dx]
                    self.array[i] = temp 
                    self.array[i].dir_next='done'
                    self.array[i-dx] = ''
                elif right_moving:
                    # Fill: Only right particle moves into the gap
                    temp = self.array[i+dx]
                    self.array[i] = temp 
                    self.array[i].dir_next='done'
                    self.array[i+dx] = ''
            
            # Case 2: Handle edge particles that cannot move
            elif i == 0 and self.array[i] != '' and self.array[i].dir_next == 'left':
                # Particle at left edge wants to move left (out of bounds)
                self.array[i].dir_next = 'done'
            elif i == L-dx and self.array[i] != '' and self.array[i].dir_next == 'right':
                # Particle at right edge wants to move right (out of bounds)
                self.array[i].dir_next = 'done'
        return
    def all_left(self,protein_count,protein_num):
        if(self.dna_protein_num==0 and protein_count==protein_num):
            return True
        return False

    def all_filled(self,protein_count,protein_num):
        if(self.dna_protein_num==self.length):
            return True
        return False
class protein:
    def __init__(self,ID_num,dir_next,time_on_dna):
        self.ID_num = ID_num
        self.dir_next = dir_next
        self.time_on_dna = time_on_dna
