import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
def reset_directions(array):
    for i in range(len(array)):
        if array[i] != "":
            n, _ = array[i]
            r = random.uniform(0, 1)
            new_dir = "left" if r > 0.5 else "right"
            array[i] = (n, new_dir)
    return array
def distribute_particles(N, L):
    """
    Distribute N particles randomly across an array of length L.
    
    Args:
        N: Number of particles (1 <= N <= L)
        L: Length of the array
    
    Returns:
        List of length L with particles as (n, dir) tuples and unfilled spaces as ""
    """
    # Initialize array with empty strings
    array = ["" for _ in range(L)]
    
    # Create list of particles with random directions
    particles = []
    for n in range(0, N):
        r = random.uniform(0, 1)
        dir = "left" if r > 0.5 else "right"
        particles.append((n, dir))
    
    # Select N unique random positions
    positions = random.sample(range(L), N)
    
    # Place particles into the array
    for pos, particle in zip(positions, particles):
        array[pos] = particle
    
    return array

def resolve_collisions(array):
    """
    Resolve collisions in an array of particles until no gaps remain.
    
    Args:
        array: List containing either "" (empty) or tuples (n, dir), where n is an integer
               and dir is "left", "right", or "done".
    
    Returns:
        The resolved array with all gaps handled and particles set to "done" where applicable.
    """
    L = len(array)
    while True:
        changes_made = False
        # Create a copy to apply changes after scanning
        new_array = array.copy()
        
        # Scan each position in the array
        for i in range(L):
            # Case 1: Handle gaps (empty spots)
            if array[i] == '':
                # Check if particle on left is moving right towards the gap
                left_moving = (i > 0 and array[i-1] != '' and array[i-1][1] == 'right')
                # Check if particle on right is moving left towards the gap
                right_moving = (i < L-1 and array[i+1] != '' and array[i+1][1] == 'left')
                
                if left_moving and right_moving:
                    # Collision: Two particles moving towards the gap
                    n_left = array[i-1][0]
                    n_right = array[i+1][0]
                    new_array[i-1] = (n_left, 'done')
                    new_array[i+1] = (n_right, 'done')
                    changes_made = True
                elif left_moving:
                    # Fill: Only left particle moves into the gap
                    n = array[i-1][0]
                    new_array[i] = (n, 'done')
                    new_array[i-1] = ''
                    changes_made = True
                elif right_moving:
                    # Fill: Only right particle moves into the gap
                    n = array[i+1][0]
                    new_array[i] = (n, 'done')
                    new_array[i+1] = ''
                    changes_made = True
            
            # Case 2: Handle edge particles that cannot move
            elif i == 0 and array[i] != '' and array[i][1] == 'left':
                # Particle at left edge wants to move left (out of bounds)
                n = array[i][0]
                new_array[i] = (n, 'done')
                changes_made = True
            elif i == L-1 and array[i] != '' and array[i][1] == 'right':
                # Particle at right edge wants to move right (out of bounds)
                n = array[i][0]
                new_array[i] = (n, 'done')
                changes_made = True
        
        # Update the array with the new state
        array = new_array
        
        # If no changes were made, all gaps are resolved
        if not changes_made:
            break
    
    return array
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
    plt.text(0.05, 0.95, f'Average MSD: {average_msd:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Squared Displacement')
    plt.title('MSD Trajectories with Best Fit Lines')
    plt.show()
    
    return
