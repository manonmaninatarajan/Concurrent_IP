import numpy as np
import json
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Settings
learning_rate = 0.01
num_iterations = 150
max_processes = 8
# Globals
runtime = []  
X_coordinate = (0, 0)
Y_coordinate = (0, 0)
existing_stores_count = 0
new_stores_count = 0

# Read data from file
def read_data(filename):
    global X_coordinate, Y_coordinate, existing_stores, new_stores, existing_stores_count, new_stores_count

    print(f"Reading data from {filename}...")
    with open("Data/" + filename, 'r') as f:
        data = json.load(f)
    print("Data successfully loaded.")
    print(f"City size: X={data['X_coordinate']}, Y={data['Y_coordinate']}")
    print(f"Number of existing stores: {len(data['Existing_stores'])}")
    print(f"Number of new stores: {len(data['New_stores'])}")

    X_coordinate = data["X_coordinate"]
    Y_coordinate = data["Y_coordinate"]
    existing_stores = np.array(data["Existing_stores"])
    new_stores = np.array(data["New_stores"])
    existing_stores_count = len(existing_stores)
    new_stores_count = len(new_stores)

# Distance cost between stores
def distance_cost(x1, y1, x2, y2):
    return np.exp(-0.2 * ((x1 - x2)**2 + (y1 - y2)**2))

# Boundary cost function
def boundary_cost(x, y):
    dist_x = min(abs(x - X_coordinate[0]), abs(x - X_coordinate[1]))
    dist_y = min(abs(y - Y_coordinate[0]), abs(y - Y_coordinate[1]))
    dist = min(dist_x, dist_y)
    if dist > 0:
        return 1 / (1 + np.exp(-0.25 * dist**2)) 
    else:
        100

# Total_Price function
def Total_Price(Existing_stores, New_stores):
    total_cost = 0  
    for New_store in New_stores:
        total_distance = 0
        for existing in Existing_stores:
            distance = distance_cost(New_store[0], New_store[1], existing[0], existing[1])
            total_distance += distance 
        distance_to_existing_stores = total_distance / len(Existing_stores)
        distance_to_boundary = boundary_cost(New_store[0], New_store[1])
        total_cost += distance_to_existing_stores + distance_to_boundary
    return total_cost

# Gradient calculation 
def calculate_gradient(existing_stores, new_stores, j, k):
    small_val = np.zeros_like(new_stores)
    small_val[j, k] = 1e-5  
    cost_plus = Total_Price(existing_stores, new_stores + small_val)
    cost_minus = Total_Price(existing_stores, new_stores - small_val)
    gradient = (cost_plus - cost_minus) / (2 * 1e-5)  # Finite difference approximation
    return gradient

# Visualization
def visualize_stores(existing_stores, new_stores, title, save=None):
    plt.figure(figsize=(8, 8))
    plt.scatter(existing_stores[:, 0], existing_stores[:, 1], color='blue', label='Existing Stores')
    plt.scatter(new_stores[:, 0], new_stores[:, 1], color='red', label='New Stores')
    plt.title(title)
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{save}.png")  # Save the figure
    plt.show(block=False)
    plt.pause(0.5)  # Pause to display the plot
    plt.close()
   

# Main program
if __name__ == "__main__":
    read_data("Manonmani_Natarajan_14_new19.json")
    visualize_stores(existing_stores, new_stores, "Initial Store Locations","Initial Store Locations")
    

    print("Starting optimization...")
    for processes in range(1, max_processes + 1):
        start_time = time.time()
        objective_values = []

        with Pool(processes=processes) as pool:
            for iteration in range(num_iterations):
                print(f"\rStarting iteration {iteration+1}...", end='', flush=True)
                # Parallelize gradient computation
                store_indices = list(range(new_stores_count))
                gradients = pool.starmap(
                    calculate_gradient,
                    [(existing_stores, new_stores, j, k) for j in store_indices for k in range(2)]
                )

                # Reshape and apply gradients
                grad = np.array(gradients).reshape(new_stores.shape)
                new_stores -= learning_rate * grad
                objective_values.append(Total_Price(existing_stores, new_stores))

                # Learning rate decay
                learning_rate *= 0.99

        end_time = time.time()
        runtime.append(end_time - start_time)        
        print(f"\nFinished optimization with {processes} processes in {runtime[-1]} seconds")
        print(f"Objective value after optimization with {processes} processes: {objective_values[-1]}")

    print("Optimization complete.")
    visualize_stores(existing_stores, new_stores, "Optimized Store Locations","Optimized Store Locations")
    print(f"Final objective value: {objective_values[-1]}")
    print(f"Final new store locations:\n{new_stores}")