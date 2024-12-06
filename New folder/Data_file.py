import numpy as np
import json


X_Coordinate = (-10, 10)  # x-Coordinates of the city
Y_Coordinate = (-10, 10)  # y-Coordinates of the city

N_Existing_stores = np.random.randint(3, 20)  # Generating number of existing stores
M_New_stores = 200  # Generating random number of new stores

if __name__ == "__main__":
    print(f"Generating N = {N_Existing_stores} number of existing stores in the city")
    N = np.random.uniform(low=-10, high=10, size=(N_Existing_stores, 2))

    print(f"Generating M = {M_New_stores} number of new stores in the city")
    M = np.random.uniform(low=-10, high=10, size=(M_New_stores, 2))

    Data = {
        "X_coordinate": X_Coordinate,
        "Y_coordinate": Y_Coordinate,
        "Existing_stores": N.tolist(),  
        "New_stores": M.tolist()       
    }

    filename = f"Manonmani_Natarajan_{N_Existing_stores}_new{M_New_stores}.json"
    print(f"Saving as {filename}")

    with open(f"Data/{filename}", 'w') as f:
        json.dump(Data, f, indent=4)

 





