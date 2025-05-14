import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Load the food dataset
df = pd.read_csv("/mnt/data/food_db_cleaned.csv")

# Dynamically fetch nutrient columns (we will assume all columns related to nutrients are numeric)
nutrient_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove columns that are not nutrients (e.g., 'Ingredient', 'Main category', etc.)
non_nutrient_columns = ['Ingredient', 'Main category', 'Subcategory', 'Detailed category']
nutrient_columns = [col for col in nutrient_columns if col not in non_nutrient_columns]

# Show nutrient columns to confirm
print("Nutrient columns detected:", nutrient_columns)

# Convert nutrient columns to numeric (in case there are non-numeric values like strings)
nutrients = df[nutrient_columns]
nutrients = nutrients.apply(pd.to_numeric, errors='coerce')  # Convert to numeric

# Drop rows with NaN values in the nutrient data
nutrients = nutrients.dropna()

# Collect user input for the target nutrient conditions
print("Please enter the nutritional goals you want to satisfy:")

# Collecting user inputs for each nutrient they care about
constraints = {}
for column in nutrient_columns:
    nutrient_goal = input(f"Enter the minimum target for {column} (or press Enter to skip): ")
    if nutrient_goal:
        constraints[column] = float(nutrient_goal)

# Step 1: Prepare the matrix of nutrient coefficients for the constraints
A_ub = []
b_ub = []

# Populate the matrix for constraints
for column, goal in constraints.items():
    A_ub.append(nutrients[column].values)  # Coefficients for each nutrient
    b_ub.append(goal)  # Target minimum for the nutrient

# Convert A_ub and b_ub to NumPy arrays
A_ub = np.array(A_ub)
b_ub = np.array(b_ub)

# Step 2: Objective Function: Maximize the sum of ingredient weights (i.e., m1 + m2 + ... + mm)
# This is simply to maximize the total amount of ingredients used
c = -np.ones(len(df))  # Maximizing the sum of m (minimize -sum(m))

# Step 3: Solve using Linear Programming (Simplex or HiGHS)
result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * len(df), method='highs')

# Step 4: Display results
if result.success:
    print("Optimal ingredient combination found!")
    print("Ingredient weights (amount to use):", result.x)
    
    # Prepare the output DataFrame with ingredients and their amounts
    result_df = df.copy()
    result_df['amount_to_use (g)'] = result.x * 100  # Convert from per 100g to grams
    for nutrient in constraints.keys():
        result_df[f'total_{nutrient}_g'] = result_df['amount_to_use (g)'] * result_df[nutrient] / 100
    
    # Display the results in a table
    print(result_df[['Ingredient', 'amount_to_use (g)'] + [f'total_{nutrient}_g' for nutrient in constraints.keys()]])
else:
    print("No valid combinations found.")
