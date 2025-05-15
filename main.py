import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, PULP_CBC_CMD

# Load your full dataset
df = pd.read_csv("food_db_cleaned.csv")



# Identify nutrient columns dynamically
nutrient_columns = df.select_dtypes(include=['number']).columns.tolist()


# Remove non-nutrient columns if present (adjust these as needed)
non_nutrient_cols = ['Ingredient', 'Main category', 'Subcategory', 'Detailed category']
nutrient_columns = [col for col in nutrient_columns if col not in non_nutrient_cols]


# Replace NaN or infinite values in nutrient columns with 0 (or any appropriate fill)
df[nutrient_columns] = df[nutrient_columns].fillna(0)
df[nutrient_columns] = df[nutrient_columns].replace([float('inf'), -float('inf')], 0)

print("Available nutrients:")
for col in nutrient_columns:
    print(f" - {col}")

# Get user input for constraints
print("\nEnter upper bound constraints (nutrient name and max value). Type 'done' to finish.")
upper_constraints = {}
while True:
    nut = input("Nutrient for upper bound (or 'done'): ").strip()
    if nut.lower() == 'done':
        break
    if nut not in nutrient_columns:
        print("Invalid nutrient name, try again.")
        continue
    val = input(f"Max allowed for {nut}: ").strip()
    try:
        val = float(val)
    except:
        print("Invalid number, try again.")
        continue
    upper_constraints[nut] = val

print("\nEnter lower bound constraints (nutrient name and min value). Type 'done' to finish.")
lower_constraints = {}
while True:
    nut = input("Nutrient for lower bound (or 'done'): ").strip()
    if nut.lower() == 'done':
        break
    if nut not in nutrient_columns:
        print("Invalid nutrient name, try again.")
        continue
    val = input(f"Min required for {nut}: ").strip()
    try:
        val = float(val)
    except:
        print("Invalid number, try again.")
        continue
    lower_constraints[nut] = val

# Get max ingredients limit
while True:
    max_ing_str = input("\nEnter maximum number of ingredients to use (integer): ").strip()
    try:
        max_ingredients = int(max_ing_str)
        if max_ingredients <= 0:
            print("Must be positive integer.")
            continue
        break
    except:
        print("Invalid integer, try again.")

print(f"\nSolving with max {max_ingredients} ingredients...")

# Big M value (large number for linking usage and amount)
M = 10000

# Define MILP problem (maximize total amount of ingredients)
prob = LpProblem("Ingredient_Selection", LpMaximize)

# Variables
amount_vars = {ing: LpVariable(f"amt_{ing}", lowBound=0) for ing in df['Ingredient']}
use_vars = {ing: LpVariable(f"use_{ing}", cat=LpBinary) for ing in df['Ingredient']}

# Objective: maximize total amount
prob += lpSum(amount_vars.values())

# Link amount and usage
for ing in df['Ingredient']:
    prob += amount_vars[ing] <= M * use_vars[ing]

# Ingredient count limit
prob += lpSum(use_vars.values()) <= max_ingredients

# Nutrient constraints upper bounds
for nut, val in upper_constraints.items():
    prob += lpSum(amount_vars[ing] * df.loc[df['Ingredient'] == ing, nut].values[0]/100
                  for ing in df['Ingredient']) <= val

# Nutrient constraints lower bounds
for nut, val in lower_constraints.items():
    prob += lpSum(amount_vars[ing] * df.loc[df['Ingredient'] == ing, nut].values[0]/100
                  for ing in df['Ingredient']) >= val

# Solve MILP
prob.solve(PULP_CBC_CMD(msg=1))  # Set msg=1 to see solver output, 0 to hide

# Print solution
if prob.status == 1:  # Optimal
    print("\nOptimal ingredient amounts (grams):")
    for ing in df['Ingredient']:
        amt = amount_vars[ing].varValue
        used = use_vars[ing].varValue
        if used > 0.5:
            print(f"{ing}: {amt:.2f} g")
else:
    print("No optimal solution found with given constraints.")
