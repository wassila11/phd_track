import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Load your full food database
df = pd.read_csv("food_db_cleaned.csv")

# Dynamically detect nutrient columns (numeric columns only)
nutrient_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove columns that are NOT nutrients
non_nutrient_cols = ['Main category', 'Subcategory', 'Detailed category']  # adjust as needed
nutrient_columns = [col for col in nutrient_columns if col not in non_nutrient_cols]

print("Available nutrients:")
for nut in nutrient_columns:
    print(f" - {nut}")

# Collect user input for constraints
print("\nEnter upper bound constraints (nutrient name and max value). Type 'done' when finished.")
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
    except ValueError:
        print("Invalid number, try again.")
        continue
    upper_constraints[nut] = val

print("\nEnter lower bound constraints (nutrient name and min value). Type 'done' when finished.")
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
    except ValueError:
        print("Invalid number, try again.")
        continue
    lower_constraints[nut] = val


# Prepare matrices for constraints
A_ub = []
b_ub = []

# Upper bound constraints (<=)
for nut, val in upper_constraints.items():
    A_ub.append(df[nut].values)
    b_ub.append(val)

# Lower bound constraints (>=) converted to <= by multiplying by -1
A_ub_lower = []
b_ub_lower = []
for nut, val in lower_constraints.items():
    A_ub_lower.append(-df[nut].values)
    b_ub_lower.append(-val)

# Combine all constraints
if A_ub and A_ub_lower:
    A_ub = np.vstack([A_ub, A_ub_lower])
    b_ub = np.hstack([b_ub, b_ub_lower])
elif A_ub_lower:
    A_ub = np.array(A_ub_lower)
    b_ub = np.array(b_ub_lower)
else:
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)


c = np.ones(len(df))

# Solve LP using HiGHS
result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * len(df), method='highs')

# Show results

if result.success:
    # Convert amounts from per 100g units to grams
    amounts_g = result.x * 100
    
    # Round amounts to nearest gram (or any precision you want)
    rounded_amounts = np.round(amounts_g)
    
    print("\nOptimal ingredient amounts (rounded to nearest gram):")
    for ing, amt in zip(df['Ingredient'], rounded_amounts):
        if amt > 0:
            print(f"{ing}: {int(amt)} g")
    
    # Calculate and display nutrient totals with rounded amounts
    print("\nTotal nutrients in the mix (rounded):")
    for nut in set(list(upper_constraints.keys()) + list(lower_constraints.keys())):
        total = sum(rounded_amounts / 100 * df[nut])  # divide by 100 because nutrients per 100g
        print(f"{nut}: {total:.2f}")
else:
    print("No solution found.")

