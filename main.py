import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Create a tiny fake dataset with 3 ingredients and 3 nutrients
data = {
    'Ingredient': ['Apple', 'Banana', 'Carrot'],
    'Protein (g/100g)': [0.3, 1.1, 0.9],
    'Fat (g/100g)': [0.2, 0.3, 0.1],
    'Sugar (g/100g)': [10, 12, 5]
}

df = pd.DataFrame(data)

nutrient_columns = ['Protein (g/100g)', 'Fat (g/100g)', 'Sugar (g/100g)']

print("Available nutrients:", nutrient_columns)

# User input simulation: instead of input(), we hardcode constraints here for test
upper_constraints = {
    'Sugar (g/100g)': 15  # max 15g sugar total
}
lower_constraints = {
    'Protein (g/100g)': 1.0,  # at least 1g protein total
    'Fat (g/100g)': 0.3       # at least 0.3g fat total
}

# Prepare matrices for upper bound constraints: sum(m_i * nutrient_i) <= max_value
A_ub = []
b_ub = []
for nut, val in upper_constraints.items():
    A_ub.append(df[nut].values)
    b_ub.append(val)

# Prepare matrices for lower bound constraints: sum(m_i * nutrient_i) >= min_value
A_ub_lower = []
b_ub_lower = []
for nut, val in lower_constraints.items():
    A_ub_lower.append(-df[nut].values)
    b_ub_lower.append(-val)

# Combine upper and lower constraints
if A_ub and A_ub_lower:
    A_ub = np.vstack([A_ub, A_ub_lower])
    b_ub = np.hstack([b_ub, b_ub_lower])
elif A_ub_lower:
    A_ub = np.array(A_ub_lower)
    b_ub = np.array(b_ub_lower)
else:
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

# Objective: maximize total amount => minimize negative sum
c = -np.ones(len(df))

result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*len(df), method='highs')

if result.success:
    print("Optimal ingredient amounts (per 100g units):")
    for ing, amt in zip(df['Ingredient'], result.x):
        print(f"{ing}: {amt:.2f}")

    print("\nTotal nutrients in the mix:")
    total_protein = sum(result.x * df['Protein (g/100g)'])
    total_fat = sum(result.x * df['Fat (g/100g)'])
    total_sugar = sum(result.x * df['Sugar (g/100g)'])
    print(f"Protein: {total_protein:.2f} g")
    print(f"Fat: {total_fat:.2f} g")
    print(f"Sugar: {total_sugar:.2f} g")
else:
    print("No solution found.")
