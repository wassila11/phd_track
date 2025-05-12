import pandas as pd
import numpy as np
from scipy.optimize import linprog



df = pd.read_csv("food_db_cleaned.csv")

print(df.count())
L = df['Ingredient'].unique().tolist()



counts = df['Ingredient'].value_counts()
repeated = counts[counts > 1]

print(f"Number of repeated entries: {repeated}")

#no null or repeted values of alim_nom_fr
print(len(list(df.columns)))
print(list(df.columns))
M = [0]*67
print(df.info())



# each row is a food item, and each column is a nutrient

# Let's say you have n food items and you're interested in first 6 nutrients
# and AN1_1 is the 1st nutrient, AN1_2 the 2nd, etc.


# Choose only the relevant nutrients
nutrients = df.iloc[1:, 13:19]  # Example: 6 nutrients (AN1_1 to AN1_6)

nutrients = nutrients.T  # Rows: nutrients, Columns: food items

# ========================
# PART 1: Maximize m1 + m2 + ... + mm
# under:
#     sum(mi * ANi_1) < 50
#     sum(mi * ANi_2) < 20
#     sum(mi * ANi_3) < 10
# ========================

A_ub = nutrients.iloc[0:3].values  # 3 constraints with '<'
b_ub = [50, 20, 10]

c = -np.ones(nutrients.shape[1])  # Maximize sum(m) <=> minimize -sum(m)

# Solve first LP
res1 = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')

print("=== Maximize ===")
print("Status:", res1.message)
print("m (weights):", res1.x)
print("Objective (max m sum):", -res1.fun)

# Save max bounds
max_m = res1.x

# ========================
# PART 2: Minimize m1 + m2 + ... + mm
# under:
#     sum(mi * ANi_4) > 15
#     sum(mi * ANi_5) > 60
#     sum(mi * ANi_6) > 5
# ========================

A_ub = -nutrients.iloc[3:6].values  # negate to convert to <=
b_ub = [-15, -60, -5]

c = np.ones(nutrients.shape[1])  # Minimize sum(m)

# Solve second LP
res2 = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')

print("=== Minimize ===")
print("Status:", res2.message)
print("m (weights):", res2.x)
print("Objective (min m sum):", res2.fun)

# Save min bounds
min_m = res2.x

# ========================
# Valid range filtering
# ========================
valid_indices = []
for i, (mi_min, mi_max) in enumerate(zip(min_m, max_m)):
    if mi_min <= mi_max:
        valid_indices.append(i)

print("Valid indices (food items):", valid_indices)
