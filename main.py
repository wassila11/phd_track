import pandas as pd
import numpy as np
import math
from scipy.optimize import linprog



df = pd.read_csv("phd_track/Table Ciqual 2020_ENG_2020 07 07.csv")




#no null or repeted values of alim_nom_fr
print(len(list(df.columns)))
print(list(df.columns))
M = [0]*67
print(df.info())



# each row is a food item, and each column is a nutrient

# Let's say you have n food items and you're interested in first 6 nutrients
# and AN1_1 is the 1st nutrient, AN1_2 the 2nd, etc.


# Choose only the relevant nutrients
nutrients = df.iloc[0:10, 13:19]  # Example: 6 nutrients (AN1_1 to AN1_6)

nutrients = nutrients.T  # Rows: nutrients, Columns: food items

# ========================
# PART 1: Maximize m1 + m2 + ... + mm
# under:
#     sum(mi * ANi_1) < 50
#     sum(mi * ANi_2) < 20
#     sum(mi * ANi_3) < 10
# ========================

A_ub = nutrients.values
for i in range (len(A_ub)):
    for j in range(len(A_ub[i])):  # 3 constraints with '<'
        if A_ub[i][j] == '-':
            A_ub[i][j] = 0
        if type(A_ub[i][j]) != int and ',' in A_ub[i][j]:
            x = A_ub[i][j].replace(',','.')
            A_ub[i][j] = x
        if type(A_ub[i][j]) != int and '<' in A_ub[i][j]:
            x = A_ub[i][j].replace('<','')
            A_ub[i][j] = x
b_ub = [500.0, 500.0, 500.0 , 100000.0, 100000.0, 100000.0]

c= [11.01321586,6.53594771,6.51890482,5.8685446,6.57894737,8.29187396,6.59630607,6.89655172,6.88705234,5.36480687]
for i in range(len(c)):
    c[i] = -c[i]

# Solve first LP
res1 = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')

print("=== Maximize ===")
print("Status:", res1.message)
print("m (weights):", res1.x)
#print("Objective (max m sum):", -res1.fun)

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
