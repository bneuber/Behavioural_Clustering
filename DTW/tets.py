# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 16:55:15 2025

@author: Bastian
"""

import numpy as np

# Beispiel-Kostenmatrix
cost = np.random.rand(5, 6)  # deine Matrix der Größe (n, m)

n, m = cost.shape
accum_cost = np.zeros_like(cost)

# Initialisiere Startpunkt
accum_cost[0, 0] = cost[0, 0]

# Erste Zeile
for i in range(1, n):
    accum_cost[i, 0] = cost[i, 0] + accum_cost[i - 1, 0]

# Erste Spalte
for j in range(1, m):
    accum_cost[0, j] = cost[0, j] + accum_cost[0, j - 1]

# Fülle den Rest
for i in range(1, n):
    for j in range(1, m):
        accum_cost[i, j] = cost[i, j] + min(
            accum_cost[i - 1, j],    # von oben
            accum_cost[i, j - 1],    # von links
            accum_cost[i - 1, j - 1] # diagonal
        )

print("Akkumulierte Kostenmatrix:")
print(accum_cost)
