#!/usr/bin/env python3
import numpy as np
import random, os, sys

def str2bool(v):
  return v.lower() in ("true", "1", "linear")

q = 200 # numebr of rows
n = 20 # number of columns (e.g., features)
m = 5 # number of needed columns (e.g., features). m may be unknown and need to be optimized as part of the search

linear = True
if(len(sys.argv) > 1):
    linear = str2bool(sys.argv[1]) 

A = np.random.rand(q,n) 
b = np.zeros(q)

col_list = random.sample(range(n), m)

x = np.random.rand(n, 1)
# print(x)

for row in range(q):
    b[row] = 0 
    for col in col_list:
        if linear:
            b[row] += A[row, col] *  x[col] # here uses linear model to start with, but real model may be non-linear b = A(x)
        else:
            b[row] += A[row, col] **  x[col] # here shows an example of non-linear, yet non-linear has many other kinds
    b[row] = round(b[row]/len(col_list)*100) 

print("\nA:\n", A, "\nx:\n", x, "\ncol_list:\n", col_list, "\nb:\n", b)

print("Is the generated label based on linear model of features: ", linear)
