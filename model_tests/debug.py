import numpy as np 
import matplotlib.pyplot as plt
import pickle
import torch

n = 30 # num of series 
T = 20000 # len of series

data = np.zeros((n,T))
data_params = np.zeros((n,3))
for series in range(n):
    t0 = np.random.randint(50,101)
    w = np.random.randint(40,51) 
    scale_factor = 0.3
    ran_seed = np.random.randint(0,2)
    tfun = np.sin if ran_seed else np.cos
    data_params[series] = np.array([t0, w, ran_seed])
    for t in range(T):
        bias = np.random.randn()
        data[series, t] = tfun((t -t0)/w) + scale_factor*bias

errors_num = 4
errors = []
for i in range(errors_num):
    t0 = np.random.randint(50,101)
    w = np.random.randint(40,51) 
    scale_factor = 1
    ran_seed = np.random.randint(0,2)
    tfun = np.sin if ran_seed else np.cos
    len_of_er = np.random.randint(30,91)
    
    err = []
    for t in range(len_of_er):
        bias = np.random.randn()
        err.append(tfun((t -t0)/w) + scale_factor*bias)

    errors.append(err)

errors = np.array(errors, dtype=object)

n_er_inos = 3
ser_w_er = []
er_start = np.zeros(errors_num, dtype=int)
er_end = np.zeros(errors_num, dtype=int)

for i, error in enumerate(errors):
    er_start[i] = np.random.randint(10001, T-90)
    er_end[i] = er_start[i] + len(error)

er_sort_ind = np.argsort(er_start)
er_start = er_start[er_sort_ind]
er_end = er_end[er_sort_ind]
errors = errors[er_sort_ind]

for er_i in range(n_er_inos):
    ser_num = np.random.randint(0,n)
    while ser_num in ser_w_er:
        ser_num = np.random.randint(0,n)
    ser_w_er.append(ser_num)

data_w_er = np.copy(data)
er_pos = dict()
dil = [2,4,8,16]
for i in range(n_er_inos):
    for j in range(errors_num):
        data_w_er[ser_w_er[i], er_start[j]:er_end[j]:dil[j]] = errors[j][::dil[j]]
        if ser_w_er[i] in er_pos:
            er_pos[ser_w_er[i]].append([er_start[j], er_end[j]])
        else:
            er_pos[ser_w_er[i]] = [[er_start[j], er_end[j]]]