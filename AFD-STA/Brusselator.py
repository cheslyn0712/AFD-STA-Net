import numpy as np
import os
import pandas as pd

def f(x, t):
    return (np.pi**2 + 2) * np.exp(-t) * np.sin(np.pi * x) - x * (1 - x) * np.exp(-3 * t) * np.sin(np.pi * x)**2

def g(x, t):
    return x * (x - 1) * np.exp(-t) - 2 * np.exp(-t) * np.sin(np.pi * x) + x * (1 - x) * np.exp(-3 * t) * np.sin(np.pi * x) + 2 * np.exp(-t)

def initialv(x):
    return x * (1 - x)

def initialu(x):
    return np.sin(np.pi * x)


def produce(L, M, t_gap,t_start, N, t_span):

    a = 1
    b = 2
    c = 0

    xrange = 1
    trange = 1
    x = np.linspace(0, xrange, N)
    dx = x[1] - x[0]

    u = np.zeros(N)
    v = np.zeros(N)
    uu = []
    vv = []
    data_list = []

    
    for j in range(1,N):
        u[j] = initialu(x[j])
        v[j] = initialv(x[j])

    u[0] = 0
    u[-1] = 0
    v[0] = 0
    v[-1] = 0

    for step in range(1,int(t_start/t_gap)):
        t = step*t_gap
        utemp = u.copy()
        for j in range(1, N-1):
            u[j] = (c + u[j]**2 * v[j] - (b + 1) * u[j] + a * (u[j + 1] - 2 * u[j] + u[j - 1]) / dx**2 + f(x[j], t)) * t_gap + u[j]
            v[j] = (b * utemp[j] - utemp[j]**2 * v[j] + a * (v[j + 1] - 2 * v[j] + v[j - 1]) / dx**2 + g(x[j], t)) * t_gap + v[j]    


    for step in range(1,M + L + t_span * L + 1):
        t = t_start + step * t_gap
        uu.append(u)
        vv.append(v)
        utemp = u.copy()
        for j in range(1, N-1):
            u[j] = (c + u[j]**2 * v[j] - (b + 1) * u[j] + a * (u[j + 1] - 2 * u[j] + u[j - 1]) / dx**2 + f(x[j], t)) * t_gap + u[j]
            v[j] = (b * utemp[j] - utemp[j]**2 * v[j] + a * (v[j + 1] - 2 * v[j] + v[j - 1]) / dx**2 + g(x[j], t)) * t_gap + v[j]

    uu = np.array(uu).T
    vv = np.array(vv).T
    data_list = np.vstack((uu, vv))
    return data_list

def process_train_data(data,t_span,L,M):
    train_data=[]
    for i in range(0,t_span+1):
        start_time=(i*L)
        train_data.append(data[:,start_time:start_time+M])
    return train_data
    
def process_label_data(data,t_span,L,M,N):
    label_data=[]
    for i in range(N):
        single_label=[]
        for j in range(t_span+1):
            start_time=(j*L)
            current_label=data[i,start_time:start_time+L+M]
            single_label.append(current_label)
        label_data.append(single_label)
    return label_data

    
def integration(L, M, t_gap,t_start, N, t_span):

    raw_data = produce(L, M, t_gap,t_start, N, t_span)  

    data=process_train_data(raw_data,t_span,L,M)
    label=process_label_data(raw_data,t_span,L,M,N)
    data=np.array(data)
    label=np.array(label)

    return data,label
