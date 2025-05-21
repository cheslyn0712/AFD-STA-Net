import numpy as np
import os
import pandas as pd


def produce(L, M, t_gap,t_start, N, t_span):

    x = np.arange(1, N+1)
    u = np.cos(x / 16)
    v = np.fft.fft(u)
    h = t_gap
    a=-1
    b=1
    k = np.concatenate((np.arange(0, N/2), [0], np.arange(-N/2+1, 0))) / 16
    L_ = k**2 - k**4
    E = np.exp(h * L_)
    E_2 = np.exp(h * L_ / 2)
    M_1 = 16
    r = np.exp(1j * np.pi * (np.arange(1, M_1+1) - 0.5) / M_1)
    LR = h * np.tile(L_, (M_1, 1)).T + np.tile(r, (N, 1))
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    
        

    g = -0.5j*k
    for i in range(1,int(t_start/t_gap)):
        Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = E_2*v + Q*Nv
        Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E_2*v + Q*Na
        Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E_2*a + Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    
    data_list = np.empty((0,N))
    for i in range(1, M+L+t_span*L+1):
        Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = E_2*v + Q*Nv
        Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E_2*v + Q*Na
        Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E_2*a + Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        u = np.real(np.fft.ifft(v))
        data_list = np.append(data_list, np.array([u]), axis=0)
    
    data = data_list.T
    return data

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

    


    



