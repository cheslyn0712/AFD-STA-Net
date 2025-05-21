import numpy as np
import os
import pandas as pd
from numpy.fft import fft, ifft

def produce(L_time, M, t_gap, t_start, N, t_span):

    Lx = 4 * np.pi                    
    r = 0.1                           
    b_2 = 0.1                         
    theta = 0.5                       
    nsave = 10                         
    
    x = (Lx / N) * np.arange(N)       
    u = np.cos(x)                     
    
    kx = np.zeros(N, dtype=int)
    half_N = N // 2
    kx[0:half_N] = np.arange(0, half_N)
    kx[half_N+1:] = np.arange(-half_N+1, 0)
    alpha = 2 * np.pi * kx / Lx       
    
    bf_c = r - 1.0
    L = bf_c + 2 * alpha**2 - alpha**4  
    
    dt = t_gap  
    A = 1 - (1 - theta) * dt * L
    A_inv = 1.0 / A
    B = 1 + theta * dt * L
    
    v = fft(u) 
    N1 = b_2 * fft(np.real(ifft(v))**2 - fft(np.real(ifft(v))**3))
    N0 = N1.copy()
    
    warmup_steps = int(t_start / dt)
    for _ in range(warmup_steps):
        
        N0 = N1
        
        N1 = b_2 * fft((np.real(ifft( v )))*(np.real(ifft( v )))) - fft((np.real(ifft( v )))**3)
        v = A_inv * (B * v + 1.5*dt*N1 - 0.5*dt*N0)
        
    
    U = np.empty((1,N))
    U[0] = u                        

    for _ in range(1, M+L_time+t_span*L_time+1):   
        N0 = N1
        N1 = b_2 * fft((np.real(ifft( v )))*(np.real(ifft( v )))) - fft((np.real(ifft( v )))**3)
        v = A_inv * (B * v + 1.5*dt*N1 - 0.5*dt*N0)
        u_phys = np.real(ifft(v))

        U=np.append(U,np.array([u_phys]),axis=0)
    
    return U.T

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

    


    



