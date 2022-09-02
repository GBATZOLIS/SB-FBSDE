import torch
import numpy as np
import matplotlib.pyplot as plt

def f1():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])

    grid_x, grid_y = torch.meshgrid(x, y)

    def create_mesh_points(x,y):
        z = []
        for i in range(x.size(0)):
            for j in range(y.size(0)):
                z.append([x[i], y[j]])
        return torch.tensor(z)

    print(create_mesh_points(x,y))

def f3(t0, T, log_snr_max, log_snr_min, sigma_max, alpha_max):
    log_snr_max, log_snr_min = torch.tensor(log_snr_max), torch.tensor(log_snr_min)
    sigma_max, alpha_max = torch.tensor(sigma_max), torch.tensor(alpha_max)

    s0 = torch.exp(log_snr_max)
    sT = torch.exp(log_snr_min)

    a0 = (torch.sqrt(sT)*sigma_max - alpha_max)/(T-t0)
    b0 = alpha_max - a0 * t0

    c0 = (sigma_max - alpha_max/torch.sqrt(s0))/(T-t0)
    d0 = sigma_max - c0 * T

    def alpha_fn(s):
        t = t_fn(s)
        return a0*t+b0

    def sigma_fn(s):
        t = t_fn(s)
        return c0*t+d0
    
    def t_fn(s):
        return (b0 - torch.sqrt(s)*d0)/(torch.sqrt(s)*c0 - a0)

    def snr_fn(s):
        return alpha_fn(s)**2/sigma_fn(s)**2
    
    logsnr = torch.linspace(log_snr_min, log_snr_max, 100)
    snr = torch.exp(logsnr)
    plt.figure()
    plt.plot(snr, alpha_fn(snr), label='alpha')
    plt.plot(snr, sigma_fn(snr), label='sigma')
    #plt.plot(snr, snr_fn(snr), label='snr')
    plt.xscale("log")
    plt.legend()
    plt.savefig('/data/Georgios/code/debug/new_prior.png')

def f2(t0, T, log_snr_max, log_snr_min, sigma_max):
    t0 = 1e-2
    T = 1
    sigma_max = 8.4
    s_0 = log_snr_max = 10
    s_T = log_snr_min = -10
    a = (s_0 - s_T)/(t0 - T)
    b = s_0 - t0 * (s_0 - s_T)/(t0 - T)
    c = (sigma_max**2-1)/(T-t0)
    d = 1 - c * t0

    def snr_fn(t):
        t=torch.tensor(t)
        return torch.exp(a*t+b)
    
    def t_fn(snr):
        return (torch.log(snr)-b)/a

    def h_fn(t):
        return c*t+d
        
    def sigma_fn(snr):
        return 1/torch.sqrt(snr)*alpha_fn(snr)
        

    a_T = sigma_max * torch.sqrt(snr_fn(T)/(1+snr_fn(T)))
    a_0 = 0.9999
    w = (a_T-a_0)/(T-t0)
    r = a_0 - w * t0

    def alpha_fn(snr):
        return w*t_fn(snr)+r

    ts = torch.linspace(t0, T, 100)
    snrs = snr_fn(ts)
    alphas = alpha_fn(snrs)
    sigmas = sigma_fn(snrs)

    plt.figure()
    plt.plot(snrs, alphas, label='alpha(snr)')
    plt.plot(snrs, sigmas, label='sigma(snr)')
    plt.xscale("log")
    plt.legend()
    plt.savefig('/home/ma-user/debug/new_prior.png')

f3(t0=1e-2, T=1, log_snr_max=10, log_snr_min=-10, sigma_max=6, alpha_max=1.)