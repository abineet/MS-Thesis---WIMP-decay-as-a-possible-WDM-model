import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import pandas as pd

def func(y, x, m_X, cross):
    #print(x)
    #print(y)
    const = 3 * 5**0.5 * g_X / (2**2.5 * np.pi**3 * g_net**0.5 * np.exp(1) * np.log(10))
    dim = m_X * cross / G**0.5
    dy_dt = const * dim * ( x**3 * np.exp(2*(1-x)) * 10**(-y[0]) - 10**y[0] ) / x**2
    #print(dy_dt)
    #input('lol')
    return [dy_dt]

h = 0.68
c = 3e8
h_cross = 6.626e-34 / (2*np.pi)
GeV = 1.602e-10
T0 = 2.348e-13 #GeV
p_crit = 8.098*h**2 * 10**(-47) #GeV^4 / (h_cross c)^3
G = 6.67e-11 * GeV**2 / (h_cross * c**5) #h_cross c^5 / GeV^2
g_X = 2
g_net = 91.5
m_X = 100 #GeV

start, end = 0.0, 3.0
x = 10**np.linspace(start,end,1000, endpoint=True)
'''
cross = 10**np.linspace(-38, -32, 100)
N = np.array([])
x_f = np.array([])
sigma = np.array([])
for i in range(len(cross)):
    y_eq = (1-x)/np.log(10) + 1.5 * np.log10(x)
    y = odeint(func, y0 = [y_eq[0]], t = x, args = (m_X, cross[i] * GeV**2 / (h_cross**2 * c**3)))
    f = interp1d(y.reshape(len(y)), x)
    x_f = np.append(x_f, f(y[-1] + 0.1))
    N = np.append(N, g_X * T0**3 * 10**y[-1][0] / ( (2*np.pi)**1.5 * np.exp(1)) )
    sigma = np.append(sigma, cross[i] * np.sqrt(x_f[-1]/2.0) / c )
df = pd.DataFrame({"Thermal Cross section":cross, "x_f":x_f, "Cross section":sigma, "Relic abundance":N})
df.to_csv("data.csv", index = False)
'''
cross = np.array([-26.0, -28.0, -30.0]) #sigma_v in cm^3 / s
y_eq = (1-x)/np.log(10) + 1.5 * np.log10(x)
fig, ax = pl.subplots(nrows=1, ncols=1)
ax.plot(np.log10(x), y_eq, '--', label = 'Equilibirum')
for i in range(len(cross)):
    temp = 10**cross[i] *1e-6 * GeV**2 / (h_cross**2 * c**3) #h_cross^2 c^3 / GeV^2
    y = odeint(func, y0 = [y_eq[0]], t = x, args = (m_X, temp))
    #sol = solve_ivp(func, [x[0], x[-1]], [y_eq[0]], t_eval = x, args=(m_X, cross), method = 'RK45')
    #y = sol.y
    ax.plot(np.log10(x), y, label = '$\\langle \sigma v \\rangle = 1e$'+str(int(cross[i]))+' cm$^3$ s$^{-1}$' )
    N = g_X * T0**3 * 10**y[-1][0] / ( (2*np.pi)**1.5 * np.exp(1))
    omega = m_X * N / ( p_crit )
    m_L = 0.255 * p_crit / N
    a_d = 10**(-5) * 2 * 0.001 * m_L/m_X
    x_d = a_d * m_X/T0
    print(N, omega, m_L, a_d, x_d)

x_ticks = np.array(range(int(start), int(end)+1, 1))
x_ticklabels = 10.0**x_ticks
y_ticks = np.array(range(-20, 1, 5))
y_ticklabels = 10.0**y_ticks
ax.set(xlabel = 'Epoch: log$_{10}$ $x$ ; $x = a m_H / T_0$', ylabel = 'Comoving no. density : log$_{10}$ $N_x/N^{(0)}_{x=1}$', title = 'Relic abundance for $m_H = '+str(m_X)+'$ GeV',
       #xticks = x_ticks, xticklabels = x_ticklabels,
       ylim = (-20, 1))
       #yticks = y_ticks, yticklabels = y_ticklabels)
ax.legend()
ax.grid()
pl.show()
