from turtle import color
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from scipy.interpolate import interp1d

def tau_dot(r_b0, a):
    Y_p = 0.24
    cut_off = 3.0
    dtau_dt = 2.3e-5 * h * r_b0 * (1 - Y_p/2) / a**2
    if dtau_dt>cut_off:
        dtau_dt = cut_off
    return -dtau_dt

def integrand(a, r, a_d):
    p_dm = 0.0
    p_drad = 0.0
    if a<a_d:
        p_dm = Omega_dm0 * a_d * r**2 / ( (1 + r**2)**0.5 * ( (1 + (a_d*r)**2 )**0.5 - 1) * a**3 )
    else:
        p_dm = Omega_dm0 * ( (1 + a**2*r**2) / (1 + r**2) )**0.5 / a**4
        p_drad = Omega_drad0/a**4

    temp = (Omega_v0 + p_dm + Omega_b0/a**3 + (Omega_nv0 + Omega_g0)/a**4 + p_drad + Omega_c0/a**2 )
    return 1/ (a**2 * temp**0.5)

def boole(X,Y):
    h=X[1]-X[0]
    return 2*h*(7*(Y[0]+Y[-1])+32*np.sum(Y[1:-1:2])+12*np.sum(Y[2:-1:4])+14*np.sum(Y[4:-1:4]))/45.0

def func(a, y, K, r, a_d):
    #print(a)
    #print(y)
    global Omega_nv0,flag_decay
    p_dm, P_dm, p_drad = 0.0, 0.0, 0.0
    if a<a_d:
        p_dm = Omega_dm0 * a_d * r**2 / ( (1 + r**2)**0.5 * ( (1 + (a_d*r)**2 )**0.5 - 1) * a**3 )
    else:
        p_dm = Omega_dm0 * ( (1 + a**2*r**2) / (1 + r**2) )**0.5 / a**4
        P_dm = Omega_dm0 / ( 3 * a**4 * ( (1 + r**2) * (1 + a**2*r**2) )**0.5 )
        if flag_decay==False:
            Omega_nv0 += Omega_drad0
            #print(Omega_nv0)
            flag_decay = True

    temp = (Omega_v0 + p_dm + Omega_b0/a**3 + (Omega_nv0 + Omega_g0)/a**4 + Omega_c0/a**2 )
    R = (3 * a * Omega_b0) / (4*Omega_g0)
    conf_time = quad(integrand, 0, a, args = (r, a_d))[0] #in units of hubble time

    #print( r_g0 * y[7], r_nv0 * y[7+n_g_moments], sigma_h )
    psi = -3.0 * ( Omega_nv0 * y[5+n_dm_moments+n_g_moments] + a**4 * 3.0 * (p_dm + P_dm) * y[3] / (4 + 3*a*a*r*r) ) / ((a * K)**2) - y[0]
    phi_dot = psi/a + ( - (K)**2 * y[0] + 1.5 * a**2 * (p_dm * y[1] + Omega_b0*y[1+n_dm_moments]/a**3 + Omega_g0*y[3+n_dm_moments]/a**4 + Omega_nv0*y[3+n_dm_moments+n_g_moments]/a**4 ) ) / ( 3 * a**3 * temp )
    #print(psi)

    factor = (1 + a*a*r*r)**0.5
    Psi_dm = []
    if a<a_d:
        Psi_dm.append( K * y[2] / (a**2 * temp**0.5) - 3 * phi_dot )
        Psi_dm.append( -y[2]/a - K * psi / ( a**2 * temp**0.5 ) )
        Psi_dm.append( 0.0 )
    else:
        Psi_dm.append( (4 + 3*a*a*r*r) * ( (K) * y[2] / ( 3 * a**2 * temp**0.5 ) - phi_dot ) / factor**2 )
        Psi_dm.append( -(a*r*r) * (2+3*a*a*r*r) * y[2] / (factor**2 * (4 + 3*a*a*r*r)) - K * ( (y[1] - 2*y[3]) / (4 + 3*a*a*r*r) + psi ) / ( a**2 * temp**0.5 ) )
        i = 2
        Psi_dm.append( K * (-y[1+i-1] * 3 * factor/(4+3*a*a*r*r) - (i+1)*y[1+i]/(K*conf_time) ) / ( factor * a**2 * temp**0.5 ) ) 
        #Psi_dm.append( ( (K/(2*i+1)) * (i * y[1+i-1] - (i+1) * y[1+i+1]) ) / ( factor * a**2 * temp**0.5 ) )
    
    delta_b_dot = -3*phi_dot + K*y[1+n_dm_moments+1] / ( a**2 * temp**0.5 )
    
    v_b_dot = 0.0
    f_g = []
    f_g.append(- (K) * y[4+n_dm_moments] / ( a**2 * temp**0.5 ) - 4 * phi_dot)

    if a<1/1101.0:
        v_b_dot = -y[2+n_dm_moments]/a + ( -(K) * psi + c*tau_dot(Omega_b0, a) * (y[2+n_dm_moments] + 3 * y[4+n_dm_moments]/4 )/R ) / ( a**2 * temp**0.5 )
        f_g.append( ( (K/3.0) * (y[3+n_dm_moments] -2*y[5+n_dm_moments] + 4*psi) + c*tau_dot(Omega_b0, a) * (y[4+n_dm_moments] + 4*y[2+n_dm_moments]/3.0) ) / ( a**2 * temp**0.5 ) )
        for i in range(2, n_g_moments):
            if i==(n_g_moments-1):
                f_g.append( ( K * y[3+n_dm_moments+i-1] - (i+1) * y[3+n_dm_moments+i] / conf_time + c*tau_dot(Omega_b0, a)*y[3+n_dm_moments+i] ) / ( a**2 * temp**0.5 ) )
            else:
                f_g.append( ( (K/(2*i+1)) * (i * y[3+n_dm_moments+i-1] - (i+1) * y[3+n_dm_moments+i+1]) + c*tau_dot(Omega_b0, a)*y[3+n_dm_moments+i] ) / ( a**2 * temp**0.5 ) )
    else:
        v_b_dot = -y[2+n_dm_moments]/a + ( -(K) * psi ) / ( a**2 * temp**0.5 )
        f_g.append( ( (K/3.0) * (y[3+n_dm_moments] -2*y[5+n_dm_moments] + 4*psi) ) / ( a**2 * temp**0.5 ) )
        for i in range(2, n_g_moments):
            if i==(n_g_moments-1):
                f_g.append( ( K * y[3+n_dm_moments+i-1] - (i+1) * y[3+n_dm_moments+i] / conf_time ) / ( a**2 * temp**0.5 ) )
            else:
                f_g.append( ( (K/(2*i+1)) * (i * y[3+n_dm_moments+i-1] - (i+1) * y[3+n_dm_moments+i+1]) ) / ( a**2 * temp**0.5 ) )
    
    f_nv = []
    f_nv.append(- (K) * y[4+n_dm_moments+n_g_moments] / ( a**2 * temp**0.5 ) - 4 * phi_dot)
    f_nv.append( ( (K/3.0) * (y[3+n_dm_moments+n_g_moments] -2*y[5+n_dm_moments+n_g_moments] + 4*psi) ) / ( a**2 * temp**0.5 ) )
    for i in range(2, n_nv_moments):
        if i==(n_nv_moments-1):
            f_nv.append( K * (y[3+n_dm_moments+n_g_moments+i-1] - (i+1)*y[3+n_dm_moments+n_g_moments+i]/(K*conf_time) ) / ( a**2 * temp**0.5 ) )
        else:
            f_nv.append( ( (K/(2*i+1)) * (i * y[3+n_dm_moments+n_g_moments+i-1] - (i+1) * y[3+n_dm_moments+n_g_moments+i+1]) ) / ( a**2 * temp**0.5 ) )

    dydt = [phi_dot] + Psi_dm + [delta_b_dot, v_b_dot] + f_g + f_nv
    #print(dydt)
    #input('LOL')                                      
    return dydt

def solve(K, r, a_d):
    a = 10**np.linspace(start,end,1000, endpoint=True)
    conf_time = quad(integrand, 0, a[0], args = (r, a_d))[0] #in units of hubble time
    #print(conf_time)

    #Initial Conditions
    #const = 10**(-9.3)
    #phi = (50 * np.pi**2 * (K)**(n-1)/ (9*(K/c)**3) * (2*np.pi)**3 * const)**0.5
    phi = 1.0
    psi = - phi / (1 + 0.4 * (conf_time/a[0])**2 * Omega_nv0 )
    #print(psi)
    
    y0 = [phi]
    y0 = y0 + [-1.5 * psi, -0.5 * K * conf_time * psi, 0.0] + [0.0 for i in range(n_dm_moments-3)]
    y0 = y0 + [-1.5 * psi, -0.5 * K * conf_time * psi]
    y0 = y0 + [-2 * psi, 2 * K * conf_time * psi / 3.0, 0.0] + [0.0 for i in range(n_g_moments-3)]
    y0 = y0 + [-2 * psi, 2 * K * conf_time * psi / 3.0, 2 * psi * (K * conf_time)**2 / 15.0] + [0.0 for i in range(n_nv_moments-3)]

    sol = solve_ivp(func, [a[0], a[-1]], y0, t_eval = a, args = (K, r, a_d), method='RK45')
    y = sol.y

    f = Omega_dm0 * ( ((1 + a**2*r**2)/(1+r**2))**0.5 - 2 / ((1+r**2*a**2)*(1+r**2))**0.5 ) + Omega_b0 * a - (Omega_g0 + Omega_nv0)
    a_mr = interp1d(f, a, kind = 'cubic')
    a_mr = a_mr(0.0)
    #print(a_mr)
    #v_cdm = y[2] * 1.5 * x_d * ( (x_d/2.0)**2 + (a*m_L_cdm/T0)**2 )**0.5 / ( (x_d)**2 + 3*(a*m_L_cdm/T0)**2 )
    #y = np.concatenate((y[:2], v_cdm.reshape(1, len(v_cdm)), y[ 1+n_cdm_moments : 3+n_cdm_moments + n_g_moments + n_nv_moments], Y0.reshape(1, len(Y0)), Y1.reshape(1, len(Y1)), Y2.reshape(1, len(Y2)) ), axis = 0)

    return np.log10(a), y[1], a_mr

c = 2997.92458  #100 km s^-1
h = 0.6871
rho_crit = 8.098e-11 * h**2   #eV^4 / (h_cross c)^3
flag_decay = False

#Photons
T0 = 2.34865e-4   #eV / Kb
Omega_g0 = np.pi**2 * T0**4 / (15 * rho_crit)
n_g_moments = 3

#CDM
Omega_dm0 = 0.255
r = [1e3, 1e4, 1e5, 1e6]
a_d = [1e-10]
n_dm_moments = 3

#Baryons
Omega_b0 = 0.045

Omega_m0 = Omega_dm0 + Omega_b0

#Massless neutrinos
T_nv0 = (4/11.0)**(1/3.0) * T0
Omega_nv0 = 7 * np.pi**2 * T_nv0**4 / (40 * rho_crit)
n_nv_moments = 5

#Decay radiation
Omega_drad0 = Omega_dm0/(1 + r[0]**2)**0.5

Omega_r0 = Omega_g0 + Omega_nv0 + Omega_drad0

Omega_c0 = 0.0#1 - r_v0 - r_m0 - r_r0

#DE
Omega_v0 = 1 + Omega_c0 - Omega_m0 - Omega_r0

a_recomb = 1/1101.0
n = 0.9660499 #spectral index of initial power spectrum

start, end = -8, -1

k = 0.6
#k = np.array([0.1, 0.01, 0.001, 0.0001])

fig, ax = pl.subplots(nrows=1, ncols=1)

ymax, ymin = 0.0, 0.0
#for i in range(len(k)):
for j in range(len(r)):
    for i in range(len(a_d)):
        print(i, j)
        #print(i)
        #df_cdm = pd.read_csv(r'D:\Education\BS-MS\Curriculum\Year_5\Thesis_Shiv_K_Sethi_RRI\Codes and Plots\Codework_7_WIMP_as_DarkMatter\k_' + str(k[i]) + r'_data.csv')
        #y_cdm = np.log10( np.abs(df_cdm['cdm overdensity']) )
        flag_decay = False
        Omega_nv0 = 7 * np.pi**2 * T_nv0**4 / (40 * rho_crit)
        Omega_drad0 = Omega_dm0/(1 + r[j]**2)**0.5
        Omega_r0 = Omega_g0 + Omega_nv0 + Omega_drad0
        
        x, y_wdm, a_eq = solve(c*k, r[j], a_d[i])
        y_wdm = np.log10(np.abs(y_wdm))
        index = y_wdm>-3
        x,y_wdm = x[index], y_wdm[index]

        #ax.plot(df_cdm['a'], y_cdm, '--', label = 'CDM, k = '+str(k[i])+"h Mpc$^{-1}$")
        #ax.plot(x, y_wdm, label = "WDM, $m_L$ = "+str(np.round(m_L_cdm/10**6,5))+" MeV, k = "+str(k[i])+"h Mpc$^{-1}$")
        ax.plot(x, y_wdm, label = 'WDM ($r = 10^{' + str( np.round(np.log10(r[j]), 2) ) + '}$, $a_d = 10^{' + str( np.round(np.log10(a_d[i]), 2) ) + '}$), b, $\gamma$, $\\nu$')

        #ymax = np.max(np.array([ymax, np.max(y_cdm), np.max(y_hdm)]))
        #ymin = np.min(np.array([ymin, np.min(y_cdm), np.min(y_hdm)]))
        ymax = np.max(np.array([ymax, np.max(y_wdm)]))
        ymin = np.min(np.array([ymin, np.min(y_wdm)]))

df_cdm = pd.read_csv(r'D:\Education\BS-MS\Curriculum\Year_5\Thesis_Shiv_K_Sethi_RRI\Codes and Plots\Codework_7_WIMP_as_DarkMatter\cdm_overdensity_k_' + str(k) + r'.csv')
y_cdm = np.log10( np.abs(df_cdm['overdensity']) )
ax.plot(df_cdm['a'], y_cdm, '--', label = 'CDM, b, $\gamma$, $\\nu$', color = 'black')
#ymax = np.max(np.array([ymax, np.max(y_cdm)]))
#ymin = np.min(np.array([ymin, np.min(y_cdm)]))

ax.plot(np.log10(np.array([a_eq, a_eq])), [ymax, ymin], '--')
ax.annotate('Radiation - matter equality', (np.log10(a_eq), ymax), textcoords = 'offset pixels', xytext = (5,-190), rotation = 90)
ax.plot(np.log10(np.array([a_recomb, a_recomb])), [ymax, ymin], '--')
ax.annotate('Last Scattering Surface', (np.log10(a_recomb), ymax), textcoords = 'offset pixels', xytext = (5,-170), rotation = 90)

x_ticks = np.array(range(int(np.ceil(x[0])), int(np.floor(x[-1]))+1, 1))
x_ticklabels = 10.0**x_ticks
ax.set(xlabel = 'Scale factor a', ylabel = 'log$_{10} |\delta_{dm}|$', title = "WIMP decay WDM overdensity at $k = "+str(k)+"$ h Mpc$^{-1}$",
        xticks = x_ticks, xticklabels = x_ticklabels,
        xlim = (start, end))
ax.legend()
ax.grid()
pl.show()
