import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from scipy.integrate import solve_ivp
from scipy import interpolate
from scipy.integrate import quad
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons
from scipy.special import spherical_jn

def tau_dot(r_b0, a):
    Y_p = 0.24
    cut_off = 3.0
    dtau_dt = 2.3e-5 * h * r_b0 * (1 - Y_p/2) / a**2
    if dtau_dt>cut_off:
        dtau_dt = cut_off
    return -dtau_dt

def integrand(a, r_v0, r_m0, r_r0, r_c0):
    E = (q**2 + (a*m_v/T_nv0)**2)**0.5
    p_h = T_nv0**4 * (boole(q, q**2 * E * f0)) / (np.pi**2 * a**4)
    temp = (r_v0 + r_m0/a**3 + r_r0/a**4 + r_c0/a**2 + p_h / p_crit )
    return 1/ (a**2 * temp**0.5)

def boole(X,Y):
    h=X[1]-X[0]
    return 2*h*(7*(Y[0]+Y[-1])+32*np.sum(Y[1:-1:2])+12*np.sum(Y[2:-1:4])+14*np.sum(Y[4:-1:4]))/45.0

def func(a, y, K, r_v0, r_m0, r_cdm0, r_b0, r_r0, r_g0, r_nv0, r_c0):
    #print(a)
    #print(y)
    E = (q**2 + (a*m_v/T_nv0)**2)**0.5
    p_h = T_nv0**4 * (boole(q, q**2 * E * f0)) / (np.pi**2 * a**4) #eV^4 / (h_cross c)^3
    temp = (r_v0 + r_m0/a**3 + r_r0/a**4 + r_c0/a**2 + p_h / p_crit)
    #print(p_h / p_crit)
    #print(temp)
    R = (3 * a * r_b0) / (4*r_g0)
    
    conf_time = quad(integrand, 0, a, args=(r_v0, r_m0, r_r0, r_c0))[0] #in units of hubble time
    
    int = boole(q, q**2 * E * f0 * np.array(y[5+n_g_moments+n_nv_moments : 5+n_g_moments+n_nv_moments+len(q)]) )
    delta_h = T_nv0**4 * int / (np.pi**2 * a**4 * p_crit)

    int = boole(q, q**4 * f0 * np.array(y[5+n_g_moments+n_nv_moments+2*len(q) : 5+n_g_moments+n_nv_moments+3*len(q)]) / E )
    sigma_h = T_nv0**4 * int / (np.pi**2 * p_crit)
    #print(delta_h, sigma_h)

    #print( r_g0 * y[7], r_nv0 * y[7+n_g_moments], sigma_h )
    psi = -3.0 * ( r_nv0 * y[7+n_g_moments] + sigma_h) / ((a * K)**2) - y[0]
    phi_dot = psi/a + ( - (K)**2 * y[0] + 1.5 * a**2 * (r_cdm0*y[1]/a**3 + r_b0*y[3]/a**3 + r_g0*y[5]/a**4 + r_nv0*y[5+n_g_moments]/a**4 + delta_h ) ) / ( 3 * a**3 * temp )
    #print(psi)
    delta_cdm_dot = -3*phi_dot + K*y[2] / ( a**2 * temp**0.5 )
    v_cdm_dot = -y[2]/a - (K) * psi / ( a**2 * temp**0.5 )
    
    delta_b_dot = -3*phi_dot + K*y[4] / ( a**2 * temp**0.5 )
    
    v_b_dot = 0.0
    f_g = []
    f_g.append(- (K) * y[6] / ( a**2 * temp**0.5 ) - 4 * phi_dot)

    if a<1/1101.0:
        v_b_dot = -y[4]/a + ( -(K) * psi + c*tau_dot(r_b0, a) * (y[4] + 3 * y[6]/4 )/R ) / ( a**2 * temp**0.5 )
        f_g.append( ( (K/3.0) * (y[5] -2*y[7] + 4*psi) + c*tau_dot(r_b0, a) * (y[6] + 4*y[4]/3.0) ) / ( a**2 * temp**0.5 ) )
        for i in range(2, n_g_moments):
            if i==(n_g_moments-1):
                f_g.append( ( K * y[5+i-1] - (i+1) * y[5+i] / conf_time + c*tau_dot(r_b0, a)*y[5+i] ) / ( a**2 * temp**0.5 ) )
            else:
                f_g.append( ( (K/(2*i+1)) * (i * y[5+i-1] - (i+1) * y[5+i+1]) + c*tau_dot(r_b0, a)*y[5+i] ) / ( a**2 * temp**0.5 ) )
    else:
        v_b_dot = -y[4]/a + ( -(K) * psi ) / ( a**2 * temp**0.5 )
        f_g.append( ( (K/3.0) * (y[5] -2*y[7] + 4*psi) ) / ( a**2 * temp**0.5 ) )
        for i in range(2, n_g_moments):
            if i==(n_g_moments-1):
                f_g.append( ( K * y[5+i-1] - (i+1) * y[5+i] / conf_time ) / ( a**2 * temp**0.5 ) )
            else:
                f_g.append( ( (K/(2*i+1)) * (i * y[5+i-1] - (i+1) * y[5+i+1]) ) / ( a**2 * temp**0.5 ) )
    
    f_nv = []
    f_nv.append(- (K) * y[6+n_g_moments] / ( a**2 * temp**0.5 ) - 4 * phi_dot)
    f_nv.append( ( (K/3.0) * (y[5+n_g_moments] -2*y[7+n_g_moments] + 4*psi) ) / ( a**2 * temp**0.5 ) )
    for i in range(2, n_nv_moments):
        if i==(n_nv_moments-1):
            f_nv.append( K * (y[5+n_g_moments+i-1] - (i+1)*y[5+n_g_moments+i]/(K*conf_time) ) / ( a**2 * temp**0.5 ) )
            #f_nv.append( ( (K/(2*i+1)) * (i * y[5+n_g_moments+i-1] - (i+1) * 2 * spherical_jn(i+1, K*conf_time) / (1 + 0.4*r_nv0/r_r0) ) ) / ( a**2 * temp**0.5 ) )
        else:
            f_nv.append( ( (K/(2*i+1)) * (i * y[5+n_g_moments+i-1] - (i+1) * y[5+n_g_moments+i+1]) ) / ( a**2 * temp**0.5 ) )

    Psi_h_0, Psi_h_1, Psi_h_2 = [], [], []
    for i in range(len(q)):
        Psi_h_0.append( -q[i] * K * y[5 + n_g_moments + n_nv_moments+len(q)+i] / (E[i] * a**2 * temp**0.5) + phi_dot * dlnf_dlnq[i])
        Psi_h_1.append( ( q[i] * K * (y[5+n_g_moments+n_nv_moments+i] - 2 * y[5+n_g_moments+n_nv_moments+2*len(q)+i]) / (3 * E[i]) + E[i] * np.exp(q[i]) * f0[i] * K * psi / (3) ) / (a**2 * temp**0.5) )
        Psi_h_2.append( K * (q[i] * y[5+n_g_moments+n_nv_moments+len(q)+i] / E[i] - 3 * y[5+n_g_moments+n_nv_moments+2*len(q)+i]/(K*conf_time) ) / ( a**2 * temp**0.5 ) ) 

    dydt = [phi_dot, delta_cdm_dot, v_cdm_dot, delta_b_dot, v_b_dot] + f_g + f_nv + Psi_h_0 + Psi_h_1 + Psi_h_2
    #print(dydt)
    #input('LOL')                                      
    return dydt

def expint(a, r_v0, r_m0, r_r0, r_c0):
    return quad(integrand, 0, a, args=(r_v0, r_m0, r_r0, r_c0))[0]

def solve(K, r_v0, r_m0, r_cdm0, r_b0, r_r0, r_g0, r_nv0, r_c0, start, end):
    a = 10**np.linspace(start,end,1000, endpoint=True)
    conf_time = quad(integrand, 0, a[0], args=(r_v0, r_m0, r_r0, r_c0))[0] #in units of hubble time
    #print(conf_time)
    
    #Initial Conditions
    #const = 10**(-9.3)
    #phi = (50 * np.pi**2 * (K)**(n-1)/ (9*(K/c)**3) * (2*np.pi)**3 * const)**0.5
    phi = 1.0
    psi = - phi / (1 + 0.4 * (conf_time/a[0])**2 * ( r_nv0 + T_nv0**4 * boole(q, q**4 * f0**2 * np.exp(q)) / (4*np.pi**2*p_crit) ) )
    #print(T0**4 * boole(q, q**4 * f0**2 * np.exp(q)) / (4*np.pi**2*p_crit))
    #print(psi)
    
    y0 = [phi, -1.5 * psi, -0.5 * K * conf_time * psi, -1.5 * psi, -0.5 * K * conf_time * psi]
    y0 = y0 + [-2 * psi, 2 * K * conf_time * psi / 3.0, 0.0] + [0.0 for i in range(n_g_moments-3)]
    y0 = y0 + [-2 * psi, 2 * K * conf_time * psi / 3.0, 2 * psi * (K * conf_time)**2 / 15.0] + [0.0 for i in range(n_nv_moments-3)]
    y0 = y0 + list(-0.25 * y0[5 + n_g_moments] * dlnf_dlnq) + list(-0.25 * y0[6 + n_g_moments] * dlnf_dlnq) + list(-0.25 * y0[7 + n_g_moments] * dlnf_dlnq)
    
    sol = solve_ivp(func, [a[0], a[-1]], y0, t_eval = a, args=(K, r_v0, r_m0, r_cdm0, r_b0, r_r0, r_g0, r_nv0, r_c0), method='RK45')
    y = sol.y

    Y0, Y1, Y2 = np.array([]), np.array([]), np.array([])
    for j in range(len(a)):
        E = (q**2 + (a[j]*m_v/T_nv0)**2)**0.5
        p_h = boole(q, q**2 * E * f0)
        P_h = boole(q, q**4 * f0 / E)/3.0

        temp = y[5+n_g_moments+n_nv_moments: 5+n_g_moments+n_nv_moments + len(q),j]
        h_term = ( boole(q, q**2 * E * f0 * temp.T) ) /(p_h)
        Y0 = np.append(Y0, h_term)

        temp = y[5+n_g_moments+n_nv_moments + len(q): 5+n_g_moments+n_nv_moments + 2*len(q),j]
        h_term = ( boole(q, q**3 * f0 * temp.T) ) /(p_h + P_h)
        Y1 = np.append(Y1, h_term)

        temp = y[5+n_g_moments+n_nv_moments + 2*len(q): 5+n_g_moments+n_nv_moments + 3*len(q),j]
        h_term = 2*( boole(q, q**4 * f0 * temp.T / E) ) / (3 * ( p_h + P_h ))
        Y2 = np.append(Y2, h_term)
    y = np.concatenate((y[ :5 + n_g_moments + n_nv_moments], Y0.reshape(1, len(Y0)), Y1.reshape(1, len(Y1)), Y2.reshape(1, len(Y2)) ), axis = 0)

    vec_expint = np.vectorize(expint)
    hori = vec_expint(a, r_v0, r_m0, r_r0, r_c0)
    f = interpolate.interp1d(hori, a)
    try:
        a_hori = f(1/K)
        return np.log10(a), y, a_hori
    except:
        return np.log10(a), y, False

def draw(x, y, visibility, a_eq, a_recomb, a_hori):
    ymax = 0
    ymin = 0
    for i in range(len(visibility)):
        if visibility[i]:
            Y = np.log10(np.abs(y[i]))
            index = Y>-3
            X, Y = x[index], Y[index]
            ax.plot(X, Y, label = labels[i])
            ymax = np.max(np.array([ymax, np.max(Y)]))
            ymin = np.min(np.array([ymin, np.min(Y)]))

    x_ticks = np.array(range(int(np.ceil(x[0])), int(np.floor(x[-1]))+1, 1))
    x_ticklabels = 10.0**x_ticks

    #matter-rad equality and recomb and horizon crossing
    ax.plot(np.log10(np.array([a_eq, a_eq])), [ymax, ymin], '--')
    ax.annotate('Radiation - matter equality', (np.log10(a_eq), ymax), textcoords = 'offset pixels', xytext = (5,-190), rotation = 90)
    ax.plot(np.log10(np.array([a_recomb, a_recomb])), [ymax, ymin], '--')
    ax.annotate('Last Scattering Surface', (np.log10(a_recomb), ymax), textcoords = 'offset pixels', xytext = (5,-170), rotation = 90)
    if a_hori!=False:
        ax.plot(np.log10(np.array([a_hori, a_hori])), [ymax, ymin], '--')
        ax.annotate('Horizon crossing', (np.log10(a_hori), ymax), textcoords = 'offset pixels', xytext = (5,-120), rotation = 90)

    ax.set(xlabel = 'a', ylabel = 'Perturbation',
            xticks = x_ticks, xticklabels = x_ticklabels,
            title = "k = "+str(np.round(k,7))+" h Mpc$^{-1}$",
            xlim = (slower.val, supper.val))
    ax.legend()
    ax.grid()

c = 2997.92458  #100 km s^-1
h = 0.6871
p_crit = 8.098e-11 * h**2   #eV^4 / (h_cross c)^3

#CDM
r_cdm0 = 0.255

#Baryons
r_b0 = 0.045

r_m0 = r_cdm0 + r_b0

#Photons
T0 = 2.34865e-4   #eV / Kb
r_g0 = np.pi**2 * T0**4 / (15 * p_crit)
n_g_moments = 3

#Massless Neutrinos
T_nv0 = (4/11.0)**(1/3.0) * T0
r_nv0 = 7 * np.pi**2 * T_nv0**4 / (40 * p_crit)
n_nv_moments = 5

r_r0 = r_g0 + r_nv0

#Massive Neutrinos
n_h_moments = 3
m_v = 0.1    #eV
q = np.linspace(0, 20, 21)
f0 = 1/(np.exp(q) + 1)
dlnf_dlnq = -q * np.exp(q) / (np.exp(q) + 1)
E = (q**2 + (m_v/T_nv0)**2)**0.5
r_h0 = T_nv0**4 * (boole(q, q**2 * E * f0)) / (np.pi**2 * p_crit)

r_c0 = 0.0#1 - r_v0 - r_m0 - r_r0

#DE
r_v0 = 1 + r_c0 - r_m0 - r_r0 - r_h0

print(r_cdm0, r_b0, r_g0, r_nv0, r_h0, r_v0)

n = 0.9660499 #spectral index of initial power spectrum
a_eq = r_r0/r_m0
a_recomb = 1/1101.0

k = 0.1
start, end = -8, -1

visibility = np.full((1+2+2+n_g_moments+n_nv_moments+n_h_moments), False)
visibility[[0,1,3,5,8,13]] = True
labels = np.array(['$\Phi$', '$\delta_{cdm}$', '$v_{cdm}$', '$\delta_{b}$', '$v_{b}$']
                    +['$\delta_{\\gamma,'+str(i)+'}$' for i in range(n_g_moments)]
                    +['$\delta_{\\nu,'+str(i)+'}$' for i in range(n_nv_moments)]
                    +['$\delta_{h}$', '$v_{h}$', '$\sigma_{h}$'])

fig, ax = pl.subplots(nrows=1, ncols=1)
pl.subplots_adjust( left = 0.05, bottom = 0.25, right = 0.85 )

axcolor = 'lightgoldenrodyellow'

axk = pl.axes([0.1, 0.12, 0.75, 0.03], facecolor=axcolor)
axlower = pl.axes([0.1, 0.075, 0.75, 0.03], facecolor=axcolor)
axupper = pl.axes([0.1, 0.02, 0.75, 0.03], facecolor=axcolor)
axcheck = pl.axes([0.88, 0.2, 0.1, 0.7], facecolor=axcolor, title = 'Variables')

sk = Slider(axk, 'k (h Mpc$^{-1}$, log scale)', -7, 2, valinit=np.log10(k))
slower = Slider(axlower, '$a_{min}$ (log scale)', -8, -2, valinit=start)
supper = Slider(axupper, '$a_{max}$ (log scale)', -8, 2, valinit=end)
check = CheckButtons(axcheck, labels, visibility)

x, y, a_hori = solve(c*k, r_v0, r_m0, r_cdm0, r_b0, r_r0, r_g0, r_nv0, r_c0, start, end)
#df = pd.DataFrame(data = {"a":x, "cdm overdensity":y[1+2+2+n_g_moments+n_nv_moments]})
#df.to_csv('k_'+str(k)+'_data.csv', index = False)
draw(x, y, visibility, a_eq, a_recomb, a_hori)

def k_update(val):
    global k, start, end, x, y, a_hori
    k = 10**sk.val

    ax.cla()

    x, y, a_hori = solve(c*k, r_v0, r_m0, r_cdm0, r_b0, r_r0, r_g0, r_nv0, r_c0, start, end)
    draw(x, y, visibility, a_eq, a_recomb, a_hori)
    
    fig.canvas.draw()

def lim_update(val):
    ax.set(xlim = (slower.val, supper.val))
    fig.canvas.draw()

def check_update(label):
    global visibility, k, start, end, x, y, a_hori
    ax.cla()
    visibility[labels == label] = np.logical_not(visibility[labels == label])
    draw(x, y, visibility, a_eq, a_recomb, a_hori)

    fig.canvas.draw()

sk.on_changed(k_update)
slower.on_changed(lim_update)
supper.on_changed(lim_update)
check.on_clicked(check_update)
pl.show()