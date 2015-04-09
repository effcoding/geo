import numpy as np


def compute_Sg(S, angles=(0,0,0)):
    
    alpha, beta, gamma = np.radians(angles)
    
    Rg = np.array([[np.cos(alpha) * np.cos(beta),  
                    np.sin(alpha) * np.cos(beta),  
                    -np.sin(beta)],
                   [np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma), 
                    np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),  
                    np.cos(beta) * np.sin(gamma)],
                   [np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma), 
                    np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma),  
                    np.cos(beta) * np.cos(gamma)]])
                  
    return np.dot(Rg.T, np.dot(S,Rg))


def compute_Sb(S, angles_G=(0,0,0), angles_B=(0,0)):
    
    delta, phi = np.radians(angles_B)
    
    Rb = np.array([[ np.cos(delta) * np.cos(phi),  np.sin(delta) * np.cos(phi), -np.sin(phi)],
                   [ -np.sin(delta), np.cos(delta),  0.0],
                   [ np.cos(delta) * np.sin(phi), np.sin(delta) * np.sin(phi), np.cos(phi)]])
    
    Sg = compute_Sg(S, angles_G)
    
    return np.dot(Rb, np.dot(Sg,Rb.T))


def compute_wellbore_stress(S, nu, theta, DP):
    
    theta = np.radians(theta)
    
    sZZ = (S[2,2] -  2. * nu * (S[0,0] - S[1,1]) * np.cos(2 * theta) - 
           4. * nu * S[0,1] * np.sin(2 * theta))
    
    stt = (S[0,0] + S[1,1] - 2. * (S[0,0] - S[1,1]) * np.cos(2 * theta) - 
           4 * S[0,1] * np.sin(2. * theta) - DP)
    
    ttz = 2. * (S[1,2] * np.cos(theta) - S[0,2] * np.sin(theta))
    
    srr = DP
    
    return (sZZ, stt, ttz, srr)


def compute_max_tangent_stress(S, nu, theta, DP):
    
    sZZ, stt, ttz, srr = compute_wellbore_stress(S, nu, theta, DP)
    
    return 0.5 * (sZZ + stt + np.sqrt((sZZ - stt) ** 2. + 4.0 * ttz ** 2.))


def compute_min_tangent_stress(S, nu, theta, DP):
    
    sZZ, stt, ttz, srr = compute_wellbore_stress(S, nu, theta, DP)
    
    return 0.5 * (sZZ + stt - np.sqrt((sZZ - stt) ** 2. + 4.0 * ttz ** 2.))


def compute_breakout_width(S, Pp, Pm, nu, C0, mu, angles_G=(0,0,0), angles_B=(0,0)):
    
    Sb = compute_Sb(S, angles_G, angles_B)
    
    Sb_eff = Sb - Pp * np.eye(3)
    
    theta = np.linspace(0, 360, num=90)
    
    smax = np.array([ compute_max_tangent_stress(Sb_eff, nu, i, (Pm-Pp)) for i in theta])
    smin = np.array([ compute_min_tangent_stress(Sb_eff, nu, i, (Pm-Pp)) for i in theta])
    
    breakout_bool_array = C0 < (smax) - ( np.sqrt(mu ** 2 + 1) + mu ) ** 2. * (smin)
    
    return np.round(breakout_bool_array.sum() * 2.0)
In [184]:
S = np.array([[145, 0, 0],[0,125,0],[0,0,70]])

delta = np.linspace(0, 360, num=50)
phi = np.linspace(0, 90, num=50)

res = np.array([ (np.cos(np.radians(d))*np.sin(np.radians(p)), 
                  np.sin(np.radians(d))*np.sin(np.radians(p)), 
                  compute_breakout_width(S, 33, 33, 0.2, 34, 1.0, angles_G=(30,0,0), angles_B=(d,p))) 
                  for d in delta for p in phi])
In [185]:
import scipy.interpolate
import matplotlib.pyplot as plt
%matplotlib inline

x = np.linspace(-1,1, num=500)
y = np.linspace(-1,1, num=500)
grid_x, grid_y = np.meshgrid(x,y)

X = res[:,0]
Y = res[:,1]
disp_x = scipy.interpolate.griddata((X, Y), res[:,2], (grid_x, grid_y), method='linear')
plt.figure()
plt.gca().set_aspect('equal')
plt.contourf(grid_x, grid_y, disp_x, cmap="coolwarm")#,levels=np.linspace(90,135,20))
plt.colorbar();
plt.title("Breakout width");

