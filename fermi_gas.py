import math
import numpy as np
import matplotlib.pyplot as plt


#all in natural units
#particle class with attributes for each particle
class particle:
    def __init__(self, mass, spin, charge, baryon_number, strangeness):
        self.mass = mass
        self.spin = spin
        self.charge = charge
        self.baryon_number = baryon_number
        self.strangeness = strangeness
        self.degeneracy = 2 * self.spin + 1
        self.con = self.degeneracy / (6 * pow(math.pi, 2))
    def k_f(self, fermi_energy):
        return np.sqrt(np.power(fermi_energy, 2) - np.power(self.mass, 2))
    def number_density(self, fermi_energy):
        return self.con * np.power(self.k_f(fermi_energy), 3)
    def pressure(self, fermi_energy):
        return self.con * (fermi_energy * (np.power(self.k_f(fermi_energy), 3) / 4 - 3 * np.power(self.mass, 2) * self.k_f(fermi_energy) / 8) + 3 * np.power(self.mass, 4) * np.log((self.k_f(fermi_energy) + fermi_energy) / self.mass) / 8)
    def energy_density(self, fermi_energy):
        return 3 * self.con * (fermi_energy * (np.power(self.mass, 2) * self.k_f(fermi_energy) / 8 + np.power(self.k_f(fermi_energy), 3) / 4) - np.power(self.mass, 4) * np.log((self.k_f(fermi_energy) + fermi_energy) / self.mass) / 8)
    def dP(self, fermi_energy):
        return self.con * (self.k_f(fermi_energy) * (4* np.power(fermi_energy, 2) - pow(self.mass, 2)))/ 4
    def dE(self, fermi_energy):
        return 3 * self.con * self.k_f(fermi_energy) * np.power(fermi_energy, 2)
    def d2P(self, fermi_energy):
        return 3 * self.con * fermi_energy * (4* np.power(fermi_energy, 2) - 3*pow(self.mass, 2))/(4*self.k_f(fermi_energy))

#declare particles in the class 'particle'
#octet
proton = particle(938272089.43, 0.5, 1, 1, 0)
neutron = particle(939565421.94, 0.5, 0, 1, 0)
sigma_plus = particle(1189370000, 0.5, 1, 1, -1)
sigma_0 = particle(1192642000, 0.5, 0, 1, -1)
sigma_minus = particle(1197449000, 0.5, -1, 1, -1)
lambda_0 = particle(1115683000, 0.5, 0, 1, -1)
xi_minus = particle(1321710000, 0.5, -1, 1, -2)
xi_0 = particle(1314860000, 0.5, 0, 1, -2)

#decuplet
delta_plusplus = particle(1232000000, 1.5, 2, 1, 0)
delta_plus = particle(1232000000, 1.5, 1, 1, 0)
delta_0 = particle(1232000000, 1.5, 0, 1, 0)
delta_minus = particle(1232000000, 1.5, -1, 1, 0)
sigma_star_plus = particle(1382800000, 1.5, 1, 1, -1)
sigma_star_0 = particle(1383700000, 1.5, 0, 1, -1)
sigma_star_minus = particle(1387200000, 1.5, -1, 1, -1)
xi_1530_minus = particle(1535000000, 1.5, -1, 1, -2)
xi_1530_0 = particle(1531800000, 1.5, 0, 1, -2)
omega = particle(1672450000, 1.5, -1, 1 , -3)
particles = [proton, neutron, sigma_0, sigma_minus, sigma_plus, lambda_0, xi_minus, xi_0, delta_plusplus, delta_plus, delta_0, delta_minus, sigma_star_minus, sigma_star_0, sigma_star_plus, xi_1530_minus, xi_1530_0, omega]

#system variables
def pressure(mu_B, mu_Q, mu_S):
    p = 0
    for part in particles:
        mu = part.baryon_number * mu_B + part.charge * mu_Q + part.strangeness * mu_S
        p = p + part.pressure(mu)
    return p
def energy_density(mu_B, mu_Q, mu_S):
    e = 0
    for part in particles:
        mu = part.baryon_number * mu_B + part.charge * mu_Q + part.strangeness * mu_S
        e = e + part.energy_density(mu)
    return e
def n_Q(mu_B, mu_Q, mu_S):
    n = 0
    for part in particles:
        mu = part.baryon_number * mu_B + part.charge * mu_Q + part.strangeness * mu_S
        n = n + part.number_density(mu) * part.charge
    return n
def n_B(mu_B, mu_Q, mu_S):
    n = 0
    for part in particles:
        mu = part.baryon_number * mu_B + part.charge * mu_Q + part.strangeness * mu_S
        n = n + part.number_density(mu) * part.baryon_number
    return n
def X_BB(mu_B, mu_Q, mu_S):
    n = 0
    for part in particles:
        mu = part.baryon_number * mu_B + part.charge * mu_Q + part.strangeness * mu_S
        n = n + part.d2P(mu) * part.baryon_number * part.baryon_number
    return n
def X_SB(mu_B, mu_Q, mu_S):
    n = 0
    for part in particles: 
        mu = part.baryon_number * mu_B + part.charge * mu_Q + part.strangeness * mu_S
        n = n + part.d2P(mu) * part.strangeness * part.baryon_number
    return n
def X_QB(mu_B, mu_Q, mu_S):
    n = 0
    for part in particles:
        mu = part.baryon_number * mu_B + part.charge * mu_Q + part.strangeness * mu_S
        n = n + part.d2P(mu) * part.charge * part.baryon_number
    return n

muB = np.linspace(1000000000, 5000000000, 1000)
muQ = 0 #np.linspace(-500000000, 0, 100)
muS = np.linspace(-500000000, 2000000000, 1000)

#print(X_BB(omega.mass, 0 , 0))

# mB, mQ = np.meshgrid(muB, muQ)
ax = plt.axes()
ax.plot(1e-24*energy_density(muB, muQ, muS), 1e-24*pressure(muB, muQ, muS))
ax.set_ylabel('Pressure(MeV^4)')
ax.set_xlabel('Energy density(MeV^4)')
plt.figure()
ax2 = plt.axes()
ax2.plot(1e-18*n_B(muB, muQ, muS)/(0.16*pow(197.3, 3)), n_B(muB, muQ, muS)/(muB*X_BB(muB, muQ, muS) + muS*X_SB(muB, muQ, muS) + muQ*X_QB(muB, muQ, muS)))
ax2.set_ylabel('c_s^2')
ax2.set_xlabel('n_B/n_sat')
plt.figure()
ax3 = plt.axes()
ax3.plot(1e-6*muB, 1e-18*n_B(muB, muQ, muS)/pow(197.3, 3))
ax3.set_ylabel('n_B(1/fm^3)')
ax3.set_xlabel('mu_B(MeV)')
plt.figure()
ax4 = plt.axes()
ax4.plot(1e-6*muB, 1e-18*n_Q(muB, muQ, muS)/pow(197.3, 3))
ax4.set_ylabel('n_Q(1/fm^3)')
ax4.set_xlabel('mu_B(MeV)')
plt.figure()
ax5 = plt.axes()
ax5.plot(1e-6*muB, n_Q(muB, muQ, muS)/n_B(muB, muQ, muS))
ax5.set_ylabel('y_Q')
ax5.set_xlabel('mu_B(MeV)')
plt.show()
ax6 = plt.axes()
ax6.plot(1e-6*muB, 1e6*X_SB(muB, muQ, muS))
ax6.set_ylabel('X_BB(1/MeV)')
ax6.set_xlabel('mu_B(MeV)')
plt.show()
