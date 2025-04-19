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
    def number_density(self, fermi_energy):
        k_f = np.sqrt(np.power(fermi_energy, 2) - pow(self.mass, 2))
        return self.con * np.pow(k_f, 3)
    def pressure(self, fermi_energy):
        k_f = np.sqrt(np.power(fermi_energy, 2) - pow(self.mass, 2))
        return self.con * (
                fermi_energy * (np.power(k_f, 3) / 4 - 3 * pow(self.mass, 2) * k_f / 8) + 3 * pow(self.mass, 4) * np.log(
            (k_f + fermi_energy) / self.mass) / 8)
    def energy_density(self, fermi_energy):
        k_f = np.sqrt(np.power(fermi_energy, 2) - pow(self.mass, 2))
        return 3 * self.con * (
                    fermi_energy * (pow(self.mass, 2) * k_f / 8 + np.power(k_f, 3) / 4) - pow(self.mass, 4) * np.log((k_f + fermi_energy) / self.mass) / 8)
    def dP(self, fermi_energy):
        k_f = np.sqrt(np.power(fermi_energy, 2) - pow(self.mass, 2))
        return self.con * (k_f * (4* np.power(fermi_energy, 2) - pow(self.mass, 2)))/ 4
    def dE(self, fermi_energy):
        k_f = np.sqrt(np.power(fermi_energy, 2) - pow(self.mass, 2))
        return 3 * self.con * k_f * np.power(fermi_energy, 2)

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
xi_1530_minus = particle(1535000000, 0.5, -1, 1, -2)
xi_1530_0 = particle(1531800000, 0.5, 0, 1, -2)
omega = particle(1672450000, 1.5, -1, 1 , -3)
particles = [proton, neutron]

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
def dP(mu_B, mu_Q, mu_S):
    dPi = 0
    for part in particles:
        mu = part.baryon_number * mu_B + part.charge * mu_Q + part.strangeness * mu_S
        dPi = dPi + part.dP(mu)
    return dPi
def dE(mu_B, mu_Q, mu_S):
    dU = 0
    for part in particles:
        mu = part.baryon_number * mu_B + part.charge * mu_Q + part.strangeness * mu_S
        dU = dU + part.dE(mu)
    return dU
def cs2(mu_B, mu_Q, mu_S):
    return dP(mu_B, mu_Q, mu_S)/dE(mu_B, mu_Q, mu_S)
muB = np.linspace(neutron.mass, 1500000000, 100)
muQ = 0
muS = 0
ax = plt.axes()
ax.plot(10e-24*energy_density(muB, muQ, muS), 10e-24*pressure(muB, muQ, muS))
ax.set_ylabel('Pressure(MeV^4)')
ax.set_xlabel('Energy density(MeV^4)')
plt.figure()
ax2 = plt.axes()
ax2.plot(10e-6*muB, cs2(muB, muQ, muS))
ax2.set_ylabel('c_s^2')
ax2.set_xlabel('mu_B(MeV)')
plt.figure()
ax3 = plt.axes()
ax3.plot(10e-6*muB, 10e-18*n_B(muB, muQ, muS))
ax3.set_ylabel('n_B(MeV^3)')
ax3.set_xlabel('mu_B(MeV)')
plt.figure()
ax4 = plt.axes()
ax4.plot(10e-6*muB, 10e-18*n_Q(muB, muQ, muS))
ax4.set_ylabel('n_Q(MeV^3)')
ax4.set_xlabel('mu_B(MeV)')
plt.show()
