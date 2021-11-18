#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Solucion problema 4 (Laboratorio 4)
# Parte A)
# Primero, se importan las librerias y funciones necesarias.
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Se definen las variables aleatorias A y Z
vaA = stats.norm(5, np.sqrt(0.2))
vaZ = stats.uniform(0, np.pi/2)

# Se definen las constantes
w = np.pi

# Creación del vector de tiempo
T = 100      # número de elementos
t_final = 5  # tiempo en segundos
t = np.linspace(0, t_final, T)

# Inicialización del proceso aleatorio X(t) con N realizaciones
N = 20
X_t = np.empty((N, len(t)))	# N funciones del tiempo x(t) con T puntos

# Creación de las muestras del proceso x(t) (A y Z independientes)
for i in range(N):
	A = vaA.rvs()
	Z = vaZ.rvs()
	x_t = A * np.cos(w*t + Z)
	X_t[i,:] = x_t
	plt.plot(t, x_t)

# Promedio de las N realizaciones en cada instante (cada punto en t)
P = [np.mean(X_t[:,i]) for i in range(len(t))]
plt.plot(t, P, lw=6)

# Graficar el resultado teórico del valor esperado
E = (10/np.pi) * (np.cos(w*t)-np.sin(w*t))
plt.plot(t, E, '-.', lw=4)

# Mostrar las realizaciones, y su promedio calculado y teórico
plt.title('Realizaciones del proceso aleatorio $X(t)$')
plt.xlabel('$t$')
plt.ylabel('$x_i(t)$')
plt.show()

# Parte B) Creación de las muestras del proceso x(t) (w y theta constantes)
# w = pi
theta = 0
for i in range(N):
	A = vaA.rvs()
	y_t = A * np.cos(w*t + theta)
	X_t[i,:] = y_t

# T valores de desplazamiento tau
desplazamiento = np.arange(T)
taus = desplazamiento/t_final

# Inicialización de matriz de valores de correlación para las N funciones
corr = np.empty((N, len(desplazamiento)))

# Nueva figura para la autocorrelación
plt.figure()

# Cálculo de correlación para cada valor de tau
for n in range(N):
	for i, tau in enumerate(desplazamiento):
		corr[n, i] = np.correlate(X_t[n,:], np.roll(X_t[n,:], tau))/T
	plt.plot(taus, corr[n,:])

# Valor teórico de correlación
Rxx = 25.2 * np.cos(w*t+theta) * np.cos(w*(t+taus)+theta)

# Gráficas de correlación para cada realización
plt.plot(taus, Rxx, '-.', lw=4, label='Correlación teórica')
plt.title('Funciones de autocorrelación de las realizaciones del proceso')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$R_{XX}(\tau)$')
plt.legend()
plt.show()

print('Valor teórico de la media E:')
print(E)
print()
print('Valor teórico de la correlación Rxx:')
print(Rxx)

