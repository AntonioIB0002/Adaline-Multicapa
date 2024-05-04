import numpy as np
import pandas as pd
from numpy import random
import matplotlib.pyplot as plt
import prueba


class MLP():
    # constructor
    def __init__(self,xi,d,w_1,w_2,us,uoc,precision,epocas,fac_ap,n_ocultas,n_entradas,n_salida):
        # Variables de inicialización 
        self.xi = np.transpose(xi)
        self.d = d
        self.w1 = w_1
        self.w2 = w_2
        self.us = us
        self.uoc = uoc
        self.precision = precision
        self.epocas = epocas
        self.fac_ap = fac_ap
        self.n_entradas = n_entradas
        self.n_ocultas = n_ocultas
        self.n_salida = n_salida
        # Variables de aprendizaje
        self.di = 0 # Salida deseada en iteracion actual
        self.error_red = 1 # Error total de la red en una conjunto de iteraciones
        self.Ew = 0 # Error cuadratico medio
        self.Error_prev = 0 # Error anterior
        self.Errores = []
        self.Error_actual = np.zeros((len(d))) # Errores acumulados en un ciclo de muestras
        self.Entradas = np.zeros((1,n_entradas))
        self.un = np.zeros((n_ocultas,1)) # Potencial de activacion en neuronas ocultas
        self.gu = np.zeros((n_ocultas,1)) # Funcion de activacion de neuronas ocultas
        self.Y = 0.0 # Potencial de activacion en neurona de salida
        self.y = 0.0 # Funcion de activacion en neurona de salida
        self.epochs = 0
        # Variables de retropropagacion
        self.error_real = 0
        self.ds = 0.0 # delta de salida
        self.docu = np.zeros((n_ocultas,1)) # Deltas en neuronas ocultas
        
    def Operacion(self):
        respuesta = np.zeros((len(self.xi[0]),1))
        for p in range(len(self.xi[0])):
            self.Entradas = self.xi[:,p]
            self.Propagar()
            respuesta[p,:] = self.y
        return respuesta.tolist()
    
    def Aprendizaje(self):
        Errores = [] # Almacenar los errores de la red en un ciclo
        self.Error_prev = self.Ew
        for i in range(len(self.d)):
            self.Entradas = self.xi[:,i] # Senales de entrada por iteracion
            self.di = self.d[i]
            self.Propagar()
            self.Backpropagation()
            self.Propagar()
            self.Error_actual[i] = (0.5)*((self.di - self.y)**2)
            # error global de la red

        filename = self.graficar()
        self.Error()
        Errores.append(self.error_red)
            
        return self.epochs,self.w1,self.w2,self.us,self.uoc,Errores,self.error_red,filename
                
    def graficar(self):
        filename = prueba.plot_decision_boundary(np.transpose(self.xi),self.d,self.w1,self.w2,self.us,self.uoc,self.precision,self.epocas,
                  self.fac_ap,self.n_ocultas,len(self.xi),self.n_salida)
        
        return filename
    def Propagar(self):
        # Operaciones en la primer capa
        for a in range(self.n_ocultas):
            self.un[a,:] = np.dot(self.w1[a,:], self.Entradas) + self.uoc[a,:]
        
        # Calcular la activacion de la neuronas en la capa oculta
        for o in range(self.n_ocultas):
            self.gu[o,:] = tanh(self.un[o,:])
        
        # Calcular Y potencial de activacion de la neuronas de salida
        self.Y = (np.dot(self.w2,self.gu) + self.us)
        # Calcular la salida de la neurona de salida
        self.y = tanh(self.Y)
    
    def Backpropagation(self):
        # Calcular el error
        self.error_real = (self.di - self.y)
        # Calcular ds
        self.ds = (dtanh(self.Y) * self.error_real)
        # Ajustar w2
        self.w2 = self.w2 + (np.transpose(self.gu) * self.fac_ap * self.ds)
        # Ajustar umbral us
        self.us = self.us + (self.fac_ap * self.ds)
        # Calcular docu
        self.docu = dtanh(self.un) * np.transpose(self.w2) * self.ds
        # Ajustar los pesos w1
        for j in range(self.n_ocultas):
            self.w1[j,:] = self.w1[j,:] + ((self.docu[j,:]) * self.Entradas * self.fac_ap)
        
        # Ajustar el umbral en las neuronas ocultas
        for g in range(self.n_ocultas):
            self.uoc[g,:] = self.uoc[g,:] + (self.fac_ap * self.docu[g,:])
        
    def Error(self):
        # Error cuadratico medio
        self.Ew = ((1/len(self.d)) * (sum(self.Error_actual)))
        self.error_red = (self.Ew - self.Error_prev)

# Funcion para obtener la tanh
def tanh(x):
    return np.tanh(x)

# Funcion para obtener la derivada de tanh x
def dtanh(x):
    return 1.0 - np.tanh(x)**2

# Funcion sigmoide de x
def sigmoide(x):
    return 1/(1+np.exp(-x))

# Funcion para obtener la derivada de de la funcion sigmoide
def dsigmoide(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)