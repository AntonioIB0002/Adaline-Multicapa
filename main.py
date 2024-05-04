import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QMessageBox, QFileDialog,QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from mw import Ui_MainWindow
import numpy as np
import random
from numpy import random
import adaline
import time

import prueba



class AdalineThread(QThread):
    update_signal = pyqtSignal(list,list,float,list,str)

    def __init__(self, parent=None):
        super(AdalineThread, self).__init__(parent)
        self.entradas = []
        self.salidas_deseadas = []
        self.limite_de_epocas = 0
        self.factor_de_aprendizaje = 0
        self.N_capas = 1
        self.w_1 = None
        self.w_2 = None
        self.bs = None
        self.boc = None
        self.presicion = 0.0000001
        self.n_entradas = 0
        self.n_ocultas = 0
        self.n_salidas = 1
        self.error_red = 1
        self.epochs = 0

    def run(self):
        red = adaline.MLP(self.entradas,self.salidas_deseadas,self.w_1,self.w_2,self.bs,self.boc,self.presicion,self.limite_de_epocas,
                  self.factor_de_aprendizaje,self.n_ocultas,len(self.entradas[0]),self.n_salidas)
        while(np.abs(self.error_red) > self.presicion):

            epochs,w1_a,w2_a,us_a,uoc_a,E,error_red,filename = red.Aprendizaje()
            self.error_red = error_red
            self.epochs +=1
            w1_a = w1_a.tolist()
            w2_a = w2_a.tolist()
            uoc_a = uoc_a.tolist()
            self.update_signal.emit(w1_a,w2_a,us_a,uoc_a,filename)
            if self.epochs > self.limite_de_epocas:
                break

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 750, 750)
        self.entradas = []
        self.salidas_deseadas = []
        self.limite_de_epocas = 0
        self.factor_de_aprendizaje = 0
        self.N_capas = 1
        self.w_1 = None
        self.w_2 = None
        self.bs = None
        self.boc = None
        self.presicion = 0.0000001
        self.n_entradas = 0
        self.n_ocultas = 0
        self.n_salidas = 1
        self.Cartesiano("plano_cartesiano.png")
        self.ui.pushButton_graficar.clicked.connect(self.grafica)
        self.ui.pushButton_reset.clicked.connect(self.reset)
        self.ui.pushButton_exportar.clicked.connect(self.AbrirArchivo)
        self.ui.pushButton.clicked.connect(self.Archivo_Salidas)

    def AbrirArchivo(self):
        archivo, _ = QFileDialog.getOpenFileName(None, "Seleccionar archivo", "", "Archivos de texto (*.txt)")
        try:
            with open(archivo, 'r') as f:
                for linea in f:
                    x, y = map(float, linea.strip().split(','))
                    # Ajuste de coordenadas
                    self.entradas.append([x, y])   

                print(self.entradas)

        except Exception as e:
            QMessageBox.warning(self, 'Error', 'Archivo no válido')

    def Archivo_Salidas(self):
        archivo, _ = QFileDialog.getOpenFileName(None, "Seleccionar archivo", "", "Archivos de texto (*.txt)")
        try:
            with open(archivo, 'r') as f:
                for linea in f:
                    linea = linea.strip()
                    self.salidas_deseadas.append(float(linea))

                self.inicializar_pesos()
                nombre = prueba.plano_cartesiano(self.entradas,self.salidas_deseadas)        
                self.scene.clear()
                self.Cartesiano(nombre)   
                print(self.salidas_deseadas)
        except Exception as e:
            QMessageBox.warning(self, 'Error', 'Archivo no válido')

    def adaline(self):
        self.thread = AdalineThread()
        self.thread.entradas = self.entradas
        self.thread.salidas_deseadas = self.salidas_deseadas
        self.thread.limite_de_epocas = self.limite_de_epocas
        self.thread.factor_de_aprendizaje = self.factor_de_aprendizaje
        self.thread.N_capas = self.N_capas
        self.thread.w_1 = self.w_1
        self.thread.w_2 = self.w_2
        self.thread.bs = self.bs
        self.thread.boc = self.boc
        self.thread.presicion = self.presicion
        self.thread.n_entradas = self.n_entradas
        self.thread.n_ocultas = self.n_ocultas
        self.thread.update_signal.connect(self.actualizar_interfaz)
        self.thread.start()

    def actualizar_interfaz(self,w1_a,w2_a,us_a,uoc_a,filename):
        self.boc = uoc_a
        self.bs = us_a
        self.w_1 = w1_a
        self.w_2 = w2_a
        pesos = 'capa entrada \n' + str(w1_a) +'\n capa de salida \n' +  str(w2_a)
        bias =  'capa entrada \n' + str(uoc_a) + '\n capa de salida \n' + str((us_a))
        self.ui.textBrowser_pesos.setText(pesos)
        self.ui.textBrowser_bias.setText(bias)
        self.scene.clear()
        self.Cartesiano(filename)

    def inicializar_pesos(self):
        print(len(self.entradas))
        self.n_ocultas  = int(self.ui.spinBox_neuronas.text())
        random.seed(0)
        self.w_1 = random.rand(self.n_ocultas,len(self.entradas[0]))
        self.w_2 = random.rand(self.n_salidas,self.n_ocultas)
        self.boc = np.ones((self.n_ocultas,1),float)
        self.bs = 1.0
        pesos = 'capa entrada \n' + str(self.w_1) +'\n capa de salida \n' +  str(self.w_2)
        bias =  'capa entrada \n' + str(self.boc) + '\n capa de salida \n' + str((self.bs))
        self.ui.textBrowser_pesos.setText(pesos)
        self.ui.textBrowser_bias.setText(bias)

    def grafica(self):
        if self.validacion():
            self.adaline()
    def validacion(self):
        try:
            self.factor_de_aprendizaje = float(self.ui.lineEdit_factor.text())
            self.limite_de_epocas = float(self.ui.lineEdit_limite.text())
            if self.limite_de_epocas < 0:
                QMessageBox.warning(self, 'Captura no válida', 'Ingrese solo números enteros o reales positivos.')
                return False
            if len(self.entradas) == 0:
                QMessageBox.warning(self, 'Ingrese entradas', 'Seleccione entradas en el plano')
                return False
            return True
        except ValueError:
            QMessageBox.warning(self, 'Captura no válida', 'Ingrese solo números enteros o reales positivos.')
            return False
    
    def Cartesiano(self, filename):
        pixmap = QPixmap(filename)
        existing_pixmaps = [item for item in self.scene.items() if isinstance(item, QGraphicsPixmapItem)]
    
        # Verificar si el elemento pixmap ya está presente en la escena
        if not existing_pixmaps:
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.ui.graphicsView.setScene(self.scene)
    def reset(self):
        self.scene.clear()
        self.entradas.clear()
        self.Cartesiano("plano_cartesiano.png")
        self.salidas_deseadas.clear()
        self.w_1 = None
        self.w_2 = None
        self.bs = None
        self.boc = None
        self.ui.textBrowser_pesos.clear()
        self.ui.textBrowser_bias.clear()



app = QApplication(sys.argv)
ventana = Window()
ventana.show()
sys.exit(app.exec_())
