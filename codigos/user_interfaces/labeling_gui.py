# Librerias

from PyQt5.QtGui import QIcon, QPixmap

from PyQt5.QtWidgets import QMainWindow, QWidget,QLabel,QApplication,QGridLayout, QPushButton, QVBoxLayout, QFormLayout, QGroupBox, QFileDialog, QErrorMessage, QMessageBox, QDialog, QLineEdit,QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt
import sys

import json
import pandas as pd
import numpy as np

# Librerias para mostrar los gráficos de plotly
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.express as px
import plotly.graph_objects as go

import os
import multiprocessing
import time

import sys
sys.path.append("..")  # Agrega el directorio anterior al sistema de rutas (directorio base del proyecto)

# Modulos propios
from common_functions import graphs, find_motifs

from labeling_time_series import Exhaustive_Fixed_Window, Evo_Variable_Window, Evo_Fixed_Window

class LabelingDlg(QMainWindow):

	def __init__(self):
		"""
		Constructor de la clase. Inicializa la aplicación.

		:return: 
		"""

		super().__init__()
		self.init_user_interface()

	def init_user_interface(self):
		"""
		Función encargada de inicializar toda la interfaz de usuario de la aplicación.

		:return: 
		"""
		self.current_language = "spanish"
		properties = self.load_language_property(self.current_language)

		main_widget = QWidget()

		# Panel Graficos

		graph_layout = QGridLayout()
		self.graph_label = QLabel(properties["TITLE_GRAPH"])
		self.graph_label.setObjectName("title_label")
		self.graph_label.setAlignment(Qt.AlignCenter)
		self.graph_widget = QWebEngineView(self)
		graph_layout.addWidget(self.graph_label, 0,0,1,1)
		graph_layout.addWidget(self.graph_widget,1,0,100,1)

		# Panel Derecha (Menus)

		menus_layout = QVBoxLayout()
		menus_layout.addSpacing(38)

		self.button_load_dataset = QPushButton(properties["SELECT_DATESET"])
		self.button_to_configure_hyperparameters = QPushButton(properties["CONFIGURE_HYPERPARAMETERS"])
		self.button_to_label = QPushButton(properties["TRAIN"])
		self.button_to_change_graph = QPushButton(properties["CHANGE_GRAPH"])
		self.button_to_change_graph.setCheckable(True)

		self.button_to_label.clicked.connect(self.label_time_serie)
		self.button_load_dataset.clicked.connect(self.load_dataset)
		self.button_to_configure_hyperparameters.clicked.connect(self.show_popup_menu)
		self.title_form_hiperparametros = QLabel(properties["SELECTED_HYPERPARAMETERS"])
		hiperparametros_form_layout = QFormLayout()
		hiperparametros_form_layout.addRow(self.title_form_hiperparametros)
		self.label_name_dataset = QLabel(properties["NO_CHOICE"])
		self.label_dataset = QLabel(properties["DATASET"])
		hiperparametros_form_layout.addRow(self.label_dataset, self.label_name_dataset)
		self.text_selected_algorith = QLabel("") 
		self.window_size_motifs = QLabel("")
		self.label_window_size = QLabel(properties["WINDOW_RANGE"])
		self.label_selected_algorith = QLabel(properties["SELECTED_ALG"])

		hiperparametros_form_layout.addRow(self.label_window_size, self.window_size_motifs)
		hiperparametros_form_layout.addRow(self.label_selected_algorith, self.text_selected_algorith)
		self.group_box_form = QGroupBox()
		self.group_box_form.setLayout(hiperparametros_form_layout)

		menus_layout.addWidget(self.button_load_dataset)
		menus_layout.addWidget(self.button_to_configure_hyperparameters)
		menus_layout.addWidget(self.button_to_label)
		menus_layout.addWidget(self.button_to_change_graph)
		#menus_layout.addWidget(self.title_form_hiperparametros)
		menus_layout.addWidget(self.group_box_form)
		#self.language_menu = QGroupBox()
		
		self.button_spanish = QLabel(self)
		self.button_spanish.setPixmap(QPixmap("imagenes/Spanish_btn.png").scaled(50,30))
		self.button_english = QLabel(self)
		self.button_english.setPixmap(QPixmap("imagenes/English_btn.png").scaled(50,30))

		self.button_english.mousePressEvent = self.change_to_english
		self.button_spanish.mousePressEvent = self.change_to_spanish

		self.language_menu = QHBoxLayout()
		self.language_menu.addWidget(self.button_spanish)
		self.language_menu.addWidget(self.button_english)

		self.language_menu.setAlignment(Qt.AlignRight)
		
		menus_layout.addStretch() 
		menus_layout.addLayout(self.language_menu)
		main_layout = QGridLayout()
		main_layout.addLayout(menus_layout,0,3)

		main_layout.addLayout(graph_layout,0,0,1,3)
		main_widget.setLayout(main_layout)

		self.setCentralWidget(main_widget) #Asignar a la ventana la distribucion de los controles
		title = u"".join([properties["TITLE_APP"]," - Christian Luna"])
		self.setWindowTitle(title)
		self.resize(1500, 750)
		#self.setWindowIcon(QIcon('imagenes/Logo_UCO.png'))

		self.df = None
		self.current_hiperparameter = None
		self.selected_algorith = None
		self.set_styles()
		self.show()

	def load_language_property(self,current_language:str):
		"""
		Función encargada de cargar el idioma proporcionado.
		:param current_language: Idioma que se desea cargar

		:return: Diccionario que contiene los diferentes textos de la aplicación para el idioma indicado.
		"""

		path = "internationalization/"
		name_file_properties = f"{current_language}_properties.json"
		try:
			file_properties = open(path + name_file_properties)
		except:
			print("Error: Ha ocurrido un problema al cargar el idioma elegido, se utilizará el idioma por defecto.")
			default_file_properties = "spanish_properties.json"
			self.current_language = "spanish"
			file_properties = open(path + default_file_properties)
		try:
			properties = json.load(file_properties)
		except:
			print("Error: Ha ocurrido un problema al procesar el fichero del idioma seleccionado.")
			file_properties.close() # Cerramos el fichero de propieades que se ha intentado procesar y cargamos el nuevo
			self.current_language = "spanish"
			default_file_properties = "spanish_properties.json"
			file_properties = open(path + default_file_properties)
			properties = json.load(file_properties)

		file_properties.close()		

		return properties

	def show_graph(self):
		"""
		Función encargada de mostrar el gráfico de la aplicación tras cargar el dataset.
		
		:return: 
		"""
		if self.df is None:
			warning_dialog = QMessageBox()
			warning_dialog.setWindowTitle("TITLE_ISSUE")
			warning_dialog.setText("ERROR_NOT_DATABASE")
			warning_dialog.setIcon(QMessageBox.Warning)
			self.set_styles(warning_dialog)
			warning_dialog.exec_()
		else: 
			fig = graphs.generate_interactive_time_series_plot(self.df)			
			self.graph_widget.setHtml(fig.to_html(include_plotlyjs='cdn'))

	def load_dataset(self):
		"""
		Función encargada de solicitar el fichero de la base de datos, leer y gráficarlo. En caso de error notificará el error encontrado.
		
		:return: 
		"""

		filter_type = "Data File (*.csv)"
		properties = self.load_language_property(self.current_language)
		caption_load_dataset = properties["CAPTION_LOAD_DATASET"]
		response = QFileDialog.getOpenFileName(parent = self, caption=caption_load_dataset, directory = os.getcwd(), filter = filter_type)
	
		#Comprobamos que se ha seleccionado un fichero 
		if response[0] != response[1]:
			name_file_dataset = response[0]
			sub_index_name_file = name_file_dataset.rfind("/")+1
			index_extension = name_file_dataset.rfind(".csv")
			name_dataset = name_file_dataset[sub_index_name_file:index_extension]
			try:
				self.df = pd.read_csv(name_file_dataset)
				if len(self.df.columns) != 2:
					error_dialog =  QMessageBox()
					error_dialog.setWindowTitle(properties["TITLE_ISSUE"])
					error_dialog.setText(properties["ERROR_NUM_COLUMNS_DATASET"])
					error_dialog.setIcon(QMessageBox.Critical)
					self.set_styles(error_dialog)
					error_dialog.exec_()
					del self.df
					self.df = None				
			except:
				error_dialog = QMessageBox()
				error_dialog.setWindowTitle(properties["TITLE_ISSUE"])
				error_dialog.setText(properties["ERROR_READ_DATASET"] + name_dataset)
				error_dialog.setIcon(QMessageBox.Critical)
				self.set_styles(error_dialog)
				error_dialog.exec_()
				self.df = None
			
			if not(self.df is None):
				self.label_name_dataset.setText(name_dataset)
				self.show_graph()

	def show_popup_menu(self):
		"""
		Función encargada de mostrar el menú de configuración.
		
		:return: 
		"""
		properties = self.load_language_property(self.current_language)
		if self.df is None:
			warning_dialog = QMessageBox()
			warning_dialog.setWindowTitle(properties["TITLE_ISSUE"])
			warning_dialog.setText(properties["ERROR_NOT_DATABASE"])
			warning_dialog.setIcon(QMessageBox.Warning)
			self.set_styles(warning_dialog)
			warning_dialog.exec_()
		else:
			self.menu_hyperparameter_dialog = QDialog()
			self.menu_hyperparameter_dialog.setWindowTitle(properties["CONF_HYPERPARAMETERS"])
			
			tmp = QVBoxLayout()
			
			menu_hyper_layout = QHBoxLayout()

			hyper_matrix_profile_form_layout = QFormLayout()
			hyper_matrix_profile_form_layout.addRow(QLabel(properties["MP_HYPERPARAMETERS"]))
			self.window_size_motifs_selector = QComboBox()
			if not(self.df is None):
				len_time_serie = self.df.shape[0]
				for size_motif in range(10, len_time_serie // 3):
					if len_time_serie % size_motif == 0:
						self.window_size_motifs_selector.addItem(str(size_motif))
			
			self.window_size_motifs_selector.setCurrentIndex(self.window_size_motifs_selector.count()-1)
			self.window_size_motifs_selector.currentIndexChanged.connect(self.check_size_window_motifs)
			
			self.num_cpu_selector = QComboBox()
			num_cpu_available = self.get_number_cpu_available()
			for i in range(1, num_cpu_available):
				self.num_cpu_selector.addItem(str(i))

			self.algorithm_selector = QComboBox()
			algoritm_names_list = ["Exh_Sec", "Exh_Par", "EVO_VF", "EVO_VV"]
			for i in algoritm_names_list:
				self.algorithm_selector.addItem(properties[i])
			
			self.algorithm_selector.currentIndexChanged.connect(self.check_available_selector_by_algorithm)

			hyper_matrix_profile_form_layout.addRow(QLabel(properties["WINDOW_RANGE"]), self.window_size_motifs_selector)
			hyper_matrix_profile_form_layout.addRow(QLabel(properties["NUM_CPU"]), self.num_cpu_selector)
			hyper_matrix_profile_form_layout.addRow(QLabel(properties["SELECT_ALG"]), self.algorithm_selector)


			hyper_matrix_profile_group_box_form = QGroupBox()
			hyper_matrix_profile_group_box_form.setLayout(hyper_matrix_profile_form_layout)

			self.population_size_line_edit = QLineEdit()
			self.num_epocs_line_edit = QLineEdit()
			self.crossover_operator_selector = QComboBox()
			type_crossover_op = ["CROOSOVER_1PT", "CROOSOVER_2PT"]
			
			for i in type_crossover_op:
				self.crossover_operator_selector.addItem(properties[i])
			
			self.mutate_operator_selector = QComboBox()
			type_mutate_op = ["MUTATE_1", "MUTATE_2", "MUTATE_3"]
			for i in type_mutate_op:
				self.mutate_operator_selector.addItem(properties[i])
			self.replace_operator_selector = QComboBox()
			type_replace_op = ["REPLACE_1", "REPLACE_2", "REPLACE_3"]
			for i in type_replace_op:
				self.replace_operator_selector.addItem(properties[i])
		
			hyper_genetic_alg_form_layout = QFormLayout()
			hyper_genetic_alg_form_layout.addRow(QLabel(properties["SELECTED_HYPERPARAMETERS"]))
			hyper_genetic_alg_form_layout.addRow(QLabel(properties["SELECTED_ALG"]), self.population_size_line_edit)
			hyper_genetic_alg_form_layout.addRow(QLabel(properties["NUM_EPOCS"]), self.num_epocs_line_edit)
			hyper_genetic_alg_form_layout.addRow(QLabel(properties["SELECT_CROOSOVER_OP"]), self.crossover_operator_selector)
			hyper_genetic_alg_form_layout.addRow(QLabel(properties["SELECT_MUTATE_OP"]), self.mutate_operator_selector)
			hyper_genetic_alg_form_layout.addRow(QLabel(properties["SELECT_REPLACE_OP"]), self.replace_operator_selector)
			hyper_genetic_alg_group_box_form = QGroupBox()
			hyper_genetic_alg_group_box_form.setLayout(hyper_genetic_alg_form_layout)
			menu_hyper_layout.addWidget(hyper_matrix_profile_group_box_form)
			menu_hyper_layout.addWidget(hyper_genetic_alg_group_box_form)
			#prueba.clicked.connect(self.menu_hyperparameter_dialog.close)

			menu_group_box = QGroupBox()
			menu_group_box.setLayout(menu_hyper_layout)
			tmp.addWidget(menu_group_box)
			button_close_menu_hyper = QPushButton(properties["SAVE_CHANGES"])
			button_close_menu_hyper.clicked.connect(self.save_hyperparameter)
			tmp.addWidget(button_close_menu_hyper)
			self.menu_hyperparameter_dialog.setLayout(tmp)
			self.menu_hyperparameter_dialog.setMinimumSize(1018, 450)
			self.set_styles(self.menu_hyperparameter_dialog)
			self.check_available_selector_by_algorithm(0)
			status = self.menu_hyperparameter_dialog.exec_()
	
	def check_size_window_motifs(self, size_window):
		"""
		Función encargada de determinar la complejidad del algoritmo exhaustivo y notitificar en caso de que sea muy elevada.
		:param size_window: Indica el indice del tamaño de ventana elegido
		:return: 
		"""

		a = time.time()
		size_window = int(self.window_size_motifs_selector.currentText())
		len_time_serie = self.df.shape[0]
		num_motifs_necesarios = len_time_serie // size_window 
		time_serie = self.df[self.df.columns[1]].to_numpy()
		num_find_motifs = find_motifs.find_num_motifs(time_serie, size_window)
		num_soluciones = num_motifs_necesarios ** num_find_motifs
		notacion_cientifica = format(num_soluciones, "e")
		mantisa, exponente = notacion_cientifica.split("e")
		b = time.time()
		properties = self.load_language_property(self.current_language)
		if int(exponente) > 8:
			warning_dialog = QMessageBox()
			warning_dialog.setWindowTitle(properties["WARNING_TITLE"])
			warning_dialog.setText( properties["WARNING_WINDOW_SIZE"] + str(notacion_cientifica))
			warning_dialog.setIcon(QMessageBox.Warning)
			self.set_styles(warning_dialog)
			warning_dialog.exec_()
			return 0
	
	def check_available_selector_by_algorithm(self, selected_algorithm:int):
		"""
		Función encargada de habilitar o deshabilitar los selector de la configuración de hiperparámetros dependiendo del algoritmo seleccionado.
		:param selected_algorithm: Indica el indice del algoritmo de etiquetado de series temporales seleccionado.

		:return: 
		"""

		algoritm_names_list = ["Exh_Sec", "Exh_Par", "EVO_VF", "EVO_VV"]
		properties = self.load_language_property(self.current_language)
		activate_cpu_selector = True
		activate_size_window_motifs_selector = True
		activate_size_population_line_edit = True
		activate_crossover_operator_selector = True
		activate_mutate_operator_selector = True
		activate_replace_operator_selector = True
		activate_num_epocs_line_edit = True

		if algoritm_names_list[selected_algorithm] == "Exh_Sec":
			activate_cpu_selector = activate_size_population_line_edit = False
			activate_crossover_operator_selector = activate_mutate_operator_selector = False
			activate_replace_operator_selector = activate_num_epocs_line_edit = False
		elif algoritm_names_list[selected_algorithm] == "Exh_Par":
			activate_size_population_line_edit = activate_replace_operator_selector = False
			activate_crossover_operator_selector = activate_mutate_operator_selector = False
			activate_num_epocs_line_edit = False
		elif algoritm_names_list[selected_algorithm] == "EVO_VF":
			activate_cpu_selector = False
		elif algoritm_names_list[selected_algorithm] == "EVO_VV":
			activate_size_window_motifs_selector = False

		self.num_cpu_selector.setEnabled(activate_cpu_selector)
		self.window_size_motifs_selector.setEnabled(activate_size_window_motifs_selector)
		self.population_size_line_edit.setEnabled(activate_size_population_line_edit)
		self.crossover_operator_selector.setEnabled(activate_crossover_operator_selector)
		self.mutate_operator_selector.setEnabled(activate_mutate_operator_selector)
		self.replace_operator_selector.setEnabled(activate_replace_operator_selector)
		self.num_epocs_line_edit.setEnabled(activate_num_epocs_line_edit)

	def save_hyperparameter(self):
		"""
		Función encargada de guardar la configuración de hiperparámetros elegida.
		
		:return: 
		"""

		algoritm_names_list = ["Exh_Sec", "Exh_Par", "EVO_VF", "EVO_VV"]
		properties = self.load_language_property(self.current_language)
		check_values = False
		if self.window_size_motifs_selector.count() != 0:
			try:
				value_line_edit = self.window_size_motifs_selector.currentText()
				size_window_motifs = int(value_line_edit)
			except:
				warning_dialog = QMessageBox()
				warning_dialog.setWindowTitle(properties["TITLE_ISSUE"])
				warning_dialog.setText(properties["ERROR_NOT_INTEGER_WINDOW_SIZE"] + value_line_edit)
				warning_dialog.setIcon(QMessageBox.Critical)
				self.set_styles(warning_dialog)
				warning_dialog.exec_()
				return 0
		else:
			size_window_motifs = -1
		if self.population_size_line_edit.isEnabled():
			try:
				value_line_edit = self.population_size_line_edit.text()
				size_population = int(value_line_edit)
			except:
				warning_dialog = QMessageBox()
				warning_dialog.setWindowTitle(properties["TITLE_ISSUE"])
				warning_dialog.setText(properties["ERROR_NOT_INTEGER_POPULATION_SIZE"] + value_line_edit)
				warning_dialog.setIcon(QMessageBox.Critical)
				self.set_styles(warning_dialog)
				warning_dialog.exec_()
				return 0
		else: 
			size_population = 0

		if self.num_epocs_line_edit.isEnabled():
			try:
				value_num_epocs = self.num_epocs_line_edit.text()
				num_epocs = int(value_num_epocs)
			except:
				warning_dialog = QMessageBox()
				warning_dialog.setWindowTitle(properties["TITLE_ISSUE"])
				warning_dialog.setText(properties["ERROR_NOT_INTEGER_NUM_EPOCS"] + value_num_epocs)
				warning_dialog.setIcon(QMessageBox.Critical)
				self.set_styles(warning_dialog)
				warning_dialog.exec_()
				return 0
		else: 
			num_epocs = 0

		#self.population_size.setText(str(size_population))
		current_algorithm = algoritm_names_list[self.algorithm_selector.currentIndex()]
		
		self.current_hiperparameter = {"algorithm":current_algorithm, "window_size":size_window_motifs,
										"size_population" : size_population,
										"type_crossover" : (self.crossover_operator_selector.currentIndex() + 1),
										"type_mutation" : (self.mutate_operator_selector.currentIndex() + 1),
										"type_replacement" : (self.replace_operator_selector.currentIndex() + 1),
										"num_epocs" : num_epocs,
										"num_cpu" : int(self.num_cpu_selector.currentText())
									}
		if current_algorithm == "EVO_VV":
			self.window_size_motifs.setText(properties["SIZE_VARIABLE"])
		else:
			self.window_size_motifs.setText(str(size_window_motifs))
		self.selected_algorith = current_algorithm

		self.text_selected_algorith.setText(properties[self.selected_algorith])
		info_dialog = QMessageBox()
		info_dialog.setWindowTitle(properties["TITLE_SAVE_CONF"])
		info_dialog.setText(properties["TEXT_SAVE_CONF"])
		info_dialog.setIcon(QMessageBox.Information)
		self.set_styles(info_dialog)
		info_dialog.exec_()
		self.menu_hyperparameter_dialog.close()

	def set_styles(self, other_dialog:QDialog = None):
		"""
		Función de establecer el estilo css de la aplicación.
		:param other_dialog: Indica el QDialog donde se aplicarán los estilos.

		:return: 
		"""

		style_file = open("style/dark_theme.qss")
		style = style_file.read()
		style_file.close()
		
		self.setStyleSheet(style)
		if other_dialog != None:
			other_dialog.setStyleSheet(style)

	def change_to_english(self,event):
		"""
		Función encargada de cambiar el idioma a ingles.
		:param event: Evento de hacer click en el desencadenante.

		:return: 
		"""

		self.current_language = "english"
		self.change_language()

	def change_to_spanish(self,event):
		"""
		Función encargada de cambiar el idioma a español.
		:param event: Evento de hacer click en el desencadenante.
		
		:return: 
		"""

		self.current_language = "spanish"
		self.change_language()
		
	def change_language(self):
		"""
		Función encargada de cambiar todos los textos de la aplicación al idioma seleccionado.
		
		:return: 
		"""

		properties = self.load_language_property(self.current_language)
		self.graph_label.setText(properties["TITLE_GRAPH"])
		self.button_load_dataset.setText(properties["SELECT_DATESET"])
		self.button_to_configure_hyperparameters.setText(properties["CONFIGURE_HYPERPARAMETERS"])
		self.button_to_label.setText(properties["TRAIN"])
		self.button_to_change_graph.setText(properties["CHANGE_GRAPH"])
		self.title_form_hiperparametros.setText(properties["SELECTED_HYPERPARAMETERS"])
		self.label_dataset.setText(properties["DATASET"])
		if self.df is None:
			self.label_name_dataset.setText(properties["NO_CHOICE"])
		self.label_window_size.setText(properties["WINDOW_RANGE"])
		self.label_selected_algorith.setText(properties["SELECTED_ALG"])
		if not (self.selected_algorith is None):
			self.text_selected_algorith.setText(properties[self.selected_algorith])
		title = u"".join([properties["TITLE_APP"]," - Christian Luna"])
		self.setWindowTitle(title)

	def label_time_serie(self):
		"""
		Función encargada de etiquetar la serie temporal y mostrar los resultados obtenidos.
		
		:return: 
		"""
		properties = self.load_language_property(self.current_language)
		if self.df is None:
			warning_dialog = QMessageBox()
			warning_dialog.setWindowTitle(properties["TITLE_ISSUE"])
			warning_dialog.setText(properties["ERROR_NOT_DATABASE"])
			warning_dialog.setIcon(QMessageBox.Critical)
			self.set_styles(warning_dialog)
			warning_dialog.exec_()
		else:
			if self.current_hiperparameter is None:
				warning_dialog = QMessageBox()
				warning_dialog.setWindowTitle(properties["TITLE_ISSUE"])
				warning_dialog.setText(properties["ERROR_NOT_ALGORITHM"])
				warning_dialog.setIcon(QMessageBox.Critical)
				self.set_styles(warning_dialog)
				warning_dialog.exec_()
			else:
				time_serie = self.df[self.df.columns[1]].to_numpy()
				labels_of_time_serie = self.labeling(time_serie, self.current_hiperparameter["algorithm"], self.current_hiperparameter)
				date = self.df[self.df.columns[0]].to_numpy()
				graphs_with_labels = self.button_to_change_graph.isChecked()
				df_estimate = graphs.generate_struct_to_graph(time_serie, date, labels_of_time_serie[0], graphs_with_labels)
				fig = graphs.generate_interactive_time_series_plot(df_estimate)
				del time_serie, date, df_estimate
				self.graph_widget.setHtml(fig.to_html(include_plotlyjs='cdn'))

	def get_number_cpu_available(self):
		"""
		Función encargada de determinar el número de hilos que posee el equipo donde se ejecuta el script
		:return: Devuelve el número de hilos que posee el equipo donde se ejecuta el script
		"""

		return multiprocessing.cpu_count()

	# Función del controlador
	def labeling(self, time_serie:np.array, algorithm:str, parameter:dict):
		"""
		Función encargada de realizar el etiquetado de la serie temporal proporcionada a partir del algoritmo y los parámetros proporcionados.
		:param time_serie: Indica la serie temporal a etiquetar
		:param algorithm: Indica el algoritmo que se utilizará para el etiquetado
		:param parameter: Indica el diccionario con los hiperparámetros del etiquetado
		:return: Devuelve la mejor configuración de etiquetas encontradas por los algoritmos
		"""

		if algorithm in ["Exh_Sec", "Exh_Par"]:
			model = Exhaustive_Fixed_Window(time_serie, parameter["window_size"])
			model.prepare_motifs()
			label_ts = model.label_time_serie() if algorithm == "Exh_Sec" else model.label_time_serie_parallel(parameter["num_cpu"])
		elif algorithm == "EVO_VV":
			model = Evo_Variable_Window(time_serie, 10, parameter["size_population"], parameter["type_crossover"], 
											  parameter["type_mutation"], 1, parameter["type_replacement"])
			model.prepare_motifs()
			model.generate_population()
			label_ts = model.evolve(parameter["num_epocs"])
		elif algorithm == "EVO_VF":
			model = Evo_Fixed_Window(time_serie, parameter["window_size"], parameter["size_population"], parameter["type_crossover"], 
											  parameter["type_mutation"], 1, parameter["type_replacement"])
			model.prepare_motifs()
			model.generate_population()
			label_ts = model.evolve(parameter["num_epocs"])

		return label_ts

def main():
	app = QApplication(sys.argv)
	ex = LabelingDlg()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()