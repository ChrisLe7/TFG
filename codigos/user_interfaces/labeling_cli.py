"""terminal.py: Interfaz de línea de comandos para utilizar los algoritmos."""

__author__ = "Christian Luna Escudero"

__email__ = "chris.luna.e@gmail.com"

__contact__ = "https://github.com/ChrisLe7"

__status__ = "Prototype"

__date__ = 29 / 4 / 2023

__copyright__ = "Trabajo Fin de Grado para la Universidad de Córdoba"

__credits__ = "Trabajo Fin de Grado para la Universidad de Córdoba"

import argparse
import multiprocessing
import time

import sys
sys.path.append("..")  # Agrega el directorio anterior al sistema de rutas (directorio base del proyecto)

# Modulos propios
from common_functions import graphs, find_motifs

from labeling_time_series import Exhaustive_Fixed_Window, Evo_Variable_Window, Evo_Fixed_Window

def check_epocs(value_epocs):
	"""
	Función encargada de comprobar que el número de épocas proporcionado no es un valor negativo o igual a 0.
	:param value_epocs: Número de épocas proporcionado por el usuario
	:return: 
	"""

	ivalue = int(value_epocs)
	if ivalue <= 0:
		raise argparse.ArgumentTypeError(f"El número de épocas no puede ser  <= 0 valor proporcionado {value_epocs}")
	return ivalue

def check_population(value_population):
	"""
	Función encargada de comprobar que el tamaño de la población proporcionado no es un valor negativo o igual a 0.
	:param value_population: Tamaño de la población proporcionado por el usuario
	:return: 
	"""

	ivalue = int(value_population)
	if ivalue <= 0:
		raise argparse.ArgumentTypeError(f"El tamaño de la población no puede ser  <= 0 valor proporcionado {value_population}")
	return ivalue

def check_size_window_motifs(time_serie, window_size):
	"""
	Función encargada de determinar la complejidad del algoritmo exhaustivo y notificar en caso de que sea muy elevada.
	:param time_serie: Indica la serie temporal a analizar
	:param window_size: Indica el indice del tamaño de ventana elegido
	:return: 
	"""

	a = time.time()
	len_time_serie = len(time_serie)
	num_motifs_necesarios = len_time_serie // window_size 
	num_find_motifs = find_motifs.find_num_motifs(time_serie, window_size)
	num_soluciones = num_motifs_necesarios ** num_find_motifs
	notacion_cientifica = format(num_soluciones, "e")
	mantisa, exponente = notacion_cientifica.split("e")
	b = time.time()

	if int(exponente) > 8:
		print("WARNING: El tamaño de motif proporcionado posee una complejidad demasiada elevada para el exhaustivo:" + str(notacion_cientifica))
		
	

parser = argparse.ArgumentParser(description='Script que permite hacer uso de los algoritmos desarrollados para el etiquetado de series temporales' )

parser.add_argument('algorithm_selected', type=str,	metavar='ALGORITHM', choices=["EXHAUSTIVO_SEC","EXHAUSTIVO_PAR","EVOLUTIVO_VF","EVOLUTIVO_VV"], help='Algoritmo que se desea utilizar para el etiquetado de la serie temporal')

parser.add_argument('name_dataset_file', type=str, metavar='DATASET', help='Nombre del fichero donde se encuentra la serie temporal a etiquetar. Este fichero deberá de tener la extensión .csv')

parser.add_argument('-w', "--window_size", dest = "window_size", type=int, metavar='WINDOW_SIZE', help="Tamaño de ventana que se utilizará para el etiquetado de ventana fija.")

parser.add_argument('-c', "--crossover", dest = "type_crossover", type=int, metavar='CROSSOVER', choices=[1,2], help='Indica el tipo de cruce que será utilizado por el algoritmo evolutivo.  Si es 1 será utilizado el cruce en un punto, sino el cruce en dos puntos.')

parser.add_argument('-m', "--mutation", dest = "type_mutation", type=int, metavar='MUTATION', choices=[1,2,3], help='Indica el tipo de cruce que será utilizado por el algoritmo evolutivo. Indica el tipo de mutación que será utilizado por el algoritmo evolutivo. Si es 1 será utilizada la mutación de reemplazo de motifs, si es 2 será utilizado el swap de etiquetas y sino será el desplazo de una etiqueta al final de la serie temporal.')

parser.add_argument('-r', "--replacement", dest = "type_replacement", type=int, metavar='REPLACEMENT', choices=[1,2,3], help='Indica el tipo de reemplazo que será utilizado por el algoritmo evolutivo. Si es 1 será utilizado un reemplazo generacional simple, si es 2 un reemplazo generacional con elitismo y sino será utilizado un elitismo puro (quedando en todo momento en la población los mejores individuos).')

parser.add_argument('-e', "--epocs", dest = "num_epocs", type=check_epocs, metavar='NUM_EPOCS', help='Indica el número de épocas que será utilizado por el algoritmo evolutivo.')

parser.add_argument('-p', "--population_size", dest = "population_size", type=check_population, metavar='POPULATION_SIZE', help='Indica el tamaño de la población para el algoritmo evolutivo.')

parser.add_argument('-W', "--with_label", dest = "graph_with_label", action='store_true', help='Indica si se desea generar cada etiqueta de forma independiente (True) o unificarlas (False).')

parser.add_argument('-n', "--num_threads", dest = "num_threads", type=int, metavar='NUM_THREADS', choices=list(range(1, multiprocessing.cpu_count()+1)), help='Indica el número de hilos que se podrán utilizar para la paralelización.')

parser.add_argument('-s', "--separator", dest = "separator", type=str, metavar='SEPARATOR', help='Indica el separador del fichero de la base de datos. Por defecto es ,')

args = parser.parse_args()

algorithm_selected = args.algorithm_selected

name_dataset_file = args.name_dataset_file

window_size = args.window_size

type_crossover = args.type_crossover if args.type_crossover else 1 

type_mutation = args.type_mutation if args.type_mutation else 1 

type_replacement = args.type_replacement if args.type_replacement else 1 

graph_with_label = args.graph_with_label

num_threads = args.num_threads if args.num_threads else 1

num_epocs = args.num_epocs if args.num_epocs else 100

population_size = args.population_size if args.population_size else 100

separator = args.separator if args.separator else ","

import pandas as pd
import os

def check_dataset(name_dataset_file:str):
	"""
	Función encargada de comprobar el estado del fichero proporcionado.
	:param name_dataset_file: Indica el path del fichero que se desea utilizar
	:return: Devuelve un Booleano el cual indica si el fichero existe y posee la extensión adecuada.
	"""

	if not os.path.isfile(name_dataset_file):
		print("Lo sentimos el fichero no existe")
		return False
	if not name_dataset_file.endswith(".csv"):
		print("El fichero no posee la extensión de .csv, este fichero no puede ser utilizado.")
		return False

	return True

def read_dataset(name_dataset_file:str, sep:str):
	"""
	Función encargada de leer la base de datos a utilizar
	:param name_dataset_file: Indica el path del fichero que se desea utilizar
	:param sep: Indica el separador de la base de datos a utilizar
	:return: Devuelve un pd.DataFrame que contendrá la base de datos en caso de cumplir ciertos requisitos.
	"""

	df = pd.DataFrame()
	if check_dataset(name_dataset_file):
		try:
			df = pd.read_csv(name_dataset_file, sep=sep)
		except:
			print("Lo sentimos el fichero no posee el formato adecuado para la lectura")

	return df


def main():
	# Lectura y cargado de la serie temporal
	start_preprocessing_time = time.time()
	df = read_dataset(name_dataset_file, separator)
	if df.empty:
		return -1

	if len(df.columns) != 2:
		print("Lo sentimos el fichero no posee el formato adecuado para el etiquetado")
		del df
		return -1

	time_serie = df[df.columns[1]].to_numpy()
	date = df[df.columns[0]].to_numpy()
	del df
	end_preprocessing_time = time.time()
	# Se comprueba ahora el algoritmo seleccionado
	if algorithm_selected in ["EXHAUSTIVO_SEC","EXHAUSTIVO_PAR","EVOLUTIVO_VF"] and window_size is None:
		print("Lo sentimos pero el algoritmo seleccionado es de tamaño de ventana fijo y se necesita proporcionar un tamaño de ventana.")
		return -1

	if algorithm_selected in ["EXHAUSTIVO_SEC","EXHAUSTIVO_PAR"]:
		check_size_window_motifs(time_serie, window_size)
		try:
			model = Exhaustive_Fixed_Window(time_serie, window_size)
			model.prepare_motifs()
			label_ts = model.label_time_serie() if algorithm_selected == "EXHAUSTIVO_SEC" else model.label_time_serie_parallel(num_threads)
			if algorithm_selected == "EXHAUSTIVO_SEC" and num_threads != 1:
				print("Se ha utilizado el algoritmo exhaustivo secuencial a pesar de proprocionar un número de hilos distinto a 1.")
		except Exception as e:
			print(e)
			return -1

	if algorithm_selected == "EVOLUTIVO_VF":
		try: 
			model = Evo_Fixed_Window(time_serie, window_size, population_size, type_crossover, type_mutation, 1, type_replacement)
			model.prepare_motifs()
			model.generate_population()
			label_ts = model.evolve(num_epocs)
		except Exception as e:
			print(e)
			return -1

	if algorithm_selected == "EVOLUTIVO_VV":
		variable_size_window = 10
		try:
			model = Evo_Variable_Window(time_serie, variable_size_window, population_size, type_crossover, type_mutation, 1, type_replacement)
			model.prepare_motifs(num_threads)
			model.generate_population()
			label_ts = model.evolve(num_epocs)
		except Exception as e:
			print(e)
			return -1

	df_result = graphs.generate_struct_to_graph(time_serie, date, label_ts[0], graph_with_label)
	del time_serie, date

	fig = graphs.generate_interactive_time_series_plot(df_result)
	end_algorithm = time.time()
	print(f"Etiquetas utilizadas {label_ts[0]}, cometen un RMSE de {label_ts[1]}")
	print(f"Tiempo requerido para el preprocesado: {end_preprocessing_time - start_preprocessing_time}")
	print(f"Tiempo completo para el etiquetado: {end_algorithm - end_preprocessing_time}")
	fig.show()

if __name__ == '__main__':
	main()