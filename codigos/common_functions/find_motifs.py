"""find_motifs.py: Funciones para la búsqueda de motifs dentro de series temporales, para el desarrollo del etiquetado de series temporales."""

__author__ = "Christian Luna Escudero"

__email__ = "chris.luna.e@gmail.com"

__contact__ = "https://github.com/ChrisLe7"

__status__ = "Prototype"

__date__ = 29 / 4 / 2023

__copyright__ = "Trabajo Fin de Grado para la Universidad de Córdoba"

__credits__ = "Trabajo Fin de Grado para la Universidad de Córdoba"

import numpy as np
import time

import matrixprofile as mp
from multiprocessing import Pool

LIMIT_STOMP_VS_SCRIMP = 16
NUM_OF_MOTIFS = 50
VERBOSE = False

def generate_struct_motifs(time_serie:np.array, min_window_size:int, max_window_size:int, step:int = 1):
	"""
	Genera un diccionario que contiene todos los motifs encontrados dentro de la serie temporal, para los rangos de ventana proporcionados.
	:param time_serie: Serie temporal donde se pretende realizar la búsqueda de los motifs.
	:param min_window_size: Tamaño de ventana mínima que se tendrá en cuenta para la búsqueda de los motifs.
	:param max_window_size: Tamaño de ventana máximo que se tendrá en cuenta para la búsqueda de los motifs
	:param step: Entero que especifica el incremento que se tendrá en cuenta para el rango de ventana. Por defecto, tomará el valor 1.
	:return: Estructura que almacena los diferentes motifs encontrados para los diferentes tamaños de ventana indicados. La clave consiste en el tamaño 
			 de ventana y el valor consiste en una lista con los indices de inicio de los diferentes motifs.
	"""

	timeTotal = 0
	list_motifs = {}
	for i in range(min_window_size, max_window_size+1, step):
		window_size = i
		start = time.time()
		
		if window_size < LIMIT_STOMP_VS_SCRIMP:
			results = mp.algorithms.stomp(time_serie,window_size)
		else:
			results = mp.algorithms.scrimp_plus_plus(time_serie,window_size, random_state = 0)
		
		motifs = mp.discover.motifs(results,k=NUM_OF_MOTIFS)
		dic_motifs = motifs["motifs"]
		end = time.time()
		
		list_motifs[window_size] = dic_motifs
		del results, motifs, dic_motifs
		
		timeTotal += end-start
	if VERBOSE:
		print(f"Tiempo total sin paralelizar -> {timeTotal} segundos \n")
	return list_motifs

def __struct_motifs__(time_serie:np.array, window_size:int):
	"""
	Genera una tupla con el tamaño de ventana y los motifs encontrados dentro de la serie temporal del tamaño proporcionado.
	:param time_serie: Serie temporal donde se pretende realizar la búsqueda de los motifs.
	:param window_size: Tamaño de ventana mínima que se tendrá en cuenta para la búsqueda de los motifs.
	
	:return: Estructura que almacena los diferentes motifs encontrados para el tamaño de ventana proporcionado.
	"""

	if window_size < LIMIT_STOMP_VS_SCRIMP:
		results = mp.algorithms.stomp(time_serie,window_size)
	else:
		results = mp.algorithms.scrimp_plus_plus(time_serie,window_size, random_state = 0)
	motifs = mp.discover.motifs(results,k=NUM_OF_MOTIFS)
	del results
	dic_motifs = motifs["motifs"]
	del motifs
	end = time.time()

	return (window_size,dic_motifs)

def __function_to_parallelize__(arg:list):
	"""
	Función auxiliar para la paralelización de generate_struct_motifs_parallel
	:param arg: Debe proprocionar los argumentos necesarios para la función __struct_motifs__
	
	:return: Devuelve lo proporcionado por la función __struct_motifs__
	"""

	return __struct_motifs__(arg[0],arg[1])

def generate_struct_motifs_parallel(time_serie:np.array, min_window_size:int, max_window_size:int, incremento:int = 1, num_threads:int = 2):
	"""
	Genera un diccionario que contiene todos los motifs encontrados dentro de la serie temporal, para los rangos de ventana proporcionados.
	Hará uso de paralelización para minimizar el tiempo requerido.
	:param time_serie: Serie temporal donde se pretende realizar la búsqueda de los motifs.
	:param min_window_size: Tamaño de ventana mínima que se tendrá en cuenta para la búsqueda de los motifs.
	:param max_window_size: Tamaño de ventana máximo que se tendrá en cuenta para la búsqueda de los motifs
	:param step: Entero que especifica el incremento que se tendrá en cuenta para el rango de ventana. Por defecto, tomará el valor 1.
	:param num_threads: Entero que especifica el número de hilos que se utilizará para la construcción de la estructura. Por defecto, toma el valor 2.
	:return: Estructura que almacena los diferentes motifs encontrados para los diferentes tamaños de ventana indicados. La clave consiste en el tamaño 
			 de ventana y el valor consiste en una lista con los indices de inicio de los diferentes motifs.
	"""

	timeTotal = 0
	list_motifs = {}

	start = time.time()
	vectorTuplas = [[time_serie, window_size] for window_size in range(min_window_size, max_window_size+1, incremento)]
	
	with Pool(processes=num_threads) as pool:
		result = pool.map(__function_to_parallelize__, vectorTuplas)
	
	for i in result:
		if i[1]: 
			list_motifs[i[0]] = i[1]

	end = time.time()
	
	if VERBOSE:	
		print(f"Tiempo total sin paralelizar -> {end-start} segundos \n")

	return list_motifs

def find_num_motifs(time_serie:np.array, window_size:int):
	"""
	Determina el número de motifs que existen dentro de la serie temporal para el tamaño de ventana indicado.
	:param time_serie: Serie temporal donde se pretende realizar la búsqueda de los motifs.
	:param window_size: Tamaño de ventana mínima que se tendrá en cuenta para la búsqueda de los motifs.
	
	:return: Número de motifs encontrados dentro de la serie temporal
	"""

	matrix_profile_struct = mp.algorithms.scrimp_plus_plus(time_serie,window_size, random_state=0)
	if window_size < LIMIT_STOMP_VS_SCRIMP:
		matrix_profile_struct = mp.algorithms.stomp(time_serie,window_size)
	else:
		matrix_profile_struct = mp.algorithms.scrimp_plus_plus(time_serie,window_size, random_state = 0)
	motifs = mp.discover.motifs(matrix_profile_struct,k=NUM_OF_MOTIFS)["motifs"]

	num_motif = len(motifs)
	del matrix_profile_struct, motifs
	
	return num_motif

if __name__ == '__main__':
	import pandas as pd
	
	def main_test():
		global VERBOSE
		VERBOSE = True
		name_file = "../datasets/own_synthetic_datasets/synthetic_Seed_1.csv"
		df = pd.read_csv(name_file)
		time_serie = df[df.columns[1]].to_numpy()
		motifs = generate_struct_motifs(time_serie, 10, len(time_serie)//3)
		print(motifs)
		motifs = generate_struct_motifs_parallel(time_serie, 10, len(time_serie)//3, num_threads= 6)
	
	main_test()