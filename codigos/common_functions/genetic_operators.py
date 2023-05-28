"""genetic_operators.py: Funciones auxiliares para el algoritmo evolutivo, contiene todos los operadores genéticos utilizados por los algoritmos evolutivos desarrollados."""

__author__ = "Christian Luna Escudero"

__email__ = "chris.luna.e@gmail.com"

__contact__ = "https://github.com/ChrisLe7"

__status__ = "Prototype"

__date__ = 30 / 4 / 2023

__copyright__ = "Trabajo Fin de Grado para la Universidad de Córdoba"

__credits__ = "Trabajo Fin de Grado para la Universidad de Córdoba"

import random
import numpy as np

#########################################################
# Crossover Operators
#########################################################

def crossover_1pt(pattern_1, pattern_2):
	"""
	Función encargada de cruzar dos individuos, mediante el cruce en un punto. Se elige un punto de corte de forma aleatoria y se unen las partes de forma que la longitud del descendiente sea menor o igual a la de los padres.
	:param pattern_1: Padre número 1 a utilizar en el cruce.
	:param pattern_2: Padre número 2 a utilizar en el cruce.

	:return: Un nuevo individuo descendiente del cruce de los dos proporcionados.
	"""	
	crossover_point = random.randint(0, len(pattern_1) - 1)
	offspring = pattern_1[:crossover_point]
	longitud_offspring_1 = [i[1] for i in offspring]
	longitud_offspring_1 = np.array(longitud_offspring_1).sum()
	
	longitud_faltante = [i[1] for i in pattern_1[crossover_point:]]
	longitud_faltante = np.array(longitud_faltante).sum()

	longitud_actual = 0
	index = len(pattern_2) - 1
	while index >= 0:
		size_window = pattern_2[index][1] 
		if longitud_faltante - size_window >= 0:
			longitud_faltante -= size_window
			index -= 1
		else:
			index+=1
			break
	index = 0 if index == -1 else index
	offspring += pattern_2[index:]
	
	return offspring

def crossover_2pt(pattern_1, pattern_2):
	"""
	Función encargada de cruzar dos individuos, mediante el cruce en dos puntos. Se eligen los puntos de corte de forma aleatoria y se unen las partes de forma que la longitud del descendiente sea menor o igual a la de los padres.
	:param pattern_1: Padre número 1 a utilizar en el cruce.
	:param pattern_2: Padre número 2 a utilizar en el cruce.

	:return: Un nuevo individuo descendiente del cruce de los dos proporcionados.
	"""	
	crossover_point_1 = random.randint(0, len(pattern_1) - 2)
	crossover_point_2 = random.randint(crossover_point_1+1, len(pattern_1) - 1)
	#print(f"Punto de corte {crossover_point_1} y {crossover_point_2}")
	offspring_1_part_1 = pattern_1[:crossover_point_1]
	offspring_1_part_2 = pattern_1[crossover_point_2:]
	longitud_offspring_1 = [i[1] for i in offspring_1_part_1]
	longitud_offspring_1 = np.array(longitud_offspring_1).sum()
	
	longitud_faltante = [i[1] for i in pattern_1[crossover_point_1:crossover_point_2]]
	longitud_faltante = np.array(longitud_faltante).sum()
	
	longitud_actual = 0

	index = len(pattern_2) - 1
	longitud_offspring_2 = [i[1] for i in offspring_1_part_2]
	longitud_offspring_2 = np.array(longitud_offspring_2).sum()
	index_cola = len(pattern_2) - 1
	while index >= 0:
		size_window = pattern_2[index][1] 
		if longitud_offspring_2 - size_window >= 0:
			longitud_offspring_2 -= size_window
			index -= 1
			index_cola -=1
		else:
			if longitud_faltante - size_window >= 0:
				longitud_faltante -=size_window
				index -=1
			else:
				index+=1
				index_cola+=1
				break
	index = 0 if index == -1 else index
	offspring_1_part_1 += (pattern_2[index:index_cola] + offspring_1_part_2)

	return offspring_1_part_1

#########################################################
# Mutation Operators
#########################################################
	
def mutate(pattern, list_motif):
	"""
	Función encargada de mutar el patrón proporcionado. Esta mutación consiste en elegir una etiqueta (motif) del patrón e intercambiarlo por otro del mismo tamaño.
	:param pattern: Patrón que se desea modificar.
	:param list_motif: Motifs disponibles para realizar el cambio.

	:return: Patrón modificado.
	"""	
	mutate_point = random.randint(0, len(pattern) - 1)
	window_size = pattern[mutate_point][1]
	motifs_same_size = list_motif[window_size]
	replace = False
	num_motif = len(pattern)
	while replace == False and num_motif >= 0:
		if len(motifs_same_size) == 1:
			mutate_point = random.randint(0, len(pattern) - 1)
			window_size = pattern[mutate_point][1]
			motifs_same_size = list_motif[window_size]
			num_motif -= 1

		index_to_replace = random.randint(0, len(motifs_same_size) - 1)
		if motifs_same_size[index_to_replace]["motifs"][0] != pattern[mutate_point][0]:
			pattern[mutate_point][0] = motifs_same_size[index_to_replace]["motifs"][0]
			replace = True

	return pattern

def mutate_swap(pattern, list_motif):
	"""
	Función encargada de mutar el patrón proporcionado. Esta mutación consiste en intercambiar las posiciones de dos etiquetas (motifs).
	:param pattern: Patrón que se desea modificar.
	:param list_motif: Motifs disponibles para realizar el cambio.

	:return: Patrón modificado.
	"""	

	index_1, index_2 = random.sample(range(len(pattern)),2)

	pattern[index_1], pattern[index_2] = pattern[index_2], pattern[index_1]

	return pattern

def mutar_displacement(pattern, list_motif):
	"""
	Función encargada de mutar el patrón proporcionado. Esta mutación en seleccionar una etiquetas (motif) y desplazarla a otra posición.
	:param pattern: Patrón que se desea modificar.
	:param list_motif: Motifs disponibles para realizar el cambio.

	:return: Patrón modificado.
	"""	

	random_index = random.randint(0, len(pattern) - 1)
	value_to_displace = pattern.pop(random_index)
	available_positions = list(range(0, len(pattern)+1))
	available_positions.remove(random_index)
	random_index = random.choice(available_positions)
	pattern.insert(random_index, value_to_displace)

	return pattern

#########################################################
# Repair Operators
#########################################################

def repair_motif (pattern, list_motif, size_time_serie):
	"""
	Función encargada de reparar el patrón para que posea la misma longitud que la serie temporal. Para ello agregará y/o quitará elementos hasta que cumpla la condición de ser reparado.
	:param pattern: Patrón que se desea modificar.
	:param list_motif: Motifs disponibles para realizar el cambio.
	:param size_time_serie: Tamaño de la serie temporal a etiquetar.

	:return: Patrón reparado.
	"""	

	size_pattern = [i[1] for i in pattern]
	size_pattern = np.array(size_pattern).sum()
	gap = size_time_serie - size_pattern
	size_motifs_available = list(list_motif.keys())

	if gap == 0:
		return pattern

	if gap > size_time_serie //3:
		gap_tmp = gap // 3
		motifs_available = list_motif[gap_tmp]	

		motif_selected = random.choice(motifs_available)
		insert_point = random.randint(0, len(pattern) - 1)

		repair = pattern[0:insert_point] + [[motif_selected["motifs"][0], gap_tmp]] + pattern[insert_point:]
		
		return repair_motif (repair, list_motif, size_time_serie)
		
	if gap >= 10 and gap in size_motifs_available:
		motifs_available = list_motif[gap]	

		motif_selected = random.choice(motifs_available)
		insert_point = random.randint(0, len(pattern) - 1)

		repair = pattern[0:insert_point] + [[motif_selected["motifs"][0], gap]] + pattern[insert_point:]
		return repair

	remove_point = random.randint(0, len(pattern) - 1)
	# WARNING: PUEDE OCURRIR QUE EL GAP SEA MAYOR QUE EL TAMAÑO MÁXIMO size_time_serie //3
	gap += pattern.pop(remove_point)[1]
	
	if gap > size_time_serie //3:
		gap_tmp = gap // 3
		motifs_available = list_motif[gap_tmp]	

		motif_selected = random.choice(motifs_available)
		insert_point = random.randint(0, len(pattern) - 1)

		repair = pattern[0:insert_point] + [[motif_selected["motifs"][0], gap_tmp]] + pattern[insert_point:]
		
		return repair_motif (repair, list_motif, size_time_serie)
	
				
	motifs_available = list_motif[gap]	

	motif_selected = random.choice(motifs_available)
	insert_point = random.randint(0, len(pattern) - 1)

	repair = pattern[0:insert_point] + [[motif_selected["motifs"][0], gap]] + pattern[insert_point:]
	
	return repair

#########################################################
# Constructor Operators
#########################################################

def generate_individual(dic_motifs:dict, size_time_serie:int):
	"""
	Función encargada de construir un patrón a partir de los motifs (etiquetas) que son proporcionados.
	:param dic_motifs: Motifs disponibles para la creación.
	:param size_time_serie: Tamaño de la serie temporal a etiquetar.

	:return: Patrón reparado.
	"""	
	elegir_elemento = True
	len_time_serie = size_time_serie
	list_motifs = list(dic_motifs.keys())
	individuo = []
	while elegir_elemento is True:
		window_size_motif = random.choice(list_motifs)	
		if len_time_serie - window_size_motif >= 0:
			motifs_selected = random.choice(dic_motifs[window_size_motif])
			individuo.append([motifs_selected["motifs"][0],window_size_motif])
			del motifs_selected
			len_time_serie -= window_size_motif
		else:
			elegir_elemento = False
	individuo = repair_motif(individuo,dic_motifs, size_time_serie)

	return individuo


#########################################################
# Select Operators
#########################################################

def select_by_tournament(population:list,size_tournament:int):
	"""
	Función encargada de seleccionar de la población proporcionada dos individuos que ganen el torneo del tamaño indicado.
	:param population: Población de patrones.
	:param size_tournament: Tamaño del torneo.

	:return: Patrones seleccionados de la población.
	"""	

	selectedParents = []
	populationSize = len(population)

	for i in range(2):
		
		best = population[random.randint(0, populationSize - 1)]
		
		for _ in range(1,size_tournament):
		
			selected = population[random.randint(0, populationSize - 1)]
			if selected[1] < best[1]:
				best = selected

		selectedParents.append(best)


	return selectedParents

def select_random(population):
	"""
	Función encargada de seleccionar de la población proporcionada dos individuos de forma aleatoria sin repetición.
	:param population: Población de patrones.

	:return: Patrones seleccionados de la población.
	"""	

	populationSize = len(population)
	selectedParents = [population[random.randint(0, populationSize - 1)] for _ in range(2)]
	# Se puede hacer de esta otra forma si no queremos que se repitan
	#selectedParents = np.random.choice(population p=chromosome_probabilities,size = 2)
	return selectedParents

def select_by_roulette(population):
	"""
	Función encargada de seleccionar de la población proporcionada dos individuos a través de una ruleta de probabilidades.
	:param population: Población de patrones.

	:return: Patrones seleccionados de la población.
	"""	

	# Calculamos el fitness total de la poblacion
	population_fitness = sum([chromosome[1] for chromosome in population])

	# Calculamos las probabilidades de cada individuo 
	chromosome_probabilities = [chromosome[1]/population_fitness for chromosome in population]

	# Seleccionar los padres de forma aleatoria basandose en las probabilidades
	index = [i for i in range(len(population))]
	index_selectedParents = np.random.choice(index, p=chromosome_probabilities,size = 2)
	selectedParents = [population[i] for i in index_selectedParents]
	return selectedParents

#########################################################
# Replacement Operators
#########################################################

def generational(population, new_population):
	"""
	Función encargada de realizar el reemplazo generacional completo.
	:param population: Población de patrones.
	:param new_population: Población nueva.

	:return: Retorna la población nueva.
	"""	

	return new_population

def generational_with_elite (population, new_population):
	"""
	Función encargada de realizar el reemplazo generacional con un 10% de elite de la población anterior.
	:param population: Población de patrones.
	:param new_population: Población nueva.

	:return: Retorna la población actualizada.
	"""	

	elite = len(population)//10
	return new_population[:(len(population) - elite)] + population[:elite]

def full_elite(population, new_population):
	"""
	Función encargada de realizar el reemplazo quedandose con solamente los mejores individuos de ambas poblaciones.
	:param population: Población de patrones.
	:param new_population: Población nueva.

	:return: Retorna la población actualizada con los mejores de ambas poblaciones.
	"""	

	population = population + new_population
	population.sort(key = lambda x: x[1])
	return population[:len(new_population)]