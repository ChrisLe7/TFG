import unittest
import copy
import random
import pandas as pd
import numpy as np
import plotly
import matrixprofile as mp 

from common_functions import graphs, find_motifs, genetic_operators
import labeling_time_series

NAME_DATASET_TEST = "../datasets/own_synthetic_datasets/synthetic_Seed_1.csv"

class Test_Genetic_Operator(unittest.TestCase):
	
	def setUp(self):
		self.pattern_1 = [["A",3],["B",1],["C",4],["E",3],["L",4]]
		self.pattern_2 = [["B",1],["B",1],["A",3],["A",3],["D",2],["B",1],["A",3]]
		self.size_time_serie = 15
		self.list_motifs = {1:[{"motifs":["B"]},{"motifs":["Z"]}],2:[{"motifs":["D"]},{"motifs":["J"]}],3:[{"motifs":["A"]},{"motifs":["E"]}],4:[{"motifs":["C"]},{"motifs":["L"]}]}
	
	def test_crossover1(self):
		offspring_1 = genetic_operators.crossover_1pt(self.pattern_1, self.pattern_2)
		self.assertNotEqual(offspring_1, self.pattern_1)

	def test_crossover2(self):
		offspring_1 = genetic_operators.crossover_2pt(self.pattern_1, self.pattern_2)
		self.assertNotEqual(offspring_1, self.pattern_1)
	
	def test_mutate(self):
		copia_aux = copy.deepcopy(self.pattern_1)
		mutate_offspring = genetic_operators.mutate(copia_aux, self.list_motifs)
		self.assertNotEqual(mutate_offspring, self.pattern_1, "Fail to mutate pattern_1")
		copia_aux = copy.deepcopy(self.pattern_2)
		mutate_offspring = genetic_operators.mutate(copia_aux, self.list_motifs)
		self.assertNotEqual(mutate_offspring, self.pattern_2, "Fail to mutate pattern_1")
	
	def test_mutate_swap(self):
		copia_aux = copy.deepcopy(self.pattern_1)
		mutate_offspring = genetic_operators.mutate_swap(copia_aux, self.list_motifs)
		self.assertNotEqual(mutate_offspring, self.pattern_1)

	def test_mutate_displacement(self):
		copia_aux = copy.deepcopy(self.pattern_1)
		mutate_offspring = genetic_operators.mutar_displacement(copia_aux, self.list_motifs)
		self.assertNotEqual(mutate_offspring, self.pattern_1)

	def test_mutate_displacement(self):
		copia_aux = copy.deepcopy(self.pattern_1)
		mutate_offspring = genetic_operators.mutar_displacement(copia_aux, self.list_motifs)
		self.assertNotEqual(mutate_offspring, self.pattern_1)

	'''
	def test_repair_motif(self):
		copy_aux = copy.deepcopy(self.pattern_1)
		random_index = random.randint(0, len(copy_aux) - 1)
		value_to_displace = copy_aux.pop(random_index)
		offspring_1 = genetic_operators.repair_motif(copy_aux, self.list_motifs, self.size_time_serie)
		size_time_serie = [i[1] for i in copy_aux]
		size_time_serie = sum(size_time_serie)
		self.assertEqual(size_time_serie, self.size_time_serie)
	'''

class Test_Find_Motifs(unittest.TestCase):
	def setUp(self):
		name_file = NAME_DATASET_TEST
		self.df = pd.read_csv(name_file)
		self.time_serie = self.df[self.df.columns[1]].to_numpy()

	def test_generate_struct_motifs(self):
		window_size = 100
		list_motifs = {}
		if window_size < 16:
			matrix_profile_struct = mp.algorithms.stomp(self.time_serie,window_size)
		else:
			matrix_profile_struct = mp.algorithms.scrimp_plus_plus(self.time_serie,window_size, random_state = 0)
		motifs = mp.discover.motifs(matrix_profile_struct,k=50)
		
		list_motifs[window_size] = motifs["motifs"]

		resultados_libreria = find_motifs.generate_struct_motifs(self.time_serie, window_size, window_size, 1)
	
		self.assertEqual(resultados_libreria, list_motifs)

	def test_generate_struct_motifs_parallel(self):	
		window_size = 100
		resultados_libreria = find_motifs.generate_struct_motifs(self.time_serie, window_size, window_size, 1)
		resultados_libreria_parallel = find_motifs.generate_struct_motifs_parallel(self.time_serie, window_size, window_size, 1)
		self.assertEqual(resultados_libreria, resultados_libreria)

	def test_find_num_motifs(self):
		window_size = 100
		resultados_libreria = find_motifs.generate_struct_motifs(self.time_serie, window_size, window_size, 1)
		num_struct = len(resultados_libreria[window_size])
		num_motifs = find_motifs.find_num_motifs(self.time_serie, window_size)
		self.assertEqual(num_struct, num_motifs)


class Test_Graphs(unittest.TestCase):
	def setUp(self):
		name_file = NAME_DATASET_TEST
		self.df = pd.read_csv(name_file)
		self.time_serie = self.df[self.df.columns[1]].to_numpy()
		self.date = self.df[self.df.columns[0]].to_numpy()
		size_motif = 250
		model = labeling_time_series.Exhaustive_Fixed_Window(self.time_serie, size_motif)
		model.prepare_motifs()
		self.label_ts = model.label_time_serie()
		del model, size_motif

	def test_graph(self):
		fig = graphs.generate_interactive_time_series_plot(self.df)		

		self.assertIsInstance(fig, plotly.graph_objs._figure.Figure)

	def test_graph_with_labels(self):	

		df_aux = graphs.generate_struct_to_graph(self.time_serie, self.date, self.label_ts[0], True)
		self.assertIsInstance(df_aux, pd.DataFrame)
		self.assertGreaterEqual(len(df_aux.columns), 3)
		fig = graphs.generate_interactive_time_series_plot(self.df)		
		self.assertIsInstance(fig, plotly.graph_objs._figure.Figure)
		

	def test_graph_without_labels(self):
		df_aux = graphs.generate_struct_to_graph(self.time_serie, self.date, self.label_ts[0], False)
		self.assertIsInstance(df_aux, pd.DataFrame)
		self.assertEqual(len(df_aux.columns), 3)
		fig = graphs.generate_interactive_time_series_plot(self.df)		

		self.assertIsInstance(fig, plotly.graph_objs._figure.Figure)


class Test_Graphs(unittest.TestCase):
	def setUp(self):
		name_file = NAME_DATASET_TEST
		self.df = pd.read_csv(name_file)
		self.time_serie = self.df[self.df.columns[1]].to_numpy()
		self.date = self.df[self.df.columns[0]].to_numpy()
		self.size_motif = 250
		

	def test_exhaustive_fixed_window(self):
		model = labeling_time_series.Exhaustive_Fixed_Window(self.time_serie, self.size_motif)
		model.prepare_motifs()
		label_ts = model.label_time_serie()
		self.assertIsInstance(label_ts, list)
		self.assertIsInstance(label_ts[0], list)
		self.assertIsInstance(label_ts[1], float)
		self.assertIsInstance(label_ts[0][0][1], int)
		self.assertEqual(label_ts[0][0][1], self.size_motif)

	def test_evolutive_fixed_window(self):	

		model = labeling_time_series.Evo_Fixed_Window(self.time_serie, self.size_motif, 100, 1, 2, 1, 1)
		model.prepare_motifs()
		model.generate_population()
		label_ts = model.evolve(100)
		self.assertIsInstance(label_ts, list)
		self.assertIsInstance(label_ts[0], list)
		self.assertIsInstance(label_ts[1], float)		
		self.assertIsInstance(label_ts[0][0][1], int)
		self.assertEqual(label_ts[0][0][1], self.size_motif)

	def test_evolutive_variable_window(self):
		size_motif = 10
		model = labeling_time_series.Evo_Variable_Window(self.time_serie, size_motif, 100, 1, 2, 1, 1)
		model.prepare_motifs()
		model.generate_population()
		label_ts = model.evolve(100)
		self.assertIsInstance(label_ts, list)
		self.assertIsInstance(label_ts[0], list)
		self.assertIsInstance(label_ts[1], float)
		self.assertIsInstance(label_ts[0][0][1], np.int64)
		self.assertGreaterEqual(label_ts[0][0][1], size_motif)

if __name__ == '__main__':
	unittest.main()