"""graphs.py: Funciones para la representación de la serie temporal, permitiendo construir la serie temporal estimada a partir del conjunto de etiquetas seleccionadas."""

__author__ = "Christian Luna Escudero"

__email__ = "chris.luna.e@gmail.com"

__contact__ = "https://github.com/ChrisLe7"

__status__ = "Prototype"

__date__ = 29 / 4 / 2023

__copyright__ = "Trabajo Fin de Grado para la Universidad de Córdoba"

__credits__ = "Trabajo Fin de Grado para la Universidad de Córdoba"

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time

VERBOSE = False


def generate_interactive_time_series_plot(dataframe:pd.DataFrame):
	"""
	Construye un gráfico interactivo a partir de un pandas.DataFrame en el cuál deberá de encontrarse definidas al menos dos columnas (unidad de tiempo y serie temporal real)
	:param dataframe: Estructura que contiene la información que se desea graficar (valor de la serie temporal y estimación propuesta o motifs utilizados).
	:return: Devuelve el objeto que contiene el gráfico generado. Este podrá ser almacenado o mostrado posteriormente.
	"""

	columns = list(dataframe.columns)
	fig = px.line(dataframe, x = columns[0], y = columns[1:])
	
	fig.update_layout(bargap=0.1,legend_title_text='Serie Temporal')
	fig.update_yaxes(title_text="Valor")
	fig.update_xaxes(title_text="Unidad de Tiempo")
	
	fig.update_xaxes(
	rangeslider_visible=True,
	rangeselector=dict(
		buttons=list([
			dict(count=1, label="1m", step="month", stepmode="backward"),
			dict(count=6, label="6m", step="month", stepmode="backward"),
			dict(count=1, label="YTD", step="year", stepmode="todate"),
			dict(count=1, label="1y", step="year", stepmode="backward"),
			dict(step="all")
			])
		)
	)
	
	return fig

def create_time_series_estimation(time_serie:np.array, labels:list):
	"""
	Función encargada de generar la estimación completa de la serie temporal proporcionada a partir de las etiquetas proporcionadas.
	:param time_serie: Serie temporal donde se pretende realizar la búsqueda de los motifs.
	:param labels:  Lista que contiene las diferentes etiquetas que componen la serie temporal estimada. Cada item se compone de una lista que contiene el inicio de la subsecuencia y el tamaño de esta.
	:return: Devuelva una lista de la serie temporal estimada.
	"""

	time_serie_estimated = []
	inicio_subsecuencia = 0
	for motif in labels:
		inicio_subsecuencia = motif[0]
		size_window_motif = motif[1]
		fin_subsecuencia = inicio_subsecuencia + size_window_motif
		subsecuencia_motif = list(time_serie[inicio_subsecuencia:fin_subsecuencia])
		if len(time_serie_estimated) != 0:
			time_serie_estimated  = time_serie_estimated + subsecuencia_motif
		else:
			time_serie_estimated = subsecuencia_motif
	
	return time_serie_estimated

def create_time_series_estimation_with_labels(time_serie:np.array, labels:list):
	"""
	Función encargada de generar un diccionario que contenga las diferentes subsecuencias que han sido utilizadas para estimar la serie temporal.
	:param time_serie: Serie temporal donde se pretende realizar la búsqueda de los motifs.
	:param labels:  Lista que contiene las diferentes etiquetas que componen la serie temporal estimada. Cada item se compone de una lista que contiene el inicio de la subsecuencia y el tamaño de esta.
	:return: Estructura que almacena los diferentes índices de inicio de cada subsecuencia (etiqueta) utilizadas. La clave consiste en el índice donde se encuentra la etiqueta y el valor consiste en la subsecuencia de dicha etiqueta
	"""

	time_serie_estimated = {}
	inicio_subsecuencia = 0
	indice = 0
	for motif in labels:
		inicio_subsecuencia = motif[0]
		size_window_motif = motif[1]
		fin_subsecuencia = inicio_subsecuencia + size_window_motif
		subsecuencia_motif = list(time_serie[inicio_subsecuencia:fin_subsecuencia])
		time_serie_estimated[indice] = [inicio_subsecuencia, subsecuencia_motif]
		indice += size_window_motif
	
	return time_serie_estimated

def generate_struct_to_graph(time_serie:np.array, date:np.array, labels:list, with_labels:bool):
	"""
	Genera un diccionario que contiene todos los motifs encontrados dentro de la serie temporal, para los rangos de ventana proporcionados.
	:param time_serie: Serie temporal donde se pretende realizar la búsqueda de los motifs.
	:param date: Unidades de tiempo de cada elemento de la serie temporal.
	:param labels:  Lista que contiene las diferentes etiquetas que componen la serie temporal estimada. Cada item se compone de una lista que contiene el inicio de la subsecuencia y el tamaño de esta.
	:param with_labels: Indica si se desea generar cada etiqueta de forma independiente (True) o unificarlas (False).
	:return: Estructura que contiene la información de la serie temporal real y la estimación propuesta (ya sea de forma única o por diferentes etiquetas).
	"""

	start = time.time()
	if with_labels:
		time_serie_estimated = create_time_series_estimation_with_labels(time_serie, labels)
		df_to_graph = pd.DataFrame({'Date': date, 'Real': time_serie})
		aux = list(time_serie_estimated.keys())
		conteo_rep = {}
		for i in range(len(aux)):
			date_aux = date[aux[i]: aux[i] + labels[i][1]]
			name_column = f"Motif ({time_serie_estimated[aux[i]][0]}, {labels[i][1]})"
			if name_column in conteo_rep.keys():
				conteo_rep[name_column] += 1
				name_column+=f" " * conteo_rep[name_column]
			else:
				conteo_rep[name_column] = 1
			df_aux = pd.DataFrame({"Date":date_aux, name_column:time_serie_estimated[aux[i]][1]})
			df_to_graph = pd.merge(df_to_graph, df_aux, on="Date", how="left")
			del df_aux
		del conteo_rep
	else:
		time_serie_estimated = create_time_series_estimation(time_serie, labels)
		df_to_graph = pd.DataFrame({'Date': date, 'Real': time_serie, 'Estimada': time_serie_estimated})
	end = time.time()

	if VERBOSE:	
		print(f"Tiempo total sin paralelizar -> {end-start} segundos \n")

	return df_to_graph


if __name__ == '__main__':
	import labeling_time_series

	def main_test():
		global VERBOSE
		VERBOSE = True

		name_file = "../datasets/own_synthetic_datasets/synthetic_Seed_1.csv"
		df = pd.read_csv(name_file)
		time_serie = df[df.columns[1]].to_numpy()
		date = df[df.columns[0]].to_numpy()
		size_motif = 250
		model = labeling_time_series.Exhaustive_Fixed_Window(time_serie, size_motif)
		model.prepare_motifs()
		label_ts = model.label_time_serie()

		df_aux = generate_struct_to_graph(time_serie, date, label_ts[0], True)
		fig = generate_interactive_time_series_plot(df_aux)
		#fig.show()
		df_aux = generate_struct_to_graph(time_serie, date, label_ts[0], False)
		fig = generate_interactive_time_series_plot(df_aux)
		#fig.show()		
	
	main_test()