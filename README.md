
# Etiquetado de Series Temporales (Trabajo Fin de Grado)

En este repositorio se encontrará todo el código desarrollado para la realización del Trabajo de Fin de Grado, para el Grado de Ingeniería Informática de la Universidad de Córdoba.

## Estructura del Repositorio
Este repositorio se encuentra dividido en diferentes carpetas con el fin de permitir una mejor comprensión del proyecto desarrollado, a continuación se encontrará el árbol de directorios para cada una de las carpetas principales.
### Codigos
En este directorio se encuentran todos los archivos necesarios para hacer uso de los algoritmos y la aplicación desarrollada. Se puede observar como en el se encuentra el fichero ```requirements.txt``` el cual permitirá llevar a cabo la instalación de las diferentes librerías que se requieren para un uso correcto. 


    ├── codigos
    │   ├── common_functions
    │   │   ├── find_motifs.py
    │   │   ├── genetic_operators.py
    │   │   └── graphs.py
    │   ├── labeling_time_series.py
    │   ├── requirements.txt
    │   ├── test.py
    │   └── user_interfaces
    │       ├── imagenes
    │       │   ├── English_btn.png
    │       │   └── Spanish_btn.png
    │       ├── internationalization
    │       │   ├── english_properties.json
    │       │   └── spanish_properties.json
    │       ├── labeling_cli.py
    │       ├── labeling_gui.py
    │       └── style
    │           └── dark_theme.qss

### Datasets
En este directorio se encuentran todas las bases de datos utilizadas en la Experimentación realizada en el Manual Técnico del Trabajo de Fin de Grado. Con el fin de que se puedan analizar y ser utilizadas fácilmente por aquellas personas que deseen comprobar el funcionamiento de los algoritmos desarrollados.

    ├── datasets
    │   ├── daily-min-temperatures.csv
    │   ├── heart_rate_pacient_1.csv
    │   ├── heart_rate_pacient_2.csv
    │   ├── heart_rate_pacient_3.csv
    │   ├── heart_rate_pacient_4.csv
    │   └── own_synthetic_datasets
    │       ├── synthetic_Seed_1.csv
    │       ├── synthetic_Seed_2.csv
    │       ├── synthetic_Seed_3.csv
    │       ├── synthetic_Seed_4.csv
    │       ├── synthetic_Seed_5.csv
    │       └── synthetic_with_white_noise
    │           ├── synthetic_Seed_1_with_white_noise.csv
    │           ├── synthetic_Seed_2_with_white_noise.csv
    │           ├── synthetic_Seed_3_with_white_noise.csv
    │           ├── synthetic_Seed_4_with_white_noise.csv
    │           └── synthetic_Seed_5_with_white_noise.csv


### Ficheros
En el se encontrarán todas los ficheros auxiliares, imagenes y resultados de las pruebas que han sido utilizados a la hora de desarrollar el TFG. 

    ├── ficheros
    │   ├── imagenes
    │   │   ├── manualUsuario
    │   │   │   ├── Configure.png
    │   │   │   ├── Home-Dataset.png
    │   │   │   ├── Home-Ingles.png
    │   │   │   ├── Home-View.png
    │   │   │   ├── LabelDataset.png
    │   │   │   ├── SaveConfigure.png
    │   │   │   ├── SaveConfigureWindow.png
    │   │   │   ├── Select-Dataset.png
    │   │   │   ├── Warning.png
    │   │   │   └── Warning-WindowSize.png
    │   │   └── pruebas_tiempos
    │   │       ├── Daily_Temperatures_time_test.png
    │   │       ├── Heart-Rate-P1-time-test.png
    │   │       └── Heart-Rate-P3-time-test.png
    │   ├── prueba_tiempos
    │   │   ├── Daily_Temperatures_time_test.csv
    │   │   ├── Heart_Rate_P1_time_test.csv
    │   │   └── Heart_Rate_P3_time_test.csv
    │   └── resultados_experimentacion
    │       ├── fitness_evo_vf_ajuste_hiper.csv
    │       ├── fitness_evo_vv_ajuste_hiper.csv
    │       ├── results_comparativa.csv
    │       ├── results_evo_exp_2.csv
    │       ├── results_evo_parallel_exp_2.csv
    │       ├── results_exhaustivo_exp_2.csv
    │       ├── results_exhaustivo_parallel_exp_2.csv
    │       ├── results_vv_evo_variable_.csv
    │       ├── tiempos_evo_vf_ajuste_hiper.csv
    │       └── tiempos_evo_vv_ajuste_hiper.csv

Finalmente en este proyecto se encuentra el ```README.md``` que pretende explicar con detalle todo el proyecto.

    └── README.md


## Autor

- [Christian Luna Escudero](https://www.github.com/ChrisLe7) Alumno de 4º curso del Grado de Ingeniería Informática – Doble Mención: Software y Computación.

## Director
- [Jose Maria Luna Ariza](https://github.com/jmluna) Profesor Contratado Doctor del Dpto. de Informática y Análisis Numérico.


## Instalación

A la hora de desear hacer uso de cualquier parte de este proyecto, se recomienda para un uso correcto utilizar Python 3.9.

### Instalación de Python
Para la instalación de Python, se pueden seguir los pasos para cada sistema operativo expuestos a continuación o hacer uso del videotutorial desarrollado, el cuál se puede encontrar en Youtube, a través del siguiente [enlace](https://www.youtube.com/watch?v=TDQdaDHtyGA).

#### Windows
1. Diríjase a la página oficial de Python y descargue la versión 3.9 del instalador de Python
para Windows.

2. Ejecute el archivo de instalación descargado y siga las instrucciones del asistente de instalación.

3. En la ventana de configuración, asegúrese de seleccionar la opción “Agregar Python 3.9 a PATH”. Esto permitirá utilizar Python desde la línea de comandos de Windows.

4. Continúe con la instalación hasta que se complete.

#### Linux 

Python suele estar incluido en casi todas las distribuciones de GNU/Linux. Si se diera el caso de que no estuviera instalado en nuestro equipo, o que a versión instalada no fuera la 3.9 se deberán de realizar los siguientes pasos para su instalación.
1. Abra un terminal y actualice el índice de paquetes apt con el comando:
```bash
sudo apt-get update
```
2. Instale Python con el comando:
```bash
sudo apt-get install python3.9
```
3. Instale el gestor de entornos de Python:
```bash
sudo apt-get install python3.9-venv
```
4. Para la creación y activación del entorno se podrá hacer uso de los siguientes comandos:
```bash
python3.9 -m venv ~/TFG
source ~/TFG/bin/activate
```
###  Instalación de las librerías requeridas
Para la instalación de las diferentes librerías se puede realizar todo fácilmente mediante el fichero `requirements.txt` que se encuentra dentro del directorio `codigos`. Se recomienda tener actualizado `pip` a la última versión disponible.

```bash
pip install -r requirements.txt
```

## Desinstalación

### Desinstalación de Python
##### Windows
Para desinstalar Python, se deberá de hacer

- Abra el menú de Inicio y busque “Agregar o quitar programas” (o “Programas y características” en versiones más recientes de Windows).
- Busca “Python” en la lista de programas instalados y selecciona la versión que se desea desinstalar.
- Haga clic en “Desinstalar” y siga las instrucciones del asistente de desinstalación.
- Una vez que se haya completado el proceso de desinstalación, asegúrese de eliminar cualquier archivo o carpeta relacionada con Python que aún pueda existir en su sistema.

#### Linux
Python no se puede desinstalar en Linux, ya que algunas partes del sistema lo necesitan para funcionar. Pero si se puede desinstalar la versión concreta realizada, para ello se pueden seguir los siguientes pasos:
1. Elimine el entorno creado:
```bash
rm -r ~/TFG
```
2. Desinstalación del gestor de entornos de python3.9 y python3.9.
```bahs
python remove python3.9-venv
python remove python3.9
```