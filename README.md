## Análisis de la Relación entre Consumo de Café, Sueño y Estrés

Este proyecto se enfoca en analizar el Global Coffee Health Dataset para entender la relación entre el consumo de café y los niveles de estrés. El objetivo principal es desarrollar un modelo capaz de estimar el nivel de estrés a partir de diferentes variables de salud y estilo de vida.


## Objetivos del Proyecto
Análisis Exploratorio de Datos (EDA): Realizar un análisis inicial para identificar patrones, valores atípicos y la calidad general del conjunto de datos.

Limpieza y Preprocesamiento de Datos: Manejar valores faltantes, valores duplicados y valores atípicos, y preparar las variables categóricas para el modelado.

Modelado Predictivo: Desarrollar un modelo que pueda predecir los niveles de estrés basándose en el consumo de café y otros hábitos de vida.

Visualización de Hallazgos: Presentar los hallazgos a través de visualizaciones claras y fáciles de interpretar.


## Pasos para la Reproducción del Proyecto

1. Configuración del Entorno
Crear el entorno virutal utilizando uv e inicializar el entorno con el comando de uv init y asegurarse que tenga las librerías necesarias, como pandas, numpy, matplotlib.pyplot, etc.

2. Obtención de los Datos
El conjunto de datos utilizado, Global Coffee Health Dataset, se encuentra en la carpeta de `data/raw` también está disponible en Kaggle en el siguiente enlace:

https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset/data

3. Ejecución del Código
Primeramente se puede encontrar el notebook `00_informe_final.ipynb` el cual contiene diferentes apartados:
- Introducción
- Antecedentes
- Objetivos
- Planteamiento del problema
- EDA
- Data Wrangling
- Conclusiones
- Referencias

Cada uno de los apartados tiene una desripción detallada sobre el contenido de este proyecto, este archivo es el reporte donde se documenta el proceso llevado a cabo en este proyecto por lo que np hay código.


El EDA se encuentra en el notebook `01_eda_incial.ipynb`. En este archivo se encuentra el código para obtener estadísticas descriptivas, visualizaciones con una explicación detallada de los hallazgos encontrados.

Dentro del archivo se puede encontrar lo siguiente:

- Carga de Datos.
- Análisis Exploratorio de Datos (EDA):
    - Se identifica la cantidad de registros y columnas.
    - Se revisa el tipo de dato de cada columna y se encuentra valores faltantes.
    - Visualizaciones para la distribución de las variables numéricas y categóricas.
    - Se calcula y visualiza la matriz de correlación para identificar relaciones entre las variables.

Por otra parte hay un tercer notebook llamado `02_data_wrangling.ipynb`, el cual a partir de las conclusiones preliminares y visualizaciones que se hicieron en el EDA, se aplicaron los tratamientos y preprocesamientos necesarios para limpiar el dataset y trabajar con los datos limpios.

Dentro de este archivo se puede ver el código necesario para la limpieza de los datos, dentro de las decisiones que se tomaron con base en el EDA se encontrará lo siguiente:

- Eliminación de la columna 'Health_Issue' por tener más del 50% valores nulos.
- Eliminación de la columna 'Caffeine_mg' por la alta correlación que mostró con la variable de consumo de café
- Codificación de las variables categóricas para que puedan ser usadas en el modelo.
- Balanceo de la variable objetivo Stress_Level ya que si no se balancea puede afectar el desempeño del modelo.

Una vez hecho el Data Wrangling, dentro de la carpeta de `processed` en la carpeta data, se encuentra el archivo `dataset_limipio.csv` el cual es el archivo ya con los datos limpios.


## Conclusiones y Futuro del Proyecto
Como se mencionó anteriormente el archivo `00_informe_final.ipynb` detalla las decisiones tomadas durante el proceso de elaboración del proyecto. 

Las siguientes etapas del proyecto se centrarán en el entrenamiento de modelos de machine learning y la evaluación de su rendimiento para predecir los niveles de estrés de manera efectiva.
