## Análisis de la Relación entre Consumo de Café, Sueño y Estrés
Este proyecto se enfoca en analizar el Global Coffee Health Dataset para entender la relación entre el consumo de café y los niveles de estrés. El objetivo principal es desarrollar un modelo capaz de estimar el nivel de estrés a partir de diferentes variables de salud y estilo de vida.
---

## Objetivos del Proyecto
Análisis Exploratorio de Datos (EDA): Realizar un análisis inicial para identificar patrones, valores atípicos y la calidad general del conjunto de datos.

Limpieza y Preprocesamiento de Datos: Manejar valores faltantes, valores duplicados y valores atípicos, y preparar las variables categóricas para el modelado.

Modelado Predictivo: Desarrollar un modelo que pueda predecir los niveles de estrés basándose en el consumo de café y otros hábitos de vida.

Visualización de Hallazgos: Presentar los hallazgos a través de visualizaciones claras y fáciles de interpretar.
---

## Pasos para la Reproducción del Proyecto

1. Configuración del Entorno
Inicializar el entorno y asegurarse que tenga las librerías necesarias.

2. Obtención de los Datos
El conjunto de datos utilizado, Global Coffee Health Dataset, se encuentra en la carpeta de data/raw, también está disponible en Kaggle en el siguiente enlace:

https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset/data

3. Ejecución del Código
El EDA (Análisis Descriptivo de Datos) se encuentra en el notebook 01_eda_incial.ipynb. Para correr el archivo puede ser celda por celda para un análisis paso a paso, o run all.

El notebook realiza las siguientes tareas:

Carga de Datos: Lee el archivo CSV y muestra un resumen inicial.

Análisis Exploratorio de Datos (EDA):

Identifica la cantidad de registros y columnas.

Revisa el tipo de dato de cada columna y encuentra valores faltantes.

Detecta y analiza la distribución de las variables numéricas y categóricas.

Calcula y visualiza la matriz de correlación para identificar relaciones entre las variables, como la alta correlación entre Coffee_Intake y Caffeine_mg.

Limpieza de Datos (Data Wrangling):

Elimina columnas con un alto porcentaje de valores nulos (e.g., Health_Issue).

Remueve columnas con alta correlación para evitar la multicolinealidad (e.g., Caffeine_mg).

Codifica las variables categóricas para que puedan ser usadas en el modelo.

Realiza un balanceo de la variable objetivo Stress_Level.

Modelado (Pendiente): Las conclusiones del informe indican que la siguiente fase del proyecto es definir el conjunto de entrenamiento y prueba, seleccionar métricas y aplicar modelos predictivos para estimar el nivel de estrés.

4. Conclusiones y Futuro del Proyecto
El informe 00_informe_final.ipynb detalla las decisiones clave tomadas durante el proceso, como la eliminación de ciertas variables y el balanceo de los datos. Las siguientes etapas del proyecto se centrarán en la selección y entrenamiento de modelos de machine learning y la evaluación de su rendimiento para predecir los niveles de estrés de manera efectiva.
---