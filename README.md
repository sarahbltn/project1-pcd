## Análisis de la Relación entre Consumo de Café, Sueño y Estrés

Este proyecto analiza el Global Coffee Health Dataset con el objetivo de estudiar la relación entre el consumo de café, los hábitos de sueño y los niveles de estrés.
Además, se desarrolla un modelo predictivo capaz de estimar el nivel de estrés a partir de variables de salud y estilo de vida.


## Objetivos del Proyecto

**Análisis Exploratorio de Datos (EDA)**: Realizar un análisis inicial para identificar patrones, valores atípicos y la calidad general del conjunto de datos.

**Limpieza y Preprocesamiento de Datos**: Manejar valores faltantes, valores duplicados y valores atípicos, y preparar las variables categóricas para el modelado.

**Modelado Predictivo**: Desarrollar un modelo que pueda predecir los niveles de estrés basándose en el consumo de café y otros hábitos de vida.

**Visualización de Hallazgos**: Presentar los hallazgos a través de visualizaciones claras y fáciles de interpretar.


## Pasos para la Reproducción del Proyecto

1. Configuración del Entorno

Crear el entorno virutal utilizando uv e inicializar el entorno con el comando de uv init, uv sync y asegurarse que tenga las librerías necesarias, como pandas, numpy, matplotlib.pyplot, etc.

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

El siguiente paso se encuentra en el notebook `03_training_model.ipynb`, donde se lleva a cabo el entrenamiento, evaluación y registro de modelos**.  
En este archivo se implementan distintas etapas.

Dentro del archivo se realiza lo siguiente:

- División del conjunto de datos: 
  Se divide el dataset en subconjuntos de entrenamiento, validación y prueba mediante `train_test_split`:
  - 60% entrenamiento (`train`)
  - 20% validación (`val`)
  - 20% prueba (`test`)
  
  Cada división utiliza un `random_state=42` para garantizar resultados reproducibles.

- Preprocesamiento:
  El preprocesamiento se realiza a través de funciones personalizadas (`preprocessing_train` y `preprocessing_eval`) que implementan las siguientes transformaciones:
  - Escalado de variables numéricas mediante `StandardScaler`.
  - Codificación de variables categóricas con `DictVectorizer` (one-hot encoding).
  - Eliminación de variables con alta correlación (|r| > 0.9).
  - Selección de características relevantes usando información mutua.
  - Balanceo de clases en el conjunto de entrenamiento para mitigar el sesgo del modelo.

  Tanto el `DictVectorizer` como el `StandardScaler` se guardan como artefactos junto con el modelo dentro de MLflow, asegurando que el proceso de transformación sea el mismo en futuras ejecuciones o despliegues.

- Entrenamiento y evaluación de modelos:
  Se entrenan y comparan diferentes modelos supervisados (p. ej., Regresión Logística, Árbol de Decisión, Random Forest) utilizando las métricas de desempeño adecuadas.  
  Cada experimento se registra en MLflow Tracking, almacenando:
  - Parámetros de entrenamiento  
  - Métricas de evaluación  
  - Artefactos (modelos, preprocesadores, gráficos)

- Reproducibilidad:  
  Para asegurar consistencia en los resultados:
  - Todas las funciones que involucran aleatoriedad definen `random_state=42`.  
  - El preprocesamiento se guarda junto con el modelo para evitar data leakage.  
  - Las dependencias del entorno se documentan en `pyproject.toml` o `requirements.txt`.

Con este flujo se garantiza que los resultados puedan ser replicados íntegramente, desde la carga de datos limpios hasta la evaluación final del modelo. Hay que asegurarse de tener su archivo .env con su TOKEN y HOST de databricks y cambiar su correo electronico en el apartado "<tu_correo>"


Para el paso final, `00_informe_final.ipynb` se encuentra una recopilación final de todo el proyecto. Por último, en el infrome final se agregó la Servicio de inferencia (FastAPI), donde el resultado final es que el cliente pueda realizar predicciones.

La arquitectura del servicio: preprocesadores locales donde se cargan desde el contenedor los artefactos guardados en MLflow (DictVectorizer, StandardScaler y lista de features); Modelos en mlflow registry donde se descarga desde databricks.

Validación y procesamiento de inputs con esquema Pydantic y transformaciones aplicadas. Los endpoints (GET y POST) donde se puede consultar el estado del servicio y realizar predicciones respectivamente.

Streamlit, que UI actua como interfaz principal para el usuario, donde puede hacer predicciones.

El flujo de comunicación donde básicamente el usuario interactúa, la UI construye el JSON, envía un POST a la API y por último s emuestra en pantalla.

Contenerización: Contenedor Backend, Frontend y la orquestación de contenedores.

Despliegue en la nube, Space del backend, space del forntend y la integración.

## Variables de entorno 

Se creó un archivo `.env` con **DATABRICKS_HOST** y **DATABRICKS_TOKEN** para la conexión con Databricks.

## Inferencia del Modeelo

Es la etapa final del proyecto para poder rpedecir el **nivel de estrés**, lo que se realiza es la carga del modelo entrenado, se prepara el entorno de ejecución, se procesa la entrada del usuario, y se ejecuta el modelo para obtener la predicción. 


## Conclusiones 





