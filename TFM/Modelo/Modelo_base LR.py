# Databricks notebook source
# MAGIC %md
# MAGIC <center><h1>Trabajo Fin de Master</header1></center>
# MAGIC <left><h1>Exploración y modelo de predicción de precios de inmuebles en Colombia</header1></left>

# COMMAND ----------

# MAGIC %md
# MAGIC Presentado por: Erika Tatiana Arias Zuluaga y César Jiménez <br>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importación de todas las librerias y paquetes

# COMMAND ----------

# MAGIC %run "/Workspace/Shared/TFM/Modelo/Funciones_comunes"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lectura y construcción del dataset

# COMMAND ----------

folders = ["/Bronce/Finca_Raiz/", "/Bronce/Mercado_Libre/"]  # Arreglo de carpetas a explorar
final_df = read_dataset(folders)
# Mostrar el DataFrame resultante
display(final_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Estrategía y plan de acción
# MAGIC A continuación de muestra el orden de ejecución de cada tarea dentro del noteebok.
# MAGIC
# MAGIC     1. Exploración de datos
# MAGIC     2. Pre-procesamiento de datos
# MAGIC     3. Selección y extracción de caracteristicas
# MAGIC     4. Modelos predictivos
# MAGIC     5. Resultado
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Exploración de datos
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Revisar variables que estan en el dataset

# COMMAND ----------

log_error_list =[]

# COMMAND ----------

if len(log_error_list)>0:
    print("Previous Error")
else:
    try:
        print(final_df.columns.tolist())

    except Exception as e:
        print("Failed! Due to: {}".format(str(e)))
        log_error_list.append("Error: {}".format(str(e)))

# COMMAND ----------

#Código para cargar el Dataset
if len(log_error_list)>0:
    print("Previous Error")
else:
    try:
        columns_delete = ['url']
        final_df = final_df.drop(columns_delete, axis=1)

        target = 'precio'
        features = [i for i in final_df.columns if i not in [target]]
        dataset = final_df
    except Exception as e:
        print("Failed! Due to: {}".format(str(e)))
        log_error_list.append("Error: {}".format(str(e)))

# COMMAND ----------

if len(log_error_list)>0:
    print("Previous Error")
else:
    try:
        # código para comprobar la dimensión de los datos en el Dataset y conocer el numero de instancias y atributos de entrada
        print('\n\033[1mInference:\033[0m The Dataset consists of {} features & {} samples.'.format(len(features), dataset.shape[0]))
    except Exception as e:
        print("Failed! Due to: {}".format(str(e)))
        log_error_list.append("Error: {}".format(str(e)))

# COMMAND ----------

if len(log_error_list)>0:
    print("Previous Error")
else:
    try:
        # código para comprobar y conocer los tipos de datos de cada uno de los atributos de entrada y la variable objetivo del modelo
        dataset.info()
    except Exception as e:
        print("Failed! Due to: {}".format(str(e)))
        log_error_list.append("Error: {}".format(str(e)))

# COMMAND ----------

# código para comprobar el número distinto de valores que toma cada atributo
dataset.nunique().sort_values()

# COMMAND ----------

import seaborn as sns
import numpy as np

# Visualizar un boxplot de la columna que contiene los datos
sns.boxplot(x=dataset['precio'])

# Calcular estadísticas descriptivas para identificar outliers
Q1 = np.percentile(dataset['precio'], 25)
Q3 = np.percentile(dataset['precio'], 75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Identificar outliers en la columna de interés
outliers = dataset[(dataset['precio'] < limite_inferior) | (dataset['precio'] > limite_superior)]

# COMMAND ----------

#Revisión rapida de la distribucion conjunta de un par de columnas de el set de datos.
sns.pairplot(dataset[["habitaciones", "baños", "administracion", "parqueaderos","area"]], diag_kind="kde")

# COMMAND ----------

# MAGIC %md
# MAGIC -mirar lo de los outliers
# MAGIC -separa por features categoricas y numericas
# MAGIC -revisar vacios en cada una de las features
# MAGIC -imputación de datos faltantes
# MAGIC -descriptiva con los datos completos
# MAGIC -mirar duplicados por la llave y eliminar duplicados
# MAGIC -graficar matriz de correlación
# MAGIC -eliminar caractristicas no relevantes
# MAGIC -codificación y normalización de caracteristicas (mirar como guardar el paso para cuando se haga una petición desde la pagina web )
# MAGIC -calsificación de precios con rangos de 20 M
# MAGIC -serparación de datos en entrenamiento, test y validación
# MAGIC -creación de modelos y algoritmos
# MAGIC -calculos de metricas para evaluación de rendimiento
# MAGIC -comparación de los modelos a traves de graficas
# MAGIC -guardar el mejor modelo en .pkl

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Pre-procesamiento de datos

# COMMAND ----------

if len(log_error_list)>0:
    print("Previous Error")
else:
    try:
        df_reducido_cual = dataset[['id_inmueble','tipo','estado','ciudad','departamento','antiguedad','tipo_conjunto','ascensor','cuarto_util','inmobiliaria','balcon','estrato']]
        print(df_reducido_cual.shape)

        df_reducido_cuan = dataset[['precio','administracion','area','habitaciones','baños','pisos_interiores','piso','parqueaderos']]
        print(df_reducido_cuan.shape)

    except Exception as e:
        print("Failed! Due to: {}".format(str(e)))
        log_error_list.append("Error: {}".format(str(e)))

# COMMAND ----------

if len(log_error_list)>0:
    print("Previous Error")
else:
    try:
        pruebasNulls = missing_values_table(df_reducido_cuan)
        nulos = pd.DataFrame(pruebasNulls)
        nulos.reset_index(inplace=True)
        display(nulos)

    except Exception as e:
        print("Failed! Due to: {}".format(str(e)))
        log_error_list.append("Error: {}".format(str(e)))

# COMMAND ----------

if len(log_error_list)>0:
    print("Previous Error")
else:
    try:
        pruebasNulls = missing_values_table(df_reducido_cual)
        nulos = pd.DataFrame(pruebasNulls)
        nulos.reset_index(inplace=True)
        display(nulos)

    except Exception as e:
        print("Failed! Due to: {}".format(str(e)))
        log_error_list.append("Error: {}".format(str(e)))

# COMMAND ----------

if len(log_error_list)>0:
    print("Previous Error")
else:
    try:
        Columns = df_reducido_cual.columns.tolist()
        #Imputar por el valor más frecuente
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

        #Imputar las variables categóricas por el más frecuente
        df_reducido_cual = pd.DataFrame(imputer.fit_transform(df_reducido_cual), columns=Columns)

        print(df_reducido_cual.shape)

    except Exception as e:
        print("Failed! Due to: {}".format(str(e)))
        log_error_list.append("Error: {}".format(str(e)))

# COMMAND ----------

if len(log_error_list)>0:
    print("Previous Error")
else:
    try:
        Columns = df_reducido_cuan.columns.tolist()
        print(Columns)
        #Imputar por el valor más frecuente
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')

        #Imputar las variables categóricas por el más frecuente
        df_reducido_cuan = pd.DataFrame(imputer.fit_transform(df_reducido_cuan), columns=Columns)

        print(df_reducido_cuan.shape)

    except Exception as e:
        print("Failed! Due to: {}".format(str(e)))
        log_error_list.append("Error: {}".format(str(e)))

# COMMAND ----------

if len(log_error_list)>0:
    print("Previous Error")
else:
    try:
        df_reducido_fin = pd.merge(df_reducido_cuan, df_reducido_cual, left_index=True, right_index=True, how='left')
        print(df_reducido_fin.shape)

    except Exception as e:
        print("Failed! Due to: {}".format(str(e)))
        log_error_list.append("Error: {}".format(str(e)))

# COMMAND ----------

# código para conocer la distribución estadistica de los datos, por cada atributo
summary = df_reducido_fin.describe()
summary.pop("precio")
summary = summary.transpose()
summary

# COMMAND ----------

# Verificar si hay duplicados en la columna 'precio'
duplicados = df_reducido_fin[df_reducido_fin.duplicated(subset=['id_inmueble'], keep=False)]
cantidad_duplicados = duplicados.shape[0]

if cantidad_duplicados > 0:
    print("Se encontraron", cantidad_duplicados, "duplicados en la columna")
    print("Filas duplicadas:")
    print(duplicados['id_inmueble'])
else:
    print("No se encontraron duplicados en la columna")

# COMMAND ----------

df_sin_duplicados = df_reducido_fin.drop_duplicates(subset=['id_inmueble'])

# COMMAND ----------

df_sin_duplicados.shape

# COMMAND ----------

columns_with_nulls = df_sin_duplicados.columns[df_sin_duplicados.isnull().any()]
if not columns_with_nulls.empty:
    print("Columnas con valores nulos:", columns_with_nulls)
else:
    print("No existe columnas con valores nulos")

# COMMAND ----------

#eliminar variables que estan umericas pero son categoricas
df_final = df_sin_duplicados[["precio","administracion","area","habitaciones","baños","pisos_interiores","estrato","piso","parqueaderos"]]

# Calcular la matriz de correlación
correlation_matrix = df_final.corr()

# Crear un mapa de calor con la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()

# COMMAND ----------

minimo_precio = 10000000
maximo_precio = 6000000000
condition1 = df_sin_duplicados['precio'] < minimo_precio
condition2 = df_sin_duplicados['precio'] > maximo_precio

df_filtered = df_sin_duplicados[~condition1 & ~condition2]


# COMMAND ----------

df_filtered.shape

# COMMAND ----------


# df_ordenado = df_sin_duplicados.sort_values(by='precio', ascending=False)
# # Mostrar el DataFrame resultante
# display(df_ordenado)
# df_ordenado.to_csv("/dbfs/mnt/adls/inmuebles/Bronce/dfanalisisprecio.csv",sep=';')

# COMMAND ----------

# # Dividir los rangos y extraer el primer valor como numérico
# df_sin_duplicados['primer_valor'] = df_sin_duplicados['Grupo_Precio'].str.split('-', expand=True)[0].astype(int)
# df_ordenado = df_sin_duplicados.sort_values(by='primer_valor')
# # Mostrar el DataFrame resultante
# display(df_ordenado)

# COMMAND ----------

# # Definir los límites de los rangos
# # Encontrar el valor mínimo de la columna 'A'
# min_value_price = minimo_precio
# #min_value_price = df_sin_duplicados['precio'].min().astype(int)

# # Encontrar el valor máximo de la columna 'A'
# max_value_price = df_filtered['precio'].max().astype(int)
# fraccion = 200000000

# rangos = list(range(min_value_price, max_value_price + fraccion, fraccion))

# # Etiquetas para los rangos
# etiquetas = [f'{limite}-{limite + fraccion - 1}' for limite in rangos[:-1]]

# # Categorizar la columna de precio en clases por fracción de 20,000,000
# df_filtered['Grupo_Precio'] = pd.cut(df_filtered['precio'], bins=rangos, labels=etiquetas, right=False)

# display(df_filtered)

# COMMAND ----------

# código para comprobar el numero total de instancias perteneciente a cada clase dentro del conjunto de datos
total_instance_per_class = df_filtered['precio'].value_counts()
total_instance_per_class

# COMMAND ----------

cantidad_valores_unicos = df_filtered['precio'].nunique()
print("La cantidad de valores únicos en la columna es:", cantidad_valores_unicos)

# COMMAND ----------

#Gráfico para conocer la distribución visual de el numero de instancias por clase
total_instance_per_class.plot(kind = 'bar')
plt.show ()

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.hist(df_filtered['precio'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de la Columna')
plt.show()

# COMMAND ----------

# #código para conocer la cantidad de valores por categorias que contienen cada uno de los atributos
# lista_categoria = []
# columnas = ['administracion', 'area', 'tipo', 'habitaciones', 'baños', 'pisos_interiores', 'estado', 'balcon', 'estrato', 'ciudad', 'departamento', 'antiguedad', 'tipo_conjunto', 'piso', 'parqueaderos', 'ascensor', 'cuarto_util', 'inmobiliaria']
# for columna in columnas:
#   conteo_valores = pd.concat([df_filtered[columna].value_counts(), pd.Series(lista_categoria)])
#   display(conteo_valores)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Selección y extracción de caracteristicas

# COMMAND ----------

normalization = {}
columns_norm = ["administracion","area","habitaciones","baños","pisos_interiores","piso","parqueaderos"]
df, norm = preprocess_data_normalization (df_filtered, columns_norm)
for col in columns_norm:
    normalization[col] = norm

# COMMAND ----------

one_hot_encoding = {}
columns_one_hot = ["tipo","estado","tipo_conjunto","ascensor","cuarto_util","inmobiliaria","balcon"]
df, one_hot = preprocess_data_one_hot (df, columns_one_hot)
for col in columns_one_hot:
    one_hot_encoding[col] = one_hot

# COMMAND ----------

Label_encoding = {}
columns_label = ["estrato", "antiguedad", "ciudad", "departamento"]
df, label_encoder = preprocess_data_label_encoding (df, columns_label)
for col in columns_label:
    Label_encoding[col] = label_encoder

# COMMAND ----------

display(df)

# COMMAND ----------

X = df.drop(columns = ['id_inmueble', 'precio'])
y = df['precio']
#Dividir los datos en conjuntos de entrenamiento y prueba
perm = np.random.permutation(len(X))
X, y = X.iloc[perm], y.iloc[perm]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# COMMAND ----------

X_train.shape

# COMMAND ----------

X_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Modelos predictivos de Clasificación

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelo de Clasificación propuestos

# COMMAND ----------

#Código de ajuste del modelo de clasificación 1: Árboles de decisión
modelRF= LinearRegression()
modelRF.fit(X_train,y_train)
predictionsRF = modelRF.predict(X_test)

# COMMAND ----------

modelSVM = KNeighborsRegressor(n_neighbors=2)
# Entrenar el modelo
modelSVM.fit(X_train, y_train)
predictionsSVM = modelSVM.predict(X_test)

# COMMAND ----------

#Código de ajuste del modelo de clasificación 3: Ensemble Learning Boosting XGBoost
modelXG = Lasso(alpha=0.1)
modelXG.fit(X_train, y_train)
predictionsXG = modelXG.predict(X_test)


# COMMAND ----------

class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')> 0.85):
              print("\nAlcanzado el 85% de precisión, se cancela el entrenamiento!!")
              self.model.stop_training = True

# COMMAND ----------

#Definición de un modelo secuencial con dos capas ocultas densamente conectadas y una capa de salida que devuelve 7 salidas.
callbacks = myCallback()
def build_model():
  model = keras.Sequential([
    Dense(64, activation='relu', input_shape=(18,)),
    Dense(64, activation='relu'),
    Dense(1)
  ])

  model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy'])
  return model

# COMMAND ----------

#Los pasos de construcción del modelo se envuelven en una función, build_model
modelDL = build_model()

# COMMAND ----------

#Uso del método .summary para imprimir una descripción simple del modelo
modelDL.summary()

# COMMAND ----------

# Definición de la cantidad de epocas y entrenamiento del modelo de aprendizaje profundo
EPOCHS = 1000
history = modelDL.fit(X_train, y_train, epochs=EPOCHS, verbose=2, validation_data=(X_test, y_test))

# COMMAND ----------

#progreso de entrenamiento del modelo usando las estadísticas almacenadas en el objeto history. y generación de predicción con el set de test
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
test_predictions = modelDL.predict(X_test)
hist.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Resultados

# COMMAND ----------

print(mean_squared_error(y_test,predictionsRF))


# COMMAND ----------

metricsRF = eval_model(y_test, predictionsRF)
# metricsSVM = eval_model(y_test, predictionsSVM)
# metricsXG = eval_model(y_test, predictionsXG)


# COMMAND ----------

#Código para mostrar la comparación de métricas de desempeño de las 3 propuestas
#Resumen de métricas por modelo
models_compare = pd.concat([metricsRF,metricsSVM, metricsXG], ignore_index=True)
models_compare.head(6)

# COMMAND ----------

#Código para mostrar la comparación de métricas de desempeño de las dos propuestas en gráfica
models = []
models.append(('RamdomForest',RandomForestClassifier()))
models.append('SVC',SVC())
models.append(('XGB',XGBClassifier()))
results =[]
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits = 10,random_state=None)
    cv_results = cross_val_score(model,X_train,y_train,cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)'% (name,cv_results.mean(),cv_results.std()))
    
plt.boxplot(results,labels=names)
plt.title('Comparación de modelos implementados')
plt.show()

# COMMAND ----------

test_loss, test_acc = modelDL.evaluate(X_test,  y_test, verbose=2)
metrics_modelDP = {"Nombre Modelo": 'Red Neuronal',
                   "Accuracy": [test_acc]}
metricsDL = pd.DataFrame(metrics_modelDP)
metricsDL
# Para mostrar las métricas de precisión y recall por clase
from sklearn.metrics import classification_report

y_pred = np.argmax(modelDL.predict(X_test), axis=-1)
print(classification_report(y_test, y_pred))

# COMMAND ----------

#Código para mostrar la comparación de métricas de desempeño de las dos propuestas en tabla
#Resumen de métricas por modelo
models_compare = pd.concat([metricsTC,metricsLR, metricsDL], ignore_index=True)
models_compare.head(3)

# COMMAND ----------

model_package = {
    'model': model,
    'label_encoders': Label_encoding,
    'one_hot_encoder': one_hot_encoding,
    'normalization': normalization
}

# COMMAND ----------

import pickle

# Ruta del archivo en Databricks
file_path = "/dbfs/inmuebles/Silver/modelo_entrenado.pkl"

# Cargar el archivo .pkl
with open(file_path, 'wb') as file:
     pickle.dump(model_package, file)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discusión de los resultados obtenidos y argumentos sobre cómo se podrían mejorar de dichos resultados