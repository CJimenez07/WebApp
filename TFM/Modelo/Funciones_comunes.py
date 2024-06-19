# Databricks notebook source
# MAGIC %md
# MAGIC ## Importación de todas las librerias y paquetes

# COMMAND ----------

# MAGIC %pip install --upgrade tensorflow
# MAGIC %pip install --upgrade protobuf
# MAGIC %pip install xgboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from functools import reduce
import pathlib
import math
import numpy as np
from IPython.display import display

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print(tf.__version__)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score,mean_gamma_deviance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# COMMAND ----------

def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
#         mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        mis_val_table_ren_columns = mis_val_table_ren_columns.sort_values('% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"  "There are " + str(mis_val_table_ren_columns.shape[0]) +   " columns that have missing values.")
        return mis_val_table_ren_columns

# COMMAND ----------

def read_dataset(folders):
    file_paths = []
    for folder in folders:
        files = dbutils.fs.ls("/mnt/adls/inmuebles" + folder)
        # Obtiene el nombre de la carpeta anterior en la ruta
        nombre_carpeta_anterior = folder.strip("/").split("/")[-1]
        for file in files:
            nombre_archivo = file.name
            if nombre_carpeta_anterior in nombre_archivo:
                file_paths.append(file.path)
            else:
                print(f"Archivo eliminado: {nombre_archivo}")
    # Eliminar ":" y agregar "/"
    modified_paths = [path.replace(':', '').replace('dbfs/', '/dbfs/') for path in file_paths]
    # # Leer los archivos en file_paths y cargarlos en un DataFrame
    dfs = []
    for file_path in modified_paths:
        df = pd.read_csv(file_path, sep=';', encoding='latin-1')
        if "finca_raiz/finca_raiz.csv" in file_path:
            df['cuarto_util'] = np.NA
        dfs.append(df)

    # Combinar todos los DataFrames en uno solo
    final_df = pd.concat(dfs, ignore_index=True)
    return (final_df)

# COMMAND ----------

def preprocess_data_normalization(data, columns):
    norm = MinMaxScaler()
    #["administracion","area","habitaciones","baños","pisos_interiores","piso","parqueaderos"]
    # Ajustar y transformar los datos de las columnas seleccionadas con el scaler
    scaled_data = norm.fit_transform(data[columns])
    
    # Reemplazar las columnas originales con las columnas escaladas
    data_scaled = data.copy()
    data_scaled[columns] = scaled_data
    return data_scaled, norm



# COMMAND ----------

def preprocess_data_one_hot(data, columns):
    one_hot = OneHotEncoder(sparse=False)
    for col in columns:
        if col in data.columns:
            # Reshape the data column to a 2D array
            reshaped_col = data[col].values.reshape(-1, 1)
            data[col] = one_hot.fit_transform(reshaped_col)
    return data, one_hot
 

# COMMAND ----------

def preprocess_data_label_encoding(data, columns):
    #estrato, antiguedad, ciudad, departamento
    label_encoder = LabelEncoder()    
    for col in columns:
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])    
    return data,label_encoder

# COMMAND ----------

#cálculo métricas de rendimiento y evaluación del modelo Decisión Tree
def eval_model(y_test, prediction):
    total_instance = prediction.size
    mcTC = confusion_matrix(y_test, prediction)
    class_correct = np.diag(mcTC).sum()
    class_incorrect = total_instance - class_correct
    accuTC = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction,average='weighted')
    recall = recall_score(y_test, prediction, average='weighted')
    especificidad = recall_score(y_test, prediction, average='weighted')
    FP_Rate = 1 - especificidad
    F1_Score = f1_score(y_test, prediction, average='weighted')
    metrics_modelTC = {"Nombre Modelo": 'Tree Classifer',
                "Instancias Totales": [total_instance], 
                "Intancias bien Clasificadas": [class_correct],
                "Intancias mal Clasificadas": [class_incorrect],
                "Accuracy": [accuTC],
                "Precisión":[precision],
                "Recall (TP Rate)":[recall],
                "Especificidad (TN Rate)": [especificidad],
                "FP Rate": [FP_Rate],
                "F1 Score":[F1_Score]}
    metricsTC = pd.DataFrame(metrics_modelTC)
    return metricsTC