# Databricks notebook source
# MAGIC %md
# MAGIC ### Instalar Librerias

# COMMAND ----------

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import pandas as pd
import os

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuración del Driver
# MAGIC
# MAGIC Se busca el driver en la ruta especificada, luego se configuran las opciones del navegador para finalmente iniciar con su ejecución

# COMMAND ----------

# Instalar el driver Chronium
dbutils.notebook.run("/Workspace/Shared/TFM/Driver/Instalacion_Driver", timeout_seconds=0)

# COMMAND ----------

# Configuración del servicio de ChromeDriver
service = Service('/tmp/chrome/latest/chromedriver_linux64/chromedriver')
service.start()

# Configuración del navegador Chrome
options = webdriver.ChromeOptions()
options.binary_location = "/tmp/chrome/latest/chrome-linux/chrome"
options.add_argument('headless') # Ejecutar en sin interfaz grafica
options.add_argument('--disable-extensions')  # Deshabilitar las extensiones
options.add_argument('--disable-gpu')  # Desactivar la aceleración de gráficos
options.add_argument('--disable-notifications') # Deshabilitar notificaciones
options.add_argument('--disable-infobars')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--no-sandbox')
options.add_argument('--remote-debugging-port=9222')
options.add_argument('--homedir=/tmp/chrome/chrome-user-data-dir')
options.add_argument('--user-data-dir=/tmp/chrome/chrome-user-data-dir')
prefs = {"download.default_directory":"/tmp/chrome/chrome-user-data-di",
         "download.prompt_for_download":False
}
options.add_experimental_option("prefs",prefs)

# Iniciar Chrome
driver = webdriver.Chrome(service=service, options=options)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Páginas a Escanear
# MAGIC Se enlistan las url's de Finca Raiz que serán escaneadas iterando página a página, se identificó una limitación de 400 páginas cada una con 25 ítems, en total se puede traer un máximo de 10.000 items, es por esto que se escanearán multiples url's con filtros aplicados.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Escanear URL's para Obtener elementos
# MAGIC Una vez se tiene la lista de url's se pasan a un ciclo FOR para iterar por cada una y obtener las url's de los inmuebles de cada página

# COMMAND ----------

# Lista vacia para guardar las url's
url_inmuebles = []

# for pagina in url_paginas:
#     for id_pagina in range(0,401):

#         # Generar la url dinámica
#         url = f"{pagina}{id_pagina}"

#Lanzar petición a la url generada
driver.get('https://www.metrocuadrado.com/apartamento/venta/usado')

# Esperar a que la página se cargue completamente (Segundos)
driver.implicitly_wait(20)

# Encontrar todos los elementos de etiqueta "a" que contiene las url's
elementos = driver.find_elements(By.XPATH, "//a[@class='sc-bdVaJa ebNrSm']")


# Iterar sobre los elementos encontrados y guardar la url en la lista
for elemento in elementos:
    url_inmuebles.append(elemento.get_attribute('href'))


# Cerrar el navegador
driver.quit()


# COMMAND ----------

elementos

# COMMAND ----------

# MAGIC %md
# MAGIC ### Limpiar Listas

# COMMAND ----------

# Eliminar url de otros objetos que no son necesarios
filtro = "inmueble"
url_inmuebles = [url for url in url_inmuebles if filtro in url]

# Eliminar url duplicadas
url_inmuebles = list(set(url_inmuebles))

len(url_inmuebles)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Guardar URL's en ADLS3
# MAGIC ##### Archivo ya guardado

# COMMAND ----------

try:
    #Archivo ya guardado
    ruta_archivo = '/dbfs/mnt/adls/inmuebles/Logs/Finca_Raiz/url_inmuebles_finca_raiz.csv'
    # Leer archivo de URL's
    inmuebles_file = pd.read_csv(ruta_archivo, sep=";", dtype={"Procesado":str})
except:
    dummy = {
    'URL': [],
    'Procesado': []
}
    inmuebles_file = pd.DataFrame(dummy)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Datos Nuevos

# COMMAND ----------

#Convertir lista a DF
nombre_columna = ['URL']
# Convertir la lista a un DataFrame de Pandas
inmuebles = pd.DataFrame(url_inmuebles, columns=nombre_columna)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Unión de archivos

# COMMAND ----------

result = pd.merge(inmuebles_file, inmuebles, on="URL", how="outer")
result.fillna("0", inplace=True)
result = result.drop_duplicates()

# Variables Función
ruta_archivo = '/dbfs/mnt/adls/inmuebles/Logs/Finca_Raiz/'
nombre_archivo = 'url_inmuebles_finca_raiz.csv'

# Crear el directorio si no existe
os.makedirs(ruta_archivo, exist_ok=True)

#Guardar csv
result.to_csv(ruta_archivo + nombre_archivo, index=False, sep=";")

len(result)