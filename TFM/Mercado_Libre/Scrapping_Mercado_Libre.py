# Databricks notebook source
# MAGIC %md
# MAGIC ### Instalar Librerias

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import pytz
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# COMMAND ----------

# Zona horaria de Colombia
colombia_timezone = pytz.timezone('America/Bogota')

# Fecha y hora actual en UTC
fecha_hora_actual_utc = datetime.now(pytz.utc)

# Convertir la fecha y hora actual a la zona horaria de Colombia
fecha_hora_actual = fecha_hora_actual_utc.astimezone(colombia_timezone).strftime('%Y%m%d_%H%M')

print(fecha_hora_actual)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Extraer Variables Necesarias por Cada Inmueble

# COMMAND ----------

ruta_archivo = '/dbfs/mnt/adls/inmuebles/Logs/Mercado_Libre/url_inmuebles_mercado_libre.csv'

# Leer archivo de URL's
inmuebles = pd.read_csv(ruta_archivo, sep=";", dtype={"Procesado":str})

#Filtrar por No Procesados
inmuebles_a_procesar = inmuebles[inmuebles["Procesado"]=="0"]

# Convertir a lista
url_inmuebles = inmuebles_a_procesar['URL'].tolist()

print("Total: " + str(len(inmuebles)))
print("Por Procesar: " + str(len(url_inmuebles)))

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

datos_inmuebles = []
url_procesadas = []
url_no_procesadas = []

for index, inmueble in enumerate(url_inmuebles):
    # Realizar la solicitud GET a la página
    response = requests.get(inmueble)
    # Parsear el contenido HTML
    soup = BeautifulSoup(response.content, "html.parser")

    try:

        #Lanzar petición a la url generada
        driver.get(inmueble)

        # Esperar a que la página se cargue completamente (Segundos)
        driver.implicitly_wait(5)

        #Parsear el html
        html_code = driver.page_source
        soup = BeautifulSoup(html_code, "html.parser")

        #Encontrar valores
        precio = soup.find("span", class_="andes-money-amount__fraction").text.strip()
        url = inmueble
        id_inmueble_pagina = inmueble[inmueble.find("mercadolibre.com.co/")+20:inmueble.find("-", inmueble.find("-")+1)]
        ubicacion = soup.find_all("p", class_="ui-pdp-color--BLACK ui-pdp-size--SMALL ui-pdp-family--REGULAR ui-pdp-media__title")
        ubicacion = ubicacion[-1].text.strip()
        tipo = soup.find("span", class_="ui-pdp-subtitle").text.strip()
        inmobiliaria = soup.find("h2", class_="ui-pdp-color--BLACK ui-pdp-size--MEDIUM ui-pdp-family--REGULAR ui-vip-seller-profile__header mb-24").text.strip()

        #Variables obtenidas de tablas
        desc_variable=[]
        descripcion_caracteristicas = soup.find_all("div", class_="andes-table__header__container")
        for item_desc in descripcion_caracteristicas:
            desc_variable_act = item_desc.text.strip()
            desc_variable.append(desc_variable_act)

        val_variable=[]
        valor_caracteristicas = soup.find_all("span", class_="andes-table__column--value")
        for item_val in valor_caracteristicas:
            val_variable_act = item_val.text.strip()
            val_variable.append(val_variable_act)

        variables = dict(zip(desc_variable, val_variable))

        administracion = variables.get("Administración")
        area = variables.get("Área construida")
        habitaciones = variables.get("Habitaciones")
        baños = variables.get("Baños")
        pisos_interiores = variables.get("Cantidad de pisos")
        balcon = variables.get("Balcón")
        estrato = variables.get("Estrato social")
        antiguedad = variables.get("Antigüedad")
        estrato = variables.get("Estrato social")
        piso = variables.get("Número de piso de la unidad")
        parqueaderos = variables.get("Estacionamientos")
        ascensor = variables.get("Ascensor")
        tipo_conjunto = variables.get("Con barrio cerrado")
        cuarto_util = variables.get("Depósitos")

        # Generar esquema
        datos_inmueble_actual = {
            "id_inmueble_pagina": id_inmueble_pagina,
            "precio": precio,        
            "administracion": administracion,
            "area": area,
            "tipo": tipo,
            "ubicacion": ubicacion,
            "habitaciones": habitaciones,
            "baños": baños,        
            "pisos_interiores": pisos_interiores,
            "balcon": balcon,        
            "estrato": estrato,                
            "antiguedad": antiguedad,
            "tipo_conjunto": tipo_conjunto,
            "piso": piso,
            "parqueaderos": parqueaderos,
            "ascensor": ascensor,
            "inmobiliaria": inmobiliaria,
            "cuarto_util": cuarto_util,
            "url": inmueble
        }

        # Agregar el diccionario a la lista de datos_inmuebles
        #print(datos_inmueble_actual)
        print(str(index/len(url_inmuebles)*100) + " %")
        datos_inmuebles.append(datos_inmueble_actual)

        # Generar diccionario de url's procesadas
        url_procesadas_actual = {
            "URL": inmueble,
            "Procesado": "1"
        }

        url_procesadas.append(url_procesadas_actual)

        del id_inmueble_pagina
        del precio       
        del administracion
        del area
        del tipo
        del ubicacion
        del habitaciones
        del baños      
        del pisos_interiores
        del balcon        
        del estrato                
        del antiguedad
        del tipo_conjunto
        del piso
        del parqueaderos
        del ascensor
        del inmobiliaria
        del cuarto_util
        del inmueble

    except Exception as error:
      
      # Generar diccionario de url's no procesadas
      url_no_procesadas_actual = {
            "URL": inmueble,
            "Procesado": "0"
      }

      url_no_procesadas.append(url_no_procesadas_actual)
      print(f"Error {index}: {error} en {inmueble}")    


# COMMAND ----------

# Cerrar el navegador
driver.quit()

#Creación de dataframe vacio
nombre_columnas = ["id_inmueble_pagina", "precio", "administracion", "area", "tipo", "ubicacion", "habitaciones", "baños",  "pisos_interiores", "balcon", "estrato", "antiguedad", "tipo_conjunto", "piso", "parqueaderos", "ascensor",  "inmobiliaria", "cuarto_util", "url"]
info_inmuebles = pd.DataFrame(datos_inmuebles, columns=nombre_columnas)
info_inmuebles.head(10)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Guardar URL's en ADLS
# MAGIC #### URL Procesadas

# COMMAND ----------

# Función para verificar si una fila puede ser codificada en 'latin-1', en caso de no serlo se elimina
def encode_latin1(row):
    try:
        row.to_string().encode('latin-1')
        return True
    except UnicodeEncodeError:
        return False

# Filtrar las filas que se pueden codificar en 'latin-1' esto generará inconsistencia en la cantidad de registros que se escriben en la siguiente capa, ya que en las url procesadas las que generen error por encoding se catalogan como procesadas
info_inmuebles = info_inmuebles[info_inmuebles.apply(encode_latin1, axis=1)]
len(info_inmuebles)

# COMMAND ----------

# Variables Función
ruta_archivo = '/dbfs/mnt/adls/inmuebles/RAW/Mercado_Libre/Datos/'
nombre_archivo = 'datos_mercado_libre_'
extensión = '.csv'

# Crear el directorio si no existe
os.makedirs(ruta_archivo, exist_ok=True)

#Guardar csv usando latin-1 de encoding
info_inmuebles.to_csv(ruta_archivo + nombre_archivo + fecha_hora_actual + extensión, index=False, sep=";", encoding= "latin-1")

print("Archivo creado: " + ruta_archivo + nombre_archivo + fecha_hora_actual + extensión)

# COMMAND ----------

# MAGIC %md
# MAGIC #### URL No Procesadas

# COMMAND ----------

#Columnas del Dataframe
columnas = ["URL", "Procesado"]

#Convertir en df de pandas las url procesadas
df_url_procesadas = pd.DataFrame(url_procesadas, columns=columnas)

#Unir con el archivo de url, para generar un flag a los registros procesados
url_df = pd.merge(inmuebles, df_url_procesadas, on="URL", how="left")

# Esté ya procesado en el archivo o se haya procesado en esta ejecución, se deja la columna Procesado como 1
url_df['Procesado'] = np.where((url_df['Procesado_x'] == "1") | (url_df['Procesado_y'] == "1"), "1", "0")

#Seleccionar Columnas
url_df = url_df[["URL", "Procesado"]]

len(url_df)

# COMMAND ----------

# Variables Función
ruta_archivo = '/dbfs/mnt/adls/inmuebles/Logs/Mercado_Libre/'
nombre_archivo = 'url_inmuebles_mercado_libre.csv'

# Crear el directorio si no existe
os.makedirs(ruta_archivo, exist_ok=True)

#Guardar csv
url_df.to_csv(ruta_archivo + nombre_archivo, index=False, sep=";")

print('Procesadas: ' + str(len(info_inmuebles)) + 
      '\nFallaron por encoding (Se considera procesada): ' + str(len(url_procesadas)-len(info_inmuebles)) + 
      '\nTotal Procesadas (Procesadas + Fallo Encoding): ' + str(len(url_procesadas)) +
      '\nFallaron por scrapping: ' + str(len(url_no_procesadas)) + 
      '\nPendientes por procesar: ' + str(len(url_inmuebles)-len(url_procesadas)) +
      '\nTotal Procesados: ' + str(len(inmuebles)-len(url_inmuebles)+len(url_procesadas)) +
      '\nTotal URLs: ' + str(len(inmuebles)))