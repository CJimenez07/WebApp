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

ruta_archivo = '/dbfs/mnt/adls/inmuebles/Logs/Finca_Raiz/url_inmuebles_finca_raiz.csv'

# Leer archivo de URL's
inmuebles = pd.read_csv(ruta_archivo, sep=";", dtype={"Procesado":str})

#Filtrar por No Procesados
inmuebles_a_procesar = inmuebles[inmuebles["Procesado"]=="0"]

# Convertir a lista
url_inmuebles = inmuebles_a_procesar['URL'].tolist()

print("Total: " + str(len(inmuebles)))
print("Por Procesar: " + str(len(url_inmuebles)))

# COMMAND ----------

url_inmuebles = ["https://www.fincaraiz.com.co/apartamento-en-venta/10604863", "https://www.fincaraiz.com.co/apartamento-en-venta/10951591"]

# COMMAND ----------

datos_inmuebles = []
url_procesadas = []
url_no_procesadas = []

for index, inmueble in enumerate(url_inmuebles):

    time.sleep(5)
    # Realizar la solicitud GET a la página
    response = requests.get(inmueble)
    # Parsear el contenido HTML
    soup = BeautifulSoup(response.content, "html.parser")
    

    try:
      #Obtener elemento con la información requerida
      posibles_box = ["MuiBox-root jss331 jss329", "MuiBox-root jss330 jss328", "MuiBox-root jss315 jss313", "MuiBox-root jss285 jss283", "MuiBox-root jss304 jss302", "jsx-3405404036 container"]
      for id_box in posibles_box:
            try:
                  items_box_descripcion = soup.find("div", class_=posibles_box)
                  items_text_description = items_box_descripcion.find_all("p")
                  break
            except Exception as error:
                  pass
      

      # Crear un diccionario para almacenar las variables y sus valores
      variables = {}

      # Recorrer la lista en pasos de 2
      for i in range(0, len(items_text_description), 2):
            nombre_variable = items_text_description[i].text.strip()  # Elemento en la posición par (nombre de la variable)
            valor_variable = items_text_description[i+1].text.strip() # Elemento en la posición impar (valor de la variable)
            # Asignar al diccionario
            variables[nombre_variable] = valor_variable

      administracion = variables.get("Administración")
      antiguedad = variables.get("Antigüedad")
      area = variables.get("Área construída")
      baños = variables.get("Baños")
      estado = variables.get("Estado")
      piso = variables.get("Piso N°")
      parqueaderos = variables.get("Parqueaderos")
      pisos_interiores = variables.get("Pisos interiores")
      habitaciones = variables.get("Habitaciones")
      estrato = variables.get("Estrato")

      posibles_id_inmueble = ["jss65 jss74 jss307 jss265", "jss65 jss74 jss306 jss264", "jss65 jss74 jss291 jss249", "jss65 jss74 jss261 jss219", "jss65 jss74 jss280 jss238", "jss65 jss74 jss292 jss250"]
      for id_inm in posibles_id_inmueble:
            try:
                  id_inmueble_pagina = soup.find("p", class_=id_inm).text.split(":")[1].strip()
                  break
            except Exception as error:
                  pass

      #Otras Variables a partir de etiquetas fijas
      inmobiliaria = soup.find("span", class_="MuiChip-label").text.strip()
      precio = soup.find("p", class_="jss65 jss72 jss102").text
      ubicacion = soup.find("p", class_= "jss65 jss73 jss166").text.strip()

      items_box_caracteristicas = soup.find("div", id="characteristics")
      if items_box_caracteristicas is None:
            items_text_caracteristicas = []
      else:
            items_text_caracteristicas = items_box_caracteristicas.find_all("p")

      # Extraer el texto de cada elemento y guardarlo en una lista
      lista_valores = [item.get_text().lower() for item in items_text_caracteristicas]

      ascensor = 1 if any('ascensor' in item for item in lista_valores) else 0
      balcon = 1 if any('balcón' in item for item in lista_valores) else 0

      # Función para validar si al menos uno de los elementos cumple con cierta condición
      # Función para transformar valores
      def transformar_valor(lista):
         for valor in lista:
            if valor == "en casa":
                  return "Casa"
            elif valor == "en edificio":
                  return "Edificio"
            elif valor == "en conjunto cerrado":
                  return "Conjunto"
         return np.NaN 

      # Validar si al menos uno de los elementos cumple con la condición
      tipo_conjunto = transformar_valor(lista_valores)

      # Generar esquema
      datos_inmueble_actual = {
         "id_inmueble_pagina": id_inmueble_pagina,
         "precio": precio,        
         "administracion": administracion,
         "area": area,
         "ubicacion": ubicacion,
         "habitaciones": habitaciones,
         "baños": baños,        
         "pisos_interiores": pisos_interiores,
         "estado": estado,
         "balcon": balcon,        
         "estrato": estrato,                
         "antiguedad": antiguedad,
         "tipo_conjunto": tipo_conjunto,
         "piso": piso,
         "parqueaderos": parqueaderos,
         "ascensor": ascensor,
         "inmobiliaria": inmobiliaria,
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
      del ubicacion
      del habitaciones
      del baños      
      del pisos_interiores
      del estado
      del balcon        
      del estrato                
      del antiguedad
      del tipo_conjunto
      del piso
      del parqueaderos
      del ascensor
      del inmobiliaria
      del inmueble

    except Exception as error:
      
      # Generar diccionario de url's no procesadas
      url_no_procesadas_actual = {
            "URL": inmueble,
            "Procesado": "-1"
      }

      url_no_procesadas.append(url_no_procesadas_actual)
      print(f"Error {index}: {error} en {inmueble}")
     

# COMMAND ----------

items_box_descripcion = soup.find("div", class_="jsx-3405404036 container")
# items_text_description = items_box_descripcion.find_all("p")
soup

# COMMAND ----------

#Creación de dataframe vacio
nombre_columnas = ["id_inmueble_pagina", "precio", "administracion", "area", "ubicacion", "habitaciones", "baños",  "pisos_interiores",  "estado", "balcon", "estrato", "antiguedad", "tipo_conjunto", "piso", "parqueaderos", "ascensor",  "inmobiliaria", "url"]
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
ruta_archivo = '/dbfs/mnt/adls/inmuebles/RAW/Finca_Raiz/Datos/'
nombre_archivo = 'datos_finca_raiz_'
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
ruta_archivo = '/dbfs/mnt/adls/inmuebles/Logs/Finca_Raiz/'
nombre_archivo = 'url_inmuebles_finca_raiz.csv'

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