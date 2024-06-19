# Databricks notebook source
# MAGIC %md
# MAGIC ### Instalar Librerias

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ### Páginas a Escanear
# MAGIC Se enlistan las url's de Mercado Libre que serán escaneadas iterando página a página, se identificó una limitación de 2000 items, donde se tienen 48 por página, en total se puede traer un máximo de 2000 items por url, es por esto que se escanearán multiples url's con filtros aplicados.

# COMMAND ----------

archivo_url = pd.read_excel('/dbfs/mnt/adls/inmuebles/Logs/General/URL_Busqueda.xlsx', sheet_name="Mercado_Libre")

url_paginas= archivo_url["URL"].tolist()
print(url_paginas)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Escanear URL's para Obtener elementos
# MAGIC Una vez se tiene la lista de url's se pasan a un ciclo FOR para iterar por cada una y obtener las url's de los inmuebles de cada página

# COMMAND ----------

# Lista vacia para guardar las url's
url_inmuebles = []


for pagina in url_paginas:
    for id_pagina in range(1,2000,48):
        #Generar la url dinámica
        url = f"{pagina}_Desde_{id_pagina}_NoIndex_True"

        try:
            # Realizar la solicitud GET a la página
            response = requests.get(url)
            time.sleep(1)
            # Parsear el contenido HTML
            soup = BeautifulSoup(response.content, "html.parser")

            elementos = soup.find_all("div", class_="ui-search-item__group__element ui-search-item__title-grid")
            # Extraer y imprimir el atributo href de cada elemento encontrado
            for div in elementos:
                a_tag = div.find('a')  # Buscar la etiqueta <a> dentro del <div>
                if a_tag and 'href' in a_tag.attrs:
                    href = a_tag['href']
                    url_inmuebles.append(href)
                    #print(href)
            print(url)
        except Exception as error:
            print("Error en :" + url)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Limpiar Listas

# COMMAND ----------

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
    ruta_archivo = '/dbfs/mnt/adls/inmuebles/Logs/Mercado_Libre/'
    nombre_archivo = 'url_inmuebles_mercado_libre.csv'
    # Leer archivo de URL's
    inmuebles_file = pd.read_csv(ruta_archivo + nombre_archivo, sep=";", dtype={"Procesado":str})
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
result = result[["URL", "Procesado"]]

# Crear el directorio si no existe
os.makedirs(ruta_archivo, exist_ok=True)

#Guardar csv
result.to_csv(ruta_archivo + nombre_archivo, index=False, sep=";")

len(result)