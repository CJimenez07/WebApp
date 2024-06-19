# Databricks notebook source
# MAGIC %md
# MAGIC ### Instalar Librerias

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extraer Variables Necesarias por Cada Inmueble

# COMMAND ----------

ruta_archivo = 'url_inmuebles_finca_raiz.csv'

# Leer archivo de URL's
inmuebles = pd.read_csv(ruta_archivo, sep=";")

# Convertir a lista
url_inmuebles = inmuebles['URL'].tolist()

# COMMAND ----------

# URL de la página
url_inmuebles = ["https://www.fincaraiz.com.co/inmueble/apartamento-en-venta/colinas-de-suba/bogota/10310081",
                 "https://www.fincaraiz.com.co/inmueble/apartamento-en-venta/los-portales---nuevo-rey/cali/10293088", 
                 "https://www.fincaraiz.com.co/inmueble/apartamento-en-venta/bocagrande/cartagena/10810194", 
                 "https://www.fincaraiz.com.co/inmueble/casa-en-venta/pradera-norte/bogota/10641011",
           #NoExisteMas      "https://www.fincaraiz.com.co/inmueble/apartamento-en-venta/claret/bogota/10710788",
                 "https://www.fincaraiz.com.co/inmueble/casa-en-venta/arroyo-de-piedra/cartagena/10394965",
                 "https://www.fincaraiz.com.co/inmueble/apartamento-en-venta/ciudad-verde/soacha/10207353",
                 "https://www.fincaraiz.com.co/inmueble/casa-en-venta/el-rodeo/la-calera/10179867"]

# COMMAND ----------

#Obtener elemento con la información requerida
items_box_caracteristicas = soup.find("div", class_="MuiBox-root jss838 jss260")
items_text_caracteristicas = items_box_caracteristicas.find_all("p")

ascensor = 1 if any('ascensor' in item.text.lower() for item in items_text_caracteristicas) else 0
balcon = 1 if any('balcón' in item.text.lower() for item in items_text_caracteristicas) else 0

# Función para validar si al menos uno de los elementos cumple con cierta condición
def validar_condicion(items):
    for item in items:
        if item is not None and "en casa" in item.text.lower():
            valor = "Casa"
        elif item is not None and "en edificio" in item.text.lower():
            valor = "Edificio"
        elif item is not None and "en conjunto cerrado" in item.text.lower():
            valor = "Conjunto"
        else:
            valor = ""
    return valor

# Validar si al menos uno de los elementos cumple con la condición
tipo_conjunto = validar_condicion(items_text_caracteristicas)

# COMMAND ----------

datos_inmuebles = []
url_no_procesadas = []

for inmueble in url_inmuebles:
    # Realizar la solicitud GET a la página
    response = requests.get(inmueble)
    # Parsear el contenido HTML
    soup = BeautifulSoup(response.content, "html.parser")

    try:
      #Obtener elemento con la información requerida
      items_box_descripcion = soup.find("div", class_="MuiBox-root jss330 jss328")
      items_text_description = items_box_descripcion.find_all("p")
      #Generar variables
      administracion = items_text_description[1].text
      antiguedad = items_text_description[3].text.strip()
      area = items_text_description[5].text.replace("m²","").replace(",",".").strip()
      baños = items_text_description[7].text.strip()
      estado = items_text_description[9].text.strip()
      piso = items_text_description[11].text.replace("Otro","").strip()
      parqueaderos = items_text_description[13].text.strip()
      pisos_interiores = items_text_description[15].text.strip()
      habitaciones = items_text_description[21].text.strip()
      estrato = items_text_description[23].text.strip()

      #Otras Variables a partir de etiquetas fijas
      inmobiliaria = soup.find("span", class_="MuiChip-label").text.strip()
      id_inmueble_pagina = soup.find("p", class_="jss65 jss74 jss306 jss264").text.split(":")[1].strip()
      precio = soup.find("p", class_="jss65 jss72 jss102").text
      ubicacion = soup.find("p", class_= "jss65 jss73 jss166").text.strip()
      url = inmueble

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
      #   "balcon": balcon,        
         "estrato": estrato,                
         "antiguedad": antiguedad,
      #   "tipo_conjunto": tipo_conjunto,
         "piso": piso,
         "parqueaderos": parqueaderos,
      #   "ascensor": ascensor,
         "inmobiliaria": inmobiliaria,
         "url": url
      }

      # Agregar el diccionario a la lista de datos_inmuebles
      print(datos_inmueble_actual)
      datos_inmuebles.append(datos_inmueble_actual)
   
    except Exception as error:
      url_no_procesadas.append(url)
      print(f"Error: {error} en {url}")
     

# COMMAND ----------

#Creación de dataframe vacio
nombre_columnas = ["id_inmueble_pagina", "precio", "administracion", "area", "ubicacion", "habitaciones", "baños",  "pisos_interiores",  "estado", "estrato", "antiguedad", "piso", "parqueaderos",  "inmobiliaria", "url"]
info_inmuebles = pd.DataFrame(datos_inmuebles, columns=nombre_columnas)

#Reemplazar el valor que se tiene cuando no hay info disponible
info_inmuebles.replace("Preguntar al anunciante", np.nan, inplace=True)
info_inmuebles.head(10)

# COMMAND ----------

# Limpieza de campos
info_inmuebles["id_inmueble_pagina"] = "FR-" + info_inmuebles["id_inmueble_pagina"]
info_inmuebles["precio"] = info_inmuebles["precio"].apply(lambda x: x if pd.isna(x) or not isinstance(x, str) or not x.strip() or not "$" in x else x.split("$")[1].strip().replace(" COP", "").replace(".", ""))
info_inmuebles["administracion"] = info_inmuebles["administracion"].apply(lambda x: x if pd.isna(x) or not isinstance(x, str) or not x.strip() or not "$" in x else x.split("$")[1].strip().replace(" COP", "").replace(".", ""))
info_inmuebles["tipo"] = info_inmuebles["url"].str.split("/").str[4].str.split("-").str[0].str.title()
info_inmuebles["estado"].fillna("Usado", inplace=True)
info_inmuebles["pisos_interiores"].fillna("1", inplace=True)
info_inmuebles["barrio"] = info_inmuebles["ubicacion"].str.split("-").str[0].str.strip().str.title()
info_inmuebles["ciudad"] = info_inmuebles["ubicacion"].str.split("-").str[1].str.strip().str.title()
info_inmuebles["departamento"] = info_inmuebles["ubicacion"].str.split("-").str[-1].str.strip().str.title()
info_inmuebles["inmobiliaria"] = info_inmuebles["inmobiliaria"].str.contains("inmobiliaria", case=False).astype(int)

# COMMAND ----------

# Conversión de tipo de dato

info_inmuebles["precio"] = pd.to_numeric(info_inmuebles['precio'], errors='coerce').astype("Int64")
info_inmuebles["administracion"] = pd.to_numeric(info_inmuebles['administracion'], errors='coerce').astype("Int64")
info_inmuebles["area"] = pd.to_numeric(info_inmuebles['area'], errors='coerce').astype(float)
info_inmuebles["habitaciones"] = pd.to_numeric(info_inmuebles['habitaciones'], errors='coerce').astype("Int64")
info_inmuebles["baños"] = pd.to_numeric(info_inmuebles['baños'], errors='coerce').astype("Int64")
info_inmuebles["pisos_interiores"] = pd.to_numeric(info_inmuebles['pisos_interiores']).astype("Int64")
info_inmuebles["estrato"] = pd.to_numeric(info_inmuebles['estrato'], errors='coerce').astype("Int64")
info_inmuebles["piso"] = pd.to_numeric(info_inmuebles['piso'], errors='coerce').astype("Int64")
info_inmuebles["parqueaderos"] = pd.to_numeric(info_inmuebles['parqueaderos'], errors='coerce').astype("Int64")

info_inmuebles = info_inmuebles[["id_inmueble_pagina", "precio", "administracion", "area", "tipo", "habitaciones", "baños", "pisos_interiores", "estado", "estrato", "barrio", "ciudad", "departamento", "antiguedad", "piso", "parqueaderos", "inmobiliaria", "url"]]
info_inmuebles.head(10)

# COMMAND ----------

info_inmuebles.size