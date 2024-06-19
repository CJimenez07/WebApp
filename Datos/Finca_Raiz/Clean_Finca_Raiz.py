# Databricks notebook source
import pandas as pd
import os
import openpyxl
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ### Leer Archivos

# COMMAND ----------

#Función para leer todos los archivos csv de una carpeta
def leer_csv_con_nombre(ruta_carpeta, sep, encoding, dtype):
   
    # Lista para almacenar DataFrames
    dataframes = []

    # Iterar sobre los archivos en la carpeta
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith('.csv'):
            # Construir la ruta completa al archivo
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            # Leer el archivo CSV y agregar una columna con el nombre del archivo
            df = pd.read_csv(ruta_archivo, sep=sep, encoding=encoding, dtype=dtype)
            df['nombre_archivo'] = archivo
            # Agregar el DataFrame a la lista de DataFrames
            dataframes.append(df)

    # Concatenar todos los DataFrames en uno solo
    df_combinado = pd.concat(dataframes, ignore_index=True)

    return df_combinado


# COMMAND ----------

carpeta_raw = '/dbfs/mnt/adls/inmuebles/RAW/Finca_Raiz/Datos/'
info_inmuebles = leer_csv_con_nombre(carpeta_raw, sep=';', encoding='latin-1', dtype={"id_inmueble_pagina":"str", "pisos_interiores":"str"})
info_inmuebles.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Limpieza de Datos

# COMMAND ----------

#Reemplazar el valor que se tiene cuando no hay info disponible
info_inmuebles.replace("Preguntar al anunciante", np.nan, inplace=True)

# Limpieza de campos
info_inmuebles["id_inmueble"] = "FR-" + info_inmuebles["id_inmueble_pagina"]
info_inmuebles["precio"] = info_inmuebles["precio"].apply(lambda x: x if pd.isna(x) or not isinstance(x, str) or not x.strip() or not "$" in x else x.split("$")[1].strip().replace(" COP", "").replace(".", ""))
info_inmuebles["administracion"] = info_inmuebles["administracion"].apply(lambda x: x if pd.isna(x) or not isinstance(x, str) or not x.strip() or not "$" in x else x.split("$")[1].strip().replace(" COP", "").replace(".", "")).replace(np.NaN,0)
info_inmuebles['area'] = info_inmuebles['area'].str.replace(" m²","").str.replace(".","")
info_inmuebles["tipo"] = info_inmuebles["url"].str.split("/").str[4].str.split("-").str[0].str.title()
info_inmuebles["estado"].fillna("Usado", inplace=True)
info_inmuebles["ascensor"] = info_inmuebles["ascensor"].replace(np.NaN, "0")
info_inmuebles["estrato"] = info_inmuebles["estrato"].replace("0",np.NaN)
info_inmuebles["pisos_interiores"].fillna("1", inplace=True)
info_inmuebles["ciudad"] = info_inmuebles["ubicacion"].str.split("-").str[1].str.strip().str.title()
info_inmuebles["departamento"] = info_inmuebles["ubicacion"].str.split("-").str[-1].str.strip().str.title()
info_inmuebles["parqueaderos"] = info_inmuebles["parqueaderos"].replace(np.NaN, "0")
info_inmuebles["inmobiliaria"] = info_inmuebles["inmobiliaria"].str.contains("inmobiliaria", case=False).astype(int)
# Crear una columna vacía para cuarto util
info_inmuebles['cuarto_util'] = pd.Series(dtype=pd.Int64Dtype())
info_inmuebles.head()

# COMMAND ----------

# Conversión de tipo de dato
info_inmuebles["precio"] = pd.to_numeric(info_inmuebles['precio'], errors='coerce').astype("Int64")
info_inmuebles["administracion"] = pd.to_numeric(info_inmuebles['administracion'], errors='coerce').astype("Int64")
info_inmuebles["area"] = pd.to_numeric(info_inmuebles['area'], errors='coerce').astype(float)
info_inmuebles["habitaciones"] = pd.to_numeric(info_inmuebles['habitaciones'], errors='coerce').astype("Int64")
info_inmuebles["baños"] = pd.to_numeric(info_inmuebles['baños'], errors='coerce').astype("Int64")
info_inmuebles["pisos_interiores"] = pd.to_numeric(info_inmuebles['pisos_interiores']).astype("Int64")
info_inmuebles["balcon"] = pd.to_numeric(info_inmuebles['balcon']).astype("Int64")
info_inmuebles["estrato"] = pd.to_numeric(info_inmuebles['estrato'], errors='coerce').astype("Int64")
info_inmuebles["piso"] = pd.to_numeric(info_inmuebles['piso'], errors='coerce').astype("Int64")
info_inmuebles["parqueaderos"] = pd.to_numeric(info_inmuebles['parqueaderos'], errors='coerce').astype("Int64")
info_inmuebles["ascensor"] = pd.to_numeric(info_inmuebles['ascensor'], errors='coerce').astype("Int64")
info_inmuebles["inmobiliaria"] = pd.to_numeric(info_inmuebles['inmobiliaria'], errors='coerce').astype("Int64")

len(info_inmuebles)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Homologaciones
# MAGIC #### Homologar Tipo

# COMMAND ----------

archivo_homologaciones = '/dbfs/mnt/adls/inmuebles/RAW/Homologaciones/Homologaciones.xlsx'
# Leer archivo de Homologaciones
homologacion_tipo = pd.read_excel(archivo_homologaciones, sheet_name="Tipo")
homologacion_tipo = homologacion_tipo[homologacion_tipo["tipo"].notna()]

# COMMAND ----------

#Cruzar con homologo de tipo, para dejar la categoría homologada

info_inmuebles_h1 = pd.merge(info_inmuebles, homologacion_tipo, on="tipo", how="left")
info_inmuebles_h1 = info_inmuebles_h1.drop(columns=["tipo"])
info_inmuebles_h1 = info_inmuebles_h1.rename(columns={'tipo_homologado': 'tipo'})
info_inmuebles_h1.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Homologar Estado

# COMMAND ----------

# Leer archivo de Homologaciones
homologacion_estado = pd.read_excel(archivo_homologaciones, sheet_name="Estado")
homologacion_estado = homologacion_estado[homologacion_estado["estado"].notna()]

# COMMAND ----------

#Cruzar con homologo de estado, para dejar la categoría homologada
info_inmuebles_h2 = pd.merge(info_inmuebles_h1, homologacion_estado, on="estado", how="left")
info_inmuebles_h2 = info_inmuebles_h2.drop(columns=["estado"])
info_inmuebles_h2 = info_inmuebles_h2.rename(columns={'estado_homologado': 'estado'})
info_inmuebles_h2.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Homologar Ubicación
# MAGIC Usando datos abiertos del DANE, para obtener el codigo Dane de: https://geoportal.dane.gov.co/servicios/descarga-y-metadatos/datos-geoestadisticos/?cod=112

# COMMAND ----------

# Leer archivo de Homologaciones
homologacion_ubicacion = pd.read_excel(archivo_homologaciones, sheet_name="Ciudad", dtype={"codigo_dane":"str"})
homologacion_ubicacion = homologacion_ubicacion[homologacion_ubicacion["codigo_dane"].notna()]
homologacion_ubicacion.head()

# COMMAND ----------

# Generar archivo de municipios sin homologar
# info_inmuebles_hnew = pd.merge(info_inmuebles_h2, homologacion_ubicacion, on=["ciudad","departamento"], how="left")

# # # Leer archivo de Codigos Divipola
# archivo_divipola = '/dbfs/mnt/adls/inmuebles/RAW/Municipios/DIVIPOLA_Municipios.xlsx'

# divipola_dane = pd.read_excel(archivo_divipola, sheet_name="Municipios", skiprows=10, dtype={2:"str"})
# divipola_dane = divipola_dane.rename(columns={'Nombre': 'departamento', "Nombre.1":"ciudad", "Código .1":"codigo_dane"})
# divipola_dane = divipola_dane[["codigo_dane","ciudad", "departamento"]]
# divipola_dane = divipola_dane[divipola_dane["codigo_dane"].notna()]
# divipola_dane["ciudad2"] = divipola_dane["ciudad"].str.title()
# divipola_dane["departamento2"] = divipola_dane["departamento"].str.title()
# divipola_dane = divipola_dane[["codigo_dane","ciudad2", "departamento2"]]
# divipola_dane.head()

# #Cruzar con homologo de ciudad, para dejar la categoría homologada
# info_inmuebles_hnew2 = pd.merge(info_inmuebles_hnew, divipola_dane, on=["codigo_dane"], how="left")
# info_inmuebles_hnew2 = info_inmuebles_hnew2[info_inmuebles_hnew2["ciudad2"].isnull()]
# info_inmuebles_hnew2 = info_inmuebles_hnew2.groupby(["ciudad", "departamento"]).size().reset_index(name='count').sort_values(by='count', ascending=False)
# info_inmuebles_hnew2.to_csv('/dbfs/mnt/adls/inmuebles/Bronce/Finca_Raiz/homologar_ciudades.csv', index=False, encoding="latin-1", sep=";")

# COMMAND ----------

#Cruzar con homologo de ciudad, para dejar la categoría homologada
info_inmuebles_h3 = pd.merge(info_inmuebles_h2, homologacion_ubicacion, on=["ciudad","departamento"], how="left")
info_inmuebles_h3 = info_inmuebles_h3.drop(columns=["ciudad","departamento"])
info_inmuebles_h3.head()

# COMMAND ----------

# Leer archivo de Codigos Divipola
archivo_divipola = '/dbfs/mnt/adls/inmuebles/RAW/Municipios/DIVIPOLA_Municipios.xlsx'

divipola_dane = pd.read_excel(archivo_divipola, sheet_name="Municipios", skiprows=10, dtype={2:"str"})
divipola_dane = divipola_dane.rename(columns={'Nombre': 'departamento', "Nombre.1":"ciudad", "Código .1":"codigo_dane"})
divipola_dane = divipola_dane[["codigo_dane","ciudad", "departamento"]]
divipola_dane = divipola_dane[divipola_dane["codigo_dane"].notna()]
divipola_dane["ciudad"] = divipola_dane["ciudad"].str.title()
divipola_dane["departamento"] = divipola_dane["departamento"].str.title()
divipola_dane.head()

# COMMAND ----------

#Cruzar con homologo de ciudad, para dejar la categoría homologada
info_inmuebles_h4 = pd.merge(info_inmuebles_h3, divipola_dane, on=["codigo_dane"], how="left")
info_inmuebles_h4.head()

# COMMAND ----------

# Selección de columnas
info_inmuebles_final = info_inmuebles_h4[["id_inmueble", "precio", "administracion", "area", "tipo", "habitaciones", "baños", "pisos_interiores", "estado", "balcon", "estrato", "ciudad", "departamento", "antiguedad", "tipo_conjunto", "piso", "parqueaderos", "ascensor", "cuarto_util", "inmobiliaria", "url"]]
info_inmuebles_final.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Guardar Archivo 

# COMMAND ----------

# Variables Función
ruta_archivo = '/dbfs/mnt/adls/inmuebles/Bronce/Finca_Raiz/'
nombre_archivo = 'Finca_Raiz.csv'

# Crear el directorio si no existe
os.makedirs(ruta_archivo, exist_ok=True)

#Guardar csv
info_inmuebles_final.to_csv(ruta_archivo + nombre_archivo, index=False, encoding="latin-1", sep=";")

print("Archivo creado :" + ruta_archivo + nombre_archivo)
print(len(info_inmuebles_final))