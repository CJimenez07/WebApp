# Databricks notebook source
# MAGIC %md
# MAGIC ### Instalar Librerias

# COMMAND ----------

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from time import sleep
import secrets
import pickle
import json
import sys
import os
import pandas as pd

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

url_apartamentos = 'https://www.metrocuadrado.com/apartamento-apartaestudio-casa-finca/venta/?search=form' #96.127
driver.get(url_apartamentos)
html_code = driver.page_source

# COMMAND ----------

html_code

# COMMAND ----------

# MAGIC %pip install bs4
# MAGIC %pip install html5lib
# MAGIC %pip install lxml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from bs4 import BeautifulSoup
soup= BeautifulSoup(html_code, "lxml")

all_house_li = soup.find_all("li", class_= "sc-gPEVay.dibcyk")

# COMMAND ----------

soup

# COMMAND ----------


def AddFilter():
    input("Press Enter to continue...")


def GoToPage(page):
    driver.get(page)

def GetListingQuantity():
    ListingQuantity = driver.find_elements(By.XPATH,"/html/body/div[@id='__next']/div[@class='UtilityProvider-wqw4uj-0 gDcynq']/div[@class='appprovider__AppThemeWrapper-asxde5-0 gWkYZE']/div[@class='Layout__LayoutStyled-sc-9y7jis-0 ibZBWk page-container']/div[@class='Container-u38a83-0 jDuhNh inner-container container']/div[@class='Row-sc-2hg243-0 iUVxfs align-items-center mb-4 row']/div[@class='Col-sc-14ninbu-0 lfGZKA mb-3 mb-sm-0 col-12 col-md-8']/span[@class='Breadcrumb-sc-15mocrt-0 aAjZo Breadcrumb-sc-1df07y0-0 bLMdQH valing-center d-block breadcrumb']/h1[@class='H1-xsrgru-0 jdfXCo d-sm-inline-block breadcrumb-item active']")
    listing_quantity = None
    for quantity in ListingQuantity:
        if "Total de inmuebles encontrados: " in quantity.text:
            listing_quantity = quantity
            break
    if listing_quantity:
        return int(listing_quantity.text.split(" ")[0])
    else:
        return 0

def GetUrls():
    urls = driver.find_elements(By.XPATH,"//a[@class='sc-bdVaJa ebNrSm']")

    url_list = []

    for url in urls:
        url_list.append(url.get_attribute("href"))

    for url in url_list:
        if url == None:
            url_list.remove(url)

    return url_list

def NextPage():
    NextPageButton = driver.find_element(By.XPATH, "/html/body/div[@id='__next']/div[@class='UtilityProvider-wqw4uj-0 gDcynq']/div[@class='appprovider__AppThemeWrapper-asxde5-0 gWkYZE']/div[@class='Layout__LayoutStyled-sc-9y7jis-0 ibZBWk page-container']/div[@class='Container-u38a83-0 jDuhNh inner-container container']/div[@class='Row-sc-2hg243-0 iUVxfs row']/div[@class='Col-sc-14ninbu-0 lfGZKA col-md-8 col-lg-9']/ul[@class='sc-dVhcbM kVlaFh Pagination-w2fdu0-0 cxBThU paginator justify-content-center align-items-baseline pagination']/li[@class='item-icon-next page-item']/a[@class='sc-bdVaJa ebNrSm page-link']")
    webdriver.ActionChains(driver).move_to_element(NextPageButton).click(NextPageButton).perform()



def AcceptCookies():
    try:
        OkBtn = driver.find_elements(By.XPATH,"//a[@class='sc-bdVaJa ebNrSm sc-htoDjs brhAsq Button-bepvgg-0 dqiWxy text-center btn-disclaimer btn btn-secondary']")
        #webdriver.ActionChains(self.driver).move_to_element(NextPageButton).click(NextPageButton).perform()
        OkBtn.click()
    except:
        pass


def Stop():

    driver.quit()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Páginas a Escanear
# MAGIC Se enlistan las url's de Metro Cuadrado que serán escaneadas iterando página a página, se identificó que por cad categoría existe máximo 200 páginas cada una con 50 ítems, en total se puede traer un máximo de 10.000 items, es por esto que se escanearán multiples url's con filtros aplicados.

# COMMAND ----------

url_apartamentos = 'https://www.metrocuadrado.com/apartamento/venta/usado/?search=form' #96.127
url_casas = 'https://www.metrocuadrado.com/casa/venta/usado/?search=form' #47.897 
url_casalotes = "https://www.metrocuadrado.com/lote/venta/usado/?search=form" #11.993
url_fincas = "https://www.metrocuadrado.com/finca/venta/usado/?search=form" #3.597 


# COMMAND ----------

def NextPage():
    NextPageButton = driver.find_element(By.XPATH, "/html/body/div[@id='__next']/div[@class='UtilityProvider-wqw4uj-0 gDcynq']/div[@class='appprovider__AppThemeWrapper-asxde5-0 gWkYZE']/div[@class='Layout__LayoutStyled-sc-9y7jis-0 ibZBWk page-container']/div[@class='Container-u38a83-0 jDuhNh inner-container container']/div[@class='Row-sc-2hg243-0 iUVxfs row']/div[@class='Col-sc-14ninbu-0 lfGZKA col-md-8 col-lg-9']/ul[@class='sc-dVhcbM kVlaFh Pagination-w2fdu0-0 cxBThU paginator justify-content-center align-items-baseline pagination']/li[@class='item-icon-next page-item']/a[@class='sc-bdVaJa ebNrSm page-link']")
    webdriver.ActionChains(driver).move_to_element(NextPageButton).click(NextPageButton).perform()

# COMMAND ----------

#Guardar url's en listas
WEBDRIVER_DELAY_EXTENDED = 10
WEBDRIVER_DELAY = 5
url_paginas = [url_apartamentos]
            #   url_casas,
            #   url_casalotes,
            #   url_fincas]

# COMMAND ----------

# Go to the url in the parameter

GoToPage(str(url_apartamentos))
#scraper.GoToPage("https://www.metrocuadrado.com/")

#scraper.LogIn("raul.becerra@yopmail.com","Qw123atrxz12$")
AddFilter()

sleep(WEBDRIVER_DELAY_EXTENDED)

# Get the amount of pages that have to be iterated through
# Lisitng quantity / Total number of listings
listings = GetListingQuantity()
numpages = int(listings / 50) + 3

print("Total listings: ",listings)
print("Total pages to skip: ",numpages)

#Set dictionary to store all urls
data = {}
data['listing'] = []

#Accept terms of cookie use
AcceptCookies()

# For every page
for i in range(0,numpages):

    sleep(WEBDRIVER_DELAY)

    # Get the listing urls in the current page
    urls = GetUrls()

    # Append every url in the list to the dictionary
    for url in urls:
        data['listing'].append({
        'url':url,
        'scraped':'False'
        })

    # # write the data to a json file
    # with open(os.path.join(LOG_FOLDER,'urls.json'), 'w') as outfile:
    #     json.dump(data, outfile,indent=4)

    # Go to the next page
    try:
        NextPage()
        print(i,'\n\n')
    except NoSuchElementException:
        exit()
        #break

    sleep(WEBDRIVER_DELAY)

# quit selenium's webdriver instance
Stop()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Escanear URL's para Obtener elementos
# MAGIC Una vez se tiene la lista de url's se pasan a un ciclo FOR para iterar por cada una y obtener las url's de los inmuebles de cada página

# COMMAND ----------

# Lista vacia para guardar las url's
url_inmuebles = []

for pagina in url_paginas:
    for id_pagina in range(0,2):

        #Lanzar petición a la url generada
        driver.get(pagina)

        # Esperar a que la página se cargue completamente (Segundos)
        driver.implicitly_wait(20)
    
        # Encontrar todos los elementos de etiqueta "a" que contiene las url's
        elementos = driver.find_elements(By.XPATH, "//a[@class='sc-bdVaJa ebNrSm']")

        # Iterar sobre los elementos encontrados y guardar la url en la lista
        for elemento in elementos:
            url_inmuebles.append(elemento.get_attribute('href'))
        print(id_pagina)
        # Go to the next page
        try:
            NextPage()
            print(id_pagina,'\n\n')
        except NoSuchElementException:
            exit()
                

# Cerrar el navegador
driver.quit()
print(url_inmuebles)


# COMMAND ----------

url_inmuebles

# COMMAND ----------

# MAGIC %md
# MAGIC ### Limpiar Listas

# COMMAND ----------

# Eliminar url de otros objetos que no son necesarios
filtro = "inmueble"
if url_inmuebles is not None:
    url_inmuebles = [url for url in url_inmuebles if filtro in url]

    # Eliminar url duplicadas
    if url_inmuebles is not None:
        url_inmuebles = list(set(url_inmuebles))

    # Check the length of url_inmuebles
    if url_inmuebles is not None:
        len_inmuebles = len(url_inmuebles)
        print(len_inmuebles)
else:
    print("url_inmuebles is None")

# COMMAND ----------

# Variables Función
ruta_archivo = '/mnt/adls/inmuebles/Logs/'
nombre_archivo = 'url_inmuebles_finca_raiz.csv'
nombre_columna = ['URL']

# Convertir la lista a un DataFrame de Pandas
inmuebles = pd.DataFrame(url_inmuebles, columns=nombre_columna)

# Crear el directorio si no existe
os.makedirs('/mnt/adls/inmuebles/Logs', exist_ok=True)

#Guardar csv
inmuebles.to_csv('/mnt/adls/inmuebles/Logs/url_inmuebles_finca_raiz.csv', index=False, sep=";")

# COMMAND ----------

ls /mnt/adls/inmuebles/Logs/url_inmuebles_finca_raiz