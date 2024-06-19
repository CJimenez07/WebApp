from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Cargar el diccionario desde el archivo .pkl
with open('WebApp\modelo_entrenado.pkl', 'rb') as file:
    model_package = pickle.load(file)

# Extraer el modelo y los objetos de preprocesamiento del diccionario
model = model_package['model']
label_encoders = model_package['label_encoders']
one_hot_encoder = model_package['one_hot_encoder']
normalization = model_package['normalization']

# Archivo de Valores Permitidos
archivo_homologaciones = 'WebApp/archivos/Homologaciones.xlsx'

# Valores permitidos para el campo estado
df_estado = pd.read_excel(archivo_homologaciones, sheet_name='Estado')
df_estado = df_estado[["estado_homologado"]]
df_estado = df_estado.drop_duplicates()
estados = df_estado['estado_homologado'].tolist()

# Valores permitidos para el campo tipo
df_tipo = pd.read_excel(archivo_homologaciones, sheet_name="Tipo")
df_tipo = df_tipo[["tipo_homologado"]]
df_tipo = df_tipo.drop_duplicates()
tipos = df_tipo["tipo_homologado"].tolist()

# Archivo de Municipios
archivo_divipola = 'WebApp/archivos/DIVIPOLA_Municipios.xlsx'

divipola_dane = pd.read_excel(archivo_divipola, sheet_name="Municipios", skiprows=10, dtype={2:"str"})
divipola_dane = divipola_dane.rename(columns={'Nombre': 'departamento', "Nombre.1":"ciudad", "Código .1":"codigo_dane"})
divipola_dane = divipola_dane[["codigo_dane","ciudad", "departamento"]]
divipola_dane = divipola_dane[divipola_dane["codigo_dane"].notna()]
divipola_dane["ciudad"] = divipola_dane["ciudad"].str.title()
divipola_dane["departamento"] = divipola_dane["departamento"].str.title()

# Obtener listas de departamentos y municipios
departamentos = divipola_dane['departamento'].unique().tolist()
#municipios = divipola_dane[['departamento', 'ciudad']].to_dict('records')

@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'POST':
        # Obtener los datos del formulario
        administracion = int(request.form['administracion'])
        area = float(request.form['area'])
        tipo = request.form['tipo']
        habitaciones = int(request.form['habitaciones'])
        baños = int(request.form['banos'])
        pisos_interiores = int(request.form['pisos_interiores'])
        estado = request.form['estado']
        balcon = int(request.form['balcon'])
        estrato = int(request.form['estrato'])
        departamento = request.form['departamento']
        ciudad = request.form['ciudad']
        antiguedad_num = int(request.form['antiguedad'])
        tipo_conjunto = request.form['tipo_conjunto']
        piso = int(request.form['piso'])
        parqueaderos = int(request.form['parqueaderos'])
        ascensor = int(request.form['ascensor'])
        cuarto_util = int(request.form['cuarto_util'])
        inmobiliaria = int(request.form['inmobiliaria'])
     
        if antiguedad_num < 1:
            antiguedad = "menor a 1 año"
        elif antiguedad_num >= 1 and antiguedad_num <= 8:
            antiguedad = "1 a 8 años"
        elif antiguedad_num >= 9 and antiguedad_num <= 15:
            antiguedad = "9 a 15 años"
        elif antiguedad_num >= 16 and antiguedad_num <= 30:
            antiguedad = "16 a 30 años"
        else:
            antiguedad = "más de 30 años"
        
        #Construir un diccionario con las variables de entrada deben estar en el mismo orden del modelo entrenado
        variables_entrada = {
            'administracion': [administracion],
            'area': [area],
            'habitaciones': [habitaciones],
            'baños': [baños],
            'pisos_interiores': [pisos_interiores],
            'piso': [piso],
            'parqueaderos': [parqueaderos],
            'tipo': [tipo],
            'estado': [estado],
            'ciudad': [ciudad],
            'departamento': [departamento],
            'antiguedad': [antiguedad],
            'tipo_conjunto': [tipo_conjunto],
            'ascensor': [ascensor],
            'cuarto_util': [cuarto_util],
            'inmobiliaria': [inmobiliaria],
            'balcon': [balcon],
            'estrato': [estrato]
        }

        # Convertir el diccionario en un DataFrame
        nuevo_registro = pd.DataFrame(variables_entrada)

        # Columnas a las que se les aplican las transformaciones
        columns_normalization = ["administracion","area","habitaciones","baños","pisos_interiores","piso","parqueaderos"]
        columns_one_hot_encoding = ["tipo","estado","tipo_conjunto"]
        columns_label_encoding = ["estrato", "antiguedad", "ciudad", "departamento"]

        # Aplicar MinMaxScaler a las columnas especificadas
        nuevo_registro[columns_normalization] = normalization.transform(nuevo_registro[columns_normalization])

        # Aplicar One Hot Encoding a las columnas especificadas
        nuevo_registro_ohe = one_hot_encoder.transform(nuevo_registro[columns_one_hot_encoding])

        # Crear un DataFrame para las columnas one-hot encoded
        nuevo_registro_ohe_df = pd.DataFrame(nuevo_registro_ohe, columns=one_hot_encoder.get_feature_names_out(columns_one_hot_encoding))

        # Concatenar las columnas normalizadas y one-hot encoded
        nuevo_registro_combined = pd.concat([nuevo_registro.drop(columns_one_hot_encoding, axis=1), nuevo_registro_ohe_df], axis=1)

        # Aplicar Label Encoding a las columnas especificadas
        for column in columns_label_encoding:
            le = label_encoders[column]
            nuevo_registro_combined[column] = le.transform(nuevo_registro_combined[column])

        # Hacer predicciones
        predicciones = model.predict(nuevo_registro_combined)

        # Convertir las predicciones a las categorías originales
        predicciones_originales = label_encoders["Grupo_Precio"].inverse_transform(predicciones)
        precio = predicciones_originales[0]
        
        return render_template('index.html', tipos=tipos, estados=estados, departamentos=departamentos, **request.form, precio=precio)

    # Si es un GET o para mostrar el formulario inicial
    return render_template('index.html', tipos=tipos, estados=estados, departamentos=departamentos)

@app.route('/municipios', methods=['POST'])
def municipios():
    departamento = request.form['departamento']
    municipios_filtrados = divipola_dane[divipola_dane['departamento'] == departamento]['ciudad'].tolist()
    return jsonify(municipios_filtrados)

if __name__ == '__main__':
    app.run(debug=True)
