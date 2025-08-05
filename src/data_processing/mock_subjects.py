import os
import shutil
import re
from src import config

# Sujetos faltantes
faltantes = ["55", "71"]

# Ruta a la carpeta con los archivos
carpeta = config.RAW_DATA_DIR

# Regex para capturar el número de sujeto al principio del nombre
regex_sujeto = re.compile(r'^(\d+)_experimento')

# Paso 1: Obtener todos los sujetos presentes
archivos = os.listdir(carpeta)

print(f"Sujetos faltantes: {faltantes}")

# Paso 2: Encontrar los archivos del sujeto 1
archivos_sujeto_1 = [f for f in archivos if f.startswith("1_")]

if not archivos_sujeto_1:
    print("No se encontraron archivos del sujeto 1 para usar como base.")
    exit()

# Paso 3: Crear copias para los sujetos faltantes
for sujeto in faltantes:
    for archivo in archivos_sujeto_1:
        origen = os.path.join(carpeta, archivo)
        nuevo_nombre = re.sub(r'^1_', f'{sujeto}_', archivo)
        destino = os.path.join(carpeta, nuevo_nombre)
        shutil.copy2(origen, destino)
        print(f"Copiado: {origen} -> {destino}")

print("Mockeo completado.")
