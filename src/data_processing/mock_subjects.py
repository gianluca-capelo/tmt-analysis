import os
import re
import shutil

from src import config


def mock_subjects(source_path):
    # Sujetos faltantes
    faltantes = ["55", "71"]

    # Paso 1: Obtener todos los sujetos presentes
    archivos = os.listdir(source_path)

    # Paso 2: Encontrar los archivos del sujeto 1
    archivos_sujeto_1 = [f for f in archivos if f.startswith("1_")]

    if not archivos_sujeto_1:
        print("No se encontraron archivos del sujeto 1 para usar como base.")
        exit()

    # Paso 3: Crear copias para los sujetos faltantes
    for sujeto in faltantes:
        for archivo in archivos_sujeto_1:
            origen = os.path.join(source_path, archivo)
            nuevo_nombre = re.sub(r'^1_', f'{sujeto}_', archivo)
            destino = os.path.join(source_path, nuevo_nombre)
            shutil.copy2(origen, destino)
            print(f"Copiado: {origen} -> {destino}")

    print("Mockeo completado.")


if __name__ == "__main__":
    mock_subjects()
    print("Mockeo de sujetos completado exitosamente.")
