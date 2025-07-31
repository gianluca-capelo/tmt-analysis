import os
import re

from src import config
# Ruta a la carpeta con los archivos
carpeta = config.RAW_DATA_DIR

# Regex para extraer el número de sujeto al inicio del nombre del archivo
regex_sujeto = re.compile(r'^(\d+)_')

# Set para almacenar los sujetos afectados
sujetos_placeholder = set()

# Procesar archivos en la carpeta
for archivo in os.listdir(carpeta):
    if 'placeholder' in archivo:
        match = regex_sujeto.match(archivo)
        if match:
            sujeto = int(match.group(1))
            sujetos_placeholder.add(sujeto)

        ruta_archivo = os.path.join(carpeta, archivo)
        os.remove(ruta_archivo)
        print(f"Eliminado: {archivo}")

# Mostrar los sujetos cuyos archivos fueron eliminados
if sujetos_placeholder:
    print("\nSujetos con archivos 'placeholder' eliminados:")
    for sujeto in sorted(sujetos_placeholder):
        print(f"- Sujeto {sujeto}")
else:
    print("No se encontraron archivos con 'placeholder' en el nombre.")
