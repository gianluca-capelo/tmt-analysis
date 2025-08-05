import os
import re
import shutil

from src import config
# Ruta a la carpeta con los archivos
carpeta = config.RAW_DATA_DIR

# Carpeta de destino para los archivos sin placeholder
carpeta_destino = config.RAW_WITH_NO_PLACEHOLDERS_DIR

# Crear carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)
    print(f"Creada carpeta de destino: {carpeta_destino}")

# Regex para extraer el número de sujeto al inicio del nombre del archivo
regex_sujeto = re.compile(r'^(\d+)_')

# Sets para almacenar los sujetos procesados
sujetos_copiados = set()
sujetos_placeholder = set()

# Procesar archivos en la carpeta
for archivo in os.listdir(carpeta):
    ruta_archivo_origen = os.path.join(carpeta, archivo)

    # Solo procesar archivos (no carpetas)
    if os.path.isfile(ruta_archivo_origen):
        match = regex_sujeto.match(archivo)
        if match:
            sujeto = int(match.group(1))

            if 'placeholder' in archivo:
                sujetos_placeholder.add(sujeto)
                print(f"Saltado (placeholder): {archivo}")
            else:
                # Copiar archivo sin placeholder
                ruta_archivo_destino = os.path.join(carpeta_destino, archivo)
                shutil.copy2(ruta_archivo_origen, ruta_archivo_destino)
                sujetos_copiados.add(sujeto)
                print(f"Copiado: {archivo}")

# Mostrar resumen de la operación
print(f"\n--- Resumen ---")
print(f"Total de sujetos copiados: {len(sujetos_copiados)}")
print(f"Total de sujetos con placeholder (saltados): {len(sujetos_placeholder)}")

if sujetos_copiados:
    print("\nSujetos copiados a la nueva carpeta:")
    for sujeto in sorted(sujetos_copiados):
        print(f"- Sujeto {sujeto}")

if sujetos_placeholder:
    print("\nSujetos con archivos 'placeholder' (no copiados):")
    for sujeto in sorted(sujetos_placeholder):
        print(f"- Sujeto {sujeto}")

if not sujetos_copiados and not sujetos_placeholder:
    print("No se encontraron archivos de sujetos en la carpeta.")
