import os
import re
import shutil

from src import config
def remove_placeholders(output_path):
    # Ruta a la carpeta con los archivos
    carpeta = config.RAW_DATA_DIR

    # Crear carpeta de destino si no existe
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Creada carpeta de destino: {output_path}")

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
                    ruta_archivo_destino = os.path.join(output_path, archivo)
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


if __name__ == "__main__":
    remove_placeholders()
    print("Proceso de eliminación de placeholders completado.")