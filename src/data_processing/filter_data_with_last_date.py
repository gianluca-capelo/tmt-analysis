import os
import re
import shutil

from src import config


def filter_data_with_last_date():
    # Directorio donde están actualmente los archivos
    DATA_DIR = config.RAW_DATA_DIR

    OUTPUT_DIR = config.FILTERED_DATA_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Expresión regular para extraer:
    #  - subjectId (uno o más dígitos)
    #  - fecha/hora en formato YYYY-MM-DD_HHhMM.SS.mmm
    #  - extensión (cualquiera)
    pattern = re.compile(
        r'^(?P<subject>\d+)_experimento_condicion_\d+_(?P<date>\d{4}-\d{2}-\d{2}_\d{2}h\d{2}\.\d{2}\.\d{3})\.(?P<ext>\w+)$'
    )

    # Diccionario para almacenar la última fecha/hora de cada sujeto
    # Estructura: { subjectId: (fechaDatetime, [lista de archivos con esa fecha]) }
    latest_records = {}

    from datetime import datetime

    def parse_datetime(date_str):
        """
        Convierte cadenas tipo:
          '2024-01-22_15h59.33.260'
        en objetos datetime.
        """
        # 1) Separemos la parte de fecha y la parte de hora
        #    '2024-01-22' y '15h59.33.260'
        date_part, time_part = date_str.split('_')  # p.ej. ['2024-01-22', '15h59.33.260']

        # 2) Reemplaza 'h' por ':'
        #    '15h59.33.260' -> '15:59.33.260'
        time_part = time_part.replace('h', ':')

        # 3) Ahora tenemos algo como '15:59.33.260'
        #    donde la primera parte es HH:MM y lo siguiente son segundos.milisegundos
        #    Lo separamos en tres partes: HH:MM | SS | miliseg
        #    '15:59.33.260'.split('.') -> ['15:59', '33', '260']
        parts = time_part.split('.')
        if len(parts) == 3:
            hour_min = parts[0]  # '15:59'
            seconds = parts[1]  # '33'
            millis = parts[2]  # '260'
            # Reconstruimos: '15:59:33.260'
            time_str = f"{hour_min}:{seconds}.{millis}"
        else:
            # Si no coincidiera, manejamos la excepción o devolvemos None
            raise ValueError(f"No se puede parsear el formato de hora: {time_part}")

        # 4) Unimos con la fecha para formar '2024-01-22 15:59:33.260'
        final_str = f"{date_part} {time_str}"

        # 5) Parseamos con datetime.strptime
        return datetime.strptime(final_str, "%Y-%m-%d %H:%M:%S.%f")

    # 1) Recorremos todos los archivos en DATA_DIR y determinamos la última fecha por sujeto
    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        if not os.path.isfile(file_path):
            continue  # omitimos directorios o cosas no deseadas

        # Verificamos si el archivo cumple con el patrón
        match = pattern.match(file_name)
        if not match:
            # Si no cumple el patrón, lo ignoramos o podríamos imprimir advertencia
            continue

        subject_id = match.group('subject')
        date_str = match.group('date')
        ext = match.group('ext')

        print(subject_id)
        # Parseamos la fecha/hora
        dt = parse_datetime(date_str)

        # Verificamos si tenemos ya registro para este sujeto
        if subject_id not in latest_records:
            # Si no existe, lo añadimos
            latest_records[subject_id] = (dt, [file_path])
        else:
            current_latest_dt, file_list = latest_records[subject_id]
            if dt > current_latest_dt:
                # Si encontramos una fecha más reciente, reemplazamos
                latest_records[subject_id] = (dt, [file_path])
            elif dt == current_latest_dt:
                # Si es la misma fecha/hora, agregamos el archivo a la lista
                file_list.append(file_path)
            else:
                # fecha/hora más antigua, ignoramos
                pass

    # 2) Copiamos los archivos correspondientes a la última fecha de cada sujeto
    for subject_id, (latest_dt, file_list) in latest_records.items():
        for fpath in file_list:
            fname = os.path.basename(fpath)
            out_path = os.path.join(OUTPUT_DIR, fname)
            shutil.copy2(fpath, out_path)
            print(f"Copiado: {fpath} -> {out_path}")

    print("Proceso finalizado. Se han copiado los archivos de la última fecha por cada sujeto.")
