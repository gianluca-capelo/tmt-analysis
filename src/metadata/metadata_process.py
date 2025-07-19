import numpy as np
import pandas as pd


def process_metadata(metadata_df: pd.DataFrame):
    # These columns are not needed, there are summary columns
    process_df = metadata_df[metadata_df['Nro de participante'] != 'Control = ']
    process_df = process_df[metadata_df['Nro de participante'] != 'Caso = ']

    process_df = (process_df.rename(
        columns={
            "Nro de participante": "subject_id",
            "Género": "sex",
            "Edad": "age",
            "Grupo": "group",
            "Anteojos o lentes de contacto": "glasses",
            "Años de escolaridad": "years_of_education",
            "MMSE / 30": "MMSE",
            "Reloj / 3": "clock_drawing_test",
            "TMT A papel PB": "tmt_a_raw",
            "TMT B papel PB": "tmt_b_raw",
            "Digit Symbol pb": "digit_symbol_raw",
            "Span directo pb": "forward_digit_span_raw",
            "span inverso pb": "backward_digit_span_raw",
            "reloj": "clock_drawing_test",
            'TMT A (papel) Z': 'tmt_a_z',
            'TMT B (papel) Z': 'tmt_b_z',
            'Digit - symbol Z': 'digit_symbol_z',
            'Span directo Z': 'forward_digit_span_z',
            'Span inverso z': 'backward_digit_span_z'
        }))

    #extracted_cols = extracted_columns()
    #process_df = process_df[extracted_cols]

    process_df = process_df.replace("Caso ?", "Caso")  # TODO GIAN: consultar a gus porque hacemos esto

    process_df = process_df.reset_index(drop=True)

    process_df = process_df.replace(",", ".", regex=True)
    process_df.columns = [x.lower() for x in process_df.columns]

    process_df["sex"] = process_df["sex"].replace({"F": 1, "M": 0}).astype("Int64")
    process_df["group"] = process_df["group"].str.lower()
    process_df["group"] = process_df["group"].replace({"caso": "mci", "control": "hc"})

    process_df = process_df.replace("Suspendido", 300).replace("No logra", 300)

    process_df['digit_symbol_raw'] = process_df['digit_symbol_raw'].replace('no se lo administraron', np.nan).astype(float)

    process_df['digit_symbol_z'] = process_df['digit_symbol_z'].replace('pendiente', np.nan).astype(float)

    process_df = process_df.astype({"tmt_a_raw": float, "tmt_b_raw": float})

    # use convert_subject_id function as lambda
    process_df["subject_id"] = process_df["subject_id"].apply(lambda x: convert_subject_id(x))
    return process_df


def extracted_columns():
    return ["subject_id", "sex", "age", "group"]


# Esto solo lo hacemos porque asi lo tenemos en las metricas
def convert_subject_id(subject_id: str|int) -> str:
    if isinstance(subject_id, int):
        subject_id = str(subject_id)
    return subject_id.zfill(4)
