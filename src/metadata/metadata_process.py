import pandas as pd


def process_metadata(metadata_df: pd.DataFrame):
    metadata_df = metadata_df[metadata_df['Nro de participante'] != 'Control = ']
    metadata_df = metadata_df[metadata_df['Nro de participante'] != 'Caso = ']
    process_df = (metadata_df.rename(
        columns={
            "Nro de participante": "subject_id",
            "Género": "sex",
            "Edad": "age",
            "Grupo": "group"
        }))

    extracted_cols = extracted_columns()
    process_df = process_df[extracted_cols]

    process_df = process_df.replace("Caso ?", "Caso")  # TODO GIAN: consultar a gus porque hacemos esto

    process_df = process_df.reset_index(drop=True)

    process_df = process_df.replace(",", ".", regex=True)
    process_df.columns = [x.lower() for x in process_df.columns]

    process_df["group"] = process_df["group"].str.lower()
    process_df["group"] = process_df["group"].replace({"caso": "MCI", "control": "HC"})

    # use convert_subject_id function as lambda
    process_df["subject_id"] = process_df["subject_id"].apply(lambda x: convert_subject_id(x))
    return process_df


def extracted_columns():
    return ["subject_id", "sex", "age", "group"]


# Esto solo lo hacemos porque asi lo tenemos en las metricas
def convert_subject_id(subject_id: str) -> str:
    return subject_id.zfill(4)
