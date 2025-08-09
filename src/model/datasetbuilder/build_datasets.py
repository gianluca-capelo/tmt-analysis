import os

import numpy as np

from src.config import PROCESSED_FOR_MODEL_DIR
from src.hand_analysis.loader.load_last_split import load_last_analysis


def get_digital_tmt_vars():
    digital_tmt_vars = [
        'rt', 'total_distance', 'non_cut_correct_targets_touches', 'non_cut_zigzag_amplitude', 'non_cut_rt',
        'mean_speed', 'std_speed', 'peak_speed', 'mean_acceleration', 'std_acceleration',
        'peak_acceleration', 'mean_abs_acceleration', 'std_abs_acceleration',
        'peak_abs_acceleration', 'mean_negative_acceleration',
        'std_negative_acceleration', 'peak_negative_acceleration',
        'hesitation_time', 'travel_time', 'search_time', 'hesitation_distance',
        'travel_distance', 'search_distance', 'hesitation_avg_speed',
        'travel_avg_speed', 'search_avg_speed', 'state_transitions',
        'total_hesitations', 'average_duration',
        'max_duration', 'zigzag_amplitude', 'distance_difference_from_ideal',
        'area_difference_from_ideal', 'intra_target_time', 'inter_target_time'
    ]
    return digital_tmt_vars


def compute_ratios_B_A(df_digital_tmt):
    # 3. Compute B/A ratios
    part_a_cols = [col for col in df_digital_tmt.columns if col.endswith('_PART_A')]
    part_b_cols = [col for col in df_digital_tmt.columns if col.endswith('_PART_B')]
    common_vars = [col.replace('_PART_A', '') for col in part_a_cols if
                   f"{col.replace('_PART_A', '')}_PART_B" in part_b_cols]

    for var in common_vars:
        df_digital_tmt[f"{var}_B_A_ratio"] = df_digital_tmt[f"{var}_PART_B"] / df_digital_tmt[f"{var}_PART_A"]

    df_digital_tmt.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df_digital_tmt


def compute_validity_percentage(df):
    df_grouped = df.groupby(['subject_id', 'trial_type'])['is_valid'].sum().unstack()
    df_grouped.columns = ['is_valid_sum_A', 'is_valid_sum_B']

    # Convertir a porcentaje
    df_grouped['is_valid_sum_A'] = (df_grouped['is_valid_sum_A'] / 10) * 100
    df_grouped['is_valid_sum_B'] = (df_grouped['is_valid_sum_B'] / 10) * 100

    # Paso intermedio: quedarnos solo con esas dos columnas
    df_validity_percentage = df_grouped[['is_valid_sum_A', 'is_valid_sum_B']]

    return df_validity_percentage


def get_cognitive_group(df):
    df_subject = df.drop_duplicates(subset='subject_id', keep='first')
    df_cognitive_group = df_subject[['subject_id', 'group']].copy()
    return df_cognitive_group


def build_digital_dataset(df_valid):
    df = df_valid.copy()

    df_cognitive_group = get_cognitive_group(df)

    df_validity_percentage = compute_validity_percentage(df)

    df_digital_tmt = df.pivot_table(
        index='subject_id',
        columns='trial_type',
        values=get_digital_tmt_vars()
    )

    # Flatten column names (e.g., mean_speed_PART_A)
    df_digital_tmt.columns = [f"{var}_{ttype}" for var, ttype in df_digital_tmt.columns]

    df_digital_tmt = compute_ratios_B_A(df_digital_tmt)

    df_digital_tmt = df_digital_tmt.merge(df_validity_percentage,
                                          left_index=True, right_index=True)

    df_digital_tmt_with_target = df_digital_tmt.merge(
        df_cognitive_group, on='subject_id', validate='one_to_one'
    ).set_index('subject_id')

    df_digital_tmt_with_target['group'] = (df_digital_tmt_with_target['group'].str
                                           .replace('mci', '1')
                                           .replace('control', '0').astype(int))

    return df_digital_tmt_with_target


def filter_valid(df_raw):
    df = df_raw.copy()
    min_age = 56
    min_number_of_trials_by_type = 1

    print("Initial unique subjects:", df['subject_id'].nunique())

    df_valid = df[df["is_valid"]]
    print("After filtering valid trials:", df_valid['subject_id'].nunique())

    df_valid = df_valid[df_valid['age'] >= min_age]
    print(f"After filtering by age >= {min_age}:", df_valid['subject_id'].nunique())

    # Keep subjects with at least one valid PART_A and PART_B trial
    valid_counts = df_valid.groupby(['subject_id', 'trial_type']).size().unstack(fill_value=0)

    eligible_subjects = valid_counts[
        (valid_counts.get('PART_A', 0) >= min_number_of_trials_by_type) &
        (valid_counts.get('PART_B', 0) >= min_number_of_trials_by_type)
        ].index

    df_valid = df_valid[df_valid['subject_id'].isin(eligible_subjects)]

    print("After filtering by min number of trials:", df_valid['subject_id'].nunique())

    return df_valid


def build_datasets():
    df_raw, _ = load_last_analysis()

    df_valid = filter_valid(df_raw)

    digital_dataset = build_digital_dataset(df_valid)

    os.makedirs(PROCESSED_FOR_MODEL_DIR, exist_ok=True)

    digital_dataset.to_csv(os.path.join(PROCESSED_FOR_MODEL_DIR, 'df_digital_tmt_with_target.csv'),
                           index_label='subject_id')


if __name__ == "__main__":
    build_datasets()
    print("Datasets built successfully.")
