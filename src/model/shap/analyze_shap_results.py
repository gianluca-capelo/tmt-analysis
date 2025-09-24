from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict

import numpy as np
import pandas as pd


@dataclass
class _Accumulators:
    """Acumuladores para sumar contribuciones y contar apariciones por feature."""
    sum_signed: Dict[str, float]
    sum_abs: Dict[str, float]
    count: Dict[str, int]


def _resolve_feature_axis(expl) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Acepta ÚNICAMENTE Explanation con values de forma (n_samples, n_features, n_classes),
    donde len(feature_names) == n_features. Devuelve:
      - values_signed: (n_samples, n_classes, n_features)
      - values_abs   : (n_samples, n_classes, n_features)
      - feature_names: (n_features,)

    Si el objeto no cumple exactamente ese formato, lanza ValueError.
    """
    # 1) Nombres de features
    feature_names = getattr(expl, "feature_names", None) or getattr(expl, "data_feature_names", None)
    if feature_names is None:
        raise ValueError("No se encontraron nombres de features en la Explanation.")
    feature_names = np.asarray(feature_names)

    # 2) Valores SHAP
    values = np.asarray(expl.values)
    if values.ndim != 3:
        raise ValueError(
            f"Se esperaba values con 3 dimensiones (n_samples, n_features, n_classes); recibido {values.shape}."
        )

    n_samples, n_features, n_classes = values.shape
    if n_classes != 2:
        raise ValueError(
            f"Se esperaba n_classes=2 (binario); recibido n_classes={n_classes}."
        )
    if len(feature_names) != n_features:
        raise ValueError(
            f"Inconsistencia: len(feature_names)={len(feature_names)} != n_features={n_features}."
        )

    # (Opcional, estricto) validar base_values si existe
    validate_base_values(expl, n_classes, n_samples)

    # 3) Reordenar a (n_samples, n_classes, n_features) para el pipeline aguas abajo
    values_signed = np.swapaxes(values, -1, -2)  # (n_samples, n_classes, n_features)
    values_abs = np.abs(values_signed)

    return values_signed, values_abs, feature_names


def validate_base_values(expl, n_classes, n_samples):
    base = getattr(expl, "base_values", None)
    if base is not None:
        base_arr = np.asarray(base)
        # Permitimos (n_samples, n_classes) o (n_classes,)
        if base_arr.ndim == 2:
            if base_arr.shape != (n_samples, n_classes):
                raise ValueError(
                    f"base_values debe tener forma (n_samples, n_classes)={(n_samples, n_classes)}; "
                    f"recibido {base_arr.shape}."
                )
        elif base_arr.ndim == 1:
            if base_arr.shape[0] != n_classes:
                raise ValueError(
                    f"base_values debe tener longitud n_classes={n_classes}; recibido {base_arr.shape[0]}."
                )
        else:
            raise ValueError(
                f"base_values con dimensionalidad no soportada: {base_arr.shape}."
            )


def _collapse_classes_to_sample_feature(
        signed_any: np.ndarray, abs_any: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Colapsa el eje de clases si existe:
      - Entrada 2D: (n_samples, n_features) -> se devuelve igual.
      - Entrada 3D: (n_samples, n_classes, n_features)
          * con signo: promedio por clase
          * en valor absoluto: promedio de absolutos por clase
      Devuelve dos arrays 2D: (n_samples, n_features).
    """
    # if signed_any.ndim == 2:
    #     return signed_any, abs_any
    if signed_any.ndim == 3:
        signed_sf = signed_any.mean(axis=1)
        abs_sf = abs_any.mean(axis=1)
        return signed_sf, abs_sf
    raise ValueError("Formas SHAP inesperadas tras normalización/clase.")


def _validate_feature_alignment(n_features: int, feat_names: np.ndarray) -> None:
    """Valida que la cantidad de columnas coincida con la cantidad de nombres de features."""
    if len(feat_names) != n_features:
        raise ValueError(
            f"Longitud de feature_names ({len(feat_names)}) no coincide con n_features ({n_features}). "
            "Esto indica una incongruencia en el objeto SHAP o en el mapeo de nombres."
        )


def _accumulate_fold_contributions(
        acc: _Accumulators,
        feat_names: np.ndarray,
        shap_signed_sf: np.ndarray,
        shap_abs_sf: np.ndarray,
) -> None:
    """
    Agrega al acumulador:
      - sumas por feature (con signo y absoluto) sobre las muestras del fold
      - conteo de apariciones (n_samples por feature en este fold)
    """
    n_samples, n_features = shap_signed_sf.shape
    fold_sum_signed = shap_signed_sf.sum(axis=0)
    fold_sum_abs = shap_abs_sf.sum(axis=0)

    for j, feature_name in enumerate(feat_names):
        acc.sum_signed[feature_name] = acc.sum_signed.get(feature_name, 0.0) + float(fold_sum_signed[j])
        acc.sum_abs[feature_name] = acc.sum_abs.get(feature_name, 0.0) + float(fold_sum_abs[j])
        acc.count[feature_name] = acc.count.get(feature_name, 0) + n_samples


def _build_result_dataframe(acc: _Accumulators) -> pd.DataFrame:
    """Construye el DataFrame final (ordenado por mean_abs_shap desc)."""
    rows = []
    for feature_name, feature_count in acc.count.items():
        mean_signed = acc.sum_signed[feature_name] / feature_count if feature_count else np.nan
        mean_abs = acc.sum_abs[feature_name] / feature_count if feature_count else np.nan
        rows.append((feature_name, feature_count, mean_signed, mean_abs))

    df = pd.DataFrame(rows, columns=["feature", "count", "mean_shap", "mean_abs_shap"])
    df = df.sort_values("mean_abs_shap", ascending=False, kind="mergesort").reset_index(drop=True)
    return df


def aggregate_shap_explanations(explanations: Iterable) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrega una lista de shap.Explanation (una por fold) y devuelve:
      - df_agg: DataFrame con columnas ['feature', 'count', 'mean_shap', 'mean_abs_shap'],
                ordenado por 'mean_abs_shap' descendente.
      - top20 : DataFrame con las 20 features más importantes por 'mean_abs_shap'.

    Reglas:
      * Considera todas las muestras de test por fold.
      * Si hay eje de clases, colapsa promediando por clase tanto con signo como en valor absoluto.
      * 'count' = # apariciones (muestras × folds en los que la feature estuvo presente).
      * No imputa ceros para folds donde la feature no aparece.

    Parámetros
    ----------
    explanations : Iterable[shap.Explanation]
        Lista (o iterable) devuelta por shap_after_nested_cv(...), una Explanation por fold.

    Retorna
    -------
    df_agg : pd.DataFrame
    top20  : pd.DataFrame
    """
    explanations = list(explanations)
    if not explanations:
        raise ValueError("La lista de explanations está vacía.")

    acc = _Accumulators(sum_signed={}, sum_abs={}, count={})

    for expl in explanations:
        signed_any, abs_any, feat_names = _resolve_feature_axis(expl)
        shap_signed_sf, shap_abs_sf = _collapse_classes_to_sample_feature(signed_any, abs_any)
        _validate_feature_alignment(shap_signed_sf.shape[1], feat_names)
        _accumulate_fold_contributions(acc, feat_names, shap_signed_sf, shap_abs_sf)

    df_agg = _build_result_dataframe(acc)
    top20 = df_agg.head(20).copy()
    return df_agg, top20
