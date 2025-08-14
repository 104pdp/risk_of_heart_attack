import io
import re
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Optional


def to_snake_case(s: str) -> str:
    """Приведение строки к snake_case."""
    s = s.strip().lower().replace("-", "_")
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", "_", s)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализация названий колонок."""
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]
    return df


def try_read_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Чтение CSV с попыткой разных кодировок."""
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Не удалось прочитать CSV ни в одной из кодировок")


def cast_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Приведение категориальных признаков к строковому типу."""
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("str").str.strip()
    return df


def cast_numericals(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """Приведение числовых признаков с обработкой запятых."""
    df = df.copy()
    for c in num_cols:
        if c in df.columns:
            s = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


def align_columns(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Приведение колонок к виду, ожидаемому моделью."""
    df = df.copy()
    for c in feature_list:
        if c not in df.columns:
            df[c] = np.nan
    return df[feature_list]


def load_artifacts(
    model_path: str = "best_model.pkl",
    feature_list_path: str = "best_model_feature_list.pkl",
    threshold_path: str = "best_model_threshold.pkl",
) -> Tuple[Any, List[str], float]:
    """Загрузка артефактов модели."""
    model = joblib.load(model_path)
    feature_list = list(joblib.load(feature_list_path))
    threshold = float(joblib.load(threshold_path))
    return model, feature_list, threshold
