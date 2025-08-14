# model_utils.py
import io
import re
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Any

# ---------- утилиты ----------
def to_snake_case(s: str) -> str:
    s = s.strip().lower().replace("-", "_")
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", "_", s)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]
    return df

def try_read_csv(raw_bytes: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue
    # если передали путь вместо байтов
    try:
        return pd.read_csv(raw_bytes)  # type: ignore
    except Exception as e:
        raise ValueError(f"Не удалось прочитать CSV: {e}")

def cast_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df

def cast_numericals(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in num_cols:
        if c in df.columns:
            # поддержка '27,3' → 27.3
            s = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")
    return df

def align_columns(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in feature_list:
        if c not in df.columns:
            df[c] = np.nan
    return df[feature_list]

# ---------- обёртка для пропусков ----------
class PredictZeroIfMissing:
    """
    Если в строке есть пропуски в любой из ожидаемых фичей:
    - predict -> 0
    - predict_proba -> [1.0, 0.0]
    Иначе отдаёт предсказание базовой модели.
    """
    def __init__(self, model: Any, feature_list: List[str]):
        self.model = model
        self.feature_list = list(feature_list)

    def _miss(self, X: pd.DataFrame) -> np.ndarray:
        return X.isnull().any(axis=1).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X, columns=self.feature_list)
        miss = self._miss(X)
        y = np.zeros(len(X), dtype=int)
        if (~miss).any():
            y[~miss] = self.model.predict(X.loc[~miss, self.feature_list])
        return y

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X, columns=self.feature_list)
        miss = self._miss(X)
        proba = np.zeros((len(X), 2), dtype=float)
        proba[miss, 0] = 1.0
        if (~miss).any():
            proba[~miss] = self.model.predict_proba(X.loc[~miss, self.feature_list])
        return proba

# ---------- публичные функции ----------
def load_artifacts(
    model_path: str = "best_model.pkl",
    feature_list_path: str = "best_model_feature_list.pkl",
    threshold_path: str = "best_model_threshold.pkl",
) -> Tuple[Any, List[str], float]:
    model = joblib.load(model_path)
    feature_list = list(joblib.load(feature_list_path))
    threshold = float(joblib.load(threshold_path))
    return model, feature_list, threshold

def load_data(input_file: str) -> pd.DataFrame:
    # читаем как путь; если хочешь байты — используй try_read_csv
    df = pd.read_csv(input_file)
    return df

def preprocess_data(df: pd.DataFrame, feature_list: List[str],
                    cat_cols: List[str], num_cols: List[str]) -> pd.DataFrame:
    df = normalize_columns(df)
    df = cast_categoricals(df, cat_cols)
    df = cast_numericals(df, num_cols)
    X = align_columns(df, feature_list)
    return X

def load_model_wrapped(model_path: str, feature_list: List[str]) -> PredictZeroIfMissing:
    base = joblib.load(model_path)
    return PredictZeroIfMissing(base, feature_list)

def make_prediction(X: pd.DataFrame, wrapped_model: Any, threshold: float,
                    id_series: pd.Series | None = None) -> pd.DataFrame:
    proba = wrapped_model.predict_proba(X)[:, 1]
    pred = (proba >= float(threshold)).astype(int).astype(float)
    out = pd.DataFrame({"prediction": pred, "proba": proba})
    if id_series is not None:
        out.insert(0, "id", id_series.values)
    return out
