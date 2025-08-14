from __future__ import annotations
import io
import re
from dataclasses import dataclass
from typing import List, Any

import numpy as np
import pandas as pd


class ColumnNormalizer:
    """Нормализует названия колонок к snake_case."""

    @staticmethod
    def to_snake_case(s: str) -> str:
        s = s.strip().lower().replace("-", "_")
        s = re.sub(r"[^\w\s]", "", s)
        return re.sub(r"\s+", "_", s)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [self.to_snake_case(c) for c in df.columns]
        return df


class CSVReader:
    """Чтение CSV, перебирая популярные кодировки."""

    def __init__(self, encodings: List[str] | None = None) -> None:
        self.encodings = encodings or ["utf-8", "utf-8-sig", "cp1251", "latin1"]

    def read(self, raw_bytes: bytes) -> pd.DataFrame:
        for enc in self.encodings:
            try:
                return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("Не удалось прочитать CSV ни в одной из кодировок")


class TypeCaster:
    """Кастит типы признаков."""

    def cast_categoricals(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype("str").str.strip()
        return df

    def cast_numericals(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for c in num_cols:
            if c in df.columns:
                s = df[c].astype(str).str.replace(",", ".", regex=False)
                df[c] = pd.to_numeric(s, errors="coerce")
        return df


class FeatureAligner:
    """Выравнивает набор колонок под feature_list."""

    def align(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        df = df.copy()
        for c in feature_list:
            if c not in df.columns:
                df[c] = np.nan
        return df[feature_list]


@dataclass
class Artifacts:
    model: Any
    feature_list: List[str]
    threshold: float


class Preprocessor:
    """Готовит входной CSV к инференсу модели."""

    def __init__(
        self,
        csv_reader: CSVReader | None = None,
        normalizer: ColumnNormalizer | None = None,
        type_caster: TypeCaster | None = None,
        aligner: FeatureAligner | None = None,
    ) -> None:
        self.csv_reader = csv_reader or CSVReader()
        self.normalizer = normalizer or ColumnNormalizer()
        self.type_caster = type_caster or TypeCaster()
        self.aligner = aligner or FeatureAligner()

    def from_bytes(
        self,
        raw: bytes,
        feature_list: List[str],
        auto_cat: List[str],
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Возвращает (очищенный df, id_series, X)."""
        df = self.csv_reader.read(raw)
        df = self.normalizer.normalize(df)

        # id
        try:
            id_col = df["id"]
            if not isinstance(id_col, pd.Series):
                raise TypeError("id is not a Series")
            id_series = pd.to_numeric(id_col, errors="coerce").fillna(-1).astype(int)
        except Exception:
            id_series = pd.Series(range(len(df)), name="id")

        # типы + выравнивание
        auto_num = [c for c in feature_list if c not in auto_cat]
        df = self.type_caster.cast_categoricals(df, auto_cat)
        df = self.type_caster.cast_numericals(df, auto_num)
        X = self.aligner.align(df, feature_list)
        return df, id_series, X
