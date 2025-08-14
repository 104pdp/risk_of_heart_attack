import os
import io
import uuid
import numpy as np
import pandas as pd
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from model_utils import (
    normalize_columns, try_read_csv,
    cast_categoricals, cast_numericals, align_columns,
    load_artifacts
)

# Настройка логгирования
logger = logging.getLogger("heart-risk")
logger.setLevel(logging.INFO)
logger.propagate = False 
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)

# Папки
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Артефакты модели
MODEL_PATH = "best_model.pkl"
FEATURES_PATH = "best_model_feature_list.pkl"
THRESHOLD_PATH = "best_model_threshold.pkl"

base_model, feature_list, threshold = load_artifacts(
    MODEL_PATH, FEATURES_PATH, THRESHOLD_PATH
)

# Категориальные и числовые признаки
AUTO_CAT = [
    "diabetes", "family_history", "smoking", "obesity", "alcohol_consumption",
    "diet", "previous_heart_problems", "medication_use", "gender"
]
AUTO_NUM = [c for c in feature_list if c not in AUTO_CAT]

app = FastAPI(title="Heart-risk API")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _prepare_X_from_bytes(raw: bytes):
    """Подготовка данных для предсказания."""
    df = try_read_csv(raw)
    df = normalize_columns(df)

    # Обработка ID (как в рабочей версии)
    try:
        id_col = df["id"]
        if not isinstance(id_col, pd.Series):
            raise TypeError("id is not a Series")
        id_series = pd.to_numeric(id_col, errors="coerce").fillna(-1).astype(int)
    except Exception:
        id_series = pd.Series(range(len(df)), name="id")

    # Обработка признаков
    df = cast_categoricals(df, AUTO_CAT)
    df = cast_numericals(df, AUTO_NUM)
    X = align_columns(df, feature_list)

    return df, id_series, X


def _predict_df(id_series: pd.Series, X: pd.DataFrame):
    """Единое место инференса:"""
    proba = base_model.predict_proba(X)[:, 1]
    pred = (proba >= float(threshold)).astype(int)
    out = pd.DataFrame({"id": id_series, "prediction": pred})
    return out


@app.get("/", response_class=HTMLResponse)
def ui_index(request: Request):
    """Главная страница с формой загрузки."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "threshold": float(threshold)}
    )


@app.post("/predict_json")
def predict_json(file: UploadFile = File(...)):
    """Возвращает JSON с предсказаниями."""
    try:
        raw = file.file.read()
        if not raw:
            raise ValueError("Пустой файл")

        _, id_series, X = _prepare_X_from_bytes(raw)
        out = _predict_df(id_series, X)
        # Ровно тот же формат, что был
        return [
            {"id": int(i), "prediction": int(p)}
            for i, p in zip(out["id"], out["prediction"]) 
        ]

    except Exception as e:
        logger.error(f"Error in predict_json: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Ошибка: {str(e)}")


@app.post("/predict_submit")
def predict_submit(
    file: UploadFile = File(...),
    out_name: str = Query(default=None),
    save_copy: bool = Query(default=False)
):
    """Основной endpoint для предсказаний."""
    try:
        raw = file.file.read()
        if not raw:
            raise ValueError("Пустой файл")

        _, id_series, X = _prepare_X_from_bytes(raw)
        out = _predict_df(id_series, X)

        # Обработка имени файла
        safe_name = out_name if out_name else f"predictions_{uuid.uuid4().hex[:8]}.csv"
        safe_name = str(safe_name).replace("\n", "_").replace("\r", "_")

        # Сохранение копии если нужно
        if save_copy:
            os.makedirs(RESULT_DIR, exist_ok=True)
            save_path = os.path.join(RESULT_DIR, safe_name)
            out.to_csv(save_path, index=False)

        # Возвращаем файл
        stream = io.StringIO()
        out.to_csv(stream, index=False)
        stream.seek(0)

        return StreamingResponse(
            stream,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}"'}
        )

    except Exception as e:
        logger.error(f"Error in predict_submit: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Ошибка: {str(e)}")


@app.get("/download/{filename}")
def download(filename: str):
    """Скачивание файла из results."""
    filepath = os.path.join(RESULT_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, filename=filename)
    raise HTTPException(status_code=404, detail="Файл не найден")


@app.get("/health")
def health():
    """Проверка работоспособности API."""
    return {
        "status": "ok",
        "model": "loaded",
        "n_features": len(feature_list),
        "threshold": float(threshold)
    }