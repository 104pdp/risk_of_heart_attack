# api.py
import os
import io
import uuid
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# ==== наши утилиты и обёртка (из model_utils.py) ====
from model_utils import (
    normalize_columns, try_read_csv,
    cast_categoricals, cast_numericals, align_columns,
    load_artifacts, PredictZeroIfMissing
)

# ==== папки ====
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)  # на случай, если создаёшь шаблон позже
os.makedirs(STATIC_DIR, exist_ok=True)

# ==== артефакты ====
MODEL_PATH     = "best_model.pkl"
FEATURES_PATH  = "best_model_feature_list.pkl"
THRESHOLD_PATH = "best_model_threshold.pkl"

base_model, feature_list, threshold = load_artifacts(
    MODEL_PATH, FEATURES_PATH, THRESHOLD_PATH
)

# категориальные признаки исходных данных (как в обучении)
AUTO_CAT = [
    "diabetes", "family_history", "smoking", "obesity", "alcohol_consumption",
    "diet", "previous_heart_problems", "medication_use", "gender"
]
AUTO_NUM = [c for c in feature_list if c not in AUTO_CAT]

# оборачиваем модель политикой "0 при пропусках" (без падений пайплайна)
wrapped = PredictZeroIfMissing(base_model, feature_list)

# ==== FastAPI + UI ====
app = FastAPI(title="Heart-risk API")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==== вспомогалки ====
def _safe_preview(df: pd.DataFrame, n: int = 10):
    """Первые n строк без NaN/Inf, пригодные для JSON."""
    prev = df.head(n).replace([np.inf, -np.inf], np.nan)
    prev = prev.where(prev.notna(), None)
    return prev.to_dict(orient="records")


def _prepare_X_from_bytes(raw: bytes):
    """
    Читаем CSV (автокодировка), нормализуем имена столбцов,
    приводим типы, выравниваем под feature_list.
    Возвращает: (df_исходный_нормализованный, id_series, X_готовый_для_модели)
    """
    df = try_read_csv(raw)          # из model_utils: поддержка популярных кодировок
    df = normalize_columns(df)

    has_id = "id" in df.columns
    id_series = df["id"].copy() if has_id else pd.Series(range(len(df)), name="id")

    df = cast_categoricals(df, AUTO_CAT)  # к string
    df = cast_numericals(df, AUTO_NUM)    # к числам
    X = align_columns(df, feature_list)   # порядок/полнота признаков

    return df, id_series, X


# ===================== UI =====================
@app.get("/", response_class=HTMLResponse)
def ui_index(request: Request):
    """Главная страница с формой загрузки CSV и кнопками."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "threshold": float(threshold)}
    )


@app.post("/predict_json")
def predict_json(file: UploadFile = File(...)):
    """
    Вернёт полный JSON-список [{id, prediction}, ...] для загруженного CSV.
    Удобно для быстрого просмотра прямо в браузере.
    """
    try:
        raw = file.file.read()
        if not raw:
            raise ValueError("Пустой файл")

        _, id_series, X = _prepare_X_from_bytes(raw)

        proba = wrapped.predict_proba(X)[:, 1]
        pred  = (proba >= float(threshold)).astype(int)

        result = [{"id": int(i), "prediction": int(p)} for i, p in zip(id_series, pred)]
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"predict_json failed: {type(e).__name__}: {e}")


# ================== API: служебные ==================
@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(feature_list), "threshold": float(threshold)}


# ================== API: отладочный ==================
@app.post("/predict_file")
def predict_file(
    file: UploadFile = File(...),
    save_copy: bool = Query(False, description="Сохранять копию результата в ./results"),
    drop_unnamed: bool = Query(True, description="Убрать столбцы вроде 'unnamed_0' из сохранённого CSV")
):
    """
    Отладочный эндпоинт:
      - формирует DataFrame = исходные колонки + prediction + proba
      - возвращает JSON-превью (первые строки)
      - если save_copy=true — кладёт полный CSV в ./results и отдаёт ссылку /download/{file_id}
    """
    try:
        raw = file.file.read()
        if not raw:
            raise ValueError("Пустой файл")

        df, id_series, X = _prepare_X_from_bytes(raw)

        proba = wrapped.predict_proba(X)[:, 1]
        pred  = (proba >= float(threshold)).astype(int)

        out = df.copy()

        # ставим id первой колонкой
        if "id" in out.columns:
            first = out.pop("id")
            out.insert(0, "id", first)
        else:
            out.insert(0, "id", id_series.values)

        out["prediction"] = pred
        out["proba"] = proba

        if drop_unnamed and "unnamed_0" in out.columns:
            out = out.drop(columns=["unnamed_0"])

        download = None
        if save_copy:
            file_id = uuid.uuid4().hex
            path = os.path.join(RESULT_DIR, f"{file_id}_predictions.csv")
            out.to_csv(path, index=False)
            download = f"/download/{file_id}"

        return {
            "rows": int(len(out)),
            "threshold": float(threshold),
            "download_file": download,
            "preview": _safe_preview(out, 10)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"predict_file failed: {type(e).__name__}: {e}")


# ================== API: боевой ==================
@app.post("/predict_submit")
def predict_submit(
    file: UploadFile = File(...),
    out_name: str | None = Query(None, description="Имя итогового CSV (например, my_preds.csv)"),
    save_copy: bool = Query(False, description="Сохранять копию результата в ./results")
):
    """
    Боевой эндпоинт: возвращает CSV ровно в формате 'id,prediction'.
    - out_name задаёт имя файла-ответа (если не задано — pred_OPT_<thr>_FINAL.csv)
    - save_copy=true сохраняет копию в ./results для /download/{file_id}
    """
    try:
        raw = file.file.read()
        if not raw:
            raise ValueError("Пустой файл")

        _, id_series, X = _prepare_X_from_bytes(raw)

        proba = wrapped.predict_proba(X)[:, 1]
        pred  = (proba >= float(threshold)).astype(int)

        out = pd.DataFrame({"id": id_series, "prediction": pred})

        # имя файла ответа
        default_name = f"pred_OPT_{float(threshold):.3f}_FINAL.csv"
        safe_name = (out_name or default_name).replace("\n", "_").replace("\r", "_")

        # по желанию — сохраняем копию на сервер
        if save_copy:
            file_id = uuid.uuid4().hex
            path = os.path.join(RESULT_DIR, f"{file_id}_{safe_name}")
            out.to_csv(path, index=False)

        # отдаём напрямую как attachment
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}"'}
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"predict_submit failed: {type(e).__name__}: {e}")


# ================== API: скачивание сохранённого ==================
@app.get("/download/{file_id}")
def download(file_id: str):
    """Скачивание файла, если ранее сохраняли с save_copy=true."""
    for fname in os.listdir(RESULT_DIR):
        if fname.startswith(file_id):
            return FileResponse(os.path.join(RESULT_DIR, fname), filename=fname)
    raise HTTPException(status_code=404, detail="File not found")
