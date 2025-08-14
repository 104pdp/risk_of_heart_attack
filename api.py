from __future__ import annotations
import os
import io
import re
import uuid
import logging
from typing import List

import pandas as pd
from joblib import load as jl_load
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from model_utils import Preprocessor, Artifacts


# ---------- ЛОГГИРОВАНИЕ ----------
logger = logging.getLogger("heart-risk")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)


# ---------- КОНФИГ ----------
class AppConfig:
    def __init__(self) -> None:
        self.UPLOAD_DIR = "uploads"
        self.RESULT_DIR = "results"
        self.TEMPLATES_DIR = "templates"
        self.STATIC_DIR = "static"
        self.MODEL_PATH = "best_model.pkl"
        self.FEATURES_PATH = "best_model_feature_list.pkl"
        self.THRESHOLD_PATH = "best_model_threshold.pkl"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True)
        os.makedirs(self.TEMPLATES_DIR, exist_ok=True)
        os.makedirs(self.STATIC_DIR, exist_ok=True)


# ---------- АРТЕФАКТЫ ----------
class ArtifactService:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.artifacts = self._load()

    def _load(self) -> Artifacts:
        model = jl_load(self.cfg.MODEL_PATH)
        feature_list = list(jl_load(self.cfg.FEATURES_PATH))
        threshold = float(jl_load(self.cfg.THRESHOLD_PATH))
        logger.info("Artifacts loaded: features=%d, threshold=%.6f", len(feature_list), threshold)
        return Artifacts(model=model, feature_list=feature_list, threshold=threshold)


# ---------- ДАННЫЕ ----------
class DataService:
    def __init__(self, artifacts: Artifacts) -> None:
        self.artifacts = artifacts
        self.prep = Preprocessor()
        self.AUTO_CAT: List[str] = [
            "diabetes", "family_history", "smoking", "obesity", "alcohol_consumption",
            "diet", "previous_heart_problems", "medication_use", "gender",
        ]

    def prepare_from_bytes(self, raw: bytes) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        return self.prep.from_bytes(raw=raw, feature_list=self.artifacts.feature_list, auto_cat=self.AUTO_CAT)


# ---------- ПРЕДСКАЗАНИЯ ----------
class PredictionService:
    def __init__(self, artifacts: Artifacts) -> None:
        self.artifacts = artifacts

    def predict(self, id_series: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.artifacts.model.predict_proba(X)[:, 1]
        pred = (proba >= float(self.artifacts.threshold)).astype(int)
        return pd.DataFrame({"id": id_series, "prediction": pred})


# ---------- РЕЗУЛЬТАТЫ ----------
class ResultStorage:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def save_csv(self, df: pd.DataFrame, filename: str) -> str:
        path = os.path.join(self.cfg.RESULT_DIR, filename)
        df.to_csv(path, index=False)
        return path

    def make_download_stream(self, df: pd.DataFrame) -> io.StringIO:
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)
        return stream


# ---------- СБОРКА ПРИЛОЖЕНИЯ ----------
class HeartRiskApp:
    def __init__(self) -> None:
        self.cfg = AppConfig()
        self.art_svc = ArtifactService(self.cfg)
        self.data_svc = DataService(self.art_svc.artifacts)
        self.pred_svc = PredictionService(self.art_svc.artifacts)
        self.store = ResultStorage(self.cfg)

        self.app = FastAPI(title="Heart-risk API")
        self.templates = Jinja2Templates(directory=self.cfg.TEMPLATES_DIR)
        self.app.mount("/static", StaticFiles(directory=self.cfg.STATIC_DIR), name="static")

        self._register_routes()

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/", response_class=HTMLResponse)
        def ui_index(request: Request):
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request, "threshold": float(self.art_svc.artifacts.threshold)},
            )

        @app.post("/predict_json")
        def predict_json(file: UploadFile = File(...)):
            try:
                raw = file.file.read()
                if not raw:
                    raise ValueError("Пустой файл")
                _, id_series, X = self.data_svc.prepare_from_bytes(raw)
                out = self.pred_svc.predict(id_series, X)
                return [{"id": int(i), "prediction": int(p)} for i, p in zip(out["id"], out["prediction"])]
            except Exception as e:
                logger.error(f"Error in predict_json: {e}")
                raise HTTPException(status_code=400, detail=f"Ошибка: {e}")

        @app.post("/predict_submit")
        def predict_submit(
            file: UploadFile = File(...),
            out_name: str | None = Query(default=None),
            save_copy: bool = Query(default=False),
        ):
            try:
                raw = file.file.read()
                if not raw:
                    raise ValueError("Пустой файл")
                _, id_series, X = self.data_svc.prepare_from_bytes(raw)
                out = self.pred_svc.predict(id_series, X)

                safe_name = out_name or f"predictions_{uuid.uuid4().hex[:8]}.csv"
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(safe_name))

                if save_copy:
                    self.store.save_csv(out, safe_name)

                stream = self.store.make_download_stream(out)
                return StreamingResponse(
                    stream,
                    media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
                )
            except Exception as e:
                logger.error(f"Error in predict_submit: {e}")
                raise HTTPException(status_code=400, detail=f"Ошибка: {e}")

        @app.get("/download/{filename}")
        def download(filename: str):
            filepath = os.path.join(self.cfg.RESULT_DIR, filename)
            if os.path.exists(filepath):
                return FileResponse(filepath, filename=filename)
            raise HTTPException(status_code=404, detail="Файл не найден")

        @app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "loaded",
                "n_features": len(self.art_svc.artifacts.feature_list),
                "threshold": float(self.art_svc.artifacts.threshold),
            }


_assembled = HeartRiskApp()
app: FastAPI = _assembled.app

def create_app() -> FastAPI:
    return HeartRiskApp().app
