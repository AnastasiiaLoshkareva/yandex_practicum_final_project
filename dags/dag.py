"""
Батч-инференс SmartScore (is_high_demand).

load_data:    проверить наличие обученной модели и артефактов в S3, собрать датасет из БД, предобработать, сохранить предобработанные признаки в S3
inference:    скачать предобработанные признаки и обученную модель из S3, посчитать proba/pred, сохранить в S3
save_results: записать результат в public.smartscore_predict
"""
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

from preprocess import get_inference_date, load_source_tables, build_raw_dataset, preprocess_features
from inference import load_model, predict_scores, save_predictions
from s3_utils import (
    check_objects_exist, download_json, download_file, upload_dataframe, download_dataframe,
)

logger = logging.getLogger(__name__)

DAG_ID = "listing_scoring_dag"
POSTGRES_CONN_ID = "postgres_raw"      # Airflow Connection к PostgreSQL
S3_CONN_ID = "s3_features"             # Airflow Connection к S3
SCHEMA = "final_project"
PREDICT_TABLE = "smartscore_predict"
# Значения по умолчанию; в Airflow можно переопределить через Variables
DEFAULT_MODEL_PREFIX = "models/high_demand_catboost/v1/"
DEFAULT_WORK_PREFIX = "smartscore/inference/"


def get_model_prefix():
    return Variable.get("smartscore_model_prefix", default_var=DEFAULT_MODEL_PREFIX)


def get_work_prefix():
    return Variable.get("smartscore_work_prefix", default_var=DEFAULT_WORK_PREFIX)



with DAG(
    dag_id=DAG_ID,
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["listing_scoring"],
) as dag:

    @task(task_id="check_model_on_s3")
    def check_model_on_s3():
        """Падает, если в S3 нет модели, meta или artifacts."""
        prefix = get_model_prefix()
        check_objects_exist(S3_CONN_ID, [
            prefix + "catboost_final.cbm",
            prefix + "catboost_meta.json",
            prefix + "preprocess_artifacts.json",
        ])
        logger.info("Используется модель из S3: %s", prefix)

    @task(task_id="build_and_upload_features")
    def build_and_upload_features():
        """Читает таблицы из БД, считает признаки на дату инференса и кладёт их в S3."""
        model_prefix = get_model_prefix()

        # дата инференса = максимальная snapshot_date (строка 'YYYY-MM-DD')
        run_date = get_inference_date(postgres_conn_id=POSTGRES_CONN_ID)

        listings, calendars, reviews = load_source_tables(run_date, postgres_conn_id=POSTGRES_CONN_ID)
        df_full = build_raw_dataset(listings, calendars, reviews)

        artifacts = download_json(S3_CONN_ID, model_prefix + "preprocess_artifacts.json")
        meta = download_json(S3_CONN_ID, model_prefix + "catboost_meta.json")
        X = preprocess_features(df_full, artifacts, meta, run_date)

        X.insert(0, "id", df_full["id"].values)     # id нужен, чтобы связать предсказания с объявлениями

        features_key = f"{get_work_prefix()}run_date={run_date}/features.csv"
        upload_dataframe(S3_CONN_ID, X, features_key)
        return {"run_date": run_date, "rows": len(X), "features_key": features_key}

    @task(task_id="make_prediction")
    def make_prediction(features_info):
        """Скачивает признаки и модель из S3, считает proba/pred, сохраняет в S3."""
        run_date = features_info["run_date"]
        model_prefix = get_model_prefix()

        meta = download_json(S3_CONN_ID, model_prefix + "catboost_meta.json")
        X = download_dataframe(S3_CONN_ID, features_info["features_key"])

        # после CSV пустые тексты стали NaN, а категории могли поменять тип - возвращаем как при обучении
        for col in meta["text_features"]:
            X[col] = X[col].fillna("")
        for col in meta["categorical_features"]:
            X[col] = X[col].astype(str)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "catboost_final.cbm"
            download_file(S3_CONN_ID, model_prefix + "catboost_final.cbm", model_path)
            model = load_model(model_path)

        proba, pred = predict_scores(X, model, meta)

        result = pd.DataFrame({
            "listing_id": X["id"].values,
            "inference_date": run_date,
            "proba": proba,
            "pred": pred,
        })
        predictions_key = f"{get_work_prefix()}run_date={run_date}/predictions.csv"
        upload_dataframe(S3_CONN_ID, result, predictions_key)
        return {"run_date": run_date, "rows": len(result), "predictions_key": predictions_key}

    @task(task_id="save_to_db")
    def save_to_db(pred_info):
        """Записывает предсказания в final_project.smartscore_predict (если за дату ещё нет)."""
        result = download_dataframe(S3_CONN_ID, pred_info["predictions_key"])
        return save_predictions(result, POSTGRES_CONN_ID, SCHEMA, PREDICT_TABLE, pred_info["run_date"])

    # Порядок выполнения
    with TaskGroup("load_data"):
        check = check_model_on_s3()
        features_info = build_and_upload_features()
        check >> features_info

    with TaskGroup("inference"):
        pred_info = make_prediction(features_info)

    with TaskGroup("save_results"):
        save_to_db(pred_info)
