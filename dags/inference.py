"""Загрузка модели CatBoost, расчёт вероятностей и запись результата в БД."""
import logging

from airflow.providers.postgres.hooks.postgres import PostgresHook
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


def load_model(model_path):
    """Загружает обученную модель CatBoost из файла .cbm."""
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def predict_scores(features, model, meta):
    """Возвращает вероятность высокого спроса (proba) и класс 0/1 (pred) для каждой строки.

    features - таблица после preprocess_features (лишние колонки, например id, игнорируются)
    meta     - словарь из catboost_meta.json (список признаков и порог)
    """
    X = features[meta["features"]]      # порядок колонок как при обучении
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= meta["threshold"]).astype(int)

    logger.info("Предсказания посчитаны: %s строк, доля класса 1: %.3f", len(proba), pred.mean())
    return proba, pred


def count_predictions(postgres_conn_id, schema, table, run_date):
    """Сколько строк в витрине за дату инференса."""
    hook = PostgresHook(postgres_conn_id=postgres_conn_id)
    result = hook.get_first(
        f"SELECT COUNT(*) FROM {schema}.{table} WHERE inference_date = %s;", parameters=(run_date,)
    )
    return int(result[0])


def save_predictions(result, postgres_conn_id, schema, table, run_date):
    """Добавляет предсказания в витрину. Если за эту дату данные уже есть, ничего не добавляет."""
    hook = PostgresHook(postgres_conn_id=postgres_conn_id)

    hook.run(f"""
        CREATE TABLE IF NOT EXISTS {schema}.{table} (
            listing_id     BIGINT           NOT NULL,
            inference_date DATE             NOT NULL,
            proba          DOUBLE PRECISION NOT NULL,
            pred           SMALLINT         NOT NULL,
            created_at     TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (inference_date, listing_id)
        );
    """)

    existing = count_predictions(postgres_conn_id, schema, table, run_date)
    if existing > 0:
        logger.info("За %s в %s.%s уже есть %d строк, повторная вставка пропущена",
                    run_date, schema, table, existing)
        return {"inserted_rows": 0, "skipped": True}

    columns = ["listing_id", "inference_date", "proba", "pred"]
    rows = list(zip(*[result[c].tolist() for c in columns]))   # tolist: обычные числа Python вместо numpy
    hook.insert_rows(table=f"{schema}.{table}", rows=rows, target_fields=columns)

    logger.info("В %s.%s добавлено %d строк за %s", schema, table, len(rows), run_date)
    return {"inserted_rows": len(rows), "skipped": False}
