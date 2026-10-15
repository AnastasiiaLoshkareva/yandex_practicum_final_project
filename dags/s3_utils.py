"""Работа с S3: подключение, проверка файлов, чтение и запись."""
import io
import json
import logging

import boto3
import pandas as pd
from airflow.hooks.base import BaseHook

logger = logging.getLogger(__name__)


def get_s3_client_and_bucket(s3_conn_id):
    """Создаёт S3-клиент и возвращает bucket. Ключи берутся из Airflow Connection (поле Extra)."""
    extras = BaseHook.get_connection(s3_conn_id).extra_dejson
    client = boto3.client(
        "s3",
        aws_access_key_id=extras.get("aws_access_key_id"),
        aws_secret_access_key=extras.get("aws_secret_access_key"),
        endpoint_url=extras.get("endpoint_url"),
    )
    return client, extras["bucket"]


def check_objects_exist(s3_conn_id, keys):
    """Проверяет, что все файлы есть в S3. Если нет - head_object падает с ошибкой 404."""
    s3, bucket = get_s3_client_and_bucket(s3_conn_id)
    for key in keys:
        info = s3.head_object(Bucket=bucket, Key=key)
        logger.info("Найден в S3: %s (%.1f КБ)", key, info["ContentLength"] / 1e3)


def download_json(s3_conn_id, key):
    """Скачивает JSON-файл из S3 и возвращает словарь."""
    s3, bucket = get_s3_client_and_bucket(s3_conn_id)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def download_file(s3_conn_id, key, local_path):
    """Скачивает файл из S3 на диск (нужно для модели .cbm)."""
    s3, bucket = get_s3_client_and_bucket(s3_conn_id)
    s3.download_file(bucket, key, str(local_path))


def upload_dataframe(s3_conn_id, df, key):
    """Сохраняет таблицу в S3 как CSV без записи на диск."""
    s3, bucket = get_s3_client_and_bucket(s3_conn_id)
    s3.put_object(Bucket=bucket, Key=key, Body=df.to_csv(index=False).encode("utf-8"))
    logger.info("Загружено в S3: %s (%d строк)", key, len(df))


def download_dataframe(s3_conn_id, key):
    """Читает CSV из S3 в таблицу."""
    s3, bucket = get_s3_client_and_bucket(s3_conn_id)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))
