import re
import logging

import numpy as np
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Настройка логирования для Airflow
logger = logging.getLogger(__name__)

# Определение даты инференса
def get_inference_date(postgres_conn_id):
    """Дата инференса = максимальная дата в final_project.smartscore_test_snapshot.
    Возвращает строку 'YYYY-MM-DD'."""
    hook = PostgresHook(postgres_conn_id=postgres_conn_id)
    result = hook.get_first("SELECT MAX(snapshot_date)::date FROM final_project.smartscore_test_snapshot;")

    if result is None or result[0] is None:
        raise ValueError("В smartscore_test_snapshot нет значений в колонке snapshot_date")

    run_date = result[0].isoformat()
    logger.info("Дата инференса: %s", run_date)
    return run_date

# Загрузка данных таблиц из БД
def load_source_tables(run_date, postgres_conn_id="postgres_raw"):
    """
    Читает данные из PostgreSQL с помощью Airflow PostgresHook.
    Args:
        run_date: дата инференса в формате 'YYYY-MM-DD'.
        postgres_conn_id: идентификатор подключения Airflow для Postgres.
    Returns:
        all_listings, all_calendars, all_reviews - три датафрейма.
    """
    logger.info("Подключение к PostgreSQL через PostgresHook (conn_id=%s)", postgres_conn_id)
    hook = PostgresHook(postgres_conn_id=postgres_conn_id)

    # smartscore_test_snapshot (тестовый срез данных)
    listings_query = """
        SELECT CAST(id AS BIGINT) AS id_int, *
        FROM final_project.smartscore_test_snapshot;
    """
    all_listings = hook.get_pandas_df(listings_query)
    if all_listings.empty:
        raise ValueError("Таблица smartscore_test_snapshot пуста")

    # точный id вместо неточного float
    all_listings = all_listings.drop(columns="id").rename(columns={"id_int": "id"})

    # одна строка на объявление
    all_listings = all_listings.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Календарь: доля доступных дней с даты инференса
    calendar_query = f"""
    SELECT CAST(listing_id AS BIGINT) AS id,AVG(CASE WHEN available THEN 1 ELSE 0 END) AS calendar_availability_rate
    FROM final_project.smartscore_calendar
    WHERE date >= %(run_date)s
    GROUP BY listing_id
    """
    all_calendars = hook.get_pandas_df(calendar_query, parameters={"run_date": run_date})
    logger.info("Календарь: %d объявлений", len(all_calendars))

    # Отзывы: последние 10 за год до даты инференса
    review_query = f"""
    SELECT CAST(listing_id AS BIGINT) AS id,
    AVG(LENGTH(comments)) AS avg_recent_comment_len,
    STRING_AGG(comments, ' ') AS concat_recent_comments
    FROM (SELECT listing_id,comments,ROW_NUMBER() OVER (PARTITION BY listing_id ORDER BY date DESC) AS rn
          FROM final_project.smartscore_review
          WHERE date <= %(run_date)s
          AND date >= %(run_date)s::date - INTERVAL '1 year') sub
    WHERE rn <= 10
    GROUP BY listing_id"""
    all_reviews = hook.get_pandas_df(review_query, parameters={"run_date": run_date})
    
    logger.info("Выгружено из Postgres: all_listings: %d, all_calendars: %d, all_reviews: %d",
                len(all_listings), len(all_calendars), len(all_reviews))

    return all_listings, all_calendars, all_reviews

# объединение всех выгруженных из БД таблиц в один тадасет
def build_raw_dataset(all_listings, all_calendars, all_reviews):
    """Объединяет smartscore_test_snapshot с calendar и review"""
    df_full = all_listings.merge(all_calendars, on=["id"], how="left", validate="one_to_one")
    df_full = df_full.merge(all_reviews, on=["id"], how="left", validate="one_to_one")
    return df_full

def process_text(text, stopwords):
    """Очистка текста: нижний регистр, без знаков препинания, без стоп-слов. Возвращает строку."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)           # убираем пунктуацию
    tokens = text.split()                           # разбиваем на слова
    tokens = [w for w in tokens if w not in stopwords]
    return " ".join(tokens)  

# предобработка призанков перед обучением модели
def preprocess_features(df_full, artifacts, meta, run_date):
    df = df_full.copy()

    # Преобразуем имеющиеся признаки
    # Обрезаем слишком большие значения: всё, что больше 400, становится равным 400
    df['minimum_nights_avg_ntm'] = df['minimum_nights_avg_ntm'].clip(upper=400)
    
    # Значения больше 1125 - это "заглушка" (2 147 483 647), заменяем их на пропуск, т.к. реальные значения не известны
    df.loc[df['maximum_nights'] > 1125, 'maximum_nights'] = np.nan

    # преобразуем к числовому типу host_acceptance_rate
    df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'].astype(str).str.replace('%', ''), errors='coerce')

    # заменяем пропуски в bathrooms на значение из bathrooms_text
    # Достаём число из текста: "1.5 baths" -> 1.5, "1 shared bath" -> 1.0
    bath_from_text = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)')[0].astype(float)

    # Тексты без цифры, например "Half-bath", считаем как 0.5
    is_half = df['bathrooms_text'].str.contains('half', case=False, na=False)
    bath_from_text[is_half] = 0.5

    # Подставляем число из текста только туда, где в bathrooms был пропуск
    df['bathrooms'] = df['bathrooms'].fillna(bath_from_text)

    # bool-признаки приводим к типу str и добавляем вместо nan - 'Missing'
    bool_like = ['host_is_superhost', 'host_identity_verified', 'instant_bookable','host_has_profile_pic']
    for c in bool_like:
        df[c] = df[c].astype(str).replace({'nan':'Missing','None':'Missing'})

    # Остальных object типа признаках также заполняем nan - Missing
    cat_cols=['room_type','host_response_time','property_type']
    for col in cat_cols:
        df[col] = df[col].fillna('Missing')

    

    # преобразуем к типу datetime признаки 'first_review','last_review', 'host_since'
    df[['first_review','last_review', 'host_since']] = df[['first_review','last_review', 'host_since']].astype('datetime64[ns]')

    # преобразование текстовых признаков название и отзывы
    stopwords = set(artifacts["stopwords"])    # set ищется быстрее, чем список
    df["filtered_sentences"] = df["concat_recent_comments"].apply(lambda t: process_text(t, stopwords))
    df["filtered_name"] = df["name"].apply(lambda t: process_text(t, stopwords))
    
    # создание новых признаков
    # считаем общее количество удобств в каждом объявлении
    df['amenities_count'] = df['amenities'].str.split(',').str.len()

    # Типы жилья property_type оставляем 20 самых частых, остальные называем Other
    df['property_type_top'] = df['property_type'].where(df['property_type'].isin(artifacts['top_property_types']), 'Other')

    # Признаки по датам
    ref_date = pd.Timestamp(run_date)

    df['days_since_last_review'] = (ref_date - df['last_review']).dt.days
    df['review_span_days'] = (df['last_review'] - df['first_review']).dt.days
    df['host_tenure_days'] = (ref_date - df['host_since']).dt.days

    # Флаги
    # Календарь закрыт весь год (если данных нет, оставляем пропуск, а не 0)
    df['calendar_closed'] = (df['calendar_availability_rate'] == 0).astype(float)
    df.loc[df['calendar_availability_rate'].isna(), 'calendar_closed'] = np.nan

    # Долгая аренда: минимум от 28 ночей
    df['is_long_stay'] = (df['minimum_nights'] >= 28).astype(int)

    # Отношения вместимости
    # Студия (0 спален) считается как 1, нулевое число ванных заменяем на пропуск
    df['guests_per_bedroom'] = df['accommodates'] / df['bedrooms'].clip(lower=1)
    df['guests_per_bathroom'] = df['accommodates'] / df['bathrooms'].replace(0, np.nan)


    df = df[meta["features"]].copy()                 # нужные колонки в порядке обучения

    for col in meta["categorical_features"]:         # категориальные: строки, без NaN
        df[col] = df[col].astype(str)
    for col in meta["text_features"]:                #пустые строки вместо NaN
            df[col] = df[col].fillna("")

    return df

