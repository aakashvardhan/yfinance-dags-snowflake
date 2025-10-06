# -*- coding: utf-8 -*-
"""
DAG: ml_forecast_pipeline_dag
Purpose: Train and forecast stock prices using Snowflake ML.FORECAST and union ETL + forecast tables.
Author: Vedika & Aakash
Date: 2025-10-06
"""
from airflow import DAG
from airflow.models import Variable
from airflow.decorators import task
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# -------------------------------------------------------------------
# Utility: return a live Snowflake connection and cursor
# -------------------------------------------------------------------
def return_snowflake_conn():
    hook = SnowflakeHook(snowflake_conn_id="my_snowflake_connection")
    conn = hook.get_conn()
    conn.autocommit(True)  # ✅ Prevent nested transaction errors
    return conn, conn.cursor()

default_args = {
    "owner": "ml-eng",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# -------------------------------------------------------------------
# DAG Definition
# -------------------------------------------------------------------
with DAG(
    dag_id='ml_forecast_pipeline_dag',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ML', 'Forecast', 'Snowflake'],
    default_args=default_args,
    description="Train and forecast stock prices using Snowflake ML.FORECAST and union ETL + forecast data.",
) as dag:

    # Read from Airflow Variables
    train_input_table = Variable.get("train_input_table", default_var="raw.stock_price")
    train_view = Variable.get("train_view", default_var="raw.stock_price_view")
    forecast_table = Variable.get("forecast_table", default_var="analytics.stock_price_forecast")
    forecast_function_name = Variable.get("forecast_function_name", default_var="analytics.predict_stock_price")
    final_table = Variable.get("final_table", default_var="analytics.stock_price_final")

    # -------------------------------------------------------------------
    # Task 1: Train model using Snowflake ML.FORECAST
    # -------------------------------------------------------------------
    @task
    def train(train_input_table, train_view, forecast_function_name):
        conn, cur = return_snowflake_conn()

        create_view_sql = f"""
            CREATE OR REPLACE VIEW {train_view} AS
            SELECT DATE, CLOSE, SYMBOL
            FROM {train_input_table}
            WHERE CLOSE IS NOT NULL AND SYMBOL IS NOT NULL;
        """

        create_model_sql = f"""
            CREATE OR REPLACE SNOWFLAKE.ML.FORECAST {forecast_function_name} (
                INPUT_DATA => SYSTEM$REFERENCE('VIEW', '{train_view}'),
                SERIES_COLNAME => 'SYMBOL',
                TIMESTAMP_COLNAME => 'DATE',
                TARGET_COLNAME => 'CLOSE',
                CONFIG_OBJECT => {{ 'ON_ERROR': 'SKIP' }}
            );
        """

        evaluate_sql = f"CALL {forecast_function_name}!SHOW_EVALUATION_METRICS();"

        try:
            cur.execute(create_view_sql)
            cur.execute(create_model_sql)
            cur.execute(evaluate_sql)
            conn.commit()
            print("✅ Model training completed successfully.")
        except Exception as e:
            print("❌ Error during training:", e)
            raise
        finally:
            cur.close()
            conn.close()

    # -------------------------------------------------------------------
    # Task 2: Generate forecasts and union ETL + forecast
    # -------------------------------------------------------------------
    @task
    def predict(forecast_function_name, train_input_table, forecast_table, final_table):
        conn, cur = return_snowflake_conn()

        # ❌ Remove nested BEGIN/END block; just call the procedure directly
        make_prediction_sql = f"""
            CALL {forecast_function_name}!FORECAST(
                FORECASTING_PERIODS => 7,
                CONFIG_OBJECT => {{'prediction_interval': 0.95}}
            );
        """

        get_forecast_sql = f"""
            CREATE OR REPLACE TABLE {forecast_table} AS
            SELECT *
            FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
        """

        join_sql = f"""
            CREATE OR REPLACE TEMP TABLE joined_forecast AS
            SELECT
                f.series AS symbol_json,
                f.ts AS forecast_date,
                f.forecast,
                f.lower_bound,
                f.upper_bound
            FROM {forecast_table} f
            INNER JOIN (
                SELECT DISTINCT SYMBOL FROM {train_input_table}
            ) s
            ON REPLACE(f.series, '"', '') = s.SYMBOL;
        """

        create_final_table_sql = f"""
            CREATE OR REPLACE TABLE {final_table} AS
            SELECT
                SYMBOL,
                DATE,
                CLOSE AS actual,
                NULL AS forecast,
                NULL AS lower_bound,
                NULL AS upper_bound
            FROM {train_input_table}
            UNION ALL
            SELECT
                REPLACE(symbol_json, '"', '') AS SYMBOL,
                forecast_date AS DATE,
                NULL AS actual,
                forecast,
                lower_bound,
                upper_bound
            FROM joined_forecast;
        """

        try:
            cur.execute(make_prediction_sql)
            cur.execute(get_forecast_sql)
            cur.execute(join_sql)
            cur.execute(create_final_table_sql)
            conn.commit()
            print("✅ Forecasting and union completed successfully.")
        except Exception as e:
            print("❌ Error during prediction:", e)
            raise
        finally:
            cur.close()
            conn.close()

    # -------------------------------------------------------------------
    # DAG dependencies
    # -------------------------------------------------------------------
    train_task = train(train_input_table, train_view, forecast_function_name)
    predict_task = predict(forecast_function_name, train_input_table, forecast_table, final_table)
    train_task >> predict_task

