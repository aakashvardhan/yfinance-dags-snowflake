from airflow import DAG
from airflow.models import Variable
from airflow.decorators import task
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

from datetime import datetime
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timezone

# --- Snowflake connection parameters from Airflow Variables ---

USER_ID = Variable.get("snowflake_userid")
PASSWORD = Variable.get("snowflake_password")
ACCOUNT = Variable.get("snowflake_account")
WAREHOUSE = Variable.get("snowflake_warehouse")
DATABASE = Variable.get("snowflake_database")

# --- Configuration parameters ---
SYMBOLS = ["AAPL", "TSLA"]
PERIOD = "180d"
INTERVAL = "1d"


def return_snowflake_hook():
    hook = SnowflakeHook(snowflake_conn_id="my_snowflake_connection")
    conn = hook.get_conn()
    return conn


def get_stock_data(symbol, period, interval, max_retries=2, backoff=2.0):

    # due to sudden hiccup in yfinance libaries, max retries is required to download stock data
    for attempt in range(max_retries):
        try:
            stock_data = yf.download(
                symbol, period=period, interval=interval, group_by=symbol
            )

            if isinstance(stock_data.columns, pd.MultiIndex):
                if symbol in stock_data.columns.get_level_values(0):
                    stock_data = stock_data.xs(symbol, axis=1, level=0)

            if stock_data.empty:
                raise ValueError(f"No data available for symbol {symbol}")
            return stock_data
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error: {e}. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(
                    f"Max retries reached. Unable to download stock data for {symbol}."
                )


@task
def extract(symbols, period, interval):
    results = []
    for symbol in symbols:
        try:
            data = get_stock_data(symbol, period, interval)

            data = data.reset_index()
            date_col = "Date" if "Date" in data.columns else "index"
            data = data.rename(columns={date_col: "date"})
            data = data[["date", "Open", "High", "Low", "Close", "Volume"]]

            # storing 90 days of stock info
            for row in data.itertuples(index=False):
                results.append(
                    {
                        "symbol": symbol,
                        "date": row.date.strftime("%Y-%m-%d"),
                        "open": float(row.Open),
                        "high": float(row.High),
                        "low": float(row.Low),
                        "close": float(row.Close),
                        "volume": float(row.Volume),
                    }
                )
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")

    return results


@task
def transform(price_list):
    transformed_list = []
    for price in price_list:
        transformed_list.append(
            [
                price["symbol"],
                price["date"],
                price["open"],
                price["high"],
                price["low"],
                price["close"],
                price["volume"],
            ]
        )
    return transformed_list


@task
def load_many(con, records):
    con = con.cursor()
    target_table = f"{DATABASE}.raw.STOCK_PRICE"

    try:
        con.execute("BEGIN;")
        con.execute(
            f"""
                CREATE OR REPLACE TABLE {target_table} (
                    symbol VARCHAR(10),
                    date DATE,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume FLOAT,
                    PRIMARY KEY (symbol, date)
                );"""
        )

        # load records
        for r in records:
            _symbol = r[0].replace("'", "''")
            _date = r[1].replace("'", "''")
            _open = r[2]
            _high = r[3]
            _low = r[4]
            _close = r[5]
            _volume = r[6]

            sql = f"""
                INSERT INTO {target_table} (symbol, date, open, high, low, close, volume)
                VALUES ('{_symbol}', '{_date}', {_open}, {_high}, {_low}, {_close}, {_volume});
            """
            # print(sql)
            con.execute(sql)
        con.execute("COMMIT;")
    except Exception as e:
        con.execute("ROLLBACK;")
        raise e


with DAG(
    dag_id="yfinance_to_snowflake_etl",
    start_date=datetime(2025, 1, 1),
    tags=["ETL"],
    catchup=False,
    schedule=None,
) as dag:

    snowflake_conn = return_snowflake_hook()
    extracted_data = extract(SYMBOLS, PERIOD, INTERVAL)
    transformed_data = transform(extracted_data)
    load_many(snowflake_conn, transformed_data)
