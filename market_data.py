from io import StringIO
from xml.etree import ElementTree as xml_et

import numpy as np
import pandas as pd
import requests


MOEX_HISTORY_URL = 'https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/IMOEX.json'
BRENT_FRED_CSV_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU'
CBR_USD_RUB_XML_URL = 'https://www.cbr.ru/scripts/XML_dynamic.asp'
CBR_SOAP_URL = 'https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx'


def fetch_moex_monthly_data(start_date: str, end_date: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    offset = 0

    while True:
        response = requests.get(
            MOEX_HISTORY_URL,
            params={'from': start_date, 'start': offset},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        column_names = payload['history']['columns']
        page_rows = payload['history']['data']
        if not page_rows:
            break

        records.extend(dict(zip(column_names, row)) for row in page_rows)
        page_size = payload['history.cursor']['data'][0][2]
        offset += page_size

    daily_data = pd.DataFrame(records)
    daily_data['TRADEDATE'] = pd.to_datetime(daily_data['TRADEDATE'])
    daily_data['CLOSE'] = pd.to_numeric(daily_data['CLOSE'], errors='coerce')
    daily_data['VALUE'] = pd.to_numeric(daily_data['VALUE'], errors='coerce')
    daily_data = daily_data.sort_values('TRADEDATE')
    daily_data['log_return'] = np.log(daily_data['CLOSE'] / daily_data['CLOSE'].shift(1))
    daily_data['squared_log_return'] = daily_data['log_return'] ** 2

    monthly_data = (
        daily_data.resample('MS', on='TRADEDATE')
        .agg(
            imoex_close=('CLOSE', 'last'),
            moex_value_sum=('VALUE', 'sum'),
            rv=('squared_log_return', 'sum'),
        )
    )

    trading_days = (
        daily_data.groupby(daily_data['TRADEDATE'].dt.to_period('M'))
        .size()
        .rename('trading_days')
    )
    trading_days.index = trading_days.index.to_timestamp()
    trading_days.index.name = 'date'

    monthly_data = monthly_data.join(trading_days, how='left')
    monthly_data = monthly_data.loc[start_date:end_date]
    monthly_data.index.name = 'date'
    return monthly_data


def fetch_usd_rub_monthly_data(start_date: str, end_date: str) -> pd.DataFrame:
    start_param = pd.Timestamp(start_date).strftime('%d/%m/%Y')
    end_param = pd.Timestamp(end_date).strftime('%d/%m/%Y')
    response = requests.get(
        CBR_USD_RUB_XML_URL,
        params={
            'date_req1': start_param,
            'date_req2': end_param,
            'VAL_NM_RQ': 'R01235',
        },
        timeout=30,
    )
    response.raise_for_status()

    root = xml_et.fromstring(response.content)
    rows = [
        {
            'date': record.get('Date'),
            'usdrub': record.find('Value').text.replace(',', '.'),
        }
        for record in root.findall('Record')
    ]

    monthly_data = pd.DataFrame(rows)
    monthly_data['date'] = pd.to_datetime(monthly_data['date'], dayfirst=True)
    monthly_data['usdrub'] = pd.to_numeric(monthly_data['usdrub'])
    monthly_data = monthly_data.set_index('date').resample('MS').mean()
    monthly_data.index.name = 'date'
    return monthly_data


def fetch_brent_monthly_data(start_date: str, end_date: str) -> pd.DataFrame:
    response = requests.get(BRENT_FRED_CSV_URL, timeout=30)
    response.raise_for_status()

    monthly_data = pd.read_csv(StringIO(response.text))
    monthly_data.columns = ['date', 'brent']
    monthly_data['date'] = pd.to_datetime(monthly_data['date'])
    monthly_data['brent'] = pd.to_numeric(monthly_data['brent'], errors='coerce')
    monthly_data = monthly_data.dropna().set_index('date').sort_index()
    monthly_data = monthly_data.loc[start_date:end_date].resample('MS').mean()
    monthly_data.index.name = 'date'
    return monthly_data


def fetch_key_rate_monthly_data(start_date: str, end_date: str) -> pd.DataFrame:
    request_body = f'''<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <KeyRate xmlns="http://web.cbr.ru/">
      <fromDate>{pd.Timestamp(start_date).strftime('%Y-%m-%dT00:00:00')}</fromDate>
      <ToDate>{pd.Timestamp(end_date).strftime('%Y-%m-%dT00:00:00')}</ToDate>
    </KeyRate>
  </soap:Body>
</soap:Envelope>'''

    response = requests.post(
        CBR_SOAP_URL,
        data=request_body.encode('utf-8'),
        headers={
            'Content-Type': 'text/xml; charset=utf-8',
            'SOAPAction': 'http://web.cbr.ru/KeyRate',
        },
        timeout=30,
    )
    response.raise_for_status()

    root = xml_et.fromstring(response.content)
    rows: list[dict[str, str]] = []
    for node in root.findall('.//{*}KR'):
        rate_date = node.findtext('{*}DT')
        key_rate = node.findtext('{*}Rate')
        if rate_date and key_rate:
            rows.append({'date': rate_date, 'key_rate': key_rate})

    monthly_data = pd.DataFrame(rows)
    monthly_data['date'] = pd.to_datetime(monthly_data['date'], utc=True).dt.tz_localize(None)
    monthly_data['key_rate'] = pd.to_numeric(monthly_data['key_rate'], errors='coerce')
    monthly_data = monthly_data.dropna().set_index('date').sort_index()
    monthly_data = monthly_data['key_rate'].resample('MS').last().to_frame()
    monthly_data.index.name = 'date'
    target_index = pd.date_range(start=start_date, end=end_date, freq='MS')
    return monthly_data.reindex(target_index)
