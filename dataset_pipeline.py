import pandas as pd

from market_data import (
    fetch_brent_monthly_data,
    fetch_key_rate_monthly_data,
    fetch_moex_monthly_data,
    fetch_usd_rub_monthly_data,
)
from settings import SETTINGS


def build_macro_monthly_dataset(
    start_date: str = SETTINGS.dataset_start_date,
    end_date: str = SETTINGS.dataset_end_date,
) -> pd.DataFrame:
    moex_data = fetch_moex_monthly_data(start_date=start_date, end_date=end_date)
    usd_rub_data = fetch_usd_rub_monthly_data(start_date=start_date, end_date=end_date)
    brent_data = fetch_brent_monthly_data(start_date=start_date, end_date=end_date)
    key_rate_data = fetch_key_rate_monthly_data(start_date=start_date, end_date=end_date)

    combined_data = pd.concat(
        [moex_data, usd_rub_data, brent_data, key_rate_data],
        axis=1,
    ).sort_index()
    combined_data = combined_data.loc[start_date:end_date]
    combined_data = combined_data.dropna(subset=['usdrub', 'brent', 'key_rate'])
    combined_data.index.name = 'date'
    return combined_data


def save_datasets() -> pd.DataFrame:
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)

    moex_data = fetch_moex_monthly_data(
        start_date=SETTINGS.dataset_start_date,
        end_date=SETTINGS.dataset_end_date,
    )
    moex_data.reset_index().to_csv(SETTINGS.moex_dataset_path, index=False)

    macro_data = build_macro_monthly_dataset(
        start_date=SETTINGS.dataset_start_date,
        end_date=SETTINGS.dataset_end_date,
    )
    macro_data.to_csv(SETTINGS.macro_dataset_path)
    return macro_data
