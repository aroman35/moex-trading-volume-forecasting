from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from settings import SETTINGS


def load_target_series(csv_path: Path | None = None) -> dict[str, pd.Series]:
    dataset_path = csv_path or SETTINGS.macro_dataset_path
    dataset = pd.read_csv(dataset_path, parse_dates=['date'])
    dataset = dataset.set_index('date').sort_index().asfreq('MS')
    dataset['log_volume'] = np.log(dataset['moex_value_sum'])
    dataset['diff_log_volume'] = dataset['log_volume'].diff()
    return {
        'moex_value_sum': dataset['moex_value_sum'].dropna(),
        'log_volume': dataset['log_volume'].dropna(),
        'diff_log_volume': dataset['diff_log_volume'].dropna(),
    }


def run_adf_test(series_name: str, values: pd.Series) -> dict[str, object]:
    adf_result = adfuller(values, autolag='AIC')
    test_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    conclusion = 'Стационарен' if p_value < 0.05 else 'Нестационарен'
    return {
        'series': series_name,
        'test': 'ADF',
        'statistic': test_statistic,
        'p_value': p_value,
        'crit_1pct': critical_values.get('1%'),
        'crit_5pct': critical_values.get('5%'),
        'crit_10pct': critical_values.get('10%'),
        'conclusion': conclusion,
    }


def run_kpss_test(series_name: str, values: pd.Series) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kpss_result = kpss(values, regression='c', nlags='auto')

    test_statistic = kpss_result[0]
    p_value = kpss_result[1]
    critical_values = kpss_result[3]
    conclusion = 'Стационарен' if p_value > 0.05 else 'Нестационарен'
    return {
        'series': series_name,
        'test': 'KPSS',
        'statistic': test_statistic,
        'p_value': p_value,
        'crit_1pct': critical_values.get('1%'),
        'crit_5pct': critical_values.get('5%'),
        'crit_10pct': critical_values.get('10%'),
        'conclusion': conclusion,
    }


def run_stationarity_tests() -> pd.DataFrame:
    target_series = load_target_series()
    result_rows: list[dict[str, object]] = []

    for series_name, values in target_series.items():
        result_rows.append(run_adf_test(series_name, values))
        result_rows.append(run_kpss_test(series_name, values))

    return pd.DataFrame(
        result_rows,
        columns=[
            'series',
            'test',
            'statistic',
            'p_value',
            'crit_1pct',
            'crit_5pct',
            'crit_10pct',
            'conclusion',
        ],
    )


def save_stationarity_results() -> Path:
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_stationarity_tests()
    results.to_csv(SETTINGS.stationarity_results_path, index=False)
    return SETTINGS.stationarity_results_path
