from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

from settings import SETTINGS


plt.style.use('seaborn-v0_8-whitegrid')


def load_macro_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    dataset_path = csv_path or SETTINGS.macro_dataset_path
    dataset = pd.read_csv(dataset_path, parse_dates=['date'])
    dataset = dataset.set_index('date').sort_index().asfreq('MS')
    dataset.index.name = 'date'
    dataset['log_volume'] = np.log(dataset['moex_value_sum'])
    dataset['diff_log_volume'] = dataset['log_volume'].diff()
    dataset['seasonal_diff_log_volume'] = dataset['log_volume'].diff(12)
    return dataset


def _add_event_markers(axis: plt.Axes) -> None:
    event_markers = [
        (pd.Timestamp('2020-03-01'), 'COVID'),
        (pd.Timestamp('2022-02-01'), 'Санкции'),
    ]
    y_top = axis.get_ylim()[1]
    for event_date, event_label in event_markers:
        axis.axvline(event_date, color='firebrick', linestyle='--', linewidth=1.2, alpha=0.85)
        axis.text(
            event_date,
            y_top,
            event_label,
            rotation=90,
            va='top',
            ha='right',
            color='firebrick',
            fontsize=10,
            backgroundcolor='white',
        )


def _format_time_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axis.tick_params(axis='x', rotation=45)


def save_series_plots(dataset: pd.DataFrame, output_dir: Path | None = None) -> Path:
    destination_dir = output_dir or SETTINGS.output_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()

    volume_in_trillions = dataset['moex_value_sum'] / 1e12
    axes[0].plot(dataset.index, volume_in_trillions, color='navy', linewidth=1.8)
    axes[0].set_title('Объём торгов MOEX')
    axes[0].set_ylabel('трлн руб.')
    _add_event_markers(axes[0])
    _format_time_axis(axes[0])

    axes[1].plot(dataset.index, dataset['log_volume'], color='darkgreen', linewidth=1.8)
    axes[1].set_title('Логарифм объёма торгов')
    axes[1].set_ylabel('log(volume)')
    _format_time_axis(axes[1])

    axes[2].plot(dataset.index, dataset['diff_log_volume'], color='darkorange', linewidth=1.5)
    axes[2].set_title('Первая разность log(volume)')
    axes[2].set_ylabel('Δ log(volume)')
    _format_time_axis(axes[2])

    axes[3].plot(dataset.index, dataset['seasonal_diff_log_volume'], color='purple', linewidth=1.5)
    axes[3].set_title('Сезонная разность log(volume) с лагом 12')
    axes[3].set_ylabel('Δ12 log(volume)')
    _format_time_axis(axes[3])

    output_path = destination_dir / 'plot_series.png'
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return output_path


def save_stl_plot(dataset: pd.DataFrame, output_dir: Path | None = None) -> Path:
    destination_dir = output_dir or SETTINGS.output_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    stl_result = STL(dataset['log_volume'], period=12, robust=True).fit()

    figure, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, constrained_layout=True)
    stl_components = [
        (dataset['log_volume'], 'Исходный ряд'),
        (stl_result.trend, 'Тренд'),
        (stl_result.seasonal, 'Сезонность'),
        (stl_result.resid, 'Остаток'),
    ]

    for axis, (series, title) in zip(axes, stl_components):
        axis.plot(dataset.index, series, linewidth=1.6)
        axis.set_title(f'STL-декомпозиция: {title}')
        _format_time_axis(axis)

    output_path = destination_dir / 'plot_stl.png'
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return output_path


def save_acf_plot(dataset: pd.DataFrame, output_dir: Path | None = None) -> Path:
    destination_dir = output_dir or SETTINGS.output_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    plot_acf(dataset['log_volume'].dropna(), lags=36, ax=axes[0])
    plot_pacf(dataset['log_volume'].dropna(), lags=36, ax=axes[1], method='ywm')
    axes[0].set_title('ACF логарифма объёма торгов')
    axes[1].set_title('PACF логарифма объёма торгов')

    output_path = destination_dir / 'plot_acf.png'
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return output_path


def save_differenced_acf_plot(dataset: pd.DataFrame, output_dir: Path | None = None) -> Path:
    destination_dir = output_dir or SETTINGS.output_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    differenced_series = dataset['diff_log_volume'].dropna()
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    plot_acf(differenced_series, lags=36, ax=axes[0])
    plot_pacf(differenced_series, lags=36, ax=axes[1], method='ywm')
    axes[0].set_title('ACF первой разности log(volume)')
    axes[1].set_title('PACF первой разности log(volume)')

    output_path = destination_dir / 'plot_acf_diff.png'
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return output_path


def save_all_plots() -> list[Path]:
    dataset = load_macro_dataset()
    return [
        save_series_plots(dataset),
        save_stl_plot(dataset),
        save_acf_plot(dataset),
        save_differenced_acf_plot(dataset),
    ]
