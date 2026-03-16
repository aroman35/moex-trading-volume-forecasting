from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import breaks_cusumolsresid, het_arch
from statsmodels.tsa.seasonal import STL

from best_model import build_train_fitted_values
from models import LOG_TARGET_COLUMN, SarimaSpec, _fit_sarimax, load_model_dataset, select_sarima_spec
from settings import SETTINGS


plt.style.use('seaborn-v0_8-whitegrid')

COVID_DATE = pd.Timestamp('2020-03-01')
SANCTIONS_DATE = pd.Timestamp('2022-02-01')


def _add_reference_lines(axis: plt.Axes) -> None:
    for event_date, label in [(COVID_DATE, 'COVID'), (SANCTIONS_DATE, 'Санкции')]:
        axis.axvline(event_date, color='firebrick', linestyle='--', linewidth=1.2, alpha=0.85)
        axis.text(
            event_date,
            axis.get_ylim()[1],
            label,
            rotation=90,
            va='top',
            ha='right',
            color='firebrick',
            fontsize=10,
            backgroundcolor='white',
        )


def _compute_cusum_process(residuals: pd.Series) -> pd.Series:
    centered = residuals - residuals.mean()
    scale = centered.std(ddof=1)
    cusum_values = centered.cumsum() / scale
    return pd.Series(cusum_values.to_numpy(), index=residuals.index)


def run_cusum_analysis(dataset: pd.DataFrame) -> Path:
    sarima_spec = select_sarima_spec(dataset[LOG_TARGET_COLUMN])
    fitted_model = _fit_sarimax(dataset[LOG_TARGET_COLUMN], sarima_spec)
    residuals = pd.Series(np.asarray(fitted_model.resid), index=dataset.index).dropna()

    cusum_statistic, cusum_pvalue, critical_values = breaks_cusumolsresid(residuals.to_numpy(), ddof=0)
    cusum_process = _compute_cusum_process(residuals)
    critical_value_5pct = float(dict(critical_values)[5])

    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    axis.plot(cusum_process.index, cusum_process, color='navy', linewidth=1.8, label='CUSUM')
    axis.axhline(critical_value_5pct, color='darkorange', linestyle='--', linewidth=1.2, label='Граница 5%')
    axis.axhline(-critical_value_5pct, color='darkorange', linestyle='--', linewidth=1.2)
    axis.set_title('CUSUM для остатков SARIMA-модели')
    axis.legend()
    _add_reference_lines(axis)
    figure.savefig(SETTINGS.cusum_plot_path, dpi=150, bbox_inches='tight')
    plt.close(figure)

    print(f'CUSUM statistic: {cusum_statistic:.6f}')
    print(f'CUSUM p-value: {cusum_pvalue:.6f}')
    print(f'CUSUM 5% critical value: {critical_value_5pct:.6f}')
    return SETTINGS.cusum_plot_path


def run_stl_anomaly_analysis(dataset: pd.DataFrame) -> Path:
    stl_result = STL(dataset[LOG_TARGET_COLUMN], period=12, robust=True).fit()
    stl_residuals = pd.Series(stl_result.resid, index=dataset.index)
    residual_iqr = stl_residuals.quantile(0.75) - stl_residuals.quantile(0.25)
    anomaly_mask = stl_residuals.abs() > 3 * residual_iqr
    anomalies = dataset.loc[anomaly_mask, [LOG_TARGET_COLUMN]].copy()
    anomalies['stl_residual'] = stl_residuals.loc[anomaly_mask]

    print('STL anomalies:')
    if anomalies.empty:
        print('No anomalies detected')
    else:
        for anomaly_date, row in anomalies.iterrows():
            print(
                f'{anomaly_date.date()}: '
                f'log_volume={row[LOG_TARGET_COLUMN]:.6f}, '
                f'stl_residual={row["stl_residual"]:.6f}'
            )

    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    axis.plot(dataset.index, dataset[LOG_TARGET_COLUMN], color='navy', linewidth=1.8, label='log_volume')
    axis.scatter(anomalies.index, anomalies[LOG_TARGET_COLUMN], color='red', s=50, label='Аномалии', zorder=3)
    for anomaly_date in anomalies.index:
        axis.text(
            anomaly_date,
            anomalies.loc[anomaly_date, LOG_TARGET_COLUMN],
            anomaly_date.strftime('%Y-%m'),
            color='red',
            fontsize=9,
            ha='left',
            va='bottom',
        )
    axis.set_title('Аномалии логарифма объёма торгов по STL-остаткам')
    axis.legend()
    figure.savefig(SETTINGS.anomalies_plot_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return SETTINGS.anomalies_plot_path


def _counterfactual_sarima_forecast(dataset: pd.DataFrame) -> tuple[pd.DataFrame, SarimaSpec]:
    train_frame = dataset.loc[:'2022-01-01'].copy()
    actual_frame = dataset.loc['2022-02-01':'2022-12-01'].copy()

    sarima_spec = select_sarima_spec(train_frame[LOG_TARGET_COLUMN])
    fitted_model = _fit_sarimax(train_frame[LOG_TARGET_COLUMN], sarima_spec)
    forecast_result = fitted_model.get_forecast(steps=len(actual_frame))
    log_forecast = pd.Series(np.asarray(forecast_result.predicted_mean), index=actual_frame.index)
    confidence_interval = forecast_result.conf_int(alpha=0.05)
    confidence_interval.index = actual_frame.index

    counterfactual = pd.DataFrame(index=actual_frame.index)
    counterfactual['actual'] = np.exp(actual_frame[LOG_TARGET_COLUMN])
    counterfactual['forecast'] = np.exp(log_forecast)
    counterfactual['lower_95'] = np.exp(confidence_interval.iloc[:, 0])
    counterfactual['upper_95'] = np.exp(confidence_interval.iloc[:, 1])
    return counterfactual, sarima_spec


def run_counterfactual_analysis(dataset: pd.DataFrame) -> Path:
    counterfactual_frame, sarima_spec = _counterfactual_sarima_forecast(dataset)
    cumulative_effect = float((counterfactual_frame['forecast'] - counterfactual_frame['actual']).sum() / 1e12)
    print(f'Counterfactual SARIMA order: {sarima_spec.order}, seasonal_order: {sarima_spec.seasonal_order}')
    print(f'Cumulative effect for Feb-Dec 2022: {cumulative_effect:.6f} trillion RUB')

    history_slice = dataset.loc['2021-01-01':'2023-12-01'].copy()
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    axis.plot(history_slice.index, np.exp(history_slice[LOG_TARGET_COLUMN]) / 1e12, color='black', linewidth=2, label='Факт')
    axis.plot(counterfactual_frame.index, counterfactual_frame['forecast'] / 1e12, color='darkgreen', linewidth=2, label='Контрфактический прогноз')
    axis.fill_between(
        counterfactual_frame.index,
        counterfactual_frame['lower_95'] / 1e12,
        counterfactual_frame['upper_95'] / 1e12,
        color='darkgreen',
        alpha=0.2,
        label='95% интервал',
    )
    axis.axvline(SANCTIONS_DATE, color='firebrick', linestyle='--', linewidth=1.2)
    axis.set_title('Контрфактический прогноз объёма торгов в 2022 году')
    axis.set_ylabel('трлн руб.')
    axis.legend()
    figure.savefig(SETTINGS.counterfactual_plot_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return SETTINGS.counterfactual_plot_path


def run_arch_test(dataset: pd.DataFrame) -> None:
    train_frame = dataset.loc[:SETTINGS.train_end_date].copy()
    fitted_frame, _, _ = build_train_fitted_values(train_frame)
    arch_statistic, arch_pvalue, _, _ = het_arch(fitted_frame['residual'], nlags=12)
    conclusion = 'ARCH effects detected' if arch_pvalue < 0.05 else 'No ARCH effects detected'
    print(f'Engle ARCH statistic: {arch_statistic:.6f}')
    print(f'Engle ARCH p-value: {arch_pvalue:.6f}')
    print(conclusion)


def save_structural_artifacts() -> tuple[Path, Path, Path]:
    dataset = load_model_dataset()
    cusum_plot_path = run_cusum_analysis(dataset)
    anomalies_plot_path = run_stl_anomaly_analysis(dataset)
    counterfactual_plot_path = run_counterfactual_analysis(dataset)
    run_arch_test(dataset)
    return cusum_plot_path, anomalies_plot_path, counterfactual_plot_path
