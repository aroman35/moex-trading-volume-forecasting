from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from models import EXOG_COLUMNS, LOG_TARGET_COLUMN, RF_LAGS, RF_WINDOWS, load_model_dataset
from settings import SETTINGS


plt.style.use('seaborn-v0_8-whitegrid')

ENSEMBLE_COMPONENTS = ['rf', 'theta', 'ets']


def _split_train_full_test(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = dataset.loc[:SETTINGS.train_end_date].copy()
    full_frame = dataset.copy()
    return train_frame, full_frame


def _fit_ets_model(train_series: pd.Series):
    model = ExponentialSmoothing(
        train_series,
        trend='add',
        seasonal='add',
        seasonal_periods=12,
        initialization_method='estimated',
    )
    return model.fit(optimized=True)


def _fit_theta_components(series: pd.Series, alpha: float = 0.5) -> tuple[np.ndarray, float, float]:
    time_index = np.arange(len(series))
    slope, intercept = np.polyfit(time_index, series.to_numpy(), 1)
    trend = intercept + slope * time_index
    residuals = series.to_numpy() - trend

    smoothed = np.empty_like(residuals)
    smoothed[0] = residuals[0]
    for index in range(1, len(residuals)):
        smoothed[index] = alpha * residuals[index] + (1 - alpha) * smoothed[index - 1]
    return trend + smoothed, slope, intercept


def _forecast_theta(series: pd.Series, forecast_index: pd.DatetimeIndex, alpha: float = 0.5) -> pd.Series:
    _, slope, intercept = _fit_theta_components(series, alpha=alpha)
    time_index = np.arange(len(series))
    trend = intercept + slope * time_index
    residuals = series.to_numpy() - trend
    smoothed_residual = residuals[0]
    for residual in residuals[1:]:
        smoothed_residual = alpha * residual + (1 - alpha) * smoothed_residual

    forecast_steps = np.arange(len(series), len(series) + len(forecast_index))
    trend_forecast = intercept + slope * forecast_steps
    return pd.Series(trend_forecast + smoothed_residual, index=forecast_index)


def _build_random_forest_training_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    feature_frame = pd.DataFrame(index=dataset.index)
    for lag in RF_LAGS:
        feature_frame[f'lag_{lag}'] = dataset[LOG_TARGET_COLUMN].shift(lag)
    for window in RF_WINDOWS:
        feature_frame[f'rolling_mean_{window}'] = dataset[LOG_TARGET_COLUMN].shift(1).rolling(window).mean()
    feature_frame['month'] = dataset.index.month
    for column_name in EXOG_COLUMNS:
        feature_frame[column_name] = dataset[column_name]
    feature_frame['target'] = dataset[LOG_TARGET_COLUMN]
    return feature_frame.dropna()


def _build_random_forest_feature_row(
    history_log_values: pd.Series,
    forecast_date: pd.Timestamp,
    exog_values: pd.Series,
) -> pd.DataFrame:
    feature_values: dict[str, float | int] = {}
    for lag in RF_LAGS:
        feature_values[f'lag_{lag}'] = history_log_values.iloc[-lag]
    for window in RF_WINDOWS:
        feature_values[f'rolling_mean_{window}'] = history_log_values.iloc[-window:].mean()
    feature_values['month'] = forecast_date.month
    for column_name in EXOG_COLUMNS:
        feature_values[column_name] = exog_values[column_name]
    return pd.DataFrame([feature_values], index=[forecast_date])


def _fit_random_forest_model(train_frame: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame]:
    training_frame = _build_random_forest_training_frame(train_frame)
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(training_frame.drop(columns='target'), training_frame['target'])
    return model, training_frame


def build_train_fitted_values(train_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    ets_model = _fit_ets_model(train_frame[LOG_TARGET_COLUMN])
    ets_fitted = pd.Series(np.asarray(ets_model.fittedvalues), index=train_frame.index, name='ets')

    theta_fitted_values, _, _ = _fit_theta_components(train_frame[LOG_TARGET_COLUMN])
    theta_fitted = pd.Series(theta_fitted_values, index=train_frame.index, name='theta')

    rf_model, rf_training_frame = _fit_random_forest_model(train_frame)
    rf_fitted = pd.Series(
        rf_model.predict(rf_training_frame.drop(columns='target')),
        index=rf_training_frame.index,
        name='rf',
    )

    fitted_frame = pd.concat([rf_fitted, theta_fitted, ets_fitted, train_frame[LOG_TARGET_COLUMN]], axis=1).dropna()
    fitted_frame['ensemble'] = fitted_frame[ENSEMBLE_COMPONENTS].mean(axis=1)
    fitted_frame['residual'] = fitted_frame[LOG_TARGET_COLUMN] - fitted_frame['ensemble']

    actual_original = np.exp(fitted_frame[LOG_TARGET_COLUMN])
    ensemble_original = np.exp(fitted_frame['ensemble'])
    residual_std_original = float((actual_original - ensemble_original).std(ddof=1))
    return fitted_frame, actual_original - ensemble_original, residual_std_original


def save_residuals_plot(fitted_frame: pd.DataFrame) -> Path:
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    residuals = fitted_frame['residual']
    residual_mean = residuals.mean()
    residual_std = residuals.std(ddof=1)

    figure, axes = plt.subplots(3, 1, figsize=(16, 10), constrained_layout=True)

    axes[0].plot(residuals.index, residuals, color='navy', linewidth=1.5)
    axes[0].axhline(0.0, color='black', linestyle='--', linewidth=1.0)
    axes[0].set_title('Остатки ансамбля во времени')

    axes[1].hist(residuals, bins=20, density=True, color='steelblue', alpha=0.75)
    x_values = np.linspace(residuals.min(), residuals.max(), 300)
    normal_density = (
        1 / (residual_std * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x_values - residual_mean) / residual_std) ** 2)
    )
    axes[1].plot(x_values, normal_density, color='firebrick', linewidth=2)
    axes[1].set_title('Гистограмма остатков')

    plot_acf(residuals, lags=36, ax=axes[2])
    axes[2].set_title('ACF остатков ансамбля')

    output_path = SETTINGS.residuals_plot_path
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return output_path


def _future_exog_frame(full_frame: pd.DataFrame, forecast_index: pd.DatetimeIndex) -> pd.DataFrame:
    last_known_exog = full_frame.iloc[-1][EXOG_COLUMNS]
    future_exog = pd.DataFrame(index=forecast_index, columns=EXOG_COLUMNS, dtype=float)
    for column_name in EXOG_COLUMNS:
        future_exog[column_name] = float(last_known_exog[column_name])
    return future_exog


def forecast_random_forest_forward(full_frame: pd.DataFrame, forecast_index: pd.DatetimeIndex) -> pd.Series:
    rf_model, _ = _fit_random_forest_model(full_frame)
    future_exog = _future_exog_frame(full_frame, forecast_index)
    history_log_values = full_frame[LOG_TARGET_COLUMN].copy()
    forecasts: list[float] = []

    for forecast_date in forecast_index:
        feature_row = _build_random_forest_feature_row(history_log_values, forecast_date, future_exog.loc[forecast_date])
        next_forecast = float(rf_model.predict(feature_row)[0])
        forecasts.append(next_forecast)
        history_log_values.loc[forecast_date] = next_forecast
    return pd.Series(forecasts, index=forecast_index)


def build_two_year_forecast(full_frame: pd.DataFrame) -> pd.DataFrame:
    forecast_index = pd.date_range(start='2026-02-01', periods=24, freq='MS')

    ets_model = _fit_ets_model(full_frame[LOG_TARGET_COLUMN])
    ets_forecast = pd.Series(np.asarray(ets_model.forecast(len(forecast_index))), index=forecast_index)
    theta_forecast = _forecast_theta(full_frame[LOG_TARGET_COLUMN], forecast_index)
    rf_forecast = forecast_random_forest_forward(full_frame, forecast_index)

    ensemble_log_forecast = pd.concat(
        [rf_forecast.rename('rf'), theta_forecast.rename('theta'), ets_forecast.rename('ets')],
        axis=1,
    ).mean(axis=1)

    _, residuals_original, residual_std_original = build_train_fitted_values(full_frame.loc[:SETTINGS.train_end_date])
    forecast_original = np.exp(ensemble_log_forecast)
    lower_bound = forecast_original - 1.96 * residual_std_original
    upper_bound = forecast_original + 1.96 * residual_std_original

    return pd.DataFrame(
        {
            'date': forecast_index,
            'forecast': forecast_original.to_numpy(),
            'lower_95': lower_bound.to_numpy(),
            'upper_95': upper_bound.to_numpy(),
        }
    )


def save_two_year_forecast_plot(full_frame: pd.DataFrame, forecast_frame: pd.DataFrame) -> Path:
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    history_slice = full_frame.loc['2020-01-01':].copy()
    forecast_index = pd.to_datetime(forecast_frame['date'])

    figure, axis = plt.subplots(figsize=(16, 8), constrained_layout=True)
    axis.plot(history_slice.index, history_slice['moex_value_sum'] / 1e12, color='black', linewidth=2, label='Факт')
    axis.plot(forecast_index, forecast_frame['forecast'] / 1e12, color='darkgreen', linewidth=2, label='Прогноз ансамбля')
    axis.fill_between(
        forecast_index,
        forecast_frame['lower_95'] / 1e12,
        forecast_frame['upper_95'] / 1e12,
        color='darkgreen',
        alpha=0.2,
        label='95% интервал',
    )
    axis.axvline(pd.Timestamp('2026-02-01'), color='firebrick', linestyle='--', linewidth=1.2)
    axis.set_title('Двухлетний прогноз объёма торгов MOEX')
    axis.set_ylabel('трлн руб.')
    axis.legend()

    output_path = SETTINGS.two_year_forecast_plot_path
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return output_path


def save_best_model_artifacts() -> tuple[Path, Path, Path]:
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_model_dataset()
    train_frame, full_frame = _split_train_full_test(dataset)

    fitted_frame, residuals_original, _ = build_train_fitted_values(train_frame)
    residuals_plot_path = save_residuals_plot(fitted_frame)

    residual_mean = float(fitted_frame['residual'].mean())
    residual_std = float(fitted_frame['residual'].std(ddof=1))
    ljung_box = acorr_ljungbox(fitted_frame['residual'], lags=[12], return_df=True)
    print(f'Residual mean: {residual_mean:.6f}')
    print(f'Residual std: {residual_std:.6f}')
    print(f'Ljung-Box stat (lag 12): {float(ljung_box["lb_stat"].iloc[0]):.6f}')
    print(f'Ljung-Box p-value (lag 12): {float(ljung_box["lb_pvalue"].iloc[0]):.6f}')

    forecast_frame = build_two_year_forecast(full_frame)
    forecast_frame.to_csv(SETTINGS.two_year_forecast_path, index=False)
    forecast_plot_path = save_two_year_forecast_plot(full_frame, forecast_frame)
    return residuals_plot_path, SETTINGS.two_year_forecast_path, forecast_plot_path
