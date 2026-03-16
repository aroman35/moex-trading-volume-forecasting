from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from settings import SETTINGS


plt.style.use('seaborn-v0_8-whitegrid')

TARGET_COLUMN = 'moex_value_sum'
LOG_TARGET_COLUMN = 'log_volume'
EXOG_COLUMNS = ['usdrub', 'brent', 'key_rate', 'rv', 'trading_days']
FORECAST_COLUMNS = ['naive', 'ets', 'theta', 'sarima', 'arimax', 'rf', 'ensemble']
RF_LAGS = [1, 2, 3, 6, 12]
RF_WINDOWS = [3, 6]


@dataclass(frozen=True)
class SarimaSpec:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]


def load_model_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    dataset_path = csv_path or SETTINGS.macro_dataset_path
    dataset = pd.read_csv(dataset_path, parse_dates=['date'])
    dataset = dataset.set_index('date').sort_index().asfreq('MS')
    dataset.index.name = 'date'
    dataset[LOG_TARGET_COLUMN] = np.log(dataset[TARGET_COLUMN])
    return dataset


def split_train_test(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = dataset.loc[:SETTINGS.train_end_date].copy()
    test_data = dataset.loc[SETTINGS.test_start_date:].copy()
    return train_data, test_data


def forecast_seasonal_naive(train_series: pd.Series, horizon_index: pd.Index, seasonal_lag: int = 12) -> pd.Series:
    history = train_series.tolist()
    forecasts: list[float] = []
    for _ in horizon_index:
        next_value = history[-seasonal_lag]
        forecasts.append(next_value)
        history.append(next_value)
    return pd.Series(forecasts, index=horizon_index)


def forecast_ets(train_series: pd.Series, horizon_index: pd.Index) -> pd.Series:
    model = ExponentialSmoothing(
        train_series,
        trend='add',
        seasonal='add',
        seasonal_periods=12,
        initialization_method='estimated',
    )
    fitted_model = model.fit(optimized=True)
    forecasts = fitted_model.forecast(len(horizon_index))
    return pd.Series(np.asarray(forecasts), index=horizon_index)


def forecast_theta_fallback(train_series: pd.Series, horizon_index: pd.Index, alpha: float = 0.5) -> pd.Series:
    time_index = np.arange(len(train_series))
    slope, intercept = np.polyfit(time_index, train_series.to_numpy(), 1)
    fitted_trend = intercept + slope * time_index
    residuals = train_series.to_numpy() - fitted_trend

    smoothed_residual = residuals[0]
    for residual in residuals[1:]:
        smoothed_residual = alpha * residual + (1 - alpha) * smoothed_residual

    forecast_time_index = np.arange(len(train_series), len(train_series) + len(horizon_index))
    trend_forecast = intercept + slope * forecast_time_index
    forecasts = trend_forecast + smoothed_residual
    return pd.Series(forecasts, index=horizon_index)


def _fit_sarimax(
    endogenous: pd.Series,
    spec: SarimaSpec,
    exogenous: pd.DataFrame | None = None,
) -> object:
    model = SARIMAX(
        endogenous,
        exog=exogenous,
        order=spec.order,
        seasonal_order=spec.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.fit(disp=False)


def select_sarima_spec(train_series: pd.Series) -> SarimaSpec:
    try:
        import pmdarima as pm  # type: ignore

        auto_model = pm.auto_arima(
            train_series,
            seasonal=True,
            m=12,
            d=1,
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
        )
        return SarimaSpec(order=auto_model.order, seasonal_order=auto_model.seasonal_order)
    except ImportError:
        best_spec: SarimaSpec | None = None
        best_aic = np.inf
        for p in range(4):
            for q in range(4):
                for seasonal_p in range(3):
                    for seasonal_q in range(3):
                        for seasonal_d in (0, 1):
                            candidate_spec = SarimaSpec(
                                order=(p, 1, q),
                                seasonal_order=(seasonal_p, seasonal_d, seasonal_q, 12),
                            )
                            try:
                                fitted_model = _fit_sarimax(train_series, candidate_spec)
                            except Exception:
                                continue
                            if np.isfinite(fitted_model.aic) and fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_spec = candidate_spec
        if best_spec is None:
            raise RuntimeError('Unable to select SARIMA specification.')
        return best_spec


def forecast_sarima(train_series: pd.Series, horizon_index: pd.Index, spec: SarimaSpec) -> pd.Series:
    fitted_model = _fit_sarimax(train_series, spec)
    forecasts = fitted_model.forecast(steps=len(horizon_index))
    return pd.Series(np.asarray(forecasts), index=horizon_index)


def _scale_exog(
    train_exog: pd.DataFrame,
    test_exog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_exog)
    scaled_test = scaler.transform(test_exog)
    scaled_train_df = pd.DataFrame(scaled_train, index=train_exog.index, columns=train_exog.columns)
    scaled_test_df = pd.DataFrame(scaled_test, index=test_exog.index, columns=test_exog.columns)
    return scaled_train_df, scaled_test_df


def forecast_arimax(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    spec: SarimaSpec,
) -> pd.Series:
    scaled_train_exog, scaled_test_exog = _scale_exog(train_frame[EXOG_COLUMNS], test_frame[EXOG_COLUMNS])
    fitted_model = _fit_sarimax(train_frame[LOG_TARGET_COLUMN], spec, exogenous=scaled_train_exog)
    forecasts = fitted_model.forecast(steps=len(test_frame), exog=scaled_test_exog)
    return pd.Series(np.asarray(forecasts), index=test_frame.index)


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


def forecast_random_forest(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> pd.Series:
    training_frame = _build_random_forest_training_frame(train_frame)
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(training_frame.drop(columns='target'), training_frame['target'])

    history_log_values = train_frame[LOG_TARGET_COLUMN].copy()
    forecasts: list[float] = []
    for forecast_date, exog_values in test_frame[EXOG_COLUMNS].iterrows():
        next_features = _build_random_forest_feature_row(history_log_values, forecast_date, exog_values)
        next_forecast = float(model.predict(next_features)[0])
        forecasts.append(next_forecast)
        history_log_values.loc[forecast_date] = next_forecast
    return pd.Series(forecasts, index=test_frame.index)


def _to_original_scale(log_series: pd.Series) -> pd.Series:
    return pd.Series(np.exp(log_series.to_numpy()), index=log_series.index)


def compute_metrics(actual_values: pd.Series, predicted_values: pd.Series) -> dict[str, float]:
    return {
        'RMSE': float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        'MAE': float(mean_absolute_error(actual_values, predicted_values)),
    }


def _forecast_one_step_random_forest(train_frame: pd.DataFrame, next_row: pd.Series) -> float:
    training_frame = _build_random_forest_training_frame(train_frame)
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(training_frame.drop(columns='target'), training_frame['target'])
    feature_row = _build_random_forest_feature_row(train_frame[LOG_TARGET_COLUMN], next_row.name, next_row[EXOG_COLUMNS])
    return float(model.predict(feature_row)[0])


def compute_cross_validation_rmse(
    train_frame: pd.DataFrame,
    sarima_spec: SarimaSpec,
    min_train_size: int = 36,
) -> pd.Series:
    fold_predictions: dict[str, list[float]] = {name: [] for name in ['naive', 'ets', 'theta', 'sarima', 'arimax', 'rf']}
    actual_values: list[float] = []

    for fold_end in range(min_train_size, len(train_frame)):
        fold_train = train_frame.iloc[:fold_end].copy()
        fold_target_row = train_frame.iloc[fold_end]
        fold_target_index = pd.DatetimeIndex([fold_target_row.name])

        actual_values.append(float(np.exp(fold_target_row[LOG_TARGET_COLUMN])))
        fold_predictions['naive'].append(float(np.exp(forecast_seasonal_naive(fold_train[LOG_TARGET_COLUMN], fold_target_index).iloc[0])))
        fold_predictions['ets'].append(float(np.exp(forecast_ets(fold_train[LOG_TARGET_COLUMN], fold_target_index).iloc[0])))
        fold_predictions['theta'].append(float(np.exp(forecast_theta_fallback(fold_train[LOG_TARGET_COLUMN], fold_target_index).iloc[0])))
        fold_predictions['sarima'].append(float(np.exp(forecast_sarima(fold_train[LOG_TARGET_COLUMN], fold_target_index, sarima_spec).iloc[0])))

        next_exog_frame = train_frame.iloc[fold_end:fold_end + 1].copy()
        fold_predictions['arimax'].append(float(np.exp(forecast_arimax(fold_train, next_exog_frame, sarima_spec).iloc[0])))
        fold_predictions['rf'].append(float(np.exp(_forecast_one_step_random_forest(fold_train, fold_target_row))))

    scores = {
        model_name: float(np.sqrt(mean_squared_error(actual_values, values)))
        for model_name, values in fold_predictions.items()
    }
    return pd.Series(scores).sort_values()


def build_model_forecasts() -> tuple[pd.DataFrame, pd.DataFrame, SarimaSpec, list[str]]:
    dataset = load_model_dataset()
    train_frame, test_frame = split_train_test(dataset)

    sarima_spec = select_sarima_spec(train_frame[LOG_TARGET_COLUMN])
    print(f'SARIMA order: {sarima_spec.order}, seasonal_order: {sarima_spec.seasonal_order}')

    log_forecasts = {
        'naive': forecast_seasonal_naive(train_frame[LOG_TARGET_COLUMN], test_frame.index),
        'ets': forecast_ets(train_frame[LOG_TARGET_COLUMN], test_frame.index),
        'theta': forecast_theta_fallback(train_frame[LOG_TARGET_COLUMN], test_frame.index),
        'sarima': forecast_sarima(train_frame[LOG_TARGET_COLUMN], test_frame.index, sarima_spec),
        'arimax': forecast_arimax(train_frame, test_frame, sarima_spec),
        'rf': forecast_random_forest(train_frame, test_frame),
    }

    cross_validation_rmse = compute_cross_validation_rmse(train_frame, sarima_spec)
    best_model_names = cross_validation_rmse.index[:3].tolist()

    actual_values = _to_original_scale(test_frame[LOG_TARGET_COLUMN])
    forecasts_frame = pd.DataFrame({'date': test_frame.index, 'actual': actual_values.to_numpy()})
    forecasts_frame = forecasts_frame.set_index('date')
    for model_name, log_prediction in log_forecasts.items():
        forecasts_frame[model_name] = np.exp(log_prediction.to_numpy())
    forecasts_frame['ensemble'] = forecasts_frame[best_model_names].mean(axis=1)
    forecasts_frame = forecasts_frame.reset_index()

    metrics_rows = []
    for model_name in FORECAST_COLUMNS:
        metrics = compute_metrics(forecasts_frame['actual'], forecasts_frame[model_name])
        metrics_rows.append({'model': model_name, **metrics})
    metrics_frame = pd.DataFrame(metrics_rows).set_index('model')
    return forecasts_frame, metrics_frame, sarima_spec, best_model_names


def save_forecast_plot(dataset: pd.DataFrame, forecasts_frame: pd.DataFrame) -> Path:
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(16, 8), constrained_layout=True)

    axis.plot(dataset.index, dataset[TARGET_COLUMN] / 1e12, color='black', linewidth=2, label='Факт')
    forecast_index = pd.to_datetime(forecasts_frame['date'])
    for model_name in FORECAST_COLUMNS:
        axis.plot(
            forecast_index,
            forecasts_frame[model_name] / 1e12,
            linewidth=1.5,
            label=model_name,
        )

    axis.axvline(pd.Timestamp(SETTINGS.test_start_date), color='firebrick', linestyle='--', linewidth=1.2)
    axis.set_title('Прогнозы моделей для объёма торгов MOEX')
    axis.set_ylabel('трлн руб.')
    axis.legend(ncol=3)

    output_path = SETTINGS.forecasts_plot_path
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return output_path


def save_model_artifacts() -> tuple[Path, Path, Path]:
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_model_dataset()
    forecasts_frame, metrics_frame, _, _ = build_model_forecasts()
    forecasts_frame.to_csv(SETTINGS.forecasts_path, index=False)
    metrics_frame.to_csv(SETTINGS.metrics_path)
    plot_path = save_forecast_plot(dataset, forecasts_frame)
    return SETTINGS.forecasts_path, SETTINGS.metrics_path, plot_path
