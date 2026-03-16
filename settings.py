from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectSettings:
    base_dir: Path
    output_dirname: str = 'output_artefacts'
    dataset_start_date: str = '2013-09-01'
    dataset_end_date: str = '2026-02-01'
    train_end_date: str = '2023-12-01'
    test_start_date: str = '2024-01-01'
    macro_dataset_filename: str = 'moex_macro_monthly.csv'
    moex_dataset_filename: str = 'moex_monthly.csv'
    stationarity_results_filename: str = 'stationarity_results.csv'
    forecasts_filename: str = 'forecasts.csv'
    metrics_filename: str = 'metrics.csv'
    forecasts_plot_filename: str = 'plot_forecasts.png'
    residuals_plot_filename: str = 'plot_residuals.png'
    two_year_forecast_filename: str = 'forecast_2yr.csv'
    two_year_forecast_plot_filename: str = 'plot_forecast_2yr.png'
    cusum_plot_filename: str = 'plot_cusum.png'
    anomalies_plot_filename: str = 'plot_anomalies.png'
    counterfactual_plot_filename: str = 'plot_counterfactual.png'

    @property
    def output_dir(self) -> Path:
        return self.base_dir / self.output_dirname

    @property
    def macro_dataset_path(self) -> Path:
        return self.output_dir / self.macro_dataset_filename

    @property
    def moex_dataset_path(self) -> Path:
        return self.output_dir / self.moex_dataset_filename

    @property
    def stationarity_results_path(self) -> Path:
        return self.output_dir / self.stationarity_results_filename

    @property
    def forecasts_path(self) -> Path:
        return self.output_dir / self.forecasts_filename

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / self.metrics_filename

    @property
    def forecasts_plot_path(self) -> Path:
        return self.output_dir / self.forecasts_plot_filename

    @property
    def residuals_plot_path(self) -> Path:
        return self.output_dir / self.residuals_plot_filename

    @property
    def two_year_forecast_path(self) -> Path:
        return self.output_dir / self.two_year_forecast_filename

    @property
    def two_year_forecast_plot_path(self) -> Path:
        return self.output_dir / self.two_year_forecast_plot_filename

    @property
    def cusum_plot_path(self) -> Path:
        return self.output_dir / self.cusum_plot_filename

    @property
    def anomalies_plot_path(self) -> Path:
        return self.output_dir / self.anomalies_plot_filename

    @property
    def counterfactual_plot_path(self) -> Path:
        return self.output_dir / self.counterfactual_plot_filename


SETTINGS = ProjectSettings(base_dir=Path(__file__).resolve().parent)
