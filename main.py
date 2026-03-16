import argparse

from best_model import save_best_model_artifacts
from dataset_pipeline import save_datasets
from models import save_model_artifacts
from settings import SETTINGS
from stationarity_tests import save_stationarity_results
from structural import save_structural_artifacts
from visualization import save_all_plots


def run_dataset_command() -> None:
    dataset = save_datasets()
    print(f'Rows (months): {len(dataset)}')
    print(f'Period: {dataset.index.min().date()} - {dataset.index.max().date()}')
    print(f'Saved: {SETTINGS.output_dirname}/{SETTINGS.macro_dataset_filename}')


def run_plots_command() -> None:
    plot_paths = save_all_plots()
    saved_files = ', '.join(path.name for path in plot_paths)
    print(f'Saved: {saved_files}')


def run_stationarity_command() -> None:
    output_path = save_stationarity_results()
    print(f'Saved: {output_path.name}')


def run_models_command() -> None:
    forecasts_path, metrics_path, plot_path = save_model_artifacts()
    print(f'Saved: {forecasts_path.name}, {metrics_path.name}, {plot_path.name}')


def run_best_model_command() -> None:
    residuals_plot_path, forecast_csv_path, forecast_plot_path = save_best_model_artifacts()
    print(f'Saved: {residuals_plot_path.name}, {forecast_csv_path.name}, {forecast_plot_path.name}')


def run_structural_command() -> None:
    cusum_plot_path, anomalies_plot_path, counterfactual_plot_path = save_structural_artifacts()
    print(f'Saved: {cusum_plot_path.name}, {anomalies_plot_path.name}, {counterfactual_plot_path.name}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='MOEX project pipeline')
    parser.add_argument(
        'command',
        nargs='?',
        choices=['dataset', 'plots', 'stationarity', 'models', 'best-model', 'structural', 'all'],
        default='all',
        help='Pipeline step to run',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {'dataset', 'all'}:
        run_dataset_command()

    if args.command in {'plots', 'all'}:
        run_plots_command()

    if args.command in {'stationarity', 'all'}:
        run_stationarity_command()

    if args.command in {'models', 'all'}:
        run_models_command()

    if args.command in {'best-model', 'all'}:
        run_best_model_command()

    if args.command in {'structural', 'all'}:
        run_structural_command()


if __name__ == '__main__':
    main()
