import json
from pathlib import Path
from typing import Union

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import typer
import xarray as xr

from gmb_modeling import plots
from gmb_modeling.dataset import get_gmb_region

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
)

app = typer.Typer()
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def error_by_month(
    processed_ds: xr.Dataset, predicted_ds: xr.Dataset, months: list | None = None
) -> tuple[list, list, list, list]:
    """Calculate error separately for each month in dataset

    :param processed_ds: Dataset containing the processed GMB data ("true")
    :type processed_ds: xarray.Dataset
    :param predicted_ds: Dataset containing the predicted GMB data ("predicted"), should have the same format and coordinates as the processed_ds (e.g. dimensions [time, rgi_id] with time and rgi_id coordinates)
    :type predicted_ds: xarray.Dataset
    :param months: List of months for which to calculate errors, defaults to None
    :type months: list, optional
    :return: Tuple containing RMSE and MAE for each month, along with the true and predicted GMB values
    :rtype: tuple[list, list, list, list]
    """
    if months is None:
        months = list(range(1, 13))

    # calculate error for each month separately and store true and predicted values for plotting
    gmb_proc, gmb_pred = [], []
    rmse_all, mae_all = [], []
    for month in months:
        pred_month = predicted_ds.sel(time=predicted_ds["time.month"] == month)
        proc_month = processed_ds.sel(time=processed_ds["time.month"] == month)

        pred_data = pred_month["monthly_gmb"].values
        proc_data = proc_month["monthly_raw"].T.values

        rmse = np.sqrt(mean_squared_error(proc_data, pred_data))
        mae = mean_absolute_error(proc_data, pred_data)

        rmse_all.append(rmse)
        mae_all.append(mae)

        gmb_proc.append(proc_data.flatten())
        gmb_pred.append(pred_data.flatten())
    return rmse_all, mae_all, gmb_proc, gmb_pred


@app.command()
def main(cfg: Union[Path, dict]) -> tuple[dict, dict]:
    """Calculate statistics for predicted GMB

    :param cfg: Configuration for analysis, can be either a path to a JSON config file or a dictionary containing the config parameters. Should contain necessary paths to data and models, as well as parameters like months and region_ids to specify the subset of data to use for analysis.
    :type cfg: Union[Path, dict]
    :return: Tuple containing the configuration used for analysis (with any added paths to models or outputs) and a dictionary of results from the analysis step (e.g. error metrics by month)
    :rtype: tuple[dict, dict]
    """
    # load config from file if a path is provided, otherwise use the provided dictionary directly
    if isinstance(cfg, Path):
        logger.info(f"Loading configuration from {cfg}...")
        cfg = json.load(cfg.open())
    else:
        logger.info("Using provided configuration dictionary...")
    assert isinstance(cfg, dict)

    # set default months and region_ids if not provided in config
    months = list(cfg.get("months"))  # type: ignore
    region_ids = list(cfg.get("region_ids"))  # type: ignore

    # load predicted GMB data for test period, subset to specified months and regions
    processed_data_path = Path(cfg.get("gmb_test_processed"))  # type: ignore
    with xr.open_dataset(processed_data_path) as _proc:
        processed_ds = _proc.load()  # type: ignore
    predicted_data_path = Path(cfg.get("gmb_pred_path"))  # type: ignore
    with xr.open_dataset(predicted_data_path) as _pred:
        predicted_ds = _pred.load()  # type: ignore

    # subset to specified regions and months if provided in config
    processed_ds = get_gmb_region(processed_ds, region_ids)
    if months is not None:
        predicted_ds = predicted_ds.sel(time=predicted_ds["time.month"].isin(months))

    # calculate error metrics by month and store true and predicted GMB values for plotting
    month_rmse, month_mae, gmb_proc, gmb_pred = error_by_month(
        processed_ds, predicted_ds, months
    )

    # generate plots of error metrics and true vs predicted GMB values by month
    figures_dir = Path(cfg.get("figures_dir"))  # type: ignore
    plots.plot_monthly_error(months, month_rmse, month_mae, save_path=figures_dir)
    plots.plot_monthly_gmb_comparison(months, gmb_proc, gmb_pred, save_path=figures_dir)
    pred_times = predicted_ds["time"].values.astype("datetime64[M]")
    processed_ds = processed_ds.sel(time=pred_times)
    plots.plot_total_error_density(processed_ds, predicted_ds, save_path=figures_dir)
    plots.plot_total_error_density_by_month(
        processed_ds, predicted_ds, save_path=figures_dir
    )
    plots.plot_error_by_feature(processed_ds, predicted_ds, save_path=figures_dir)
    # plots.plot_error_by_month(processed_ds, predicted_ds)

    # record test performance metrics in report and save JSON copy
    results = {
        "month_rmse": [float(x) for x in month_rmse],
        "month_mae": [float(x) for x in month_mae],
        "mean_rmse": float(np.mean(month_rmse)),
        "mean_mae": float(np.mean(month_mae)),
    }

    return cfg, results


if __name__ == "__main__":
    app()
