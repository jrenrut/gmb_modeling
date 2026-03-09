import json
from pathlib import Path
import pickle
from typing import Union

from loguru import logger
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import typer
import xarray as xr

from gmb_modeling.dataset import get_gmb_region, get_sd_region, remove_anomaly

app = typer.Typer()


def cca_pseudoproxy(
    sd_pca: PCA,
    gmb_pca: PCA,
    cca: CCA,
    sd_test_data: np.ndarray,
    gmb_monthly_mean: xr.DataArray,
    time_coords: np.ndarray,
    rgi_coords: np.ndarray,
) -> xr.DataArray:
    """Use the CCA pseudoproxy method to predict the unseen GMB PCs from new SD PCs

    :param sd_pca: PCA object fitted on the SD training data
    :type sd_pca: sklearn.decomposition.PCA
    :param gmb_pca: PCA object fitted on the GMB training data
    :type gmb_pca: sklearn.decomposition.PCA
    :param cca: CCA object fitted on the training data
    :type cca: sklearn.cross_decomposition.CCA
    :param sd_test_data: Test data for SD, should be in the same format as the training data used to fit the PCA and CCA models (e.g. [time, lat, lon] or [time, features])
    :type sd_test_data: np.ndarray
    :param gmb_monthly_mean: Monthly mean GMB values used for anomaly removal, should be in the same format as the GMB training data monthly mean (e.g. [rgi_id] or [lat, lon])
    :type gmb_monthly_mean: xr.DataArray
    :param time_coords: Time coordinates corresponding to the SD test data, used for constructing the output xarray DataArray with correct time dimension
    :type time_coords: np.ndarray
    :param rgi_coords: RGI coordinates corresponding to the GMB test data, used for constructing the output xarray DataArray with correct RGI dimension
    :type rgi_coords: np.ndarray
    :return: Predicted GMB data as an xarray DataArray with dimensions [time, rgi_id] and the same time and RGI coordinates as the input test data
    :rtype: xr.DataArray
    """
    logger.info("Performing CCA pseudoproxy prediction...")
    # reshape test data if needed to ensure it's in the format [time, features] for PCA transformation
    if sd_test_data.shape[0] != len(time_coords):
        sd_test_data = np.moveaxis(sd_test_data, -1, 0)
    sd_test_data = sd_test_data.reshape(sd_test_data.shape[0], -1)

    # transform SD test data to PCs using the fitted SD PCA model
    sd_PCs_test = sd_pca.transform(sd_test_data)  # shape: [time, n_modes]
    # predict GMB PCs from SD PCs using the fitted CCA model
    gmb_CCA_pred = cca.predict(sd_PCs_test)  # shape: [time, n_modes]
    # inverse transform predicted GMB PCs back to original space using the fitted GMB PCA model
    gmb_pred_data = gmb_pca.inverse_transform(gmb_CCA_pred)  # shape: [time, n_features]
    gmb_pred_data = xr.DataArray(
        data=gmb_pred_data,
        dims=("time", "rgi_id"),
        coords={"time": time_coords, "rgi_id": rgi_coords},
        name="monthly_gmb",
    )
    # remove monthly mean from predictions to get anomalies, then add back the monthly mean to get final predicted GMB values

    gmb_pred_data = remove_anomaly(gmb_pred_data, monthly_mean=gmb_monthly_mean)
    return gmb_pred_data


@app.command()
def main(cfg: Union[Path, dict]) -> dict:
    """Retrieve data and fit models to perform CCA pseudoproxy method

    :param cfg: Configuration for prediction, can be either a path to a JSON config file or a dictionary containing the config parameters. Should contain necessary paths to data and models, as well as parameters like months and region_ids to specify the subset of data to use for prediction.
    :type cfg: Union[Path, dict]
    :return: Dictionary containing the configuration used for prediction (with any added paths to models or outputs)
    :rtype: dict
    """
    # load config from file if a path is provided, otherwise use the provided dictionary directly
    if isinstance(cfg, Path):
        logger.info(f"Loading configuration from {cfg}...")
        cfg = json.load(cfg.open())
    else:
        logger.info("Using provided configuration dictionary...")
    assert isinstance(cfg, dict)

    # set up paths for models and outputs based on workflow directory if provided, otherwise use default directories
    workflow_dir = Path(cfg.get("workflow_dir"))  # type: ignore
    if workflow_dir is not None:
        workflow_dir = Path(workflow_dir)

    # set default months and region_ids if not provided in config
    months = cfg.get("months")
    region_ids = cfg.get("region_ids")
    if cfg.get("months") == []:
        cfg["months"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if cfg.get("region_ids") == []:
        cfg["region_ids"] = [
            "VIL",
            "SIR",
            "NRM",
            "SEM",
            "NCM",
            "CCM",
            "SCM",
            "NIR",
            "CRM",
            "SRM",
        ]

    month_str = f"_{months[0]:02d}-{months[-1]:02d}"  # type: ignore
    region_str = (
        f"_{'-'.join(region_ids) if isinstance(region_ids, list) else region_ids}"
    )

    # load test data for SD and GMB, subset to specified months and regions
    processed_data_sd_test_path = cfg.get("sd_test_processed")
    processed_data_gmb_test_path = cfg.get("gmb_test_processed")
    assert processed_data_sd_test_path is not None, (
        "Processed SD test data path not found in config"
    )
    assert processed_data_gmb_test_path is not None, (
        "Processed GMB test data path not found in config"
    )
    with xr.open_dataset(processed_data_sd_test_path) as _sd:
        sd_ds = _sd.load()
    if months is not None:
        sd_ds = sd_ds.sel(time=sd_ds["time.month"].isin(months))
    sd_ds = get_sd_region(sd_ds, region_ids)
    sd_data = sd_ds["SNODP"].values
    sd_time = sd_ds["time"].values
    with xr.open_dataset(processed_data_gmb_test_path) as _gmb:
        gmb_ds = _gmb.load()
    if months is not None:
        gmb_ds = gmb_ds.sel(time=gmb_ds["time.month"].isin(months))
    gmb_ds = get_gmb_region(gmb_ds, region_ids)
    gmb_monthly_mean = gmb_ds["monthly_mean"]
    gmb_rgi = gmb_ds["rgi_id"].values

    # load fitted PCA and CCA models from training step
    gmb_pca_path = Path(cfg.get("pca_gmb_path"))  # type: ignore
    sd_pca_path = Path(cfg.get("pca_sd_path"))  # type: ignore
    with open(gmb_pca_path, "rb") as f:
        pca_gmb, gmb_PCs_train, gmb_eigvecs = pickle.load(f)
    with open(sd_pca_path, "rb") as f:
        pca_sd, sd_PCs_train, sd_eigvecs = pickle.load(f)
    cca_path = Path(cfg.get("cca_path"))  # type: ignore
    cfg["cca_path"] = str(cca_path)
    with open(cca_path, "rb") as f:
        cca, U, V, R, A, B, F, G = pickle.load(f)

    # perform CCA pseudoproxy prediction to get predicted GMB values for the test period based on the SD test data and the fitted PCA and CCA models
    gmb_pred_data = cca_pseudoproxy(
        pca_sd, pca_gmb, cca, sd_data, gmb_monthly_mean, sd_time, gmb_rgi
    )

    # save predictions into the workflow's predicted directory when possible
    outpath = workflow_dir / f"gmb_cca_predictions{month_str}{region_str}.nc"
    cfg["gmb_pred_path"] = str(outpath)
    gmb_pred_ds = gmb_pred_data.to_dataset(name="monthly_gmb")
    logger.info(f"Saving predicted GMB data to {outpath}...")
    gmb_pred_ds.to_netcdf(outpath, format="NETCDF4")

    return cfg


if __name__ == "__main__":
    app()
