import pickle

import numpy as np
import typer
import xarray as xr

from gmb_modeling.config import MODELS_DIR, PREDICTED_DATA_DIR, PROCESSED_DATA_DIR
from gmb_modeling.dataset import remove_anomaly

app = typer.Typer()


def cca_pseudoproxy(
    sd_pca, gmb_pca, cca, sd_test_data, gmb_monthly_mean, time_coords, rgi_coords
):
    sd_test_data = np.reshape(
        sd_test_data, (sd_test_data.shape[0], np.prod(sd_test_data.shape[1:]))
    )
    sd_PCs_test = sd_pca.transform(sd_test_data)
    gmb_CCA_pred = cca.predict(sd_PCs_test)
    gmb_pred_data = gmb_pca.inverse_transform(gmb_CCA_pred)
    gmb_pred_data = xr.DataArray(
        data=gmb_pred_data,
        dims=("time", "rgi_id"),
        coords={"time": time_coords, "rgi_id": rgi_coords},
        name="monthly_gmb",
    )
    gmb_pred_data = remove_anomaly(gmb_pred_data, monthly_mean=gmb_monthly_mean)
    return gmb_pred_data


@app.command()
def main():
    processed_data_sd_test_path = PROCESSED_DATA_DIR / "sd_data" / "test" / "data.nc"
    processed_data_gmb_test_path = PROCESSED_DATA_DIR / "gmb_data" / "test" / "data.nc"
    sd_ds = xr.open_dataset(processed_data_sd_test_path)["SNODP"]
    sd_data = sd_ds.values
    sd_time = sd_ds["time"].values
    gmb_ds = xr.open_dataset(processed_data_gmb_test_path)
    gmb_data = gmb_ds["monthly_gmb"].values
    gmb_monthly_mean = gmb_ds["monthly_mean"]
    gmb_rgi = gmb_ds["rgi_id"].values

    gmb_pca_path = MODELS_DIR / "pca_gmb_train.pkl"
    sd_pca_path = MODELS_DIR / "pca_sd_train.pkl"

    with open(gmb_pca_path, "rb") as f:
        pca_gmb, gmb_PCs_train, gmb_eigvecs = pickle.load(f)
    with open(sd_pca_path, "rb") as f:
        pca_sd, sd_PCs_train, sd_eigvecs = pickle.load(f)

    cca_path = MODELS_DIR / "cca_sd_gmb_train.pkl"
    with open(cca_path, "rb") as f:
        cca, U, V = pickle.load(f)

    gmb_pred_data = cca_pseudoproxy(
        pca_sd, pca_gmb, cca, sd_data, gmb_monthly_mean, sd_time, gmb_rgi
    )

    outpath = PREDICTED_DATA_DIR / "gmb_data" / "test" / "gmb_cca_predictions.nc"
    gmb_pred_ds = gmb_pred_data.to_dataset(name="monthly_gmb")
    gmb_pred_ds.to_netcdf(outpath, format="NETCDF4")


if __name__ == "__main__":
    app()
