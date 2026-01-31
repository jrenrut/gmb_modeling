import pickle

from loguru import logger
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import typer
import xarray as xr

from gmb_modeling.config import MODELS_DIR, PROCESSED_DATA_DIR
from gmb_modeling.plots import (
    plot_cca_modes,
    plot_pca_modes_gmb,
    plot_pca_modes_sd,
    plot_pca_variance,
)

app = typer.Typer()


def fit_pca(data, min_var=0.9, n_modes=None):
    if n_modes is None:
        n_modes = np.min(data.shape)
    pca = PCA(n_components=n_modes)
    PCs = pca.fit_transform(data)
    eigvecs = pca.components_
    explained_variance = pca.explained_variance_ratio_
    cumvar = np.cumsum(explained_variance)
    n_modes = np.argmax(cumvar >= min_var) + 1
    logger.info(f"Selected {n_modes} modes to explain {min_var * 100}% variance")
    return (
        pca,
        PCs[:, :n_modes],
        eigvecs[:n_modes, :],
        explained_variance[:n_modes],
    )


def fit_pca_gmb():
    processed_data_gmb_train_path = PROCESSED_DATA_DIR / "gmb_data" / "train" / "data.nc"
    data = xr.open_dataset(processed_data_gmb_train_path)
    gmb_data = data["monthly_gmb"].values.T
    logger.info(f"Fitting GMB PCA - initial shape: {gmb_data.shape}")
    pca_gmb_train, gmb_PCs_train, gmb_eigvecs, gmb_explained_variance = fit_pca(gmb_data)
    n_modes = len(gmb_explained_variance)
    plot_pca_variance(
        gmb_explained_variance,
        n_modes=n_modes,
    )
    pca_gmb_train, gmb_PCs_train, gmb_eigvecs, gmb_explained_variance = fit_pca(
        gmb_data, n_modes=n_modes
    )
    plot_pca_modes_gmb(
        gmb_eigvecs,
        gmb_PCs_train,
        data["lon"].values,
        data["lat"].values,
        data["time"].values,
        n_modes=n_modes,
    )
    pca_outpath = MODELS_DIR / "pca_gmb_train.pkl"
    with open(pca_outpath, "wb") as f:
        pickle.dump((pca_gmb_train, gmb_PCs_train, gmb_eigvecs), f)


def fit_pca_sd():
    processed_data_sd_train_path = PROCESSED_DATA_DIR / "sd_data" / "train" / "data.nc"
    data = xr.open_dataset(processed_data_sd_train_path)
    sd_data = data["SNODP"].values
    logger.info(f"Fitting SD PCA - initial shape: {sd_data.shape}")
    sd_data = np.swapaxes(
        np.reshape(sd_data, (sd_data.shape[0], np.prod(sd_data.shape[1:]))), 0, 1
    ).T
    pca_sd_train, sd_PCs_train, sd_eigvecs, sd_explained_variance = fit_pca(sd_data)
    n_modes = len(sd_explained_variance)
    plot_pca_variance(
        sd_explained_variance,
        n_modes=n_modes,
    )
    pca_sd_train, sd_PCs_train, sd_eigvecs, sd_explained_variance = fit_pca(
        sd_data, n_modes=n_modes
    )
    plot_pca_modes_sd(
        sd_eigvecs,
        sd_PCs_train,
        data["lon"].values,
        data["lat"].values,
        data["time"].values,
        n_modes=n_modes,
    )
    pca_outpath = MODELS_DIR / "pca_sd_train.pkl"
    with open(pca_outpath, "wb") as f:
        pickle.dump((pca_sd_train, sd_PCs_train, sd_eigvecs), f)


def fit_cca(sd_PCs, gmb_PCs):
    logger.info("Fitting CCA between SD and GMB PCs...")
    n_modes = min(sd_PCs.shape[1], gmb_PCs.shape[1])
    cca = CCA(n_components=n_modes)
    U, V = cca.fit_transform(sd_PCs, gmb_PCs)

    normx = np.empty_like(U[0, :])
    normy = np.empty_like(normx)

    for ind in range(len(U[0, :])):
        normx[ind] = np.std(U[:, ind])
        normy[ind] = np.std(V[:, ind])

        U[:, ind] = U[:, ind] / normx[ind]
        V[:, ind] = V[:, ind] / normy[ind]

    # Canonical weights
    A = cca.x_weights_
    B = cca.y_weights_
    # Canonical loadings
    F = np.cov(sd_PCs.T) @ A  # shape: [n_modes, n_modes]
    G = np.cov(gmb_PCs.T) @ B
    # Correlation for each mode
    r = [np.corrcoef(U[:, ii], V[:, ii])[0, 1] for ii in range(n_modes)]

    logger.info(
        f"CCA fitting complete. Canonical correlations: {[f'{val:.2f}' for val in r]}"
    )

    cca_outpath = MODELS_DIR / "cca_sd_gmb_train.pkl"
    with open(cca_outpath, "wb") as f:
        pickle.dump((cca, U, V), f)


@app.command()
def main():
    fit_pca_gmb()
    fit_pca_sd()

    gmb_pca_path = MODELS_DIR / "pca_gmb_train.pkl"
    sd_pca_path = MODELS_DIR / "pca_sd_train.pkl"

    with open(gmb_pca_path, "rb") as f:
        pca_gmb_train, gmb_PCs_train, gmb_eigvecs = pickle.load(f)
    with open(sd_pca_path, "rb") as f:
        pca_sd_train, sd_PCs_train, sd_eigvecs = pickle.load(f)

    fit_cca(sd_PCs_train, gmb_PCs_train)
    cca_path = MODELS_DIR / "cca_sd_gmb_train.pkl"
    with open(cca_path, "rb") as f:
        cca, U, V = pickle.load(f)

    processed_data_sd_train_path = PROCESSED_DATA_DIR / "sd_data" / "train" / "data.nc"
    sd_data = xr.open_dataset(processed_data_sd_train_path)
    sd_lon = sd_data["lon"].values
    sd_lat = sd_data["lat"].values
    train_time = sd_data["time"].values

    processed_data_gmb_train_path = PROCESSED_DATA_DIR / "gmb_data" / "train" / "data.nc"
    gmb_data = xr.open_dataset(processed_data_gmb_train_path)
    gmb_lon = gmb_data["lon"].values
    gmb_lat = gmb_data["lat"].values

    plot_cca_modes(
        cca,
        U,
        V,
        sd_eigvecs,
        gmb_eigvecs,
        sd_lon,
        sd_lat,
        gmb_lon,
        gmb_lat,
        train_time,
    )


if __name__ == "__main__":
    app()
