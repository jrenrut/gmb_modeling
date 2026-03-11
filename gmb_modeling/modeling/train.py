import json
from pathlib import Path
import pickle
from typing import Optional, Union

from loguru import logger
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import typer
import xarray as xr

from gmb_modeling.config import MODELS_DIR
from gmb_modeling.dataset import get_gcm_region, get_gmb_region
from gmb_modeling.plots import (
    plot_cca_modes,
    plot_pca_modes_gcm,
    plot_pca_modes_gmb,
    plot_pca_variance,
)

app = typer.Typer()


def fit_pca(
    data: np.ndarray, min_var: float, n_modes: Optional[int] = None, refit=False
) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a PCA model to the data and find n_modes to reach a minimum total variance

    :param data: data to fit PCA on, shape [n_samples, n_features]
    :type data: numpy.ndarray
    :param min_var: minimum total variance to explain with selected modes (between 0 and 1)
    :type min_var: float
    :param n_modes: number of modes to use, defaults to None
    :type n_modes: Optional[int], optional
    :param refit: Whether to refit the PCA with the selected number of modes to reduce noise in modes and PCs, defaults to False
    :type refit: bool, optional
    :return: PCA model, selected PCs, eigenvectors, and explained variance for each mode
    :rtype: tuple[PCA, np.ndarray, np.ndarray, np.ndarray]
    """
    if n_modes is None:
        n_modes = np.min(data.shape)
    pca = PCA(n_components=n_modes)
    PCs = pca.fit_transform(data)
    eigvecs = pca.components_
    explained_variance = pca.explained_variance_ratio_
    if refit:
        logger.info(f"Refit PCA with {n_modes} modes...")
    else:
        # find number of modes to reach minimum variance explained
        cumvar = np.cumsum(explained_variance)
        n_modes = int(np.argmax(cumvar >= min_var) + 1)
        variance_explained = cumvar[n_modes - 1]
        logger.info(
            f"Selected {n_modes} modes to explain {variance_explained * 100:.2f}% variance"
        )
    return (
        pca,
        PCs[:, :n_modes],
        eigvecs[:n_modes, :],
        explained_variance[:n_modes],
    )


def fit_pca_gmb(
    data_path: Path,
    min_var: float = 0.9,
    months: Optional[list[int]] = None,
    region_ids: Optional[Union[str, int, list]] = None,
    out_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Fit a PCA model for GMB data

    :param data_path: Path to the processed GMB training data (netCDF file)
    :type data_path: Path
    :param min_var: minimum total variance to explain with selected modes (between 0 and 1), defaults to 0.9
    :type min_var: float, optional
    :param months: Month(s) to include, defaults to None
    :type months: Optional[list[int]], optional
    :param region_ids: Region ID(s) to include, defaults to None
    :type region_ids: Optional[Union[str, int, list]], optional
    :param out_dir: Output directory for the fitted model, defaults to None
    :type out_dir: Optional[Path], optional
    :param figures_dir: Directory for saving figures, defaults to None
    :type figures_dir: Optional[Path], optional
    :return: Path to the saved PCA model
    :rtype: Path
    """
    with xr.open_dataset(data_path) as _data:
        data = _data.load()  # load into memory and close file

    # subset data by specified months and region IDs, if provided
    month_str = ""
    if months is not None:
        data = data.sel(time=data["time.month"].isin(months))
        month_str = f"_{months[0]:02d}-{months[-1]:02d}"
    region_str = ""
    if region_ids is not None:
        data = get_gmb_region(data, region_ids)
        region_str = (
            f"_{'-'.join(region_ids) if isinstance(region_ids, list) else region_ids}"
        )

    # reshape data to [n_samples, n_features] for PCA (samples are time steps, features are spatial points)
    gmb_data = data["monthly_gmb"].values.T
    logger.info(f"Fitting GMB PCA - initial shape: {gmb_data.shape}")
    pca_gmb_train, gmb_PCs_train, gmb_eigvecs, gmb_explained_variance = fit_pca(
        gmb_data,
        min_var,
    )

    # plot explained variance and PCA modes
    n_modes = len(gmb_explained_variance)
    plot_pca_variance(
        gmb_explained_variance,
        "GMB",
        n_modes=n_modes,
    )

    # refit PCA with selected number of modes to reduce noise in modes and PCs, and plot spatial patterns of selected modes
    pca_gmb_train, gmb_PCs_train, gmb_eigvecs, gmb_explained_variance = fit_pca(
        gmb_data, min_var, n_modes=n_modes, refit=True
    )

    # plot PCA modes on map
    region_ids_list = [region_ids] if isinstance(region_ids, (str, int)) else region_ids
    plot_pca_modes_gmb(
        gmb_eigvecs,
        gmb_PCs_train,
        data["lon"].values,
        data["lat"].values,
        data["time"].values,
        n_modes=n_modes,
        region_ids=region_ids_list,
        save_path=figures_dir,
    )

    # save PCA model, selected PCs, and eigenvectors to a pickle file in the output directory (workflow directory if available, otherwise MODELS_DIR)
    base = Path(out_dir) if out_dir is not None else MODELS_DIR
    base.mkdir(parents=True, exist_ok=True)
    pca_outpath = base / f"pca_gmb_train{month_str}{region_str}.pkl"
    logger.info(f"Saving GMB PCA model to {pca_outpath}...")
    with open(pca_outpath, "wb") as f:
        pickle.dump((pca_gmb_train, gmb_PCs_train, gmb_eigvecs), f)

    return pca_outpath


def fit_pca_gcm(
    data_path: Path,
    min_var: float = 0.9,
    months: Optional[list[int]] = None,
    region_ids: Optional[Union[str, int, list]] = None,
    out_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Fit a PCA model for GCM data

    :param data_path: Path to the processed GCM training data (netCDF file)
    :type data_path: Path
    :param min_var: Minimum variance to retain in the PCA, defaults to 0.9
    :type min_var: float, optional
    :param months: List of months to include in the analysis, defaults to None
    :type months: Optional[list[int]], optional
    :param region_ids: Region ID(s) to include, defaults to None
    :type region_ids: Optional[Union[str, int, list]], optional
    :param out_dir: Output directory for the fitted model, defaults to None
    :type out_dir: Optional[Path], optional
    :param figures_dir: Directory for saving figures, defaults to None
    :type figures_dir: Optional[Path], optional
    :return: Path to the saved PCA model
    :rtype: Path
    """
    with xr.open_dataset(data_path) as _data:
        data = _data.load()  # load into memory and close file

    # subset data by specified months and region IDs, if provided
    month_str = ""
    if months is not None:
        data = data.sel(time=data["time.month"].isin(months))
        month_str = f"_{months[0]:02d}-{months[-1]:02d}"
    region_str = ""
    if region_ids is not None:
        data = get_gcm_region(data, region_ids)
        region_str = (
            f"_{'-'.join(region_ids) if isinstance(region_ids, list) else region_ids}"
        )

    # reshape data to [n_samples, n_features] for PCA (samples are time steps, features are spatial points)
    gcm_data = data["GCM"].values
    logger.info(f"Fitting GCM PCA - initial shape: {gcm_data.shape}")
    gcm_data = np.swapaxes(
        np.reshape(gcm_data, (gcm_data.shape[0], np.prod(gcm_data.shape[1:]))),
        0,
        1,
    ).T.astype(np.float64)

    # fit PCA and find number of modes to reach minimum variance explained, then refit with that number of modes to reduce noise in modes and PCs
    pca_gcm_train, gcm_PCs_train, gcm_eigvecs, gcm_explained_variance = fit_pca(
        gcm_data,
        min_var,
    )

    # plot explained variance and PCA modes
    n_modes = len(gcm_explained_variance)
    plot_pca_variance(
        gcm_explained_variance,
        "GCM",
        n_modes=n_modes,
    )

    # refit PCA with selected number of modes to reduce noise in modes and PCs, and plot spatial patterns of selected modes
    pca_gcm_train, gcm_PCs_train, gcm_eigvecs, gcm_explained_variance = fit_pca(
        gcm_data, min_var, n_modes=n_modes, refit=True
    )

    # plot PCA modes on map
    region_ids_list = [region_ids] if isinstance(region_ids, (str, int)) else region_ids
    plot_pca_modes_gcm(
        gcm_eigvecs,
        gcm_PCs_train,
        data["lon"].values,
        data["lat"].values,
        data["time"].values,
        n_modes=n_modes,
        region_ids=region_ids_list,
        save_path=figures_dir,
    )

    # save PCA model, selected PCs, and eigenvectors to a pickle file in the output directory (workflow directory if available, otherwise MODELS_DIR)
    base = Path(out_dir) if out_dir is not None else MODELS_DIR
    base.mkdir(parents=True, exist_ok=True)
    pca_outpath = base / f"pca_gcm_train{month_str}{region_str}.pkl"
    logger.info(f"Saving GCM PCA model to {pca_outpath}...")
    with open(pca_outpath, "wb") as f:
        pickle.dump((pca_gcm_train, gcm_PCs_train, gcm_eigvecs), f)

    return pca_outpath


def fit_cca(
    gcm_PCs,
    gmb_PCs,
    months: Optional[list[int]] = None,
    region_ids: Optional[Union[str, int, list]] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """Fit a CCA model to find correlations between GMB and GCM Principal Components (PCs)

    :param gcm_PCs: Principal components of the GCM data, shape [n_samples, n_gcm_modes]
    :type gcm_PCs: numpy.ndarray
    :param gmb_PCs: Principal components of the GMB data, shape [n_samples, n_gmb_modes]
    :type gmb_PCs: numpy.ndarray
    :param months: List of months to include in the analysis, defaults to None
    :type months: Optional[list[int]], optional
    :param region_ids: List of region IDs to include in the analysis, defaults to None
    :type region_ids: Optional[Union[str, int, list]], optional
    :param out_dir: Output directory for the fitted CCA model, defaults to None
    :type out_dir: Optional[Path], optional
    :return: Path to the saved CCA model
    :rtype: Path
    """
    logger.info("Fitting CCA between GCM and GMB PCs...")
    n_modes = min(
        gcm_PCs.shape[1], gmb_PCs.shape[1]
    )  # number of modes is limited by smaller set of PCs
    # fit CCA model and calculate canonical correlations, weights, and loadings
    cca = CCA(n_components=n_modes)
    U, V = cca.fit_transform(gcm_PCs, gmb_PCs)

    # normalize canonical variates to have unit variance for interpretability of canonical correlations and loadings
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
    F = np.cov(gcm_PCs.T) @ A  # shape: [n_modes, n_modes]
    G = np.cov(gmb_PCs.T) @ B  # shape: [n_modes, n_modes]
    # Correlation for each mode
    R = [np.corrcoef(U[:, ii], V[:, ii])[0, 1] for ii in range(n_modes)]

    logger.info(
        f"CCA fitting complete. Canonical correlations: {[f'{val:.2f}' for val in R]}"
    )

    # save CCA model, canonical variates, correlations, weights, and loadings to a pickle file in the output directory (workflow directory if available, otherwise MODELS_DIR)
    month_str = ""
    if months is not None:
        month_str = f"_{months[0]:02d}-{months[-1]:02d}"
    region_str = ""
    if region_ids is not None:
        region_str = (
            f"_{'-'.join(region_ids) if isinstance(region_ids, list) else region_ids}"
        )
    base = Path(out_dir) if out_dir is not None else MODELS_DIR
    base.mkdir(parents=True, exist_ok=True)
    cca_outpath = base / f"cca_gcm_gmb_train_{month_str}_{region_str}.pkl"
    logger.info(f"Saving CCA model to {cca_outpath}...")
    with open(cca_outpath, "wb") as f:
        pickle.dump((cca, U, V, R, A, B, F, G), f)

    return cca_outpath


@app.command()
def main(cfg: Union[Path, dict]) -> tuple[dict, dict]:
    """Fit PCA models for GMB and GCM training data, and a CCA model to find correlations between GMB and GCM PCs. Save fitted models and results to output directory (workflow directory if available, otherwise MODELS_DIR).

    :param cfg: Configuration for training, either as a path to a JSON config file or a dictionary. Should include keys for "months", "region_ids", "pca_min_var", and paths for processed training data.
    :type cfg: Union[Path, dict]
    :return: Updated configuration dictionary with paths to fitted models, and a results dictionary containing PCA and CCA information
    :rtype: tuple[dict, dict]
    """
    # load config from file if a path is provided, otherwise use the provided dictionary
    if isinstance(cfg, Path):
        logger.info(f"Loading configuration from {cfg}...")
        cfg = json.load(cfg.open())
    else:
        logger.info("Using provided configuration dictionary...")
    assert isinstance(cfg, dict)

    # extract parameters from config, with defaults for months and region IDs if not provided
    months = cfg.get("months")
    region_ids = cfg.get("region_ids")
    pca_min_var = cfg.get("pca_min_var", 0.9)
    cfg["pca_min_var"] = pca_min_var
    if months == []:
        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if region_ids == []:
        region_ids = [
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

    # fit PCA models for GMB and GCM training data
    workflow_dir = Path(cfg.get("workflow_dir"))  # type: ignore
    figures_dir = Path(cfg.get("figures_dir"))  # type: ignore
    gmb_processed_data_path = Path(cfg.get("gmb_train_processed"))  # type: ignore
    gmb_pca_path = fit_pca_gmb(
        gmb_processed_data_path,
        months=months,
        region_ids=region_ids,
        min_var=pca_min_var,
        out_dir=workflow_dir,
        figures_dir=figures_dir,
    )
    gcm_processed_data_path = Path(cfg.get("gcm_train_processed"))  # type: ignore
    gcm_pca_path = fit_pca_gcm(
        gcm_processed_data_path,
        months=months,
        region_ids=region_ids,
        min_var=pca_min_var,
        out_dir=workflow_dir,
        figures_dir=figures_dir,
    )
    cfg["pca_gmb_path"] = str(gmb_pca_path)
    cfg["pca_gcm_path"] = str(gcm_pca_path)
    with open(gmb_pca_path, "rb") as f:
        pca_gmb_train, gmb_PCs_train, gmb_eigvecs = pickle.load(f)
    with open(gcm_pca_path, "rb") as f:
        pca_gcm_train, gcm_PCs_train, gcm_eigvecs = pickle.load(f)

    # record PCA information for report
    gmb_explained_variance = getattr(pca_gmb_train, "explained_variance_ratio_", [])
    pca_info_gmb = {
        "n_components": int(getattr(pca_gmb_train, "n_components_", None) or 0),
        "explained_variance": list(gmb_explained_variance),
    }
    gcm_explained_variance = getattr(pca_gcm_train, "explained_variance_ratio_", [])
    pca_info_gcm = {
        "n_components": int(getattr(pca_gcm_train, "n_components_", None) or 0),
        "explained_variance": list(gcm_explained_variance),
    }

    # fit CCA model to find correlations between GMB and GCM PCs
    cca_path = fit_cca(
        gcm_PCs_train,
        gmb_PCs_train,
        months=months,
        region_ids=region_ids,
        out_dir=workflow_dir,
    )
    cfg["cca_path"] = str(cca_path)
    with open(cca_path, "rb") as f:
        cca, U, V, R, A, B, F, G = pickle.load(f)

    # record CCA information for report
    cca_info = {
        "canonical_correlations": [float(val) for val in R],
        "x_weights_shape": getattr(cca, "x_weights_").shape
        if hasattr(cca, "x_weights_")
        else None,
        "y_weights_shape": getattr(cca, "y_weights_").shape
        if hasattr(cca, "y_weights_")
        else None,
    }

    # plot CCA modes on map
    processed_data_gcm_train_path = cfg.get("gcm_train_processed")
    assert processed_data_gcm_train_path is not None, (
        "Processed GCM training data path not found in config"
    )
    with xr.open_dataset(processed_data_gcm_train_path) as _gcm:
        gcm_data = _gcm.load()
    if months is not None:
        gcm_data = gcm_data.sel(time=gcm_data["time.month"].isin(months))
    gcm_data = get_gcm_region(gcm_data, region_ids)
    gcm_lon = gcm_data["lon"].values
    gcm_lat = gcm_data["lat"].values

    train_time = gcm_data["time"].values

    processed_data_gmb_train_path = cfg.get("gmb_train_processed")
    assert processed_data_gmb_train_path is not None, (
        "Processed GMB training data path not found in config"
    )
    with xr.open_dataset(processed_data_gmb_train_path) as _gmb:
        gmb_data = _gmb.load()
    if months is not None:
        gmb_data = gmb_data.sel(time=gmb_data["time.month"].isin(months))
    gmb_data = get_gmb_region(gmb_data, region_ids)
    gmb_lon = gmb_data["lon"].values
    gmb_lat = gmb_data["lat"].values

    plot_cca_modes(
        U,
        V,
        gcm_eigvecs,
        gmb_eigvecs,
        gcm_lon,
        gcm_lat,
        gmb_lon,
        gmb_lat,
        train_time,
        region_ids=region_ids,
        save_path=figures_dir,
    )

    # compile results to return
    results = {"pca_gmb": pca_info_gmb, "pca_gcm": pca_info_gcm, "cca": cca_info}

    return cfg, results


if __name__ == "__main__":
    # app()
    config_file = (
        Path(__file__).resolve().parent.parent / "cfg" / "all-BC_2016-cutoff.json"
    )
    main(config_file)
