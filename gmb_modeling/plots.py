from typing import Optional, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA
import typer
import xarray as xr

from gmb_modeling.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PREDICTED_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from gmb_modeling.dataset import (
    get_gmb_region,
    get_gmb_rgiid,
    get_sd_gridpoint,
    get_sd_region,
    load_coastline,
    load_data,
    load_regions,
    remove_anomaly,
)

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


def save_figure_and_axes(fig: Figure, axes: list, filename: str) -> None:
    fig_path = FIGURES_DIR / f"{filename}.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {fig_path}")


def plot_glacier_regions() -> None:
    bc_regions = load_regions()
    coastline = load_coastline()

    coastLon = coastline.lon
    coastLat = coastline.lat

    fig, ax = plt.subplots(figsize=(10, 10))
    bc_regions.plot(ax=ax, column="name", legend=True)
    ax.plot(coastLon, coastLat, "k-", linewidth=0.5)
    ax.set_xlim((-140, -114))
    ax.set_ylim((48, 60))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Glacier Regions of British Columbia")
    plt.savefig(FIGURES_DIR / "bc_glacier_regions.png", bbox_inches="tight", dpi=300)
    plt.show()


def plot_sd(
    month: str, raw: bool = True, region_id: Optional[Union[str, int]] = None
) -> None:
    bc_regions = load_regions()
    coastline = load_coastline()

    coastLon = coastline.lon
    coastLat = coastline.lat

    if raw:
        data_path = INTERIM_DATA_DIR / "sd_data" / "monthly_sd_clean.nc"
        ds = xr.open_dataset(data_path)
        if region_id is not None:
            ds = get_sd_region(ds, region_id)
        month_np = np.datetime64(month)
        matches = np.where(ds["time"].values.astype("datetime64[M]") == month_np)[0]
        if matches.size == 0:
            raise ValueError(f"Month {month_np} not found in dataset time coordinate.")
        month_data = ds["SNODP"].isel(time=matches[0])
    else:
        month_data = None
        try:
            ds = load_data(processed_data_path="sd_data", subset="train")
            if region_id is not None:
                ds = get_sd_region(ds, region_id)
            time_vals = ds["time"].values.astype("datetime64[M]")
            month_np = np.datetime64(month)
            matches = np.where(time_vals == month_np)[0]
            if matches.size == 0:
                raise ValueError(
                    f"Month {month_np} not found in dataset time coordinate."
                )
            month_data = ds["SNODP"].isel(time=matches[0])
        except Exception:
            ds = load_data(processed_data_path="sd_data", subset="test")
            if region_id is not None:
                ds = get_sd_region(ds, region_id)
            time_vals = ds["time"].values.astype("datetime64[M]")
            month_np = np.datetime64(month)
            matches = np.where(time_vals == month_np)[0]
            if matches.size == 0:
                raise ValueError(
                    f"Month {month_np} not found in dataset time coordinate."
                )
            month_data = ds["SNODP"].isel(time=matches[0])
    sd_lat = ds["lat"].values
    sd_lon = ds["lon"].values
    fig, ax = plt.subplots(figsize=(10, 8))
    pcm = ax.pcolormesh(
        sd_lon,
        sd_lat,
        month_data,
        cmap="coolwarm",
        shading="auto",
        zorder=1,
    )
    ax.plot(coastLon, coastLat, "k", lw=1)
    region_name = ""
    if region_id is not None:
        if isinstance(region_id, str):
            region = bc_regions[bc_regions["name"] == region_id]
            region_name = region_id
        else:
            region = bc_regions[bc_regions["id"] == region_id]
            region_name = region.iloc[0]["name"]
        region.boundary.plot(ax=ax, color="red", linewidth=1, zorder=2)
    ax.set_xlim((min(sd_lon) - 0.25, max(sd_lon) + 0.25))
    ax.set_ylim((min(sd_lat) - 0.25, max(sd_lat) + 0.25))
    fig.colorbar(pcm, ax=ax, label="Snow Depth (m)")
    if raw:
        title = f"{region_name} MERRA-2 Snow Depth for month {month} (Raw)"
    else:
        title = f"{region_name} MERRA-2 Snow Depth for month {month} (Anomaly)"
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


def plot_gmb(
    month: str, raw: bool = True, region_id: Optional[Union[str, int]] = None
) -> None:
    bc_regions = load_regions()
    coastline = load_coastline()

    coastLon = coastline.lon
    coastLat = coastline.lat

    if raw:
        data_path = INTERIM_DATA_DIR / "gmb_data" / "monthly_gmb_clean.nc"
        ds = xr.open_dataset(data_path)
        if region_id is not None:
            ds = get_gmb_region(ds, region_id)
        month_np = np.datetime64(month)
        matches = np.where(ds["time"].values.astype("datetime64[M]") == month_np)[0]
        if matches.size == 0:
            raise ValueError(f"Month {month_np} not found in dataset time coordinate.")
        month_data = ds["monthly_gmb"].isel(time=matches[0])
    else:
        month_data = None
        try:
            ds = load_data(processed_data_path="gmb_data", subset="train")
            if region_id is not None:
                ds = get_gmb_region(ds, region_id)
            time_vals = ds["time"].values.astype("datetime64[M]")
            month_np = np.datetime64(month)
            matches = np.where(time_vals == month_np)[0]
            if matches.size == 0:
                raise ValueError(
                    f"Month {month_np} not found in dataset time coordinate."
                )
            month_data = ds["monthly_gmb"].isel(time=matches[0])
        except Exception:
            ds = load_data(processed_data_path="gmb_data", subset="test")
            if region_id is not None:
                ds = get_gmb_region(ds, region_id)
            time_vals = ds["time"].values.astype("datetime64[M]")
            month_np = np.datetime64(month)
            matches = np.where(time_vals == month_np)[0]
            if matches.size == 0:
                raise ValueError(
                    f"Month {month_np} not found in dataset time coordinate."
                )
            month_data = ds["monthly_gmb"].isel(time=matches[0])
    gmb_lat = ds["lat"].values
    gmb_lon = ds["lon"].values

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(coastLon, coastLat, "k", lw=1)
    region_name = ""
    if region_id is not None:
        if isinstance(region_id, str):
            region = bc_regions[bc_regions["name"] == region_id]
            region_name = region_id
        else:
            region = bc_regions[bc_regions["id"] == region_id]
            region_name = region.iloc[0]["name"]
        region.boundary.plot(ax=ax, color="red", linewidth=1, zorder=1)
    pcm = ax.scatter(
        gmb_lon,
        gmb_lat,
        c=month_data.values,
        cmap="coolwarm",
        s=100,
        edgecolor="none",
        alpha=0.8,
        zorder=2,
    )
    ax.set_xlim((min(gmb_lon) - 0.1, max(gmb_lon) + 0.1))
    ax.set_ylim((min(gmb_lat) - 0.1, max(gmb_lat) + 0.1))
    fig.colorbar(pcm, ax=ax, label="GMB (m.w.e.)")
    if raw:
        title = f"{region_name} Modeled Glacier Mass Balance for {month} (Raw)"
    else:
        title = f"{region_name} Modeled Glacier Mass Balance for {month} (Anomaly)"
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


def plot_sd_anomaly(
    lat: float,
    lon: float,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    save: bool = False,
    filename: Optional[str] = None,
) -> None:
    data_path_raw = INTERIM_DATA_DIR / "sd_data" / "monthly_sd_clean.nc"
    ds_raw = xr.open_dataset(data_path_raw)
    ds_raw = get_sd_gridpoint(ds_raw, lat, lon)
    if start_month is not None:
        start_np = np.datetime64(start_month)
        ds_raw = ds_raw.sel(time=slice(start_np, None))
    if end_month is not None:
        end_np = np.datetime64(end_month)
        ds_raw = ds_raw.sel(time=slice(None, end_np))

    try:
        ds_processed = load_data(processed_data_path="sd_data", subset="train")
        ds_processed = get_sd_gridpoint(ds_processed, lat, lon)
    except Exception:
        ds_processed = load_data(processed_data_path="sd_data", subset="test")
        ds_processed = get_sd_gridpoint(ds_processed, lat, lon)

    months = ds_raw["time"].values.astype("datetime64[M]")
    monthly_mean = ds_raw["SNODP"].groupby("time.month").mean("time")
    anomaly = ds_raw["SNODP"].groupby("time.month") - monthly_mean
    window_size = ds_processed.attrs.get("window_size", 1)
    smoothed_anomaly = anomaly.rolling(time=window_size, center=True).mean()

    latlon_str = f"[{lat:.2f}, {lon:.2f}]"

    plt.figure(figsize=(10, 15))

    plt.subplot(4, 1, 1)
    plt.plot(months, ds_raw["SNODP"], color="blue")
    plt.xlabel("Time")
    plt.ylabel("Snow Depth [m]")
    plt.title(f"Raw Value at {latlon_str}")

    plt.subplot(4, 1, 2)
    plt.plot(MONTHS, monthly_mean, color="blue")
    plt.xlabel("Time")
    plt.ylabel("Snow Depth [m]")
    plt.title(f"Seasonal Value at {latlon_str}")

    plt.subplot(4, 1, 3)
    plt.plot(months, anomaly, color="blue")
    plt.xlabel("Time")
    plt.ylabel("Snow Depth [m]")
    plt.title(f"Anomaly at {latlon_str}")

    plt.subplot(4, 1, 4)
    plt.plot(months, smoothed_anomaly, color="blue")
    plt.xlabel("Time")
    plt.ylabel("Snow Depth [m]")
    plt.title(f"Smoothed Anomaly at {latlon_str}")

    plt.tight_layout()
    if save:
        fname = filename or f"sd_anomaly_{lat:.2f}_{lon:.2f}".replace(" ", "_")
        save_figure_and_axes(plt.gcf(), plt.gcf().axes, fname)
    plt.show()


def plot_gmb_anomaly(
    rgi_id: str,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    save: bool = False,
    filename: Optional[str] = None,
) -> None:
    data_path_raw = INTERIM_DATA_DIR / "gmb_data" / "monthly_gmb_clean.nc"
    ds_raw = xr.open_dataset(data_path_raw)
    ds_raw = ds_raw.sel(rgi_id=rgi_id)
    if start_month is not None:
        start_np = np.datetime64(start_month)
        ds_raw = ds_raw.sel(time=slice(start_np, None))
    if end_month is not None:
        end_np = np.datetime64(end_month)
        ds_raw = ds_raw.sel(time=slice(None, end_np))

    try:
        ds_processed = load_data(processed_data_path="gmb_data", subset="train")
        ds_processed = get_gmb_rgiid(ds_processed, rgi_id)
    except Exception:
        ds_processed = load_data(processed_data_path="gmb_data", subset="test")
        ds_processed = get_gmb_rgiid(ds_processed, rgi_id)

    months = ds_raw["time"].values.astype("datetime64[M]")
    monthly_mean = ds_raw["monthly_gmb"].groupby("time.month").mean("time")
    anomaly = ds_raw["monthly_gmb"].groupby("time.month") - monthly_mean
    window_size = ds_processed.attrs.get("window_size", 1)
    smoothed_anomaly = anomaly.rolling(time=window_size, center=True).mean()

    plt.figure(figsize=(10, 15))

    plt.subplot(4, 1, 1)
    plt.plot(months, ds_raw["monthly_gmb"], color="green")
    plt.xlabel("Time")
    plt.ylabel("GMB [m.w.e.]")
    plt.title(f"Raw GMB for Glacier {rgi_id}")

    plt.subplot(4, 1, 2)
    plt.plot(MONTHS, monthly_mean, color="green")
    plt.xlabel("Time")
    plt.ylabel("GMB [m.w.e.]")
    plt.title(f"Seasonal GMB for Glacier {rgi_id}")

    plt.subplot(4, 1, 3)
    plt.plot(months, anomaly, color="green")
    plt.xlabel("Time")
    plt.ylabel("GMB [m.w.e.]")
    plt.title(f"GMB Anomaly for Glacier {rgi_id}")

    plt.subplot(4, 1, 4)
    plt.plot(months, smoothed_anomaly, color="green")
    plt.xlabel("Time")
    plt.ylabel("GMB [m.w.e.]")
    plt.title(f"Smoothed GMB Anomaly for Glacier {rgi_id}")

    plt.tight_layout()
    if save:
        fname = filename or f"gmb_anomaly_{rgi_id}"
        save_figure_and_axes(plt.gcf(), plt.gcf().axes, fname)
    plt.show()


def plot_pca_variance(
    explained_variance: np.ndarray, n_modes: Optional[int] = None, color="blue"
) -> None:
    title = "Variance Explained by All Modes"
    if n_modes is not None:
        explained_variance = explained_variance[:n_modes]
        title = f"Variance Explained by First {n_modes} Modes"

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(range(len(explained_variance)), explained_variance, color=color)
    plt.xlabel("Mode Number")
    plt.ylabel("Fraction Variance Explained")
    plt.title(title)


def plot_pca_modes_gmb(
    eigvecs: np.ndarray,
    PCs: np.ndarray,
    gmb_lon: np.ndarray,
    gmb_lat: np.ndarray,
    time: np.ndarray,
    n_modes: Optional[int] = None,
    save: bool = False,
    filename: Optional[str] = None,
) -> None:
    if n_modes is None:
        n_modes = int(eigvecs.shape[0])

    bc_regions = load_regions()
    coastline = load_coastline()

    coastLon = coastline.lon
    coastLat = coastline.lat

    fig, axes = plt.subplots(
        n_modes,
        2,
        figsize=(12, 3 * n_modes),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    for i in range(n_modes):
        # Plot spatial pattern (eigenvector)
        ax1 = axes[i, 0] if n_modes > 1 else axes[0, 0]
        sc = ax1.scatter(gmb_lon, gmb_lat, c=eigvecs[i], cmap="coolwarm", s=20)
        ax1.plot(coastLon, coastLat, "k", lw=1)
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_title(rf"Spatial Pattern (Eigenvector) Mode \#{i + 1}")
        plt.colorbar(sc, ax=ax1, orientation="vertical", label="Eigenvector Value")
        ax1.set_xlim((min(gmb_lon) - 0.1, max(gmb_lon) + 0.1))
        ax1.set_ylim((min(gmb_lat) - 0.1, max(gmb_lat) + 0.1))

        # Plot Principal Component (time series)
        ax2 = axes[i, 1] if n_modes > 1 else axes[0, 1]
        ax2.plot(time, PCs[:, i], color="green")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("PC Value")
        ax2.set_title(rf"PC Time Series Mode \#{i + 1}")
        ax2.grid(True)

    plt.tight_layout()
    if save:
        fname = filename or f"pca_gmb_{n_modes}modes"
        save_figure_and_axes(fig, axes, fname)
    plt.show()


def plot_pca_modes_sd(
    eigvecs: np.ndarray,
    PCs: np.ndarray,
    sd_lon: np.ndarray,
    sd_lat: np.ndarray,
    time: np.ndarray,
    n_modes: Optional[int] = None,
    save: bool = False,
    filename: Optional[str] = None,
) -> None:
    if n_modes is None:
        n_modes = int(eigvecs.shape[0])

    bc_regions = load_regions()
    coastline = load_coastline()

    coastLon = coastline.lon
    coastLat = coastline.lat

    fig, axes = plt.subplots(
        n_modes,
        2,
        figsize=(12, 3 * n_modes),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    for i in range(n_modes):
        # Plot spatial pattern (eigenvector)
        ax1 = axes[i, 0] if n_modes > 1 else axes[0, 0]
        pcm = ax1.pcolormesh(
            sd_lon,
            sd_lat,
            eigvecs[i, :].reshape(len(sd_lat), len(sd_lon)),
            cmap="coolwarm",
            shading="auto",
            zorder=1,
        )
        ax1.plot(coastLon, coastLat, "k", lw=1)
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_title(rf"Spatial Pattern (Eigenvector) Mode \#{i + 1}")
        plt.colorbar(pcm, ax=ax1, orientation="vertical", label="Eigenvector Value")
        ax1.set_xlim((min(sd_lon) - 0.1, max(sd_lon) + 0.1))
        ax1.set_ylim((min(sd_lat) - 0.1, max(sd_lat) + 0.1))

        # Plot Principal Component (time series)
        ax2 = axes[i, 1] if n_modes > 1 else axes[0, 1]
        ax2.plot(time, PCs[:, i], color="blue")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("PC Value")
        ax2.set_title(rf"PC Time Series Mode \#{i + 1}")
        ax2.grid(True)

    plt.tight_layout()
    if save:
        fname = filename or f"pca_sd_{n_modes}modes"
        save_figure_and_axes(fig, axes, fname)
    plt.show()


def plot_cca_modes(
    cca: CCA,
    U: np.ndarray,
    V: np.ndarray,
    sd_eigvecs: np.ndarray,
    gmb_eigvecs: np.ndarray,
    sd_lon: np.ndarray,
    sd_lat: np.ndarray,
    gmb_lon: np.ndarray,
    gmb_lat: np.ndarray,
    train_time: np.ndarray,
    save: bool = False,
    filename: Optional[str] = None,
) -> None:
    bc_regions = load_regions()
    coastline = load_coastline()

    coastLon = coastline.lon
    coastLat = coastline.lat

    n_modes = min(len(sd_eigvecs), len(gmb_eigvecs))

    fig, axes = plt.subplots(n_modes, 3, figsize=(18, 4 * n_modes))

    for mi, mode in enumerate(range(n_modes)):
        eig_map = sd_eigvecs[mode, :]
        eig_map2d = eig_map.reshape((len(sd_lat), len(sd_lon)))
        ax = axes[mi, 0] if n_modes > 1 else axes[0, 0]
        pcm = ax.pcolormesh(
            sd_lon,
            sd_lat,
            eig_map2d,
            cmap="coolwarm",
            shading="auto",
        )
        ax.plot(coastLon, coastLat, "k", lw=0.5)
        ax.set_xlim([min(sd_lon), max(sd_lon)])
        ax.set_ylim([min(sd_lat), max(sd_lat)])
        ax.set_title(rf"SD Eigenvector Mode \#{mode + 1}")
        plt.colorbar(pcm, ax=ax, orientation="vertical", label="SD eigenvector")

        ax = axes[mi, 1] if n_modes > 1 else axes[0, 1]
        sc = ax.scatter(
            gmb_lon,
            gmb_lat,
            c=gmb_eigvecs[mode, :],
            cmap="coolwarm",
            s=12,
            edgecolor="none",
        )
        ax.plot(coastLon, coastLat, "k", lw=0.5)
        ax.set_xlim([min(sd_lon), max(sd_lon)])
        ax.set_ylim([min(sd_lat), max(sd_lat)])
        ax.set_title(rf"GMB Eigenvector Mode \#{mode + 1}")
        plt.colorbar(sc, ax=ax, orientation="vertical", label="GMB eigenvector")

        ax = axes[mi, 2] if n_modes > 1 else axes[0, 2]
        ax.plot(train_time, U[:, mode], label="U (SD)", lw=1.5, color="b")
        ax.plot(train_time, V[:, mode], label="V (GMB)", lw=1.5, color="g")
        ax.set_title(rf"Canonical Variates Mode \#{mode + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Score")
        ax.set_xlim([train_time[0], train_time[-1]])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    # optional saving: default filename indicates number of modes
    if save:
        fname = filename or f"cca_modes_{n_modes}modes"
        save_figure_and_axes(fig, axes, fname)
    # Caller may pass filename by wrapping this function or by saving plt.gcf() externally
    plt.show()


def compare_pred_test_glacier(rgi_id):
    gmb_test_path = PROCESSED_DATA_DIR / "gmb_data" / "test" / "data.nc"
    gmb_test_data = xr.open_dataset(gmb_test_path)
    gmb_test_ds = get_gmb_rgiid(gmb_test_data, rgi_id)
    gmb_monthly_mean = gmb_test_ds["monthly_mean"]
    gmb_test_anomaly = remove_anomaly(
        gmb_test_ds["monthly_gmb"],
        monthly_mean=gmb_monthly_mean,
    )
    gmb_test = gmb_test_anomaly.values
    test_time = gmb_test_ds["time"].values

    gmb_pred_path = PREDICTED_DATA_DIR / "gmb_data" / "test" / "gmb_cca_predictions.nc"
    gmb_pred_data = xr.open_dataset(gmb_pred_path)
    gmb_pred = get_gmb_rgiid(gmb_pred_data, rgi_id)["monthly_gmb"].values

    plt.figure(figsize=(6, 6))
    plt.plot(
        test_time,
        gmb_test,
        label="GMB Test Original",
        color="green",
    )
    plt.plot(
        test_time,
        gmb_pred,
        label="GMB Test Reconstructed from SD",
        color="red",
    )
    plt.xlabel("Time")
    plt.ylabel("GMB")
    plt.title("Original vs Reconstructed GMB Time Series for Sample Glacier")
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_pred_test_month(month: str, region_id: Optional[Union[str, int]] = None):
    bc_regions = load_regions()
    coastline = load_coastline()

    coastLon = coastline.lon
    coastLat = coastline.lat

    data_path = INTERIM_DATA_DIR / "gmb_data" / "monthly_gmb_clean.nc"
    ds = xr.open_dataset(data_path)
    if region_id is not None:
        ds = get_gmb_region(ds, region_id)
    month_np = np.datetime64(month)
    matches = np.where(ds["time"].values.astype("datetime64[M]") == month_np)[0]
    if matches.size == 0:
        raise ValueError(f"Month {month_np} not found in dataset time coordinate.")
    month_data_test = ds["monthly_gmb"].isel(time=matches[0])

    ds_pred = xr.open_dataset(
        PREDICTED_DATA_DIR / "gmb_data" / "test" / "gmb_cca_predictions.nc"
    )
    if region_id is not None:
        ds_pred = get_gmb_region(ds_pred, region_id)
    time_vals_pred = ds_pred["time"].values.astype("datetime64[M]")
    matches_pred = np.where(time_vals_pred == month_np)[0]
    if matches_pred.size == 0:
        raise ValueError(
            f"Month {month_np} not in predictions dataset (time range {time_vals_pred[0]} to {time_vals_pred[-1]})."
        )
    month_data_pred = ds_pred["monthly_gmb"].isel(time=matches_pred[0])

    gmb_lat = ds["lat"].values
    gmb_lon = ds["lon"].values

    xlim = (min(gmb_lon) - 0.1, max(gmb_lon) + 0.1)
    ylim = (min(gmb_lat) - 0.1, max(gmb_lat) + 0.1)

    month_data_diff = month_data_test - month_data_pred

    fig, axes = plt.subplots(1, 3, figsize=(3 * 12, 6))
    sc = axes[0].scatter(
        gmb_lon,
        gmb_lat,
        c=month_data_test,
        cmap="coolwarm",
        s=12,
        edgecolor="none",
        vmin=None,
        vmax=None,
    )
    axes[0].plot(coastLon, coastLat, "k", lw=0.5)
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    plt.colorbar(sc, ax=axes[0], orientation="vertical", label="GMB Value")
    axes[0].set_title(f"Original GMB Test Month {month}")
    sc = axes[1].scatter(
        gmb_lon,
        gmb_lat,
        c=month_data_pred,
        cmap="coolwarm",
        s=12,
        edgecolor="none",
        vmin=None,
        vmax=None,
    )
    axes[1].plot(coastLon, coastLat, "k", lw=0.5)
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    plt.colorbar(sc, ax=axes[1], orientation="vertical", label="GMB Value")
    axes[1].set_title(f"Reconstructed GMB Test Month {month}")
    sc = axes[2].scatter(
        gmb_lon, gmb_lat, c=month_data_diff, cmap="PiYG", s=12, edgecolor="none"
    )
    axes[2].plot(coastLon, coastLat, "k", lw=0.5)
    axes[2].set_xlim(xlim)
    axes[2].set_ylim(ylim)
    plt.colorbar(sc, ax=axes[2], orientation="vertical", label="GMB Error")
    axes[2].set_title(f"Error Map GMB Test Month {month}")
    vmin = min(np.min(month_data_test), np.min(month_data_pred))
    vmax = max(np.max(month_data_test), np.max(month_data_pred))
    axes[0].collections[0].set_clim(vmin, vmax)
    axes[1].collections[0].set_clim(vmin, vmax)
    axes[2].collections[0].set_clim(
        -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
    )
    plt.show()


@app.command()
def main():
    # plot_glacier_regions()

    # plot_sd("2016-01", raw=True, region_id="NCM")
    # plot_gmb("1995-10", raw=True, region_id="NCM")

    # plot_sd_anomaly(lat=53.0, lon=-125.0, start_month="1980-01", end_month="2020-12")
    # plot_gmb_anomaly(rgi_id="RGI60-02.00006", start_month="1980-01", end_month="2020-12")

    # compare_pred_test_glacier("RGI60-02.00007")
    compare_pred_test_month("2019-08")


if __name__ == "__main__":
    app()
