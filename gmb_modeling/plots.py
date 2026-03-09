from pathlib import Path
from typing import Optional, Union

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import typer
import xarray as xr

from gmb_modeling.dataset import (
    get_sd_gridpoint,
    load_coastline,
    load_regions,
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


def full_extent(
    ax: Union[Axes, list[Axes]], padx: float = 0.0, pady: float = 0.0
) -> Bbox:
    """Get the full extent of an axes, including axes labels, tick labels, and titles

    :param ax: The axes to get the extent of
    :type ax: matplotlib.axes.Axes or list of matplotlib.axes.Axes
    :param padx: Amount to pad the extent by in the x-direction, as a fraction of the original size (e.g. 0.1 adds 10% padding), defaults to 0.0
    :type padx: float, optional
    :param pady: Amount to pad the extent by in the y-direction, as a fraction of the original size (e.g. 0.1 adds 10% padding), defaults to 0.0
    :type pady: float, optional
    :return: The bounding box of the full extent
    :rtype: matplotlib.transforms.Bbox
    """
    if isinstance(ax, list):
        first_ax = ax[0]
        fig = first_ax.figure
        fig.canvas.draw()
        items = []
        for a in ax:
            items.extend(a.get_xticklabels() + a.get_yticklabels())
            items += [a, a.title]
    else:
        fig = ax.figure
        fig.canvas.draw()
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + padx, 1.0 + pady)


def plot_glacier_regions(save_path: Optional[Path] = None) -> None:
    """Plot the glacier regions of British Columbia with different colors

    :param save_path: Path to save the plot
    :type save_path: Optional[Path]
    """
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
    if save_path is not None:
        save_name = save_path / "glacier_regions.png"
        fig.savefig(save_name, bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_sd(
    data_path: Path,
    month: str,
    region_ids: Optional[list] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot snow depth in specified regions for a specified month

    :param data_path: Path to processed SD dataset containing "SNODP" variable with dimensions [time, lat, lon] and coordinates "time", "lat", and "lon"
    :type data_path: Path
    :param month: Month to plot, can be in any format recognized by pandas (e.g. "2020-01", "Jan 2020", "January 2020", etc.)
    :type month: str
    :param region_ids: List of region IDs to plot, defaults to None (plots all regions)
    :param save_path: Path to save the plot
    :type save_path: Optional[Path]
    :type region_ids: Optional[list], optional
    :raises ValueError: If the specified month is not found in the dataset
    """
    # load BC regions and coastline for plotting
    bc_regions = load_regions()
    coastline = load_coastline()
    coastLon = coastline.lon
    coastLat = coastline.lat

    # open dataset and get specified month
    with xr.open_dataset(data_path) as _ds:
        ds = _ds.load()
    month_np = np.datetime64(pd.to_datetime(month).strftime("%Y-%m"))
    time_vals = ds["time"].values.astype("datetime64[M]")
    matches = np.where(time_vals == month_np)[0]
    if matches.size == 0:
        raise ValueError(f"Month {month} not found in dataset time coordinate.")
    month_data = ds["SNODP"].isel(time=matches[0])
    month_str = pd.to_datetime(month).strftime("%Y-%b")
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
    if region_ids is None:
        min_lat, min_lon, max_lat, max_lon = (
            min(sd_lat),
            min(sd_lon),
            max(sd_lat),
            max(sd_lon),
        )
    else:
        min_lat, min_lon, max_lat, max_lon = (np.inf, np.inf, -np.inf, -np.inf)
    for region_id in bc_regions["name"].values:
        if region_ids is not None and region_id not in region_ids:
            continue
        region = bc_regions[bc_regions["name"] == region_id]
        region.boundary.plot(
            ax=ax, color=region["color"].values[0], linewidth=1, zorder=2
        )
        minx, miny, maxx, maxy = region.total_bounds
        min_lat = min(min_lat, miny)
        min_lon = min(min_lon, minx)
        max_lat = max(max_lat, maxy)
        max_lon = max(max_lon, maxx)
    ax.plot(coastLon, coastLat, "k", lw=1)
    ax.set_xlim((min_lon - 0.25, max_lon + 0.25))
    ax.set_ylim((min_lat - 0.25, max_lat + 0.25))
    fig.colorbar(pcm, ax=ax, label="Snow Depth (m)")
    ax.set_title(f"MERRA-2 Snow Depth for month {month_str}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if save_path is not None:
        save_name = save_path / f"sd_{month_str}.png"
        fig.savefig(save_name, bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_gmb(
    data_path: Path,
    month: str,
    region_ids: Optional[list] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot GMB for specified regions for specified month

    :param data_path: Path to processed GMB dataset containing "monthly_gmb" variable with dimensions [time, rgi_id] and coordinates "time", "rgi_id", "lat", and "lon"
    :type data_path: Path
    :param month: Month to plot, can be in any format recognized by pandas (e.g. "2020-01", "Jan 2020", "January 2020", etc.)
    :type month: str
    :param region_ids: List of region IDs to plot, defaults to None (plots all regions)
    :type region_ids: Optional[list], optional
    :param save_path: Path to save the plot
    :type save_path: Optional[Path]
    :raises ValueError: If the specified month is not found in the dataset
    """
    # load BC regions and coastline for plotting
    bc_regions = load_regions()
    coastline = load_coastline()
    coastLon = coastline.lon
    coastLat = coastline.lat

    # open dataset and get specified month
    with xr.open_dataset(data_path) as _ds:
        ds = _ds.load()
    month_np = np.datetime64(pd.to_datetime(month).strftime("%Y-%m"))
    time_vals = ds["time"].values.astype("datetime64[M]")
    matches = np.where(time_vals == month_np)[0]
    if matches.size == 0:
        raise ValueError(f"Month {month} not found in dataset time coordinate.")
    month_data = ds["monthly_gmb"].isel(time=matches[0])
    month_str = pd.to_datetime(month).strftime("%Y-%b")
    gmb_lat = ds["lat"].values
    gmb_lon = ds["lon"].values

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(coastLon, coastLat, "k", lw=1)
    if region_ids is None:
        min_lat, min_lon, max_lat, max_lon = (
            min(gmb_lat),
            min(gmb_lon),
            max(gmb_lat),
            max(gmb_lon),
        )
    else:
        min_lat, min_lon, max_lat, max_lon = (np.inf, np.inf, -np.inf, -np.inf)
    for region_id in bc_regions["name"].values:
        if region_ids is not None and region_id not in region_ids:
            continue
        region = bc_regions[bc_regions["name"] == region_id]
        region.boundary.plot(
            ax=ax, color=region["color"].values[0], linewidth=1, zorder=2
        )
        # Update bounds for zooming
        minx, miny, maxx, maxy = region.total_bounds
        min_lat = min(min_lat, miny)
        min_lon = min(min_lon, minx)
        max_lat = max(max_lat, maxy)
        max_lon = max(max_lon, maxx)
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
    ax.set_xlim((min_lon - 0.1, max_lon + 0.1))
    ax.set_ylim((min_lat - 0.1, max_lat + 0.1))
    fig.colorbar(pcm, ax=ax, label="GMB (m.w.e.)")
    ax.set_title(f"GMB for month {month_str}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if save_path is not None:
        save_name = save_path / f"gmb_{month_str}.png"
        fig.savefig(save_name, bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_sd_anomaly(
    data_path: Path,
    lat: float,
    lon: float,
    save_path: Optional[Path] = None,
) -> None:
    """Plot the original, monthly mean, and anomaly time series for a snow depth gridpoint

    :param data_path: Path to the processed snow depth data
    :type data_path: Path
    :param lat: Latitude of the gridpoint to plot
    :type lat: float
    :param lon: Longitude of the gridpoint to plot
    :type lon: float
    :param save_path: Path to save the plot
    :type save_path: Optional[Path]
    """
    # open dataset and subset to specified gridpoint and time slice
    with xr.open_dataset(data_path) as _ds_processed:
        ds_processed = _ds_processed.load()
    ds_processed = get_sd_gridpoint(ds_processed, lat, lon)
    months = ds_processed["time"].values.astype("datetime64[M]")

    latlon_str = f"[{lat:.2f}, {lon:.2f}]"

    fig, ax = plt.subplots(4, 1, figsize=(12, 12))

    ax[0].plot(months, ds_processed["monthly_raw"], color="blue")
    ax[0].set_xlim((months[0], months[-1]))
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Snow Depth [m]")
    ax[0].set_title(f"Raw Value at {latlon_str}")
    ax[0].grid(True, axis="y")

    ax[1].plot(MONTHS, ds_processed["monthly_mean"], color="blue")
    ax[1].set_xlim((0, 11))
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Snow Depth [m]")
    ax[1].set_title(f"Seasonal Value at {latlon_str}")
    ax[1].grid(True, axis="y")

    ax[2].plot(months, ds_processed["anomaly"], color="blue")
    ax[2].set_xlim((months[0], months[-1]))
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Snow Depth [m]")
    ax[2].set_title(f"Anomaly at {latlon_str}")
    ax[2].grid(True, axis="y")

    ax[3].plot(months, ds_processed["SNODP"], color="blue")
    ax[3].set_xlim((months[0], months[-1]))
    ax[3].set_xlabel("Time")
    ax[3].set_ylabel("Snow Depth [m]")
    ax[3].set_title(f"Smoothed Anomaly at {latlon_str}")
    ax[3].grid(True, axis="y")

    plt.tight_layout()
    if save_path is not None:
        fname = save_path / f"sd_anomaly_{lat:.2f}_{lon:.2f}.png"
        fig.savefig(fname, bbox_inches="tight", dpi=500)

        # save axes
        extent0 = full_extent(ax[0], padx=0.1, pady=0.1).transformed(
            fig.dpi_scale_trans.inverted()
        )
        fname0 = save_path / f"sd_anomaly_{lat:.2f}_{lon:.2f}_raw.png"
        fig.savefig(fname0, bbox_inches=extent0, dpi=500)
        extent1 = full_extent(ax[1], padx=0.1, pady=0.1).transformed(
            fig.dpi_scale_trans.inverted()
        )
        fname1 = save_path / f"sd_anomaly_{lat:.2f}_{lon:.2f}_seasonal.png"
        fig.savefig(fname1, bbox_inches=extent1, dpi=500)
        extent2 = full_extent(ax[2], padx=0.1, pady=0.1).transformed(
            fig.dpi_scale_trans.inverted()
        )
        fname2 = save_path / f"sd_anomaly_{lat:.2f}_{lon:.2f}_anomaly.png"
        fig.savefig(fname2, bbox_inches=extent2, dpi=500)
        extent3 = full_extent(ax[3], padx=0.1, pady=0.1).transformed(
            fig.dpi_scale_trans.inverted()
        )
        fname3 = save_path / f"sd_anomaly_{lat:.2f}_{lon:.2f}_smoothed.png"
        fig.savefig(fname3, bbox_inches=extent3, dpi=500)
    plt.close(fig)


def plot_gmb_anomaly(
    data_path: Path,
    rgi_id: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot the original, monthly mean, and anomaly time series for a glacier

    :param data_path: Path to the processed glacier mass balance data
    :type data_path: Path
    :param rgi_id: RGI ID of the glacier to plot
    :type rgi_id: str
    :param save_path: Path to save the plot, defaults to None
    :type save_path: Optional[Path], optional
    """
    # open dataset and subset to specified glacier and time slice
    with xr.open_dataset(data_path) as _ds_processed:
        ds_processed = _ds_processed.load()
    ds_processed = ds_processed.sel(rgi_id=rgi_id)
    months = ds_processed["time"].values.astype("datetime64[M]")

    lat = ds_processed["lat"].values.item()
    lon = ds_processed["lon"].values.item()
    latlon_str = f"[{lat:.2f}, {lon:.2f}]"

    fig, ax = plt.subplots(4, 1, figsize=(12, 12))

    ax[0].plot(months, ds_processed["monthly_raw"], color="green")
    ax[0].set_xlim((months[0], months[-1]))
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("GMB [m.w.e.]")
    ax[0].set_title(f"Raw GMB for Glacier {rgi_id} at {latlon_str}")
    ax[0].grid(True, axis="y")

    ax[1].plot(MONTHS, ds_processed["monthly_mean"], color="green")
    ax[1].set_xlim((0, 11))
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("GMB [m.w.e.]")
    ax[1].set_title(f"Seasonal GMB for Glacier {rgi_id} at {latlon_str}")
    ax[1].grid(True, axis="y")

    ax[2].plot(months, ds_processed["anomaly"], color="green")
    ax[2].set_xlim((months[0], months[-1]))
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("GMB [m.w.e.]")
    ax[2].set_title(f"GMB Anomaly for Glacier {rgi_id} at {latlon_str}")
    ax[2].grid(True, axis="y")

    ax[3].plot(months, ds_processed["monthly_gmb"], color="green")
    ax[3].set_xlim((months[0], months[-1]))
    ax[3].set_xlabel("Time")
    ax[3].set_ylabel("GMB [m.w.e.]")
    ax[3].set_title(f"Smoothed GMB Anomaly for Glacier {rgi_id} at {latlon_str}")
    ax[3].grid(True, axis="y")

    plt.tight_layout()
    if save_path is not None:
        fname = save_path / f"gmb_anomaly_{rgi_id}.png"
        fig.savefig(fname, bbox_inches="tight", dpi=500)

        # save axes
        extent0 = full_extent(ax[0], padx=0.0, pady=0.1).transformed(
            fig.dpi_scale_trans.inverted()
        )
        fname0 = save_path / f"gmb_anomaly_{rgi_id}_raw.png"
        fig.savefig(fname0, bbox_inches=extent0, dpi=500)
        extent1 = full_extent(ax[1], padx=0.1, pady=0.1).transformed(
            fig.dpi_scale_trans.inverted()
        )
        fname1 = save_path / f"gmb_anomaly_{rgi_id}_seasonal.png"
        fig.savefig(fname1, bbox_inches=extent1, dpi=500)
        extent2 = full_extent(ax[2], padx=0.1, pady=0.1).transformed(
            fig.dpi_scale_trans.inverted()
        )
        fname2 = save_path / f"gmb_anomaly_{rgi_id}_anomaly.png"
        fig.savefig(fname2, bbox_inches=extent2, dpi=500)
        extent3 = full_extent(ax[3], padx=0.1, pady=0.1).transformed(
            fig.dpi_scale_trans.inverted()
        )
        fname3 = save_path / f"gmb_anomaly_{rgi_id}_smoothed.png"
        fig.savefig(fname3, bbox_inches=extent3, dpi=500)
    plt.close(fig)


def plot_pca_variance(
    explained_variance: np.ndarray,
    n_modes: Optional[int] = None,
    color="blue",
    save_path: Optional[Path] = None,
) -> None:
    """Plot the variance explained by PCA modes

    :param explained_variance: Array of variance explained by each PCA mode, typically obtained from the PCA model's explained_variance_ratio_ attribute
    :type explained_variance: numpy.ndarray
    :param n_modes: Number of PCA modes to plot, defaults to None
    :type n_modes: Optional[int], optional
    :param color: Color for the scatter plot, defaults to "blue"
    :type color: str, optional
    :param save_path: Path to save the plot, defaults to None
    :type save_path: Optional[Path], optional
    """
    title = "Variance Explained by All Modes"
    if n_modes is not None:
        explained_variance = explained_variance[:n_modes]
        title = f"Variance Explained by First {n_modes} Modes"

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(range(len(explained_variance)), explained_variance, color=color)
    ax.set_xlabel("Mode Number")
    ax.set_ylabel("Fraction Variance Explained")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    if save_path is not None:
        fname = save_path / f"pca_variance_{n_modes}modes.png"
        plt.savefig(fname, bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_pca_modes_gmb(
    eigvecs: np.ndarray,
    PCs: np.ndarray,
    gmb_lon: np.ndarray,
    gmb_lat: np.ndarray,
    time: np.ndarray,
    n_modes: Optional[int] = None,
    region_ids: Optional[list] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot the spatial eigenvectors and temporal principal components for GMB from PCA

    :param eigvecs: Spatial eigenvectors [n_modes, n_glaciers]
    :type eigvecs: numpy.ndarray
    :param PCs: Temporal principal components [time, n_modes]
    :type PCs: numpy.ndarray
    :param gmb_lon: Longitude coordinates for GMB data
    :type gmb_lon: numpy.ndarray
    :param gmb_lat: Latitude coordinates for GMB data
    :type gmb_lat: numpy.ndarray
    :param time: Time coordinates for the principal components
    :type time: numpy.ndarray
    :param n_modes: Number of principal modes to plot, defaults to None
    :type n_modes: Optional[int], optional
    :param region_ids: List of region IDs to include in the plot, defaults to None
    :type region_ids: Optional[list], optional
    :param save_path: Path to save the figure, defaults to None
    :type save_path: Optional[Path], optional
    """
    if n_modes is None:
        n_modes = int(eigvecs.shape[0])

    bc_regions = load_regions()
    coastline = load_coastline()
    coastLon = coastline.lon
    coastLat = coastline.lat

    fig, axes = plt.subplots(
        n_modes,
        2,
        figsize=(12, 4 * n_modes),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    min_lat, min_lon, max_lat, max_lon = (np.inf, np.inf, -np.inf, -np.inf)
    regions = []
    for region_id in bc_regions["name"].values:
        if region_ids is not None and region_id not in region_ids:
            continue
        region = bc_regions[bc_regions["name"] == region_id]
        regions.append(region)
        # Update bounds for zooming
        minx, miny, maxx, maxy = region.total_bounds
        min_lat = min(min_lat, miny)
        min_lon = min(min_lon, minx)
        max_lat = max(max_lat, maxy)
        max_lon = max(max_lon, maxx)

    xlim = (min_lon - 0.1, max_lon + 0.1)
    ylim = (min_lat - 0.1, max_lat + 0.1)

    for i in range(n_modes):
        # Plot spatial pattern (eigenvector)
        ax1 = axes[i, 0] if n_modes > 1 else axes[0, 0]
        sc = ax1.scatter(gmb_lon, gmb_lat, c=eigvecs[i], cmap="coolwarm", s=20)
        ax1.plot(coastLon, coastLat, "k", lw=1)
        for region in regions:
            region.boundary.plot(
                ax=ax1, color=region["color"].values[0], linewidth=1, zorder=2
            )
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_title(rf"Eigenvector Mode \#{i + 1}")
        plt.colorbar(sc, ax=ax1, orientation="vertical", label="Eig. Value")
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        # Plot Principal Component (time series)
        ax2 = axes[i, 1] if n_modes > 1 else axes[0, 1]
        # Prepare compact time-position mapping: equal spacing inside consecutive
        # month segments, with a small fixed gap between segments so points are
        # not connected across missing-month gaps but horizontal space is compact.
        times_pd = pd.to_datetime(time)
        months_num = times_pd.year * 12 + times_pd.month

        # identify consecutive-month segments (list of (start, end) pairs)
        segs = []
        start = 0
        for j in range(1, len(months_num)):
            if months_num[j] - months_num[j - 1] > 1:
                segs.append((start, j))
                start = j
        segs.append((start, len(months_num)))

        # Build compact positions: unit spacing within segments, small fixed gap between segments
        inter_gap = 0.5
        positions = np.empty(len(times_pd), dtype=float)
        cur = 0.0
        for s, e in segs:
            length = e - s
            if length <= 0:
                continue
            for k in range(length):
                positions[s + k] = cur + k
            cur += length + inter_gap

        # Plot each consecutive segment separately so lines don't connect across gaps
        for s, e in segs:
            seg_pos = positions[s:e]
            seg_pc = PCs[s:e, i]
            if len(seg_pos) > 1:
                ax2.plot(seg_pos, seg_pc, "-", color="green")
            else:
                # single-point segment: draw a short horizontal tick (no markers)
                eps = 0.15
                ax2.plot(
                    [seg_pos[0] - eps, seg_pos[0] + eps],
                    [seg_pc[0], seg_pc[0]],
                    "-",
                    color="green",
                )

        # xtick formatting: month above, year below, limit density
        max_ticks = 10
        step_tick = max(1, int(np.ceil(len(times_pd) / max_ticks)))
        tick_idx = list(range(0, len(times_pd), step_tick))
        tick_pos = positions[tick_idx]
        tick_labels = [
            f"{times_pd[idx].strftime('%b')}\n{times_pd[idx].strftime('%Y')}"
            for idx in tick_idx
        ]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels, rotation=0, ha="center")
        pad = (
            max(0.5, 0.05 * (positions.max() - positions.min()))
            if positions.max() > positions.min()
            else 0.5
        )
        ax2.set_xlim(positions[0] - pad, positions[-1] + pad)
        ax2.set_xlabel("Month")
        ax2.set_ylabel("PC Value")
        ax2.set_title(rf"PC Time Series Mode \#{i + 1}")
        ax2.grid(True)

    plt.tight_layout()
    if save_path is not None:
        fname = save_path / f"pca_gmb_{n_modes}modes.png"
        plt.savefig(fname, bbox_inches="tight", dpi=500)

        # save modes separately
        for i in range(n_modes):
            extent = full_extent(list(axes[i, :]), padx=0.1, pady=0.1).transformed(
                fig.dpi_scale_trans.inverted()
            )
            fname_mode = save_path / f"pca_gmb_mode_{i + 1}.png"
            plt.savefig(fname_mode, bbox_inches=extent, dpi=500)
    plt.close(fig)


def plot_pca_modes_sd(
    eigvecs: np.ndarray,
    PCs: np.ndarray,
    sd_lon: np.ndarray,
    sd_lat: np.ndarray,
    time: np.ndarray,
    n_modes: Optional[int] = None,
    region_ids: Optional[list] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot the spatial eigenvectors and temporal principal components for snow depth from PCA

    :param eigvecs: Spatial eigenvectors [n_modes, n_gridpoints]
    :type eigvecs: numpy.ndarray
    :param PCs: Principal components [time, n_modes]
    :type PCs: numpy.ndarray
    :param sd_lon: Longitude coordinates for snow depth data
    :type sd_lon: numpy.ndarray
    :param sd_lat: Latitude coordinates for snow depth data
    :type sd_lat: numpy.ndarray
    :param time: Time coordinates for the principal components
    :type time: numpy.ndarray
    :param n_modes: Number of modes to plot, defaults to None
    :type n_modes: Optional[int], optional
    :param region_ids: List of region IDs to include, defaults to None
    :type region_ids: Optional[list], optional
    :param save_path: Path to save the figure, defaults to None
    :type save_path: Optional[Path], optional
    """
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

    min_lat, min_lon, max_lat, max_lon = (np.inf, np.inf, -np.inf, -np.inf)
    regions = []
    for region_id in bc_regions["name"].values:
        if region_ids is not None and region_id not in region_ids:
            continue
        region = bc_regions[bc_regions["name"] == region_id]
        regions.append(region)
        # Update bounds for zooming
        minx, miny, maxx, maxy = region.total_bounds
        min_lat = min(min_lat, miny)
        min_lon = min(min_lon, minx)
        max_lat = max(max_lat, maxy)
        max_lon = max(max_lon, maxx)

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
        for region in regions:
            region.boundary.plot(
                ax=ax1, color=region["color"].values[0], linewidth=1, zorder=2
            )
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_title(rf"Spatial Pattern (Eigenvector) Mode \#{i + 1}")
        plt.colorbar(pcm, ax=ax1, orientation="vertical", label="Eigenvector Value")
        ax1.set_xlim((min(sd_lon) - 0.1, max(sd_lon) + 0.1))
        ax1.set_ylim((min(sd_lat) - 0.1, max(sd_lat) + 0.1))

        # Plot Principal Component (time series) with compact spacing and
        # no markers: equal spacing inside consecutive-month segments,
        # small fixed gap between segments so segments are not connected.
        ax2 = axes[i, 1] if n_modes > 1 else axes[0, 1]
        times_pd = pd.to_datetime(time)
        months_num = times_pd.year * 12 + times_pd.month
        segs = []
        start = 0
        for j in range(1, len(months_num)):
            if months_num[j] - months_num[j - 1] > 1:
                segs.append((start, j))
                start = j
        segs.append((start, len(months_num)))

        inter_gap = 0.5
        positions = np.empty(len(times_pd), dtype=float)
        cur = 0.0
        for s, e in segs:
            length = e - s
            if length <= 0:
                continue
            for k in range(length):
                positions[s + k] = cur + k
            cur += length + inter_gap

        for s, e in segs:
            seg_pos = positions[s:e]
            seg_pc = PCs[s:e, i]
            if len(seg_pos) > 1:
                ax2.plot(seg_pos, seg_pc, "-", color="blue")
            else:
                eps = 0.15
                ax2.plot(
                    [seg_pos[0] - eps, seg_pos[0] + eps],
                    [seg_pc[0], seg_pc[0]],
                    "-",
                    color="blue",
                )

        max_ticks = 10
        step_tick = max(1, int(np.ceil(len(times_pd) / max_ticks)))
        tick_idx = list(range(0, len(times_pd), step_tick))
        tick_pos = positions[tick_idx]
        tick_labels = [
            f"{times_pd[idx].strftime('%b')}\n{times_pd[idx].strftime('%Y')}"
            for idx in tick_idx
        ]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels, rotation=0, ha="center")
        pad = (
            max(0.5, 0.05 * (positions.max() - positions.min()))
            if positions.max() > positions.min()
            else 0.5
        )
        ax2.set_xlim(positions[0] - pad, positions[-1] + pad)
        ax2.set_xlabel("Month")
        ax2.set_ylabel("PC Value")
        ax2.set_title(rf"PC Time Series Mode \#{i + 1}")
        ax2.grid(True)

    plt.tight_layout()
    if save_path:
        fname = save_path / f"pca_sd_{n_modes}modes.png"
        plt.savefig(fname, bbox_inches="tight", dpi=500)

        # save modes separately
        for i in range(n_modes):
            extent = full_extent(list(axes[i, :]), padx=0.1, pady=0.1).transformed(
                fig.dpi_scale_trans.inverted()
            )
            fname_mode = save_path / f"pca_sd_mode_{i + 1}.png"
            plt.savefig(fname_mode, bbox_inches=extent, dpi=500)
    plt.close(fig)


def plot_cca_modes(
    U: np.ndarray,
    V: np.ndarray,
    sd_eigvecs: np.ndarray,
    gmb_eigvecs: np.ndarray,
    sd_lon: np.ndarray,
    sd_lat: np.ndarray,
    gmb_lon: np.ndarray,
    gmb_lat: np.ndarray,
    time: np.ndarray,
    region_ids: Optional[list] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot the spatial eigenvectors of GMB and snow depth, as well as the temporal canonical variates from CCA

    :param U: Canonical variates for the snow depth dataset [time, n_modes]
    :type U: numpy.ndarray
    :param V: Canonical variates for the GMB dataset [time, n_modes]
    :type V: numpy.ndarray
    :param sd_eigvecs: Spatial eigenvectors for the snow depth dataset [n_modes, n_gridpoints]
    :type sd_eigvecs: numpy.ndarray
    :param gmb_eigvecs: Spatial eigenvectors for the GMB dataset [n_modes, n_glaciers]
    :type gmb_eigvecs: numpy.ndarray
    :param sd_lon: Longitudes for the snow depth dataset
    :type sd_lon: numpy.ndarray
    :param sd_lat: Latitudes for the snow depth dataset
    :type sd_lat: numpy.ndarray
    :param gmb_lon: Longitudes for the GMB dataset
    :type gmb_lon: numpy.ndarray
    :param gmb_lat: Latitudes for the GMB dataset
    :type gmb_lat: numpy.ndarray
    :param time: Time coordinates for the data
    :type time: numpy.ndarray
    :param region_ids: List of region IDs to include, defaults to None
    :type region_ids: Optional[list], optional
    :param save_path: Path to save the figure, defaults to None
    :type save_path: Optional[Path], optional
    """
    bc_regions = load_regions()
    coastline = load_coastline()
    coastLon = coastline.lon
    coastLat = coastline.lat

    n_modes = min(len(sd_eigvecs), len(gmb_eigvecs))

    fig, axes = plt.subplots(n_modes, 3, figsize=(18, 4 * n_modes))

    min_lat, min_lon, max_lat, max_lon = (np.inf, np.inf, -np.inf, -np.inf)
    regions = []
    for region_id in bc_regions["name"].values:
        if region_ids is not None and region_id not in region_ids:
            continue
        region = bc_regions[bc_regions["name"] == region_id]
        regions.append(region)
        # Update bounds for zooming
        minx, miny, maxx, maxy = region.total_bounds
        min_lat = min(min_lat, miny)
        min_lon = min(min_lon, minx)
        max_lat = max(max_lat, maxy)
        max_lon = max(max_lon, maxx)
    xlim = (min_lon - 0.1, max_lon + 0.1)
    ylim = (min_lat - 0.1, max_lat + 0.1)

    for mi, mode in enumerate(range(n_modes)):
        eig_map = sd_eigvecs[mode, :]
        eig_map2d = eig_map.reshape((len(sd_lat), len(sd_lon)))
        ax1 = axes[mi, 0] if n_modes > 1 else axes[0, 0]
        pcm = ax1.pcolormesh(
            sd_lon,
            sd_lat,
            eig_map2d,
            cmap="coolwarm",
            shading="auto",
        )
        ax1.plot(coastLon, coastLat, "k", lw=0.5)
        for region in regions:
            region.boundary.plot(
                ax=ax1, color=region["color"].values[0], linewidth=1, zorder=2
            )
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_title(rf"SD Eigenvector Mode \#{mode + 1}")
        plt.colorbar(pcm, ax=ax1, orientation="vertical", label="SD eigenvector")

        ax2 = axes[mi, 1] if n_modes > 1 else axes[0, 1]
        sc = ax2.scatter(
            gmb_lon,
            gmb_lat,
            c=gmb_eigvecs[mode, :],
            cmap="coolwarm",
            s=12,
            edgecolor="none",
        )
        ax2.plot(coastLon, coastLat, "k", lw=0.5)
        for region in regions:
            region.boundary.plot(
                ax=ax2, color=region["color"].values[0], linewidth=1, zorder=2
            )
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_title(rf"GMB Eigenvector Mode \#{mode + 1}")
        plt.colorbar(sc, ax=ax2, orientation="vertical", label="GMB eigenvector")

        ax3 = axes[mi, 2] if n_modes > 1 else axes[0, 2]
        # Compact spacing for canonical variates, line-only plotting.
        times_pd = pd.to_datetime(time)
        months_num = times_pd.year * 12 + times_pd.month
        segs = []
        start = 0
        for j in range(1, len(months_num)):
            if months_num[j] - months_num[j - 1] > 1:
                segs.append((start, j))
                start = j
        segs.append((start, len(months_num)))

        inter_gap = 0.5
        positions = np.empty(len(times_pd), dtype=float)
        cur = 0.0
        for s, e in segs:
            length = e - s
            if length <= 0:
                continue
            for k in range(length):
                positions[s + k] = cur + k
            cur += length + inter_gap

        # Plot U and V per segment without markers
        for s, e in segs:
            seg_pos = positions[s:e]
            seg_u = U[s:e, mode]
            seg_v = V[s:e, mode]
            if len(seg_pos) > 1:
                label_u = "U (SD)" if s == segs[0][0] else None
                label_v = "V (GMB)" if s == segs[0][0] else None
                ax3.plot(seg_pos, seg_u, "-", lw=1.5, color="b", label=label_u)
                ax3.plot(seg_pos, seg_v, "-", lw=1.5, color="g", label=label_v)
            else:
                eps = 0.15
                ax3.plot(
                    [seg_pos[0] - eps, seg_pos[0] + eps],
                    [seg_u[0], seg_u[0]],
                    "-",
                    color="b",
                )
                ax3.plot(
                    [seg_pos[0] - eps, seg_pos[0] + eps],
                    [seg_v[0], seg_v[0]],
                    "-",
                    color="g",
                )

        max_ticks = 10
        step_tick = max(1, int(np.ceil(len(times_pd) / max_ticks)))
        tick_idx = list(range(0, len(times_pd), step_tick))
        tick_pos = positions[tick_idx]
        tick_labels = [
            f"{times_pd[idx].strftime('%b')}\n{times_pd[idx].strftime('%Y')}"
            for idx in tick_idx
        ]
        ax3.set_xticks(tick_pos)
        ax3.set_xticklabels(tick_labels, rotation=0, ha="center")
        pad = (
            max(0.5, 0.05 * (positions.max() - positions.min()))
            if positions.max() > positions.min()
            else 0.5
        )
        ax3.set_xlim(positions[0] - pad, positions[-1] + pad)
        ax3.set_title(rf"Canonical Variates Mode \#{mode + 1}")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Score")
        ax3.legend()
        ax3.grid(True)

    plt.tight_layout()
    if save_path is not None:
        fname = save_path / f"cca_modes_{n_modes}modes.png"
        plt.savefig(fname, bbox_inches="tight", dpi=500)

        # save modes separately
        for i in range(n_modes):
            extent = full_extent(list(axes[i, :]), padx=0.1, pady=0.1).transformed(
                fig.dpi_scale_trans.inverted()
            )
            fname_mode = save_path / f"cca_mode_{i + 1}.png"
            plt.savefig(fname_mode, bbox_inches=extent, dpi=500)
    plt.close(fig)


def compare_pred_test_glacier(
    gmb_proc: xr.Dataset,
    gmb_pred: xr.Dataset,
    rgi_id: str,
    save_path: Optional[Path] = None,
) -> None:
    """Compare original vs. predicted GMB for a specified glacier across the test time period

    :param gmb_proc: Processed GMB data for the specified glacier
    :type gmb_proc: xarray.Dataset
    :param gmb_pred: Predicted GMB data for the specified glacier
    :type gmb_pred: xarray.Dataset
    :param rgi_id: RGI ID of the glacier
    :type rgi_id: str
    :param save_path: Path to save the plot, defaults to None
    :type save_path: Optional[Path], optional
    """
    # get same time slice from processed dataset
    test_time = gmb_pred["time"].values.astype("datetime64[M]")
    gmb_proc = gmb_proc.sel(time=test_time)

    # Build compact x-position mapping: equal spacing within consecutive-month
    # segments and a small fixed gap between segments so points remain evenly
    # spaced but segments are not connected.
    times = pd.to_datetime(test_time)
    months_num = times.year * 12 + times.month

    segs = []
    start = 0
    for j in range(1, len(months_num)):
        if months_num[j] - months_num[j - 1] > 1:
            segs.append((start, j))
            start = j
    segs.append((start, len(months_num)))

    inter_gap = 0.5
    positions = np.empty(len(times), dtype=float)
    cur = 0.0
    for s, e in segs:
        length = e - s
        if length <= 0:
            continue
        for k in range(length):
            positions[s + k] = cur + k
        cur += length + inter_gap

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each consecutive segment as a line (no markers). For single-point
    # segments draw a short horizontal tick so the value is visible.
    for si, (s, e) in enumerate(segs):
        seg_pos = positions[s:e]
        seg_proc = gmb_proc["monthly_gmb"].values[s:e]
        seg_pred = gmb_pred["monthly_gmb"].values[s:e]
        label_proc = "GMB Test Original" if si == 0 else None
        label_pred = "GMB Test Reconstructed from SD" if si == 0 else None
        if len(seg_pos) > 1:
            ax.plot(seg_pos, seg_proc, "-", color="black", label=label_proc)
            ax.plot(seg_pos, seg_pred, "-", color="green", label=label_pred)
        else:
            eps = 0.15
            ax.plot(
                [seg_pos[0] - eps, seg_pos[0] + eps],
                [seg_proc[0], seg_proc[0]],
                "-",
                color="black",
            )
            ax.plot(
                [seg_pos[0] - eps, seg_pos[0] + eps],
                [seg_pred[0], seg_pred[0]],
                "-",
                color="green",
            )

    # Mark jumps with a dashed vertical line and double-slash text
    try:
        y_max = max(
            np.nanmax(gmb_proc["monthly_gmb"].values),
            np.nanmax(gmb_pred["monthly_gmb"].values),
        )
    except Exception:
        y_max = ax.get_ylim()[1]
    for i in range(1, len(positions)):
        diff = int(months_num[i] - months_num[i - 1])
        if diff > 1:
            boundary = (positions[i] + positions[i - 1]) / 2.0
            ax.axvline(boundary, color="k", linestyle="--", linewidth=1)
            ax.text(boundary, y_max, "//", ha="center", va="top", fontsize=12)

    # Choose a subset of ticks (max ~10) to avoid crowding
    max_ticks = 10
    step = max(1, int(np.ceil(len(times) / max_ticks)))
    tick_idx = list(range(0, len(times), step))
    tick_pos = positions[tick_idx]
    tick_labels = [
        f"{times[i].strftime('%b')}\n{times[i].strftime('%Y')}" for i in tick_idx
    ]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    pad = max(0.5, 0.05 * (positions.max() - positions.min()))
    ax.set_xlim(positions[0] - pad, positions[-1] + pad)
    ax.set_xlabel("Time")
    ax.set_ylabel("GMB")
    ax.set_title(f"Original vs Reconstructed GMB Time Series for Glacier {rgi_id}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path is not None:
        fname = save_path / f"gmb_test_comparison_{rgi_id}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=500)
    plt.close(fig)


def compare_pred_test_month(
    gmb_proc: xr.Dataset,
    gmb_pred: xr.Dataset,
    month: str,
    region_ids: Optional[list] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Compare original vs. predicted GMB for a specified month across all glaciers

    :param gmb_proc: Processed GMB data for all glaciers
    :type gmb_proc: xarray.Dataset
    :param gmb_pred: Predicted GMB data for all glaciers
    :type gmb_pred: xarray.Dataset
    :param month: Month to compare, can be in any format recognized by pandas (e.g. "2020-01", "Jan 2020", "January 2020", etc.)
    :type month: str
    :param region_ids: List of region IDs to include, defaults to None
    :type region_ids: Optional[list], optional
    :param save_path: Path to save the plot, defaults to None
    :type save_path: Optional[Path], optional
    :raises ValueError: _description_
    :raises ValueError: _description_
    """
    bc_regions = load_regions()
    coastline = load_coastline()
    coastLon = coastline.lon
    coastLat = coastline.lat

    gmb_lat = gmb_pred["lat"].values
    gmb_lon = gmb_pred["lon"].values

    pred_rgi_ids = gmb_pred["rgi_id"].values
    gmb_proc = gmb_proc.sel(rgi_id=pred_rgi_ids)

    xlim = (min(gmb_lon) - 0.1, max(gmb_lon) + 0.1)
    ylim = (min(gmb_lat) - 0.1, max(gmb_lat) + 0.1)

    gmb_proc_data = gmb_proc["monthly_gmb"].values
    gmb_pred_data = gmb_pred["monthly_gmb"].values
    gmb_diff = gmb_proc_data - gmb_pred_data

    month_str = pd.to_datetime(month).strftime("%Y-%b")

    regions = []
    xmin, ymin, xmax, ymax = (np.inf, np.inf, -np.inf, -np.inf)
    for region_id in bc_regions["name"].values:
        if region_ids is not None and region_id not in region_ids:
            continue
        region = bc_regions[bc_regions["name"] == region_id]
        regions.append(region)
        # Update bounds for zooming
        minx, miny, maxx, maxy = region.total_bounds
        xmin = min(xmin, minx)
        ymin = min(ymin, miny)
        xmax = max(xmax, maxx)
        ymax = max(ymax, maxy)
    xlim = min(xlim[0], xmin - 0.1), max(xlim[1], xmax + 0.1)
    ylim = min(ylim[0], ymin - 0.1), max(ylim[1], ymax + 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(3 * 7, 6))
    sc = axes[0].scatter(
        gmb_lon,
        gmb_lat,
        c=gmb_proc_data,
        cmap="coolwarm",
        s=12,
        edgecolor="none",
        vmin=None,
        vmax=None,
    )
    axes[0].plot(coastLon, coastLat, "k", lw=0.5)
    for region in regions:
        region.boundary.plot(
            ax=axes[0], color=region["color"].values[0], linewidth=1, zorder=2
        )
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    plt.colorbar(sc, ax=axes[0], orientation="vertical", label="GMB Value")
    axes[0].set_title(f"Original GMB Test Month {month_str}")
    sc = axes[1].scatter(
        gmb_lon,
        gmb_lat,
        c=gmb_pred_data,
        cmap="coolwarm",
        s=12,
        edgecolor="none",
        vmin=None,
        vmax=None,
    )
    axes[1].plot(coastLon, coastLat, "k", lw=0.5)
    for region in regions:
        region.boundary.plot(
            ax=axes[1], color=region["color"].values[0], linewidth=1, zorder=2
        )
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    plt.colorbar(sc, ax=axes[1], orientation="vertical", label="GMB Value")
    axes[1].set_title(f"Reconstructed GMB Test Month {month_str}")
    sc = axes[2].scatter(
        gmb_lon, gmb_lat, c=gmb_diff, cmap="PiYG", s=12, edgecolor="none"
    )
    axes[2].plot(coastLon, coastLat, "k", lw=0.5)
    for region in regions:
        region.boundary.plot(
            ax=axes[2], color=region["color"].values[0], linewidth=1, zorder=2
        )
    axes[2].set_xlim(xlim)
    axes[2].set_ylim(ylim)
    plt.colorbar(sc, ax=axes[2], orientation="vertical", label="GMB Error")
    axes[2].set_title(f"Error Map GMB Test Month {month_str}")
    vmin = min(np.min(gmb_proc_data), np.min(gmb_pred_data))
    vmax = max(np.max(gmb_proc_data), np.max(gmb_pred_data))
    axes[0].collections[0].set_clim(vmin, vmax)
    axes[1].collections[0].set_clim(vmin, vmax)
    axes[2].collections[0].set_clim(
        -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
    )
    plt.tight_layout()
    if save_path is not None:
        fname = save_path / f"gmb_test_comparison_{month_str}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_monthly_error(
    months: list, rmse: list, mae: list, save_path: Optional[Path] = None
):
    """Plot the cumulative error for each month

    :param months: List of month numbers (1-12) corresponding to the error values
    :type months: list
    :param rmse: List of RMSE values for each month
    :type rmse: list
    :param mae: List of MAE values for each month
    :type mae: list
    :param save_path: Path to save the plot, defaults to None
    :type save_path: Optional[Path], optional
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(
        [i for i in range(1, len(months) + 1)],
        rmse,
        yerr=np.std(rmse),
        linestyle="none",
        label="RMSE",
        color="blue",
        capsize=5,
        marker="o",
    )
    ax.set_xticks(range(1, len(months) + 1))
    ax.set_xticklabels([MONTHS[i - 1] for i in months])
    ax.set_xlabel("Month")
    ax.set_ylabel("Error")
    ax.set_title("Average Monthly GMB Reconstruction Error")
    ax.legend()
    ax.grid(True, axis="y")
    month_str = f"_{months[0]:02d}-{months[-1]:02d}" if months is not None else ""
    if save_path is not None:
        fname = save_path / f"gmb_monthly_error{month_str}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_monthly_gmb_comparison(
    months: list,
    gmb_test_values: list,
    gmb_pred_values: list,
    save_path: Optional[Path] = None,
):
    """Plot spread of original and predicted GMB values for each month

    :param months: List of month numbers (1-12) corresponding to the GMB values
    :type months: list
    :param gmb_test_values: List of GMB values for the test dataset for each month
    :type gmb_test_values: list
    :param gmb_pred_values: List of GMB values for the predicted dataset for each month
    :type gmb_pred_values: list
    :param save_path: Path to save the plot, defaults to None
    :type save_path: Optional[Path], optional
    """
    gmb_test_mean = [np.mean(g) for g in gmb_test_values]
    gmb_test_std = [np.std(g) for g in gmb_test_values]
    gmb_pred_mean = [np.mean(g) for g in gmb_pred_values]
    gmb_pred_std = [np.std(g) for g in gmb_pred_values]
    rmse = [
        np.sqrt(mean_squared_error(gt, gp))
        for gt, gp in zip(gmb_test_values, gmb_pred_values)
    ]
    ymin0 = (
        min(
            min(gmb_test_mean) - max(gmb_test_std),
            min(gmb_pred_mean) - max(gmb_pred_std),
        )
        * 0.9
    )
    ymax0 = (
        max(
            max(gmb_test_mean) + max(gmb_test_std),
            max(gmb_pred_mean) + max(gmb_pred_std),
        )
        * 1.1
    )
    ymin1 = min(rmse) * 0.9
    ymax1 = max(rmse) * 1.1
    ymin = min(ymin0, ymin1) * 0.9
    ymax = max(ymax0, ymax1) * 1.1
    height_ratio0 = (ymax1 - ymin1) / (ymax - ymin)
    height_ratio1 = (ymax0 - ymin0) / (ymax - ymin)
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [height_ratio1, height_ratio0], "hspace": 0},
        sharex=True,
        tight_layout=True,
    )
    ax[0].boxplot(
        gmb_test_values,
        positions=np.arange(1, len(months) + 1) - 0.11,
        widths=0.2,
        labels=[MONTHS[i - 1] for i in months],
        boxprops=dict(color="blue"),
        medianprops=dict(color="blue"),
        whiskerprops=dict(color="blue"),
        capprops=dict(color="blue"),
        showfliers=False,
        label="Original",
    )
    ax[0].boxplot(
        gmb_pred_values,
        positions=np.arange(1, len(months) + 1) + 0.11,
        widths=0.2,
        labels=[MONTHS[i - 1] for i in months],
        boxprops=dict(color="red"),
        medianprops=dict(color="red"),
        whiskerprops=dict(color="red"),
        capprops=dict(color="red"),
        showfliers=False,
        label="Reconstructed",
    )
    ax[0].set_xticks(range(1, len(months) + 1))
    ax[0].set_xticklabels([])
    ax[0].set_xlabel("Month")
    ax[0].set_ylabel("GMB (m.w.e.)")
    ax[0].set_title("Average Monthly Reconstructed vs Original GMB (Test Period)")
    ax[0].legend()
    ax[0].grid(True, axis="y", linestyle="-", linewidth=1.0)
    ax[1].scatter(
        range(1, len(months) + 1), rmse, label="RMSE per Month", color="purple"
    )
    ax[1].set_xticks(range(1, len(months) + 1))
    ax[1].set_xticklabels([MONTHS[i - 1] for i in months])
    ax[1].set_ylabel("RMSE")
    ax[1].yaxis.set_label_coords(-0.07, 0.65)
    ax[1].grid(True, axis="y", linestyle="-", linewidth=1.0)
    plt.tight_layout()
    if save_path is not None:
        fname = save_path / "gmb_monthly_comparison.png"
        plt.savefig(fname, bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_total_error_density(
    gmb_test: xr.Dataset, gmb_pred: xr.Dataset, save_path: Optional[Path] = None
) -> None:
    """Make density plot original vs. predicted GMB values

    :param gmb_test: Test GMB data for all glaciers and months in the test dataset
    :type gmb_test: xarray.Dataset
    :param gmb_pred: Predicted GMB data for all glaciers and months in the test dataset
    :type gmb_pred: xarray.Dataset
    :param save_path: Path to save the plot, defaults to None
    :type save_path: Optional[Path], optional
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(
        gmb_test["monthly_raw"].values.flatten(),
        gmb_pred["monthly_gmb"].T.values.flatten(),
        gridsize=100,
        cmap="viridis",
        mincnt=1,
        bins="log",
    )
    ax.plot(
        [
            min(gmb_test["monthly_raw"].values.flatten()),
            max(gmb_test["monthly_raw"].values.flatten()),
        ],
        [
            min(gmb_test["monthly_raw"].values.flatten()),
            max(gmb_test["monthly_raw"].values.flatten()),
        ],
        "r--",
        lw=1,
    )
    ax.set_xlabel("Observed GMB")
    ax.set_ylabel("Reconstructed GMB")
    ax.grid(True)
    ax.set_title("GMB Reconstruction Error Density for All Test Data")
    plt.colorbar(hb, ax=ax, label="Density")
    if save_path is not None:
        fname = save_path / "gmb_total_error_density.png"
        plt.savefig(fname, bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_total_error_density_by_month(
    gmb_test: xr.Dataset, gmb_pred: xr.Dataset, save_path: Optional[Path] = None
) -> None:
    """Make a density plot for each month of original vs. predicted GMB values

    :param gmb_test: Test GMB data for all glaciers and months in the test dataset
    :type gmb_test: xarray.Dataset
    :param gmb_pred: Predicted GMB data for all glaciers and months in the test dataset
    :type gmb_pred: xarray.Dataset
    :param save_path: Path to save the plot, defaults to None
    :type save_path: Optional[Path], optional
    """
    months = gmb_test["time.month"].values

    for month in np.unique(months):
        month_str = MONTHS[month - 1]

        gmb_test_month = gmb_test.sel(time=gmb_test["time.month"] == month)
        gmb_pred_month = gmb_pred.sel(time=gmb_pred["time.month"] == month)

        fig, ax = plt.subplots(figsize=(8, 8))
        hb = ax.hexbin(
            gmb_test_month["monthly_raw"].values.flatten(),
            gmb_pred_month["monthly_gmb"].T.values.flatten(),
            gridsize=100,
            cmap="viridis",
            mincnt=1,
            bins="log",
        )
        ax.plot(
            [
                min(gmb_test_month["monthly_raw"].values.flatten()),
                max(gmb_test_month["monthly_raw"].values.flatten()),
            ],
            [
                min(gmb_pred_month["monthly_gmb"].T.values.flatten()),
                max(gmb_pred_month["monthly_gmb"].T.values.flatten()),
            ],
            "r--",
            lw=1,
        )
        ax.set_xlabel("Observed GMB")
        ax.set_ylabel("Reconstructed GMB")
        ax.grid(True)
        ax.set_title(f"GMB Reconstruction Error Density for {month_str}")
        plt.colorbar(hb, ax=ax, label="Density")
        if save_path is not None:
            fname = save_path / f"gmb_total_error_density_{month_str}.png"
            plt.savefig(fname, bbox_inches="tight", dpi=500)
        plt.close(fig)


def plot_error_by_feature(
    gmb_test: xr.Dataset, gmb_pred: xr.Dataset, save_path: Optional[Path] = None
) -> None:
    """Make density plot of GMB reconstruction error vs. glacier features (area, zmed, slope)

    :param gmb_test: Test GMB data for all glaciers and months in the test dataset, including glacier features
    :type gmb_test: xarray.Dataset
    :param gmb_pred: Predicted GMB data for all glaciers and months in the test dataset
    :type gmb_pred: xarray.Dataset
    :param save_path: Path to save the plot, defaults to None
    :type save_path: Optional[Path], optional
    """
    features = ["area", "zmed", "slope"]
    for feature in features:
        feature_values = [gmb_test[feature].values] * len(gmb_test["time"])
        feature_values = np.concatenate(feature_values).flatten()
        test_values = gmb_test["monthly_gmb"].values.flatten()
        pred_values = gmb_pred["monthly_gmb"].values.flatten()
        error = pred_values - test_values

        fig, ax = plt.subplots(figsize=(8, 8))
        hb = ax.hexbin(
            feature_values,
            error,
            gridsize=100,
            cmap="viridis",
            mincnt=1,
            bins="log",
        )
        ax.set_xlabel(f"{feature.capitalize()}")
        ax.set_ylabel("Reconstruction Error (Predicted - Observed)")
        ax.set_title(f"GMB Reconstruction Error vs {feature.capitalize()}")
        ax.set_xscale("log" if feature == "area" else "linear")
        ax.grid(True)
        plt.colorbar(hb, ax=ax, label="Density")
        if save_path is not None:
            fname = save_path / f"gmb_error_by_{feature}.png"
            plt.savefig(fname, bbox_inches="tight", dpi=500)
        plt.close(fig)


@app.command()
def main():
    pass


if __name__ == "__main__":
    app()
