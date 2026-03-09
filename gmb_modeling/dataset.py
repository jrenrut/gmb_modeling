from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import geopandas as gpd
from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer
import xarray as xr

from gmb_modeling.config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()


def load_coastline() -> SimpleNamespace:
    """Read BC coastline data

    :return: A SimpleNamespace containing coastline latitude and longitude arrays
    :rtype: SimpleNamespace
    """
    logger.debug("Loading coastline data...")
    coasts = pd.read_csv(EXTERNAL_DATA_DIR / "coastline.csv", header=None)
    coastLat = coasts[0]  # latitude is in the first column
    coastLon = coasts[1]  # longitude is in the second column
    coastline = SimpleNamespace(name="BC", lat=coastLat, lon=coastLon)
    logger.debug("Coastline data loaded.")
    return coastline


def load_regions() -> gpd.GeoDataFrame:
    """Read BC mountain region data

    :return: A GeoDataFrame containing the BC mountain regions with their geometries and attributes
    :rtype: geopandas.GeoDataFrame
    """
    logger.debug("Loading BC glacier regions...")
    bc_regions = gpd.read_file(EXTERNAL_DATA_DIR / "bca_glacier_regions.shp")
    bc_regions = bc_regions.set_crs(epsg=3005, allow_override=True)
    bc_regions = bc_regions.to_crs(epsg=4326)
    indices = np.arange(len(bc_regions))
    bc_regions["id"] = indices
    bc_regions["name"] = [
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
    ]  # names of the 10 BC glacier regions, in order of their region ID
    bc_regions["color"] = [
        "blue",
        "green",
        "orange",
        "purple",
        "red",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]  # colors to use for plotting each region
    logger.debug("BC glacier regions loaded.")
    return bc_regions


def load_rgi_data() -> gpd.GeoDataFrame:
    """Load Randolph Glacier Inventory data

    :return: A GeoDataFrame containing the RGI glacier data with their geometries and attributes
    :rtype: geopandas.GeoDataFrame
    """
    logger.debug("Loading RGI data...")
    rgi_path = EXTERNAL_DATA_DIR / "rgi"
    rgi = None
    for file in rgi_path.glob("*.shp"):
        rgi_ = gpd.read_file(file)
        if rgi is None:
            rgi = rgi_
        else:
            rgi = pd.concat([rgi, rgi_], ignore_index=True)
    logger.debug("RGI data loaded.")
    return rgi  # pyright: ignore[reportReturnType]


def clean_monthly_gmb_data(
    data_path: Path,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    region_ids: Optional[list[str]] = None,
    outfile_name: str = "monthly_gmb_clean.nc",
) -> Path:
    """Process and clean monthly GMB data

    :param data_path: Path to the raw monthly GMB CSV data file
    :type data_path: pathlib.Path
    :param start_month: The starting month for the data range (inclusive)
    :type start_month: pandas.Timestamp
    :param end_month: The ending month for the data range (inclusive)
    :type end_month: pandas.Timestamp
    :param region_ids: Optional list of region IDs to filter the glaciers by (if None, includes all regions)
    :type region_ids: list[str], optional
    :param outfile_name: The name of the output NetCDF file, defaults to "monthly_gmb_clean.nc"
    :type outfile_name: str, optional
    :return: Path to the cleaned monthly GMB NetCDF data file
    :rtype: pathlib.Path
    """
    logger.info("Loading monthly GMB data...")
    df = pd.read_csv(data_path)
    rgi = load_rgi_data()

    logger.info("Mapping RGI attributes to GMB data...")
    rgi_ids = df["Unnamed: 0"]
    rgi_lats, rgi_lons = [], []
    rgi_areas, rgi_zmins, rgi_zmaxs, rgi_zmeds = [], [], [], []
    rgi_slopes, rgi_apsects, rgi_lmaxs = [], [], []
    rgi_statuses, rgi_terms, rgi_surges, rgi_names = [], [], [], []
    for rgi_id in tqdm(rgi_ids, desc="Loading RGI data"):
        rgi_glacier = rgi[
            rgi["RGIId"] == rgi_id
        ]  # filter RGI dataset for the glacier with the matching RGI ID
        if len(rgi_glacier) == 0:
            logger.warning(f"RGI ID {rgi_id} not found in RGI dataset.")
        else:
            rgi_lats.append(rgi_glacier["CenLat"].values[0])
            rgi_lons.append(rgi_glacier["CenLon"].values[0])
            rgi_areas.append(rgi_glacier["Area"].values[0])
            rgi_zmins.append(rgi_glacier["Zmin"].values[0])
            rgi_zmaxs.append(rgi_glacier["Zmax"].values[0])
            rgi_zmeds.append(rgi_glacier["Zmed"].values[0])
            rgi_slopes.append(rgi_glacier["Slope"].values[0])
            rgi_apsects.append(rgi_glacier["Aspect"].values[0])
            rgi_lmaxs.append(rgi_glacier["Lmax"].values[0])
            rgi_statuses.append(rgi_glacier["Status"].values[0])
            rgi_terms.append(rgi_glacier["TermType"].values[0])
            rgi_surges.append(rgi_glacier["Surging"].values[0])
            rgi_names.append(rgi_glacier["Name"].values[0])

    time_cols = df.columns[1:]  # the first column is the RGI ID, the rest are time
    time_index = pd.to_datetime(
        time_cols, format="%Y-%m"
    )  # convert time columns to datetime

    xr_data = xr.DataArray(
        data=df[time_cols].values,
        dims=["rgi_id", "time"],
        coords={"rgi_id": rgi_ids, "time": time_index},
    )  # create xarray DataArray from the dataframe, with dimensions rgi_id and time
    xr_data = xr_data.assign_coords(
        lat=("rgi_id", rgi_lats),
        lon=("rgi_id", rgi_lons),
        area=("rgi_id", rgi_areas),
        zmin=("rgi_id", rgi_zmins),
        zmax=("rgi_id", rgi_zmaxs),
        zmed=("rgi_id", rgi_zmeds),
        slope=("rgi_id", rgi_slopes),
        apsect=("rgi_id", rgi_apsects),
        lmax=("rgi_id", rgi_lmaxs),
        status=("rgi_id", rgi_statuses),
        term=("rgi_id", rgi_terms),
        surge=("rgi_id", rgi_surges),
        name=("rgi_id", rgi_names),
    )  # add the RGI attributes as coordinates to the DataArray
    xr_data.name = "monthly_gmb"
    xr_ds = xr_data.to_dataset()

    logger.info("Removing glaciers with NaN GMB values...")
    # drop glaciers with any NaN values in their GMB time series, and log how many were dropped
    nan_glaciers = xr_ds["monthly_gmb"].isnull().any(dim="time")
    if nan_glaciers.sum() > 0:
        logger.warning(
            f"Dropping {nan_glaciers.sum().item()} glaciers with NaN GMB values."
        )
        df = df.loc[~df["Unnamed: 0"].isin(xr_ds["rgi_id"].values[nan_glaciers.values])]
    xr_ds_clean = xr_ds.sel(rgi_id=~nan_glaciers)  # drop glaciers with NaN values

    logger.info("Assigning BC region names and IDs to glaciers...")
    # for each glacier, check which BC region it falls within based on its lat/lon coordinates, and assign the region name and ID as new coordinates in the dataset
    bc_regions = load_regions()
    region_names: list[str] = []
    region_id_values: list[int] = []
    region_filter = region_ids
    for glacier_id in tqdm(
        xr_ds_clean["rgi_id"].values, desc="Assigning regions to glaciers"
    ):
        row = {
            "lon": xr_ds_clean["lon"].sel(rgi_id=glacier_id).values,
            "lat": xr_ds_clean["lat"].sel(rgi_id=glacier_id).values,
        }  # get the lat/lon coordinates of the glacier
        # create a GeoDataFrame point from the lat/lon coordinates
        point = gpd.points_from_xy([row["lon"]], [row["lat"]])
        point_gdf = gpd.GeoDataFrame(geometry=point, crs=bc_regions.crs)
        joined = gpd.sjoin(point_gdf, bc_regions, how="left", predicate="within")
        # if the point falls within a region, assign the region name and ID, otherwise assign "Unknown" and -1
        if not joined.empty and pd.notnull(joined.iloc[0]["name"]):
            name = joined.iloc[0]["name"]
            id_ = joined.iloc[0]["id"]
            if region_filter is not None and name not in region_filter:
                name = "Unknown"
                id_ = -1
        else:
            logger.warning(
                f"Glacier {glacier_id} at ({row['lat']}, {row['lon']}) not found in any region."
            )
            name = "Unknown"
            id_ = -1
        region_names.append(name)
        region_id_values.append(id_)

    # add the region names and IDs as new coordinates in the dataset
    xr_ds_clean = xr_ds_clean.assign_coords(
        region_name=("rgi_id", region_names),
        region_id=("rgi_id", region_id_values),
    )

    logger.info("Cropping data to specified time range...")
    # crop the dataset to the specified time range
    xr_ds_cropped = xr_ds_clean.sel(time=slice(start_month, end_month))

    outfile = INTERIM_DATA_DIR / "gmb_data" / outfile_name
    xr_ds_cropped.to_netcdf(outfile, format="NETCDF4")
    logger.info(f"Cleaned monthly GMB data saved to {outfile}.")

    return outfile


def clean_monthly_sd_data(
    data_path: Path,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    region_ids: Optional[list[str]] = None,
    outfile_name: str = "monthly_sd_clean.nc",
) -> Path:
    """Process and clean monthly snow depth data

    :param data_path: Path to the raw monthly snow depth NetCDF data files
    :type data_path: pathlib.Path
    :param start_month: The starting month for the data range (inclusive)
    :type start_month: pandas.Timestamp
    :param end_month: The ending month for the data range (inclusive)
    :type end_month: pandas.Timestamp
    :param region_ids: Optional list of region IDs to filter the data by (if None, includes all regions)
    :type region_ids: list[str], optional
    :param outfile_name: The name of the output file, defaults to "monthly_sd_clean.nc"
    :type outfile_name: str, optional
    :return: The path to the cleaned NetCDF file
    :rtype: pathlib.Path
    """
    logger.info("Loading monthly SD data...")
    # read in all the NetCDF files in the data path and concatenate them into a single xarray Dataset, filling any NaN values with 0
    sd_files = list(data_path.glob("MERRA2_*.nc4*"))
    with xr.open_mfdataset(sd_files, combine="by_coords") as _sd_xr:
        sd_xr = _sd_xr.load().fillna(0.0)

    logger.info("Cropping SD data to specified time range...")
    # convert time coordinate to datetime and filter to the specified time range
    sd_time = sd_xr["time"].to_numpy().astype("datetime64[M]")
    sd_time_mask = (sd_time >= np.datetime64(start_month)) & (
        sd_time <= np.datetime64(end_month)
    )
    sd_xr_cropped = sd_xr.sel(time=sd_time_mask)

    # make a new coordinate for the bc region if the grid points fall within bc
    logger.info("Assigning BC region mask to SD data...")
    bc_regions = load_regions()
    # for each grid point, check if it falls within any of the bc regions, and use the region with the most area overlap
    lons = sd_xr_cropped["lon"].values
    lats = sd_xr_cropped["lat"].values
    region_name_mask = np.full((len(lons), len(lats)), "Unknown", dtype=object)
    region_id_mask = np.full((len(lons), len(lats)), -1)
    total_pts = len(lons) * len(lats)
    for i, j in tqdm(
        product(range(len(lons)), range(len(lats))),
        total=total_pts,
        desc="Assigning region mask to SD data",
    ):
        point = gpd.points_from_xy(
            [lons[i]], [lats[j]]
        )  # create a GeoDataFrame point from the lat/lon coordinates
        point_gdf = gpd.GeoDataFrame(geometry=point, crs=bc_regions.crs)
        joined = gpd.sjoin(point_gdf, bc_regions, how="left", predicate="within")
        # if the point falls within a region, assign the region name and ID, otherwise assign "Unknown" and -1
        if not joined.empty and pd.notnull(joined.iloc[0]["id"]):
            region_id_mask[i, j] = joined.iloc[0]["id"]
            region_name_mask[i, j] = joined.iloc[0]["name"]

    sd_xr_cropped = sd_xr_cropped.assign_coords(
        region_name=(("lon", "lat"), region_name_mask),
        region_id=(("lon", "lat"), region_id_mask),
    )  # add the region names and IDs as new coordinates in the dataset

    outfile = INTERIM_DATA_DIR / "sd_data" / outfile_name
    sd_xr_cropped.to_netcdf(outfile, format="NETCDF4")
    logger.info(f"Cleaned monthly SD data saved to {outfile}.")

    return outfile


def split_data_by_month(
    data: xr.DataArray, cutoff_month: pd.Timestamp
) -> tuple[xr.DataArray, xr.DataArray]:
    """Split dataset into train and test based on cutoff month

    :param data: The input dataset to be split, with a time coordinate
    :type data: xarray.DataArray
    :param cutoff_month: The month at which to split the data (split month goes to test set)
    :type cutoff_month: pandas.Timestamp
    :return: The train and test datasets
    :rtype: tuple[xarray.DataArray, xarray.DataArray]
    """
    logger.info(f"Splitting data at cutoff month {cutoff_month}...")
    train_data = data.sel(time=data["time"].values < cutoff_month.to_numpy())
    test_data = data.sel(time=data["time"].values >= cutoff_month.to_numpy())
    return train_data, test_data


def get_monthly_mean(data: xr.DataArray) -> xr.DataArray:
    """Get mean of each month over all years

    :param data: The input dataset with a time coordinate, to calculate the monthly mean from
    :type data: xarray.DataArray
    :return: The monthly mean dataset, with the same coordinates as the input dataset except the time coordinate is replaced with a month coordinate (1-12)
    :rtype: xarray.DataArray
    """
    logger.info("Calculating monthly mean...")
    mean_monthly = data.groupby("time.month").mean(dim="time")
    return mean_monthly


def get_anomaly(data: xr.DataArray, monthly_mean: xr.DataArray) -> xr.DataArray:
    """Get deviation from mean of each month over all years

    :param data: The input dataset with a time coordinate, to calculate the anomaly from
    :type data: xarray.DataArray
    :param monthly_mean: The monthly mean dataset
    :type monthly_mean: xarray.DataArray
    :return: The anomaly dataset, with the same coordinates as the input dataset
    :rtype: xarray.DataArray
    """
    logger.info("Calculating anomaly...")
    anomaly = data.groupby("time.month") - monthly_mean
    return anomaly


def remove_anomaly(data: xr.DataArray, monthly_mean: xr.DataArray) -> xr.DataArray:
    """Remove monthly anomaly by adding back the monthly mean

    :param data: A dataset with a time coordinate, to remove the anomaly from (typically a dataset of predictions that was generated from anomaly data)
    :type data: xarray.DataArray
    :param monthly_mean: The monthly mean dataset
    :type monthly_mean: xarray.DataArray
    :return: The dataset with the anomaly removed, with the same coordinates as the input dataset
    :rtype: xarray.DataArray
    """
    logger.info("Removing anomaly to get original data...")
    uncorrected_data = data.groupby("time.month") + monthly_mean
    return uncorrected_data


def smooth_gmb_data(
    data: xr.DataArray,
    window_size: int = 3,
) -> xr.DataArray:
    """Apply smoothing to monthly data

    :param data: The input dataset with a time coordinate, to apply smoothing to
    :type data: xarray.DataArray
    :param window_size: The size of the rolling window for smoothing, defaults to 3
    :type window_size: int, optional
    :return: The smoothed dataset, with the same coordinates as the input dataset
    :rtype: xarray.DataArray
    """
    logger.info(f"Smoothing GMB data with window size {window_size}...")
    smoothed_data = np.empty_like(data.values)
    for i in tqdm(range(data.shape[1]), desc="Smoothing data"):
        smoothed_data[:, i] = np.convolve(
            data.values[:, i], np.ones(window_size) / window_size, mode="same"
        )
    smoothed_data = xr.DataArray(data=smoothed_data, coords=data.coords, dims=data.dims)
    return smoothed_data


def smooth_sd_data(
    data: xr.DataArray,
    window_size: int = 3,
) -> xr.DataArray:
    """Apply smoothing to monthly data

    :param data: The input dataset with a time coordinate, to apply smoothing to
    :type data: xarray.DataArray
    :param window_size: The size of the rolling window for smoothing, defaults to 3
    :type window_size: int, optional
    :return: The smoothed dataset, with the same coordinates as the input dataset
    :rtype: xarray.DataArray
    """
    logger.info(f"Smoothing SD data with window size {window_size}...")
    smoothed_data = np.empty_like(data.values)
    lons = data["lon"].values
    lats = data["lat"].values
    total_pts = len(lons) * len(lats)
    for i, j in tqdm(
        product(range(data.shape[1]), range(data.shape[2])),
        total=total_pts,
        desc="Smoothing data",
    ):
        smoothed_data[:, i, j] = np.convolve(
            data.values[:, i, j], np.ones(window_size) / window_size, mode="same"
        )
    smoothed_data = xr.DataArray(data=smoothed_data, coords=data.coords, dims=data.dims)
    return smoothed_data


def load_data(processed_data_path: str, subset: str = "train") -> xr.Dataset:
    """Load data from saved .nc file

    :param processed_data_path: The path to the processed data file
    :type processed_data_path: str
    :param subset: The subset of data to load, defaults to "train"
    :type subset: str, optional
    :return: The loaded dataset
    :rtype: xarray.Dataset
    """
    logger.debug(f"Loading {subset} data from {processed_data_path}...")
    data_file = PROCESSED_DATA_DIR / processed_data_path / subset / "data.nc"
    with xr.open_dataset(data_file) as _ds:
        ds = _ds.load()
    return ds


def get_gmb_region(
    ds: xr.Dataset, region_ids: Optional[Union[int, str, list]] = None
) -> xr.Dataset:
    """Get subset of GMB data for specific BC mountain region

    Can specify region by either region name or region ID, or a list of names or IDs for multiple regions. If no region is specified, returns the full dataset.

    :param ds: The input GMB dataset with a region_id and region_name coordinate, to filter for the specified region IDs
    :type ds: xarray.Dataset
    :param region_ids: The region IDs to filter for, defaults to None
    :type region_ids: Optional[Union[int, str, list]], optional
    :return: The filtered dataset
    :rtype: xarray.Dataset
    """
    if region_ids is None:
        return ds
    logger.debug(f"Filtering GMB data for region IDs {region_ids}...")
    if isinstance(region_ids, list):
        if isinstance(region_ids[0], str):
            region_ds = ds.isel(rgi_id=ds["region_name"].isin(region_ids))
        else:
            region_ds = ds.isel(rgi_id=ds["region_id"].isin(region_ids))
    elif isinstance(region_ids, str):
        region_ds = ds.isel(rgi_id=ds["region_name"] == region_ids)
    else:
        region_ds = ds.isel(rgi_id=ds["region_id"] == region_ids)
    return region_ds


def get_sd_region(
    ds: xr.Dataset, region_ids: Optional[Union[int, str, list]] = None
) -> xr.Dataset:
    """Get subset of snow depth data for specific BC mountain region

    Can specify region by either region name or region ID, or a list of names or IDs for multiple regions. If no region is specified, returns the full dataset.

    :param ds: The input snow depth dataset with a region_id and region_name coordinate, to filter for the specified region IDs
    :type ds: xarray.Dataset
    :param region_ids: The region IDs to filter for, defaults to None
    :type region_ids: Optional[Union[int, str, list]], optional
    :return: The filtered dataset
    :rtype: xarray.Dataset
    """
    if region_ids is None:
        return ds
    logger.debug(f"Filtering SD data for region IDs {region_ids}...")
    if isinstance(region_ids, list):
        if isinstance(region_ids[0], str):
            mask = ds["region_name"].isin(region_ids)
        else:
            mask = ds["region_id"].isin(region_ids)
    elif isinstance(region_ids, str):
        mask = ds["region_name"] == region_ids
    else:
        mask = ds["region_id"] == region_ids

    # get the indices of the grid points that fall within the specified region(s) based on the mask, and use those indices to subset the dataset
    lon_idx, lat_idx = np.where(mask.values)
    if lon_idx.size == 0:
        logger.warning(f"No grid points found for region IDs {region_ids}")
        return ds.isel(lon=slice(0, 0), lat=slice(0, 0))

    # get the unique longitude and latitude indices that correspond to the specified region(s), and use those to subset the dataset, then apply the mask to set any grid points outside the region(s) to NaN and fill those with 0
    lon_sel = np.unique(lon_idx)
    lat_sel = np.unique(lat_idx)
    region_ds = ds.isel(lon=lon_sel, lat=lat_sel)

    # apply the mask to set any grid points outside the region(s) to NaN, then fill those with 0
    region_mask = mask.isel(lon=lon_sel, lat=lat_sel)
    region_ds = region_ds.where(region_mask, drop=False)
    region_ds = region_ds.fillna(0.0)

    return region_ds


def get_sd_gridpoint(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    """Get snow depth data for a specific grid point based on its longitude and latitude coordinates, using nearest neighbor selection to find the closest grid point to the specified coordinates

    :param ds: The input snow depth dataset with longitude and latitude coordinates, to filter for the specified grid point
    :type ds: xarray.Dataset
    :param lat: The latitude coordinate of the grid point
    :type lat: float
    :param lon: The longitude coordinate of the grid point
    :type lon: float
    :return: The filtered dataset for the specified grid point
    :rtype: xarray.Dataset
    """
    logger.debug(f"Filtering SD data for grid point at ({lat}, {lon})...")
    point_ds = ds.sel(lon=lon, lat=lat, method="nearest")
    return point_ds


def get_gmb_rgiid(ds: xr.Dataset, rgi_id: str) -> xr.Dataset:
    """Get GMB data for a specific glacier based on its RGI ID

    :param ds: The input GMB dataset with an rgi_id coordinate, to filter for the specified glacier
    :type ds: xarray.Dataset
    :param rgi_id: The RGI ID of the glacier to filter for
    :type rgi_id: str
    :return: The filtered dataset for the specified glacier
    :rtype: xarray.Dataset
    """
    logger.debug(f"Filtering GMB data for glacier with RGI ID {rgi_id}...")
    point_ds = ds.sel(rgi_id=rgi_id)
    return point_ds


@app.command()
def main(cfg: Union[Path, dict]) -> tuple[dict, dict]:
    """Run the dataset ingestion and processing functions based on a configuration file

    :param cfg: The configuration for the dataset processing, either as a path to a JSON file or as a dictionary
    :type cfg: Union[Path, dict]
    :return: The processed configuration and results dictionaries
    :rtype: tuple[dict, dict]
    """
    # Load configuration from JSON file if a path is provided, otherwise use the provided dictionary
    if isinstance(cfg, Path):
        logger.info(f"Loading configuration from {cfg}...")
        cfg = json.load(cfg.open())
    else:
        logger.info("Using provided configuration dictionary...")
    assert isinstance(cfg, dict)

    # get the start month, end month, and cutoff month from the configuration, and convert them to pandas Timestamps
    start_month = pd.Timestamp(cfg["start_month"])
    end_month = pd.Timestamp(cfg["end_month"])
    cutoff_month = pd.Timestamp(cfg["cutoff_month"])
    region_ids = cfg.get("region_ids", None)

    gmb_data_path = RAW_DATA_DIR / "gmb_data" / "ts_monthly_const_area_lstm.csv"
    cfg["gmb_data_path"] = (
        gmb_data_path.as_posix()
    )  # save the raw data path to the config for reference
    gmb_interim = clean_monthly_gmb_data(
        gmb_data_path, start_month, end_month
    )  # process the raw GMB data and save the cleaned interim dataset, and save the path to the interim dataset in the config for reference
    cfg["gmb_interim"] = (
        gmb_interim.as_posix()
    )  # save the interim data path to the config for reference

    sd_data_path = RAW_DATA_DIR / "merra_sd_land_data"
    cfg["sd_data_path"] = (
        sd_data_path.as_posix()
    )  # save the raw data path to the config for reference
    sd_interim = clean_monthly_sd_data(
        sd_data_path, start_month, end_month
    )  # process the raw snow depth data and save the cleaned interim dataset, and save the path to the interim dataset in the config for reference
    cfg["sd_interim"] = (
        sd_interim.as_posix()
    )  # save the interim data path to the config for reference

    with xr.open_dataset(gmb_interim) as _gmb:
        gmb_data = _gmb.load()
    with xr.open_dataset(sd_interim) as _sd:
        sd_data = _sd.load()

    # split the datasets into train and test based on the cutoff month
    cutoff_month = pd.Timestamp("2016-01")
    gmb_train_orig, gmb_test_orig = split_data_by_month(
        gmb_data["monthly_gmb"], cutoff_month
    )
    sd_train_orig, sd_test_orig = split_data_by_month(sd_data["SNODP"], cutoff_month)

    # calculate the monthly mean for the train datasets
    gmb_train_mean = get_monthly_mean(gmb_train_orig)
    sd_train_mean = get_monthly_mean(sd_train_orig)

    # calculate the anomaly for the train and test datasets by subtracting the training monthly mean from the original data
    gmb_train_anomaly = get_anomaly(gmb_train_orig, gmb_train_mean)
    gmb_test_anomaly = get_anomaly(gmb_test_orig, gmb_train_mean)
    sd_train_anomaly = get_anomaly(sd_train_orig, sd_train_mean)
    sd_test_anomaly = get_anomaly(sd_test_orig, sd_train_mean)

    # apply smoothing to the anomaly datasets using a rolling mean with a specified window size
    window_size = cfg.get(
        "smoothing_window_size", 3
    )  # get the smoothing window size from the config, default to 3 if not specified
    gmb_train_smooth = smooth_gmb_data(gmb_train_anomaly, window_size=window_size)
    gmb_test_smooth = smooth_gmb_data(gmb_test_anomaly, window_size=window_size)
    gmb_train_smooth.attrs["subset"] = "train"
    gmb_test_smooth.attrs["subset"] = "test"
    gmb_train_smooth.attrs["window_size"] = window_size
    gmb_test_smooth.attrs["window_size"] = window_size
    sd_train_smooth = smooth_sd_data(sd_train_anomaly, window_size=window_size)
    sd_test_smooth = smooth_sd_data(sd_test_anomaly, window_size=window_size)
    sd_train_smooth.attrs["subset"] = "train"
    sd_test_smooth.attrs["subset"] = "test"
    sd_train_smooth.attrs["window_size"] = window_size
    sd_test_smooth.attrs["window_size"] = window_size

    # save the processed datasets to .nc files, including the smoothed anomaly data as the main variable, and the original data and monthly mean as additional variables in the same dataset, and save the paths to the processed datasets in the config for reference
    logger.info("Saving processed datasets...")

    gmb_train_ds = gmb_train_smooth.to_dataset(name="monthly_gmb")
    gmb_train_ds["monthly_raw"] = gmb_train_orig
    gmb_train_ds["monthly_mean"] = gmb_train_mean
    gmb_train_ds["anomaly"] = gmb_train_anomaly
    gmb_train_ds.attrs["window_size"] = window_size
    gmb_data_train_path = PROCESSED_DATA_DIR / "gmb_data" / "train" / "data.nc"
    logger.info(f"Saving processed GMB train dataset to {gmb_data_train_path}...")
    gmb_train_ds.to_netcdf(gmb_data_train_path, format="NETCDF4")
    gmb_train_ds.close()  # close the dataset to free up memory
    cfg["gmb_train_processed"] = gmb_data_train_path.as_posix()

    # TODO: do we need to smooth the test data?
    # gmb_test_ds = gmb_test_smooth.to_dataset(name="monthly_gmb")
    gmb_test_ds = gmb_test_anomaly.to_dataset(name="monthly_gmb")
    gmb_test_ds["monthly_raw"] = gmb_test_orig
    gmb_test_ds["monthly_mean"] = gmb_train_mean
    gmb_test_ds["anomaly"] = gmb_test_anomaly
    gmb_data_test_path = PROCESSED_DATA_DIR / "gmb_data" / "test" / "data.nc"
    logger.info(f"Saving processed GMB test dataset to {gmb_data_test_path}...")
    gmb_test_ds.to_netcdf(gmb_data_test_path, format="NETCDF4")
    gmb_test_ds.close()  # close the dataset to free up memory
    cfg["gmb_test_processed"] = gmb_data_test_path.as_posix()

    sd_train_ds = sd_train_smooth.to_dataset(name="SNODP")
    sd_train_ds["monthly_raw"] = sd_train_orig
    sd_train_ds["monthly_mean"] = sd_train_mean
    sd_train_ds["anomaly"] = sd_train_anomaly
    sd_train_ds.attrs["window_size"] = window_size
    sd_data_train_path = PROCESSED_DATA_DIR / "sd_data" / "train" / "data.nc"
    logger.info(f"Saving processed SD train dataset to {sd_data_train_path}...")
    sd_train_ds.to_netcdf(sd_data_train_path, format="NETCDF4")
    sd_train_ds.close()  # close the dataset to free up memory
    cfg["sd_train_processed"] = sd_data_train_path.as_posix()

    # sd_test_ds = sd_test_smooth.to_dataset(name="SNODP")
    sd_test_ds = sd_test_anomaly.to_dataset(name="SNODP")
    sd_test_ds["monthly_raw"] = sd_test_orig
    sd_test_ds["monthly_mean"] = sd_train_mean
    sd_test_ds["anomaly"] = sd_test_anomaly
    sd_data_test_path = PROCESSED_DATA_DIR / "sd_data" / "test" / "data.nc"
    logger.info(f"Saving processed SD test dataset to {sd_data_test_path}...")
    sd_test_ds.to_netcdf(sd_data_test_path, format="NETCDF4")
    sd_test_ds.close()  # close the dataset to free up memory
    cfg["sd_test_processed"] = sd_data_test_path.as_posix()

    # log statistics about the datasets, including the number of glaciers and time steps in the train and test sets, the shape of the snow depth datasets, and the cutoff month used for splitting, and save these statistics to a report
    results = {
        "gmb_train_glaciers": int(gmb_train_ds.sizes.get("rgi_id", 0)),
        "gmb_test_glaciers": int(gmb_test_ds.sizes.get("rgi_id", 0)),
        "gmb_train_times": int(gmb_train_ds.sizes.get("time", 0)),
        "gmb_test_times": int(gmb_test_ds.sizes.get("time", 0)),
        "sd_train_shape": tuple(sd_train_ds["SNODP"].shape),
        "sd_test_shape": tuple(sd_test_ds["SNODP"].shape),
        "cutoff_month": str(cutoff_month),
    }

    return cfg, results


if __name__ == "__main__":
    # app()
    config_file = Path(__file__).resolve().parent / "cfg" / "all-BC_2016-cutoff.json"
    main(config_file)
