from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
from loguru import logger
import numpy as np
import pandas as pd
from pyparsing import Union
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
    logger.info("Loading coastline data...")
    coasts = pd.read_csv(EXTERNAL_DATA_DIR / "coastline.csv", header=None)
    coastLat = coasts[0]
    coastLon = coasts[1]
    coastline = SimpleNamespace(name="BC", lat=coastLat, lon=coastLon)
    logger.info("Coastline data loaded.")
    return coastline


def load_regions() -> gpd.GeoDataFrame:
    logger.info("Loading BC glacier regions...")
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
    ]
    logger.info("BC glacier regions loaded.")
    return bc_regions


def load_rgi_data() -> gpd.GeoDataFrame:
    logger.info("Loading RGI data...")
    rgi_path = EXTERNAL_DATA_DIR / "rgi"
    rgi = None
    for file in rgi_path.glob("*.shp"):
        rgi_ = gpd.read_file(file)
        if rgi is None:
            rgi = rgi_
        else:
            rgi = pd.concat([rgi, rgi_], ignore_index=True)
    logger.info("RGI data loaded.")
    return rgi  # pyright: ignore[reportReturnType]


def clean_monthly_gmb_data(
    data_path: Path,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    outfile_name: str = "monthly_gmb_clean.nc",
) -> Path:
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
        rgi_glacier = rgi[rgi["RGIId"] == rgi_id]
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

    time_cols = df.columns[1:]
    time_index = pd.to_datetime(time_cols, format="%Y-%m")

    xr_data = xr.DataArray(
        data=df[time_cols].values,
        dims=["rgi_id", "time"],
        coords={"rgi_id": rgi_ids, "time": time_index},
    )
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
    )
    xr_data.name = "monthly_gmb"
    xr_ds = xr_data.to_dataset()

    logger.info("Removing glaciers with NaN GMB values...")
    nan_glaciers = xr_ds["monthly_gmb"].isnull().any(dim="time")
    if nan_glaciers.sum() > 0:
        logger.warning(
            f"Dropping {nan_glaciers.sum().item()} glaciers with NaN GMB values."
        )
        df = df.loc[~df["Unnamed: 0"].isin(xr_ds["rgi_id"].values[nan_glaciers.values])]
    xr_ds_clean = xr_ds.sel(rgi_id=~nan_glaciers)

    logger.info("Assigning region names and IDs to glaciers...")
    bc_regions = load_regions()
    region_names = []
    region_ids = []
    for glacier_id in xr_ds_clean["rgi_id"].values:
        row = {
            "lon": xr_ds_clean["lon"].sel(rgi_id=glacier_id).values,
            "lat": xr_ds_clean["lat"].sel(rgi_id=glacier_id).values,
        }
        point = gpd.points_from_xy([row["lon"]], [row["lat"]])
        point_gdf = gpd.GeoDataFrame(geometry=point, crs=bc_regions.crs)
        joined = gpd.sjoin(point_gdf, bc_regions, how="left", predicate="within")
        if not joined.empty and pd.notnull(joined.iloc[0]["name"]):
            name = joined.iloc[0]["name"]
            id_ = joined.iloc[0]["id"]
        else:
            logger.warning(
                f"Glacier {glacier_id} at ({row['lat']}, {row['lon']}) not found in any region."
            )
            name = "Unknown"
            id_ = -1
        region_names.append(name)
        region_ids.append(id_)

    xr_ds_clean = xr_ds_clean.assign_coords(
        region_name=("rgi_id", region_names),
        region_id=("rgi_id", region_ids),
    )

    logger.info("Cropping data to specified time range...")
    xr_ds_cropped = xr_ds_clean.sel(time=slice(start_month, end_month))

    outfile = INTERIM_DATA_DIR / "gmb_data" / outfile_name
    xr_ds_cropped.to_netcdf(outfile, format="NETCDF4")
    logger.info(f"Cleaned monthly GMB data saved to {outfile}.")

    return outfile


def clean_monthly_sd_data(
    data_path: Path,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    outfile_name: str = "monthly_sd_clean.nc",
) -> Path:
    logger.info("Loading monthly SD data...")
    sd_files = list(data_path.glob("MERRA2_*.nc4*"))
    sd_xr = xr.open_mfdataset(sd_files, combine="by_coords")
    sd_xr = sd_xr.fillna(0.0)

    logger.info("Cropping SD data to specified time range...")
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
    for i in tqdm(range(len(lons)), desc="Assigning region mask"):
        for j in range(len(lats)):
            point = gpd.points_from_xy([lons[i]], [lats[j]])
            point_gdf = gpd.GeoDataFrame(geometry=point, crs=bc_regions.crs)
            joined = gpd.sjoin(point_gdf, bc_regions, how="left", predicate="within")
            if not joined.empty and pd.notnull(joined.iloc[0]["id"]):
                region_id_mask[i, j] = joined.iloc[0]["id"]
                region_name_mask[i, j] = joined.iloc[0]["name"]
    sd_xr_cropped = sd_xr_cropped.assign_coords(
        region_name=(("lon", "lat"), region_name_mask),
        region_id=(("lon", "lat"), region_id_mask),
    )

    outfile = INTERIM_DATA_DIR / "sd_data" / outfile_name
    sd_xr_cropped.to_netcdf(outfile, format="NETCDF4")
    logger.info(f"Cleaned monthly SD data saved to {outfile}.")

    return outfile


def split_data_by_month(data: xr.DataArray, cutoff_month: pd.Timestamp):
    logger.info(f"Splitting data at cutoff month {cutoff_month}...")
    train_data = data.sel(time=data["time"].values < cutoff_month.to_numpy())
    test_data = data.sel(time=data["time"].values >= cutoff_month.to_numpy())
    return train_data, test_data


def get_monthly_mean(data: xr.DataArray) -> xr.DataArray:
    # get mean of each month over all years
    logger.info("Calculating monthly mean...")
    mean_monthly = data.groupby("time.month").mean(dim="time")
    return mean_monthly


def get_anomaly(data: xr.DataArray, monthly_mean: xr.DataArray) -> xr.DataArray:
    logger.info("Calculating anomaly...")
    anomaly = data.groupby("time.month") - monthly_mean
    return anomaly


def remove_anomaly(data: xr.DataArray, monthly_mean: xr.DataArray) -> xr.DataArray:
    logger.info("Removing anomaly to get original data...")
    uncorrected_data = data.groupby("time.month") + monthly_mean
    return uncorrected_data


def smooth_data(
    data: xr.DataArray,
    window_size: int = 3,
) -> xr.DataArray:
    logger.info(f"Smoothing data with window size {window_size}...")
    smoothed_data = data.rolling(time=window_size, center=True, min_periods=1).mean()
    smoothed_data.attrs["window_size"] = window_size
    smoothed_data.attrs["min_periods"] = 1
    return smoothed_data


def load_data(processed_data_path: str, subset: str = "train") -> xr.Dataset:
    logger.info(f"Loading {subset} data from {processed_data_path}...")
    data_file = PROCESSED_DATA_DIR / processed_data_path / subset / "data.nc"
    return xr.open_dataset(data_file)


def get_gmb_region(ds: xr.Dataset, region_id: Union[int, str]) -> xr.Dataset:
    logger.info(f"Filtering GMB data for region ID {region_id}...")
    if isinstance(region_id, str):
        region_ds = ds.isel(rgi_id=ds["region_name"] == region_id)
    else:
        region_ds = ds.isel(rgi_id=ds["region_id"] == region_id)
    return region_ds


def get_sd_region(ds: xr.Dataset, region_id: Union[int, str]) -> xr.Dataset:
    logger.info(f"Filtering SD data for region ID {region_id}...")
    if isinstance(region_id, str):
        mask = ds["region_name"] == region_id
    else:
        mask = ds["region_id"] == region_id

    lon_idx, lat_idx = np.where(mask.values)
    if lon_idx.size == 0:
        logger.warning(f"No grid points found for region {region_id}")
        return ds.isel(lon=slice(0, 0), lat=slice(0, 0))

    lon_sel = np.unique(lon_idx)
    lat_sel = np.unique(lat_idx)

    region_ds = ds.isel(lon=lon_sel, lat=lat_sel)

    region_mask = mask.isel(lon=lon_sel, lat=lat_sel)
    region_ds = region_ds.where(region_mask, drop=False)

    return region_ds


def get_sd_gridpoint(ds: xr.Dataset, lon: float, lat: float) -> xr.Dataset:
    logger.info(f"Filtering SD data for grid point at ({lat}, {lon})...")
    point_ds = ds.sel(lon=lon, lat=lat, method="nearest")
    return point_ds


def get_gmb_rgiid(ds: xr.Dataset, rgi_id: str) -> xr.Dataset:
    logger.info(f"Filtering GMB data for glacier with RGI ID {rgi_id}...")
    point_ds = ds.sel(rgi_id=rgi_id)
    return point_ds


@app.command()
def main():
    start_month = pd.Timestamp("1980-01")
    end_month = pd.Timestamp("2022-12")

    gmb_data_path = RAW_DATA_DIR / "gmb_data" / "ts_monthly_const_area_lstm.csv"
    gmb_interim = clean_monthly_gmb_data(gmb_data_path, start_month, end_month)

    sd_data_path = RAW_DATA_DIR / "merra_sd_land_data"
    sd_interim = clean_monthly_sd_data(sd_data_path, start_month, end_month)

    gmb_data = xr.open_dataset(gmb_interim)
    sd_data = xr.open_dataset(sd_interim)

    cutoff_month = pd.Timestamp("2015-09")
    gmb_train, gmb_test = split_data_by_month(gmb_data["monthly_gmb"], cutoff_month)
    sd_train, sd_test = split_data_by_month(sd_data["SNODP"], cutoff_month)

    gmb_train_mean = get_monthly_mean(gmb_train)
    sd_train_mean = get_monthly_mean(sd_train)

    gmb_train_anomaly = get_anomaly(gmb_train, gmb_train_mean)
    gmb_test_anomaly = get_anomaly(gmb_test, gmb_train_mean)
    sd_train_anomaly = get_anomaly(sd_train, sd_train_mean)
    sd_test_anomaly = get_anomaly(sd_test, sd_train_mean)

    window_size = 3
    gmb_train_smooth = smooth_data(gmb_train_anomaly, window_size=window_size)
    gmb_test_smooth = smooth_data(gmb_test_anomaly, window_size=window_size)
    gmb_train_smooth.attrs["subset"] = "train"
    gmb_test_smooth.attrs["subset"] = "test"
    sd_train_smooth = smooth_data(sd_train_anomaly, window_size=window_size)
    sd_test_smooth = smooth_data(sd_test_anomaly, window_size=window_size)
    sd_train_smooth.attrs["subset"] = "train"
    sd_test_smooth.attrs["subset"] = "test"

    logger.info("Saving processed datasets...")
    gmb_train_ds = gmb_train_smooth.to_dataset(name="monthly_gmb")
    gmb_train_ds["monthly_mean"] = gmb_train_mean
    gmb_train_ds.to_netcdf(
        PROCESSED_DATA_DIR / "gmb_data" / "train" / "data.nc", format="NETCDF4"
    )

    gmb_test_ds = gmb_test_smooth.to_dataset(name="monthly_gmb")
    # Use training monthly mean for test anomaly reference
    gmb_test_ds["monthly_mean"] = gmb_train_mean
    gmb_test_ds.to_netcdf(
        PROCESSED_DATA_DIR / "gmb_data" / "test" / "data.nc", format="NETCDF4"
    )

    sd_train_ds = sd_train_smooth.to_dataset(name="SNODP")
    sd_train_ds["monthly_mean"] = sd_train_mean
    sd_train_ds.to_netcdf(
        PROCESSED_DATA_DIR / "sd_data" / "train" / "data.nc", format="NETCDF4"
    )

    sd_test_ds = sd_test_smooth.to_dataset(name="SNODP")
    sd_test_ds["monthly_mean"] = sd_train_mean
    sd_test_ds.to_netcdf(
        PROCESSED_DATA_DIR / "sd_data" / "test" / "data.nc", format="NETCDF4"
    )


if __name__ == "__main__":
    app()
