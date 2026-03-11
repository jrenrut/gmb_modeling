import argparse
import json
from pathlib import Path
import sys

import jsonschema
from loguru import logger
import numpy as np
import typer
import xarray as xr

from gmb_modeling import dataset, plots
from gmb_modeling.config import MODELS_DIR, REPORTS_DIR
from gmb_modeling.modeling import analysis, predict, train

app = typer.Typer()

DEFAULT_CONFIG = Path(__file__).resolve().parent / "cfg" / "main.json"


def run(cfg_file: Path = DEFAULT_CONFIG):
    """Run full pipeline based on JSON config file.

    :param cfg_file: Path to JSON config file containing parameters and paths for the pipeline run. Should include necessary paths to data and models, as well as parameters like months and region_ids to specify the subset of data to use for training and analysis.
    :type cfg_file: Path
    """
    # set up logger
    cfg = json.load(cfg_file.open())
    # validate config against JSON schema if available
    schema_path = Path(__file__).resolve().parent / "cfg" / "config_schema.json"
    with schema_path.open("r", encoding="utf8") as sf:
        schema = json.load(sf)
    try:
        jsonschema.validate(instance=cfg, schema=schema)
        logger.info("Config validated against schema: %s", schema_path)
    except jsonschema.ValidationError as e:
        logger.error("Configuration validation failed: %s", e.message)
        raise SystemExit(f"Configuration validation failed: {e.message}")
    workflow_name = cfg.get("name", "default_workflow")
    workflow_dir = MODELS_DIR / workflow_name
    workflow_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = workflow_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cfg["workflow_dir"] = str(workflow_dir).replace("\\", "/")
    cfg["figures_dir"] = str(figures_dir).replace("\\", "/")
    log_file = workflow_dir / "run.log"
    # if log file already exists make a new one with an incremented number suffix
    if log_file.exists():
        i = 1
        while True:
            new_log_file = workflow_dir / f"run_{i}.log"
            if not new_log_file.exists():
                log_file = new_log_file
                break
            i += 1
    logger.remove()  # clear default handlers
    logger.add(sys.stderr, level="INFO")  # console
    logger.add(
        str(log_file),
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        backtrace=True,
        diagnose=True,
    )
    logger.info("Beginning gmb_modeling pipeline...")
    logger.info(f"Loaded config from {cfg_file}")

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

    months = cfg.get("months")
    region_ids = cfg.get("region_ids")

    run_dataset = cfg.get("run_dataset", True)
    run_train = cfg.get("run_train", True)
    run_predict = cfg.get("run_predict", True)
    run_analysis = cfg.get("run_analysis", True)

    summary_lines = [
        f"Config file: {cfg_file}",
        f"Months: {months}",
        f"Region IDs: {region_ids}",
    ]

    if run_dataset:
        logger.info(f"Initializing workflow '{workflow_name}'...")
        logger.info("Running dataset preprocessing...")
        cfg, dataset_results = dataset.main(cfg)
        logger.info("Dataset preprocessing complete.")
        summary_lines.append("Dataset preprocessing: completed")
        # add dataset results to summary
        for key, value in dataset_results.items():
            summary_lines.append(f"{key}: {value}")
        with open(workflow_dir / "config_used.json", "w") as f:
            json.dump(cfg, f, indent=4)

        plots.plot_glacier_regions(save_path=figures_dir)

        # plot a sample month of GCM and GMB data for visual inspection
        gcm_interim_path = cfg["gcm_interim"]
        gcm_variable = str(cfg.get("gcm_variable"))  # type: ignore

        with xr.open_dataset(gcm_interim_path) as _gcm_interim:
            gcm_interim = _gcm_interim.load()
        gmb_interim_path = cfg["gmb_interim"]
        with xr.open_dataset(gmb_interim_path) as _gmb_interim:
            gmb_interim = _gmb_interim.load()
        sample_month = np.random.choice(gcm_interim["time"].values)
        plots.plot_gcm(
            data_path=gcm_interim_path,
            gcm_variable=gcm_variable,
            month=sample_month,
            region_ids=region_ids,
            save_path=figures_dir,
        )
        plots.plot_gmb(
            data_path=gmb_interim_path,
            month=sample_month,
            region_ids=region_ids,
            save_path=figures_dir,
        )
        gcm_interim.close()
        gmb_interim.close()

        # plot a sample gmb and snow depth time series for a randomly selected point with non-zero snow depth values in the test period
        gcm_processed_data_path = cfg["gcm_train_processed"]
        with xr.open_dataset(gcm_processed_data_path) as _gcm_processed:
            gcm_processed = _gcm_processed.load()
        gmb_processed_data_path = cfg["gmb_train_processed"]
        with xr.open_dataset(gmb_processed_data_path) as _gmb_processed:
            gmb_processed = _gmb_processed.load()
        while True:  # check that the randomly selected point corresponding to an rgi_id has non-zero snow depth values in the test period
            sample_rgi_id = np.random.choice(gmb_processed["rgi_id"].values)
            sample_lat = float(gmb_processed.sel(rgi_id=sample_rgi_id)["lat"].values)
            sample_lon = float(gmb_processed.sel(rgi_id=sample_rgi_id)["lon"].values)
            sample_point_sd = gcm_processed.sel(
                lat=sample_lat, lon=sample_lon, method="nearest"
            )
            if sample_point_sd["monthly_mean"].values.sum() > 0:
                break
        plots.plot_gcm_anomaly(
            gcm_processed_data_path, sample_lat, sample_lon, save_path=figures_dir
        )
        plots.plot_gmb_anomaly(
            gmb_processed_data_path, sample_rgi_id, save_path=figures_dir
        )
        gcm_processed.close()
        gmb_processed.close()

    else:
        logger.info("Skipping dataset processing...")
        workflow_dir = cfg.get("workflow_dir", REPORTS_DIR / workflow_name)
        if not isinstance(workflow_dir, Path):
            workflow_dir = Path(workflow_dir)
        with open(workflow_dir / "config_used.json", "r") as f:
            cfg = json.load(f)

    if run_train:
        logger.info("Running model training...")
        cfg, train_results = train.main(cfg)
        logger.info("Model training complete.")
        summary_lines.append("Model training: completed")
        # add training results to summary
        for key, value in train_results.items():
            summary_lines.append(f"{key}: {value}")
        with open(workflow_dir / "config_used.json", "w") as f:
            json.dump(cfg, f, indent=4)
    else:
        logger.info("Skipping training...")
        with open(workflow_dir / "config_used.json", "r") as f:
            cfg = json.load(f)

    if run_predict:
        logger.info("Running prediction...")
        cfg = predict.main(cfg)
        summary_lines.append("Prediction complete.")
        with open(workflow_dir / "config_used.json", "w") as f:
            json.dump(cfg, f, indent=4)

        figures_dir = Path(cfg["figures_dir"])

        # plot a sample time series of predicted vs true GMB values for a randomly selected point corresponding to an rgi_id
        gmb_interim_path = cfg["gmb_interim"]
        with xr.open_dataset(gmb_interim_path) as _gmb_interim:
            gmb_interim = _gmb_interim.load()
        gmb_pred_path = cfg["gmb_pred_path"]
        with xr.open_dataset(gmb_pred_path) as _gmb_pred:
            gmb_pred = _gmb_pred.load()
        sample_rgi_id = np.random.choice(gmb_pred["rgi_id"].values)
        sample_orig_gmb_rgi = gmb_interim.sel(rgi_id=sample_rgi_id)
        sample_pred_gmb_rgi = gmb_pred.sel(rgi_id=sample_rgi_id)
        plots.compare_pred_test_glacier(
            sample_orig_gmb_rgi,
            sample_pred_gmb_rgi,
            sample_rgi_id,
            save_path=figures_dir,
        )

        # plot a smaple month of predicted vs true GMB values for all glaciers in the test period
        sample_month = np.random.choice(gmb_pred["time"].values)
        sample_orig_gmb_month = gmb_interim.sel(time=sample_month, method="nearest")
        sample_pred_gmb_month = gmb_pred.sel(time=sample_month)
        region_ids = cfg.get("region_ids")
        plots.compare_pred_test_month(
            sample_orig_gmb_month,
            sample_pred_gmb_month,
            sample_month,
            region_ids=region_ids,
            save_path=figures_dir,
        )
        gmb_interim.close()
        gmb_pred.close()

    else:
        logger.info("Skipping prediction...")
        with open(workflow_dir / "config_used.json", "r") as f:
            cfg = json.load(f)

    if run_analysis:
        logger.info("Running Analysis...")
        cfg, analysis_results = analysis.main(cfg)
        logger.info("Analysis complete.")
        # add analysis results to summary
        for key, value in analysis_results.items():
            summary_lines.append(f"{key}: {value}")
        summary_lines.append("Analysis: completed")
        with open(workflow_dir / "config_used.json", "w") as f:
            json.dump(cfg, f, indent=4)

    with open(workflow_dir / "summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    with open(workflow_dir / "summary.json", "w") as f:
        json.dump(cfg, f, indent=4)
    logger.info("Run complete.")
    # save log
    logger.info("\n".join(summary_lines))


def main():
    """Parse command-line arguments and run the pipeline.

    Usage: python -m gmb_modeling.run [CONFIG]
    If CONFIG is omitted the default `cfg/main.json` is used.
    """
    parser = argparse.ArgumentParser(
        description="Run gmb_modeling pipeline with a JSON config file"
    )
    parser.add_argument(
        "config", nargs="?", default=str(DEFAULT_CONFIG), help="Path to JSON config file"
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    run(cfg_path)


if __name__ == "__main__":
    main()
