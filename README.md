# gmb_modeling

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Exploring dynamic downscaling of climate models and correlating regional snow variables with winter alpine snowpack and glacier mass balance.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         gmb_modeling and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── gmb_modeling   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes gmb_modeling a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## MERRA-2 Data Download Instructions

*You will need a free [NASA Earthdata](https://forum.earthdata.nasa.gov/) account, and authentication stored in a cookie file like ~/.netrc or ~/.usr-cookies*

Data accessed through [NASA GES DISC](https://disc.gsfc.nasa.gov/datasets?page=1&subject=Snow%2FIce&project=MERRA-2) - MERRA-2 filtered for "Snow/Ice" subjects.

I have used the monthly mean [Land Ice Surface Diagnostics](https://disc.gsfc.nasa.gov/datasets/M2TMNXGLC_5.12.4/summary) dataset.

Click on "Subset / Get Data" on the right.

Download method: "Get File Subsets using OPeNDAP".

British Columbia lat/lon bounds: -139.06,48.3,-114.03,60.

Select variables and then click "Get Data".

The service will select all the chips with your data and create separate download links. Click "Download Links List" - it will be a .txt file. Put this in "./data/raw/merra_download_links/"

Now run `python data/raw/merra_download_links/download_merra.py --links <path-to-download-links.txt> --cookies <path-to-cookies> --name <name-of-subset>`

There will be a new directory in "./data/raw/" with the `--name` argument, containing a combined .nc dataset file.