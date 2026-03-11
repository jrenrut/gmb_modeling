from __future__ import annotations

import argparse
import concurrent.futures
import logging
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Iterable, Union

import xarray as xr

from gmb_modeling.config import RAW_DATA_DIR

LOG = logging.getLogger("download_merra")


def _curl_cmd(
    url: str,
    out_dir: Path,
    cookies_file: Path = Path("~/.netrc").expanduser(),
) -> list[str]:
    """Build a curl command list for a single URL.

    Uses resume (`-C -`), retry, and cookie support. Caller should ensure
    `curl_path` exists on PATH or provide a full path.
    """
    out_dir = Path(out_dir)
    # derive a safe output filename from the URL (strip query string)
    filename = Path(url.split("?")[0]).name
    out_file = out_dir / filename

    # follow redirects, resume, fail on HTTP errors, use compressed transfer
    args = [
        "curl",
        "-n",
        "-L",
    ]

    # add cookie support if requested (use cookie jar for session)
    if cookies_file is not None:
        cookies_file = Path(cookies_file).expanduser()
        args += ["-c", str(cookies_file), "-b", str(cookies_file)]

    # finally add output target and url
    args += ["--output", str(out_file), "--url", str(url)]

    return args


def _run_cmd(cmd: Iterable[str]) -> tuple[str, int, str]:
    """Run a command (list) and return (cmd, returncode, stdout+stderr).

    This function does not swallow FileNotFoundError; if the invoked
    program is missing the exception will propagate so the program fails
    loudly.
    """
    cmd_list = list(cmd)
    LOG.info("Running: %s", shlex.join(cmd_list))
    proc = subprocess.run(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    out_bytes = proc.stdout or b""
    try:
        out_text = out_bytes.decode("utf-8")
    except Exception:
        out_text = out_bytes.decode(errors="replace")
    return (shlex.join(cmd_list), proc.returncode, out_text)


def download_merra(
    download_links_path: Path,
    name: str,
    parallel: int = 4,
    curl_path: str = "curl",
    cookies_file: Path = Path("~/.netrc").expanduser(),
) -> Union[Path, None]:
    """Download files listed (one URL per line) in `download_links_path` using curl."""
    out_dir = RAW_DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(download_links_path, "r", encoding="utf8") as fh:
        urls = [
            ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")
        ]

    if not urls:
        LOG.warning("No URLs found in %s", download_links_path)
        return

    # require curl presence at provided path
    curl_exec = shutil.which(curl_path)
    if curl_exec is None:
        raise RuntimeError(f"curl executable not found on PATH: {curl_path}")

    # expand cookies path if provided
    if cookies_file is not None:
        cookies_file = Path(cookies_file).expanduser()

    # build curl commands
    cmds = [
        _curl_cmd(
            url,
            out_dir,
            cookies_file=cookies_file,
        )
        for url in urls
    ]

    # run in parallel using threads (subprocess is I/O bound)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {ex.submit(_run_cmd, cmd): cmd for cmd in cmds}
        for fut in concurrent.futures.as_completed(futs):
            orig_cmd = futs[fut]
            cmd_str, rc, out = fut.result()

            # determine output file from the original command list
            out_file = None
            try:
                if "--output" in orig_cmd:
                    idx = orig_cmd.index("--output")
                    if idx + 1 < len(orig_cmd):
                        out_file = Path(orig_cmd[idx + 1])
                elif "-o" in orig_cmd:
                    idx = orig_cmd.index("-o")
                    if idx + 1 < len(orig_cmd):
                        out_file = Path(orig_cmd[idx + 1])
            except Exception:
                out_file = None

            if rc != 0:
                LOG.error("FAILED (%d): %s\nOutput:\n%s", rc, cmd_str, out)
                raise RuntimeError(
                    f"Download command failed (rc={rc}): {cmd_str}\nOutput:\n{out}"
                )

            # verify file has been written and is not trivially small
            if out_file is not None:
                if not out_file.exists():
                    LOG.error(
                        "Expected output file missing: %s\nCommand: %s\nOutput:\n%s",
                        out_file,
                        cmd_str,
                        out,
                    )
                    raise RuntimeError(
                        f"Expected output file missing: {out_file}\nCommand: {cmd_str}\nOutput:\n{out}"
                    )
                try:
                    size = out_file.stat().st_size
                except Exception as e:
                    LOG.error("Could not stat output file %s: %s", out_file, e)
                    raise
                if size < 200:
                    LOG.error(
                        "Downloaded file appears empty or too small (%d bytes): %s\nCommand: %s\nOutput:\n%s",
                        size,
                        out_file,
                        cmd_str,
                        out,
                    )
                    raise RuntimeError(
                        f"Downloaded file appears empty or too small ({size} bytes): {out_file}\nCommand: {cmd_str}\nOutput:\n{out}"
                    )

            LOG.info("SUCCESS: %s", cmd_str)
            results.append((cmd_str, rc, out))

    # summary
    n_failed = sum(1 for _c, rc, _o in results if rc != 0)
    if n_failed:
        LOG.error("%d downloads failed", n_failed)
    else:
        LOG.info("All downloads completed successfully")

    return out_dir


def combine_merra_files(out_dir: Path) -> None:
    """Combine all downloaded MERRA files in `out_dir` into a single file named `{name}.nc`."""
    out_name = out_dir / f"{out_dir.name}_combined.nc"
    LOG.info("Combining MERRA files in %s into %s.nc", out_dir, out_name)
    files = sorted(out_dir.glob("MERRA2_*.nc4*"))
    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
    ds.to_netcdf(out_name, format="NETCDF4")


def _cli():
    parser = argparse.ArgumentParser(
        description="Download MERRA files using curl from a links file"
    )
    parser.add_argument(
        "links", type=Path, help="Path to a text file with one URL per line"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="merra_combined",
        help="Name for combined output file (without extension)",
    )
    parser.add_argument(
        "--parallel", type=int, default=4, help="Number of parallel downloads"
    )
    parser.add_argument(
        "--cookies", type=Path, help="Path to cookies file to load/save (optional)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    out_dir = download_merra(
        args.links,
        name=args.name,
        parallel=args.parallel,
        cookies_file=args.cookies,
    )

    assert out_dir is not None, "Download did not complete successfully"

    combine_merra_files(out_dir)


if __name__ == "__main__":
    _cli()
