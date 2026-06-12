#!/usr/bin/env python3
"""Execute a notebook headlessly with nbclient and write the executed copy.

Usage:
    LEJEPA_QUICK=1 python tools/run_notebook.py NOTEBOOK.ipynb --out EXECUTED.ipynb

The kernel working directory is pinned to the project root (the parent of this
tools/ directory) so results/ always lands next to the notebook regardless of
where the script is invoked from. timeout=None because the FULL sweep cell can
legitimately run for a long time on CPU.
"""
import argparse
import pathlib

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook")
    ap.add_argument("--out", required=True, help="path for the executed copy")
    args = ap.parse_args()

    nb = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbformat.write(nb, args.out)
    print(f"executed OK -> {args.out}")


if __name__ == "__main__":
    main()
