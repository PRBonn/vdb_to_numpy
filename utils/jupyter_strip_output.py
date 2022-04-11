#!/usr/bin/env python3
from pathlib import Path

import git
import nbconvert
import nbformat
import click
import os


def get_staged_notebooks(repo):
    notebooks = []
    for staged_file in repo.index.diff("HEAD"):
        if staged_file.a_path.endswith(".ipynb"):
            notebooks.append(Path(staged_file.a_path))
    return notebooks


def stage_notebook(repo, path):
    repo.git.add(path.absolute())


def clean_notebook_output(nb_path):
    print("Clean {}".format(nb_path.name))
    ep = nbconvert.preprocessors.ClearOutputPreprocessor(timeout=6000)

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
    try:
        ep.preprocess(nb, {"metadata": {"path": nb_path.parent}})
    except nbconvert.preprocessors.execute.CellExecutionError:
        print("Cleaning of {} failed".format(nb_path.name))
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


@click.command()
@click.argument("notebook", required=False, type=click.Path(exists=True))
def main(notebook):
    """
    Small utility to cleanup the output of a Jupyter notebook. If no argument
    is provided it will look for all staged notebooks in the current git
    repository.
    """
    if notebook:
        clean_notebook_output(Path(notebook))
    else:
        repo = git.Repo(__file__, search_parent_directories=True)
        for nb_path in get_staged_notebooks(repo):
            if os.path.exists(nb_path):
                clean_notebook_output(nb_path)
                stage_notebook(repo, nb_path)


if __name__ == "__main__":
    main()
