# Research Code for the thought virus research paper

This a Work-in-progress research code repository meant to hold each experiement.


## Setup

This repo uses [git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=mac) for large files. Install git-lfs and make sure you pull the required files when cloning this repo

```sh
# clone with lfds
git lfs clone https://github.com/Multi-Agent-Security-Initiative/thought_virus.git

# or if you already cloned this repo, then just
cd thought_virus
git lfs pull
```
-----
This repo use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage python dependencies

Use `uv sync` to install depeendices and create the virual environement 

```sh
uv sync
```


## Code Structure

Put individual experiments in `${workspaceRoot}/runs/\d+`

- runs/1
- runs/2

etc

with shared code in `${workspaceRoot}/shared`

## Run Structure

Each run represents a repeatable experiment, follow these guidelines to keep things organized/repeatable. See `runs/1` for an example.

- Each run add a Readme.md to explain the purpose of the experiment, what the expected input/output should be etc.
- Each run should include a `run_config.py` which specifices run specfiic information, (like this is run 1). This is so that if you want to modify a run to test something else, you can just copy/paste the folder and everything will work.
- runs may refere to past runs, but should not refer to future runs (ie, if you need to re-run something due to changed code in a future run, copy/paste the old run ).


## Other notes

Try not to make any breaking changes in `shared` because that could affect past runs. Instead, copy/paste, files/folders. Storage space is cheap, time is not!