# Cryptic Pockets Database

A pipeline for analyzing protein structures to identify cryptic pockets.

## Installation

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate database
```

Before running the pipeline, set the `FETCH_PATH` in `monomer_calcs.py` to your local directory for CIF files.

## Usage

Run the pipeline:
```bash
python monomer_calcs.py [options]
```

Options:
- `--n_jobs`: Number of parallel jobs (default: 16)
- `--input_list`: Use predefined PDB IDs
- `--update_previous`: Update existing data
- `--max_res`: Maximum resolution threshold in Å (default: 2.5)
- `--custom_folder`: Custom data folder (default: data)

## Project Structure

- `monomer_calcs.py`: Main pipeline for data collection and processing
- `scoring_function.py`: Scoring system for pocket analysis
- `get_sites.py`: Identifies and clusters binding sites
- `refine_smiles.py`: Processes and refines ligand information

## Data Organization

- `data/`: Main data directory
  - `monomer_calcs/`: Processed structure data
  - `xyz_files/`: Structure coordinates
  - `xyz_files_local/`: Local structure coordinates
  - `checkpoints/`: Processing checkpoints
