# CryptoBank

CryptoBank is a structural bioinformatics pipeline for building holo/apo protein pairs and scoring ligand-associated conformational changes to identify cryptic pockets.

The main workflow:

- queries the RCSB PDB for holo and apo structures
- filters out non-relevant ligands with curated exclusion lists
- pairs holo and apo entries through protein clustering metadata
- aligns structures in PyMOL and exports XYZ representations
- computes pocket-opening scores with the local scoring model
- optionally adds local AlphaFold structures as extra apo candidates based on UniProt accession

## Repository Contents

- `monomer_calcs.py`: main end-to-end pipeline
- `scoring_function.py`: feature extraction and scoring code
- `get_sites.py`: site clustering and centroid generation
- `refine_smiles.py`: ligand cleanup and local ligand-property utilities
- `data_query.json`: GraphQL field selection for RCSB metadata queries
- `exclusion_lists/`: curated ligand exclusion tables
- `3seg_distmodel_parameters.npy`, `weights1seg.pkl`: model assets used by the scoring function

## Requirements

- Linux or another environment where the `database` conda environment can be created successfully
- Conda
- network access to the RCSB APIs and structure downloads
- enough local disk space for downloaded CIF files, temporary XYZ files, and output pickles

The supplied environment includes the main runtime dependencies, including PyMOL, MDAnalysis, RDKit, pandas, scikit-learn, and the RCSB API client.

## Installation

Create the environment and activate it:

```bash
conda env create -f environment.yml
conda activate database
```

## Local Data Paths

Two local structure stores matter for normal use:

1. RCSB CIF cache for experimental structures
2. AlphaFold CIF directory for optional apo expansion

### RCSB CIF cache

By default the pipeline resolves the experimental structure cache relative to the repository:

```text
../../pdb/cif_files
```

You can override that location without editing the code:

```bash
export FETCH_PATH=/path/to/cif_files
```

If `FETCH_PATH` is not set, the default relative location above is used.

The same `FETCH_PATH` setting is used by the helper scripts that fetch experimental structures.

### AlphaFold apo structures

If local AlphaFold models are available, the pipeline automatically adds them as extra apo candidates by matching holo `uniprot_id` values.

By default the pipeline resolves the AlphaFold directory relative to the repository:

```text
../../pdb/alphafold
```

You can override that location with:

```bash
export ALPHAFOLD_DIR=/path/to/alphafold_models
```

Expected filenames are:

```text
AF-<UNIPROT>-F1-model_v*.cif
```

If more than one local AlphaFold version is present for the same UniProt accession, the pipeline always uses the highest available `model_v*` file.

AlphaFold files are not downloaded automatically by this repository. If a UniProt accession has no corresponding local AlphaFold CIF, that holo will simply use the experimental apo pool only.

## Running the Pipeline

The main entry point is:

```bash
python monomer_calcs.py [options]
```

Available CLI options:

- `--n_jobs`: number of parallel jobs for structure processing, default `16`
- `--input_list`: use a predefined holo list instead of querying all eligible structures
- `--update_previous`: refresh downloaded data and merge previous scores into the new `combined.p`
- `--max_res`: maximum crystallographic resolution threshold in angstroms, default `2.5`
- `--custom_folder`: output root for a run, default `data`

When `--input_list` is enabled:

- `holo_list.p` is required in the current working directory
- `apo_list.p` is optional
- if `apo_list.p` is absent, the pipeline still runs using apo candidates derived from filtered holo entries plus any matching local AlphaFold structures

### Examples

Default run:

```bash
python monomer_calcs.py
```

Run into a separate output folder:

```bash
python monomer_calcs.py --custom_folder run_2026_03_18
```

Small validation run with a stricter resolution cutoff and one worker:

```bash
python monomer_calcs.py --max_res 0.8 --n_jobs 1 --custom_folder smoke_test
```

Note: values of `--max_res` below `0.83` are clamped to `0.83` internally.

## What the Pipeline Produces

Under `<custom_folder>/monomer_calcs/` the main outputs are:

- `combined.p`: master dataframe containing holo/apo pairs and computed scores
- `all_holos.p`: filtered holo structures kept for pairing
- `all_apos.p`: apo candidates kept for pairing, including AlphaFold rows when available
- `holo_2.p`: raw downloaded holo metadata table
- `apo_2.p`: raw downloaded apo metadata table

Additional working directories are created under `<custom_folder>/`:

- `xyz_files/`: global-alignment XYZ exports and temporary PyMOL sessions
- `xyz_files_local/`: local-alignment XYZ exports
- `monomer_calcs/structure_dfs/`: structure-level work splits
- `monomer_calcs/checkpoints/`: structure-level result checkpoints
- `rejects/`: auxiliary output folder reserved by the pipeline

## Input Sources and Pairing Logic

The pipeline combines several information sources:

- RCSB Search API for structure ID discovery
- RCSB GraphQL data endpoint for structure metadata
- curated exclusion lists in `exclusion_lists/`
- PyMOL structure alignment for holo/apo comparison
- local AlphaFold CIF files keyed by UniProt accession

Holo and apo structures are paired through `cluster_id_95` membership after ligand filtering. When an AlphaFold model exists for a holo UniProt accession, an additional apo row is appended with `structureId_apo = AF-<UNIPROT>-F1-model_vX`.

## Notes and Caveats

- `--input_list` expects a local `holo_list.p` file in the current working directory.
- The pipeline performs network downloads and can take substantial time on full runs.
- PyMOL alignment and XYZ export make the workflow relatively I/O-heavy.
- The repository currently uses pickle files as its main storage format.
- Some helper scripts assume a local structure mirror or local scratch directories; `monomer_calcs.py` is the primary supported entry point in this repository state.

## Tested State

The current repository state was validated with:

```bash
conda run -n database python monomer_calcs.py --max_res 0.8 --n_jobs 1 --custom_folder tmp_max_res_08_af_v6
```

That test completed successfully and confirmed that local AlphaFold models were added as apo structures and scored alongside experimental apo entries.
