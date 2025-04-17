#!/usr/bin/env python3
import os
import json
import pickle
import urllib.parse
import urllib.request
import shutil
import argparse
from datetime import datetime
from joblib import Parallel, delayed
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import glob
import asyncio
import aiohttp
from pymol import cmd
from rcsbapi.search import search_attributes as attrs


# Custom module imports (adjust these paths as needed)
from scoring_function import *



FETCH_PATH = '../../pdb/cif_files/'
# -----------------------------------------------------------------------------
# GLOBAL PARAMETERS & DATA FOLDER SETUP
# -----------------------------------------------------------------------------
# Global variable for custom data folder (default is "data", can be overridden by --custom_folder)
DATA_ROOT = "data"  

# The following globals will be set after command-line parsing:
STRUCTURE_DFS_DIR = None
CHECKPOINT_DIR = None
MASTER_CHECKPOINT_FILE = None

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def ensure_dir(directory: str) -> None:
    """
    Ensures that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Path to the directory to ensure exists
        
    Returns:
        None
    """
    os.makedirs(directory, exist_ok=True)

def warn(*args, **kwargs):
    """
    A no-op warning function used to suppress warnings.
    
    Args:
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
        
    Returns:
        None
    """
    pass

import warnings
warnings.warn = warn

# -----------------------------------------------------------------------------
# DATA FETCHING & PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------
def get_ids(max_res: float = 2.5) -> (List[str], List[str]):
    """
    Retrieves PDB IDs for holo and apo structures based on resolution criteria.
    
    Args:
        max_res (float): Maximum resolution threshold in Angstroms (default: 2.5)
        
    Returns:
        tuple: A tuple containing two lists:
            - List of holo structure PDB IDs
            - List of apo structure PDB IDs
    """
    # Use the provided resolution threshold instead of interactive input
    if max_res < 0.83:
        max_res = 0.83
        print("Threshold too low, using 0.83 Å instead")
    print("### Using", max_res, "Å as threshold for resolution ###")
    q_resolution = attrs.rcsb_entry_info.resolution_combined <= max_res
    q_holo = attrs.rcsb_entry_info.nonpolymer_entity_count >= 1
    q_apo = attrs.rcsb_entry_info.nonpolymer_entity_count == 0
    cluster_id = attrs.rcsb_cluster_membership.cluster_id >= 0

    query_holos = q_holo & q_resolution & cluster_id
    print('  Starting the get_ids for holos')
    holo_ids = list(query_holos())
    print('  Finished get_ids for holos')
    print(f"  {'Initial no. of Holo Structures:':<50}{len(holo_ids):,} H")

    query_apos = q_apo & q_resolution & cluster_id
    print('  Starting the get_ids for apos')
    apo_ids = list(query_apos())
    print('  Finished get_ids for apos')
    print(f"  {'Initial no. of Apo Structures:':<50}{len(apo_ids):,} A")
    return holo_ids, apo_ids

def process_targets(targets: List[Dict]) -> Dict:
    """
    Processes target information from a list of target dictionaries.
    
    Args:
        targets (List[Dict]): List of target dictionaries containing information about
                            protein targets, their interaction types, and sources.
                            
    Returns:
        Dict: A dictionary containing processed target information including:
            - target_names: Pipe-separated list of target names
            - target_interaction_types: Pipe-separated list of interaction types
            - target_uniprots: Pipe-separated list of UniProt IDs
            - target_sources: Pipe-separated list of source databases
            - target_count: Number of unique targets
    """
    if not targets:
        return {
            'target_names': None,
            'target_interaction_types': None,
            'target_uniprots': None,
            'target_sources': None,
            'target_count': 0
        }
    names = set()
    interaction_types = set()
    uniprots = set()
    sources = set()
    for target in targets:
        if target.get('name'):
            names.add(target['name'])
        if target.get('interaction_type'):
            interaction_types.add(target['interaction_type'])
        if target.get('reference_database_accession_code'):
            uniprots.add(target['reference_database_accession_code'])
        if target.get('provenance_source'):
            sources.add(target['provenance_source'])
    return {
        'target_names': '|'.join(sorted(names)) if names else None,
        'target_interaction_types': '|'.join(sorted(interaction_types)) if interaction_types else None,
        'target_uniprots': '|'.join(sorted(uniprots)) if uniprots else None,
        'target_sources': '|'.join(sorted(sources)) if sources else None,
        'target_count': len(names)
    }

def safe_get_inchi(nonpolymer: dict) -> Optional[str]:
    """
    Safely extracts the InChI identifier from a nonpolymer entity dictionary.
    
    Args:
        nonpolymer (dict): Dictionary containing nonpolymer entity information
        
    Returns:
        Optional[str]: The InChI identifier if found, None otherwise
    """
    try:
        comp = nonpolymer.get('nonpolymer_comp') or {}
        descriptor = comp.get('rcsb_chem_comp_descriptor')
        if isinstance(descriptor, dict):
            return descriptor.get('InChI')
        return None
    except Exception:
        return None

def extract_protein_data_from_pickle(data: dict, apo: bool = False) -> pd.DataFrame:
    """
    Extracts protein data from a pickle file and converts it to a pandas DataFrame.
    
    Args:
        data (dict): Dictionary containing protein data from pickle file
        apo (bool): Whether the data represents apo structures (default: False)
        
    Returns:
        pd.DataFrame: DataFrame containing extracted protein data with columns for:
            - rcsb_id: PDB ID
            - resolution: Structure resolution
            - sequence: Protein sequence
            - and other relevant protein information
    """
    entries = data['data']['entries']
    all_rows = []
    for entry in entries:
        refine_data = entry.get('refine', [])
        refine_dict = refine_data[0] if refine_data else {}

        # Fix the resolution handling
        resolution_combined = entry.get('rcsb_entry_info', {}).get('resolution_combined')
        resolution = resolution_combined[0] if isinstance(resolution_combined, list) else None
        
        basic_info = {
            'rcsb_id': entry.get('rcsb_id'),
            'initial_release_date': entry.get('rcsb_accession_info', {}).get('initial_release_date'),
            'revision_date': entry.get('rcsb_accession_info', {}).get('revision_date'),
            'resolution': resolution,  # Use the safely extracted resolution
            'polymer_composition': entry.get('rcsb_entry_info', {}).get('polymer_composition'),
            'branched_entity_count': entry.get('rcsb_entry_info', {}).get('branched_entity_count'),
            'nonpolymer_entity_count': entry.get('rcsb_entry_info', {}).get('nonpolymer_entity_count'),
            'polymer_entity_count': entry.get('rcsb_entry_info', {}).get('polymer_entity_count'),
            'ls_R_factor_R_work': refine_dict.get('ls_R_factor_R_work'),
            'ls_R_factor_obs': refine_dict.get('ls_R_factor_obs'),
            'ls_d_res_high': refine_dict.get('ls_d_res_high')
        }
        for polymer in entry.get('polymer_entities', []) or []:
            chain_info = {
                'chain_id': polymer.get('entity_poly', {}).get('pdbx_strand_id'),
                'sequence': polymer.get('entity_poly', {}).get('pdbx_seq_one_letter_code'),
                'sequence_length': polymer.get('entity_poly', {}).get('rcsb_sample_sequence_length'),
                'auth_asym_ids': ','.join(polymer.get('rcsb_polymer_entity_container_identifiers', {}).get('auth_asym_ids', []) or []),
                'asym_ids': ','.join(polymer.get('rcsb_polymer_entity_container_identifiers', {}).get('asym_ids', []) or []),
            }
            if polymer.get('uniprots'):
                chain_info['uniprot_id'] = polymer['uniprots'][0].get('rcsb_uniprot_container_identifiers', {}).get('uniprot_id')
            ref_seq = polymer.get('rcsb_polymer_entity_container_identifiers', {}).get('reference_sequence_identifiers', [])
            if ref_seq and isinstance(ref_seq, list) and len(ref_seq) > 0:
                chain_info['uniprot_accession'] = ref_seq[0].get('database_accession')
            if polymer.get('rcsb_cluster_membership'):
                for cluster in polymer['rcsb_cluster_membership']:
                    chain_info[f'cluster_id_{cluster["identity"]}'] = cluster['cluster_id']
            base_row = {**basic_info, **chain_info}
            chain_ligands = False
            for nonpolymer in entry.get('nonpolymer_entities', []) or []:
                ligand_instances = nonpolymer.get('nonpolymer_entity_instances', []) or []
                chain_instances = [inst for inst in ligand_instances 
                                   if inst.get('rcsb_nonpolymer_entity_instance_container_identifiers', {}).get('auth_asym_id') 
                                   in chain_info['auth_asym_ids'].split(',')]
                if chain_instances:
                    chain_ligands = True
                    nonpolymer_comp = nonpolymer.get('nonpolymer_comp') or {}
                    chem_comp = nonpolymer_comp.get('chem_comp') or {}
                    ligand_info = {
                        'ligand_id': nonpolymer.get('pdbx_entity_nonpoly', {}).get('comp_id'),
                        'ligand_name': nonpolymer.get('rcsb_nonpolymer_entity', {}).get('pdbx_description'),
                        'ligand_formula_weight': nonpolymer.get('rcsb_nonpolymer_entity', {}).get('formula_weight'),
                        'ligand_number_of_molecules': nonpolymer.get('rcsb_nonpolymer_entity', {}).get('pdbx_number_of_molecules'),
                        'ligand_type': chem_comp.get('type'),
                        'ligand_inchi': safe_get_inchi(nonpolymer),
                        'ligand_drugbank_id': nonpolymer_comp.get('rcsb_chem_comp_container_identifiers', {}).get('drugbank_id')
                    }
                    smiles_descriptors = nonpolymer_comp.get('pdbx_chem_comp_descriptor', []) or []
                    smiles_openeye = next(
                        (desc.get('descriptor') for desc in smiles_descriptors 
                         if desc.get('type') == 'SMILES_CANONICAL' and 
                            desc.get('program') == 'OpenEye OEToolkits'),
                        None
                    )
                    ligand_info['ligand_smiles_openeye'] = smiles_openeye
                    smiles_cactvs = next(
                        (desc.get('descriptor') for desc in smiles_descriptors 
                         if desc.get('type') == 'SMILES_CANONICAL' and 
                            desc.get('program') == 'CACTVS'),
                        None
                    )
                    ligand_info['ligand_smiles_CACTVS'] = smiles_cactvs
                    for instance in chain_instances:
                        container = instance.get('rcsb_nonpolymer_entity_instance_container_identifiers', {})
                        instance_info = {
                            'auth_seq_id': container.get('auth_seq_id'),
                            'auth_asym_id': container.get('auth_asym_id')
                        }
                        targets = nonpolymer_comp.get('rcsb_chem_comp_target', []) or []
                        target_info = process_targets(targets)
                        complete_row = {**base_row, **ligand_info, **instance_info, **target_info}
                        all_rows.append(complete_row)
            if not chain_ligands:
                all_rows.append(base_row)

    df = pd.DataFrame(all_rows)
    for col in ['initial_release_date', 'revision_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    if not apo:
        if "auth_seq_id" in df.columns:
            df["auth_seq_id"] = pd.to_numeric(df["auth_seq_id"], errors="coerce")
            df = df[df["auth_seq_id"].notna()]
            df["auth_seq_id"] = df["auth_seq_id"].astype(int)
            df.rename(columns={"auth_seq_id": "resid"}, inplace=True)
    else:
        df.rename(columns={"auth_asym_id": "resid"}, inplace=True)
    return df

def get_data_2(pdb_list: List[str], apos: bool) -> Optional[pd.DataFrame]:
    """
    Fetches protein data for a list of PDB IDs from the RCSB database.
    
    Args:
        pdb_list (List[str]): List of PDB IDs to fetch data for
        apos (bool): Whether to fetch apo structure data
        
    Returns:
        Optional[pd.DataFrame]: DataFrame containing protein data if successful,
                              None if the fetch operation fails
    """
    pdbs_str = '","'.join(pdb_list)
    with open("./data_query.json", "r") as json_file:
        json_as_str = json_file.read().replace(" ", "")
    json_as_str = json_as_str.replace("{\n", "{").replace("\n}", "}").replace("\n", ",")[:-1]
    url_stem = 'https://data.rcsb.org/graphql?'
    url_query = f'query={{entries(entry_ids:["{pdbs_str}"]){json_as_str}}}'
    url_query = urllib.parse.quote(url_query, safe=':()=')
    data_url = url_stem + url_query
    response = urllib.request.urlopen(data_url)
    data = json.loads(response.read())
    df = extract_protein_data_from_pickle(data, apos)
    return df

def optimize_dataframe(df):
    """
    Optimizes a pandas DataFrame by reducing memory usage through type conversion.
    
    Args:
        df (pd.DataFrame): DataFrame to optimize
        
    Returns:
        pd.DataFrame: Optimized DataFrame with reduced memory usage
    """
    for col in df.columns:
        col_type = df[col].dtype
        try:
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                continue
        except:
            pass
        if col_type == object:
            try:
                if df[col].nunique() / len(df[col]) < 0.5:
                    df[col] = df[col].astype('category')
            except:
                pass
        elif col_type in ['int64', 'int32']:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type in ['float64', 'float32']:
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_datetime64_any_dtype(col_type):
            pass
    return df

def download_data(input_list: bool, max_res: float = 2.5, update_previous: bool = False) -> None:
    """
    Downloads and processes protein structure data from the RCSB database.
    
    Args:
        input_list (bool): Whether to use a predefined list of PDB IDs
        max_res (float): Maximum resolution threshold in Angstroms (default: 2.5)
        update_previous (bool): Whether to update existing data (default: False)
        
    Returns:
        None
        
    Side Effects:
        - Downloads PDB files
        - Creates/updates data files in the data directory
        - Processes and stores protein structure information
    """
    holo_path = os.path.join(DATA_ROOT, "monomer_calcs", "holo_2.p")
    apo_path = os.path.join(DATA_ROOT, "monomer_calcs", "apo_2.p")
    if os.path.exists(holo_path) and os.path.exists(apo_path) and not update_previous:
        holo_2 = pickle.load(open(holo_path, "rb"))
        apo_2 = pickle.load(open(apo_path, "rb"))
    else:
        if not input_list:
            holo_ids, apo_ids = get_ids(max_res)
        else:
            holo_ids = pd.read_pickle("holo_list.p")
            #apo_ids = pd.read_pickle("apo_list.p")
            #holo_ids = ["2HS1", "6IXD"]
            apo_ids = ["3DVD", "3DJK"]
            #holo_ids = holo_ids + apo_ids
            #holo_ids = list(set(holo_ids))
            #apo_ids = ["3DVD", "6SF4"]
            print(holo_ids)
            if len(holo_ids[0]) == 4:
                pass
            elif len(holo_ids[0]) == 6:
                holo_ids = [id[:4] for id in holo_ids]
                holo_ids = list(set(holo_ids))
            else:
                raise ValueError("Holo ids should be 4 or 6 characters long")
        n = 1000
        holo_chunks = [holo_ids[i * n:(i + 1) * n] for i in range((len(holo_ids) + n - 1) // n)]
        apo_chunks = [apo_ids[i * n:(i + 1) * n] for i in range((len(apo_ids) + n - 1) // n)]
        print('  Starting downloading data for holos')
        output_2 = Parallel(n_jobs=16)(delayed(get_data_2)(chunk, False) for chunk in holo_chunks)
        df_list_2 = [df for df in output_2 if df is not None]
        holo_2 = pd.concat(df_list_2, ignore_index=True)
        ensure_dir(os.path.join(DATA_ROOT, "monomer_calcs"))
        pickle.dump(holo_2, open(holo_path, "wb"))
        print('  Starting downloading data for apos')
        output_2 = Parallel(n_jobs=16)(delayed(get_data_2)(chunk, True) for chunk in apo_chunks)
        df_list_2 = [df for df in output_2 if df is not None]
        apo_2 = pd.concat(df_list_2, ignore_index=True)
        pickle.dump(apo_2, open(apo_path, "wb"))
    combined_df = exlude_not_ligands(holo_2, apo_2)
    combined_path = os.path.join(os.path.dirname(combined_df.iloc[0]['rcsb_id']), DATA_ROOT, "monomer_calcs")
    ensure_dir(combined_path)
    combined_df = optimize_dataframe(combined_df)
    combined_reconstructed = pd.DataFrame(combined_df.values, columns=combined_df.columns)
    with open(os.path.join(combined_path, "combined.p"), "wb") as f:
        pickle.dump(combined_reconstructed, f)

def get_asym(row, apo: bool = False):
    """
    Maps auth_asym_ids to asym_ids based on the chain in structureId.
    
    Args:
        row (pandas.Series): A row from the dataframe containing structureId, auth_asym_ids, and asym_ids
        apo (bool): Whether to use apo-specific column names (default: False)
        
    Returns:
        str: The corresponding asym_id for the chain in structureId, or None if mapping can't be determined
    """
    # Determine which fields to use based on apo parameter
    if apo:
        structure_id_field = 'structureId_apo'
        auth_asym_ids_field = 'auth_asym_ids_apo'
        asym_ids_field = 'asym_ids_apo'
    else:
        structure_id_field = 'structureId'
        auth_asym_ids_field = 'auth_asym_ids'
        asym_ids_field = 'asym_ids'
    
    # Extract the chain ID from structureId (everything after the underscore)
    if '_' not in str(row[structure_id_field]):
        return None
    
    chain = row[structure_id_field].split('_')[1]
    
    # Check if auth_asym_ids and asym_ids are present and not NaN
    if pd.isna(row[auth_asym_ids_field]) or pd.isna(row[asym_ids_field]):
        return None
    
    # Split the comma-separated chains
    auth_chains = row[auth_asym_ids_field].split(',')
    asym_chains = row[asym_ids_field].split(',')
    
    # Make sure both lists have the same length
    if len(auth_chains) != len(asym_chains):
        return None
    
    # Find the position of the chain in auth_asym_ids
    try:
        index = auth_chains.index(chain)
        # Return the corresponding chain from asym_ids
        return asym_chains[index]
    except ValueError:
        # If the chain is not found in auth_asym_ids
        return None

# -----------------------------------------------------------------------------
# LIGAND EXCLUSION FUNCTION
# -----------------------------------------------------------------------------
def exlude_not_ligands(holos: pd.DataFrame, apos: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out non-ligand entities from the protein structure data.
    
    Args:
        holos (pd.DataFrame): DataFrame containing holo structure data
        apos (pd.DataFrame): DataFrame containing apo structure data
        
    Returns:
        pd.DataFrame: Combined DataFrame with non-ligand entities excluded
        
    Note:
        Uses predefined exclusion lists for elements, ions, low mass ligands, and solvents
    """
    elements_df = pd.read_csv('./exclusion_lists/elements.csv', sep="|").replace(np.nan, "NA")
    ions_df = pd.read_csv('./exclusion_lists/ions_curated.csv', sep="|").replace(np.nan, "NA")
    low_mass_df = pd.read_csv('./exclusion_lists/low_mass_ligands_curated.csv', sep="|").replace(np.nan, "NA")
    solvents_df = pd.read_csv('./exclusion_lists/solvents.csv', sep="|").replace(np.nan, "NA")
    exclusion_list = pd.concat([elements_df, ions_df, low_mass_df, solvents_df], ignore_index=True)
    exclusion_list = exclusion_list.drop(columns=["ligandName", "ligandSmiles", "ligandMolecularWeight"])
    holos_na = holos[holos["ligand_id"].isna()].copy()
    holos_na["auth_asym_id"] = None
    holos_na["auth_asym_ids"] = holos_na["auth_asym_ids"].str.split(",")
    holos_na = holos_na.explode("auth_asym_ids")
    holos_na["auth_asym_id"] = holos_na["auth_asym_ids"]
    holos_not_na = holos[holos["ligand_id"].notna()]
    holos_merged = pd.concat([holos_not_na, holos_na], ignore_index=True)
    holos_merged["holo_and_chain"] = holos_merged["rcsb_id"] + "_" + holos_merged["auth_asym_id"]
    holos_merged.loc[holos_merged["holo_and_chain"].isna(), "holo_and_chain"] = (
        holos_merged.loc[holos_merged["holo_and_chain"].isna(), "rcsb_id"] +
        "_" +
        holos_merged.loc[holos_merged["holo_and_chain"].isna(), "auth_asym_ids"]
    )
    holos_merged = holos_merged.drop_duplicates()
    holos_merged["exclusion_list"] = holos_merged["ligand_id"].isin(exclusion_list["ligandId"])
    holos_exclusion_list = holos_merged[((holos_merged["exclusion_list"] == True) |
                                           (holos_merged["ligand_formula_weight"] < 0.06)) |
                                           (holos_merged["ligand_id"].isna())]
    holos_not_exclusion_list = holos_merged[(holos_merged["exclusion_list"] == False) &
                                              (holos_merged["ligand_formula_weight"] >= 0.06) &
                                              (holos_merged["ligand_id"].notna())]
    holos_new_apo = holos_exclusion_list[~holos_exclusion_list["holo_and_chain"].isin(holos_not_exclusion_list["holo_and_chain"])]
    cols_to_drop = [col for col in holos_new_apo.columns if col.startswith("ligand_") or
                                                 col.startswith("target_") or
                                                 col.startswith("holo_and_chain") or
                                                 col.startswith("exclusion_list") or
                                                 col.startswith("auth_seq_id")]
    holos_new_apo = holos_new_apo.drop(columns=cols_to_drop)
    holos_na_new_apo = holos_not_exclusion_list[holos_not_exclusion_list["ligand_id"].isna()]
    holos_definitive = holos_not_exclusion_list[holos_not_exclusion_list["ligand_id"].notna()]
    apos["auth_asym_id"] = apos["auth_asym_ids"].str.split(",")
    apos = apos.explode("auth_asym_id")
    all_apos = pd.concat([holos_new_apo, apos], ignore_index=True).add_suffix("_apo")
    all_apos["holo_and_chain_apo"] = all_apos["rcsb_id_apo"] + "_" + all_apos["auth_asym_id_apo"]
    holos_definitive = holos_definitive[holos_definitive["cluster_id_95"].notna()]
    holos_definitive = holos_definitive[holos_definitive["uniprot_id"].notna()]
    holos_definitive["cluster_id_95"] = holos_definitive["cluster_id_95"].astype(int)
    all_apos = all_apos[all_apos["cluster_id_95_apo"].notna()]
    all_apos = all_apos[all_apos["uniprot_id_apo"].notna()]
    intersection_holos_apos = holos_definitive[holos_definitive["cluster_id_95"].isin(all_apos["cluster_id_95_apo"])]
    intersection_apos_holos = all_apos[all_apos["cluster_id_95_apo"].isin(holos_definitive["cluster_id_95"])]
    with open(os.path.join(DATA_ROOT, "monomer_calcs", "all_holos.p"), "wb") as f:
        pickle.dump(holos_definitive, f)
    with open(os.path.join(DATA_ROOT, "monomer_calcs", "all_apos.p"), "wb") as f:
        pickle.dump(all_apos, f)
    print("Merging intersection holos and apos")
    combined_df = pd.merge(
        intersection_holos_apos,
        intersection_apos_holos,
        left_on='cluster_id_95',
        right_on='cluster_id_95_apo',
        how='left',
        suffixes=('_holos', '_apos')
    )
    combined_df.rename(columns={
        "cluster_id_95": "clusterNumber95",
        "cluster_id_95_apo": "clusterNumber95_apo",
        "holo_and_chain": "structureId",
        "holo_and_chain_apo": "structureId_apo",
        "ligand_id": "ligandId"
    }, inplace=True)
    for col in ['scoreDate', 'linear_score_3', 'linear_score_3_segments', 'rmsd',
                'linear_score_local_3', 'linear_score_3_segments_local', 'rmsd_local', 
                'linear_score_3_segments', 'linear_score_3_segments_local', 'alt_removed_ligand', 'alt_removed_holo', 
                'alt_removed_apo','ligand_contacts', "holo_full_seq", "apo_full_seq", "holo_lig_seq", "apo_lig_seq",
                'linear_score_1', 'linear_score_local_1', 'lig_around_apo_1', 'lig_around_holo_1',
                'lig_around_apo_3', 'lig_around_holo_3', 'lig_around_apo_local_1',
                'lig_around_holo_local_1', 'lig_around_apo_local_3', 'lig_around_holo_local_3']:
        combined_df[col] = np.nan
    combined_df = combined_df.drop(columns=['resid_apo'])
    combined_df = combined_df.drop_duplicates(keep='first')
    print("Sorting values")
    combined_df = combined_df.sort_values(by=['clusterNumber95', 'structureId', 'ligandId', 'resid', 'structureId_apo'])
    combined_df.reset_index(drop=True, inplace=True)
    combined_df['asym'] = combined_df.apply(get_asym, axis=1, apo = False)
    combined_df['asym_apo'] = combined_df.apply(get_asym, axis=1, apo = True)
    return combined_df

# -----------------------------------------------------------------------------
# SEQUENCE AROUND LIGAND FUNCTION
# -----------------------------------------------------------------------------
def get_center_of_mass(selection: str) -> tuple:
    """
    Calculates the center of mass for a given PyMOL selection.
    
    Args:
        selection (str): PyMOL selection string
        
    Returns:
        tuple: (x, y, z) coordinates of the center of mass
        
    Raises:
        ValueError: If no atoms are found in the selection
    """
    model = cmd.get_model(selection)
    if not model.atom:
        raise ValueError(f"No atoms found in selection: {selection}")
    coords = np.array([atom.coord for atom in model.atom])
    center = coords.mean(axis=0)
    return tuple(center)

def extract_binding_site_sequence(structure: str, lig_sel: str, distance: float = 4.0) -> str:
    """
    Extracts the protein sequence around a ligand binding site.
    
    Args:
        structure (str): Structure identifier
        lig_sel (str): Ligand selection string
        distance (float): Distance threshold in Angstroms (default: 4.0)
        
    Returns:
        str: Protein sequence around the binding site
        
    Note:
        Uses PyMOL to select residues within the specified distance of the ligand
    """
    structure_model = cmd.get_model(f"{structure} and polymer.protein")
    structure_model_residues = {}
    for atom in structure_model.atom:
        structure_model_residues[atom.resi] = atom.resn

    binding_site_model = cmd.get_model(f"{structure} and polymer.protein and byres ({lig_sel} around {distance})")
    binding_site_model_residues = {}
    for atom in binding_site_model.atom:
        binding_site_model_residues[atom.resi] = atom.resn
    return structure_model_residues, binding_site_model_residues

# -----------------------------------------------------------------------------
# ALTERNATE LOCATION REMOVAL FUNCTION
# -----------------------------------------------------------------------------
def remove_alternate_locations(object_name: str, ligand_info: Optional[List[Dict]] = None) -> (bool, bool):
    """
    Removes alternate atom locations from protein and ligand structures.
    
    Args:
        object_name (str): Name of the PyMOL object to process
        ligand_info (Optional[List[Dict]]): List of ligand information dictionaries
        
    Returns:
        tuple: (ligand_removed, protein_removed) indicating whether alternate locations
               were removed from ligands and protein respectively
    """
    lig_removed_dict = {}
    prot_removed = False
    if ligand_info is not None:
        for row in ligand_info:
            lig = row['ligandId']
            resid = row['resid']
            sel_lig = f"{object_name} and resn {lig} and resid {resid} and hetatm"
            lig_alternate_loc = f"not alt '' and resname {lig} and resid {resid}"
            count = cmd.count_atoms(lig_alternate_loc)
            lig_key = f"{lig}_{resid}"
            if count > 0:
                alt_list = []
                cmd.iterate(lig_alternate_loc, "alt_list.append(alt)", space={'alt_list': alt_list})
                unique_altloc = np.unique(alt_list)
                cmd.remove(f"{lig_alternate_loc} and not alt '' and not alt '{unique_altloc[0]}'")
                lig_removed_dict[lig_key] = True
            else:
                lig_removed_dict[lig_key] = False
            if cmd.count_atoms(sel_lig) < 1:
                print(object_name, lig, resid, "empty lig selection after alt removal")
    prot_alternate_loc = f"not alt '' and {object_name} and polymer"
    count = cmd.count_atoms(prot_alternate_loc)
    if count > 0:
        alt_list = []
        cmd.iterate(prot_alternate_loc, "alt_list.append(alt)", space={'alt_list': alt_list})
        unique_altloc = np.unique(alt_list)
        cmd.remove(f"{prot_alternate_loc} and not alt '' and not alt '{unique_altloc[0]}'")
        prot_removed = True
    if ligand_info is not None:
        return lig_removed_dict, prot_removed
    else:
        return prot_removed

# -----------------------------------------------------------------------------
# PYMOL PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------
def analyze_ligand_contacts(pdb_id, ligand_info):
    """
    Analyzes contacts between ligands and protein chains.
    
    Args:
        pdb_id (str): PDB identifier
        ligand_info (List[Dict]): List of ligand information dictionaries
        
    Returns:
        Dict: Dictionary mapping ligand identifiers to contact information with protein chains
        
    Note:
        Uses PyMOL to calculate contacts within 4 Angstroms
    """
    main_pdb = pdb_id.split("_")[0]
    official_chain = pdb_id.split("_")[1]
    cmd.reinitialize()
    cmd.feedback("disable", "all", "output")
    cmd.set('fetch_path', cmd.exp_path(FETCH_PATH), quiet=2)
    cmd.fetch(main_pdb)
    cmd.remove("sol")
    all_chains = sorted(set(cmd.get_chains()))
    ligand_contacts = {}
    for row in ligand_info:
        lig = row["ligandId"]
        resid = row["resid"]
        lig_key = f"{lig}_{resid}"
        cmd.select(f"ligand_{lig}_{resid}_{official_chain}", f"chain {official_chain} and resn {lig} and resi {resid}")
        chain_contacts_dict = {}
        for chain in all_chains:
            cmd.select(f"near_ligand_{chain}", f"byres ({main_pdb} and chain {chain} within 4 of (ligand_{lig}_{resid}_{official_chain}))")
            chain_contacts = []
            cmd.iterate(f"near_ligand_{chain} and name CA and polymer", "chain_contacts.append((resi, resn))", space={'chain_contacts': chain_contacts})
            number_of_contacts = len(chain_contacts)
            chain_contacts_dict[chain] = number_of_contacts
        ligand_contacts[lig_key] = chain_contacts_dict
    return ligand_contacts

# -----------------------------------------------------------------------------
# UPDATED PYMOL PROCESSING FUNCTION FOR STRUCTURE-LEVEL PROCESSING
# -----------------------------------------------------------------------------
def pymol_monomer_processing(apo_list: List[str], structureId: str, ligs: List[str], ligand_info: List[Dict]) -> pd.DataFrame:
    """
    Processes protein structures using PyMOL to calculate structural metrics.
    
    Args:
        apo_list (List[str]): List of apo structure identifiers
        structureId (str): Holo structure identifier
        ligs (List[str]): List of ligand identifiers
        ligand_info (List[Dict]): List of ligand information dictionaries
        
    Returns:
        pd.DataFrame: DataFrame containing calculated structural metrics including:
            - RMSD values
            - Linear scores
            - Binding site sequences
            - Contact information
        
    Note:
        This is a core function that performs most of the structural analysis
    """
    ligand_contacts = analyze_ligand_contacts(structureId, ligand_info)
    cmd.feedback("disable", "all", "output")
    cmd.reinitialize()
    cmd.set('fetch_path', cmd.exp_path(FETCH_PATH), quiet=2)
    pymolspace = {}
    cmd.fetch(structureId, discrete=1)
    cmd.remove('sol')
    xyz_path = os.path.join(DATA_ROOT, "xyz_files", f"structure_{structureId}")
    xyz_path_local = os.path.join(DATA_ROOT, "xyz_files_local", f"structure_{structureId}")
    ensure_dir(xyz_path)
    ensure_dir(xyz_path_local)
    for row in ligand_info:
        lig = row['ligandId']
        resid = row['resid']
        lig_key = f"{lig}_{resid}"
        holo_lig = f'holo_{structureId}_{lig}_{resid}'
        alt_lig_removed_dict, alt_removed_holo = remove_alternate_locations(structureId, ligand_info)
        cmd.select(holo_lig, f"({structureId} & resid {resid} & resn {lig} & ! hydrogen)")
        cmd.create(holo_lig, holo_lig)
        cmd.select(f'holo_{structureId}', f"({structureId} & polymer & ! hydrogen & ! hetatm)")
        cmd.create(f'holo_{structureId}', f'holo_{structureId}')
        cmd.save(f'{xyz_path}/holo_{structureId}_{holo_lig}.pse', f'{holo_lig},holo_{structureId}')
        cmd.save(f'{xyz_path}/holo_{structureId}_{holo_lig}.xyz', f'{holo_lig},holo_{structureId}')
        cmd.save(f'{xyz_path_local}/holo_{structureId}_{holo_lig}.xyz', f'{holo_lig},holo_{structureId}')
    session_path = os.path.join(DATA_ROOT, "xyz_files", f"pymol_temp_session_{structureId}.pse")
    cmd.save(session_path)
    holo_apo_score = []
    for apo in apo_list:
        apo_backup = apo
        cmd.reinitialize()
        cmd.set('fetch_path', cmd.exp_path(FETCH_PATH), quiet=2)
        cmd.load(session_path)
        try:
            cmd.fetch(apo, discrete=1)
        except Exception as err:
            print(err)
            print(f"Error fetching apo: trying again with full object {apo}")
            cmd.fetch(apo[:4], discrete=1)
            cmd.select(f"{apo}_temp", f"{apo[:4]} and chain {apo.split('_')[1]}")
            cmd.create(f'{apo}_temp', f'{apo}_temp')
            cmd.delete(apo[:4])
            apo_backup = apo
            apo = f"{apo}_temp"
        cmd.remove('sol')
        alt_removed_apo = remove_alternate_locations(apo)
        RMSD_name = f'RMSD_{structureId}__{apo}'
        apo_selection  = f'{apo} & n. CA & polymer'
        holo_selection = f'{structureId} & n. CA & polymer'
        for row in ligand_info:
            pymolspace[RMSD_name] = cmd.align(apo_selection, holo_selection)
            lig = row['ligandId']
            resid = row['resid']
            lig_key = f"{lig}_{resid}"
            ligand_sel = f"{structureId} and resn {lig} and resi {resid}"
            holo_full_seq, holo_lig_seq = extract_binding_site_sequence(structureId, ligand_sel, distance=4.0)
            apo_full_seq, apo_lig_seq = extract_binding_site_sequence(apo, ligand_sel, distance=4.0)
            if ligand_contacts[lig_key][structureId.split('_')[1]] == 0:
                result = {
                    'structureId': structureId,
                    'structureId_apo': apo_backup,
                    'ligandId': lig, 
                    'resid': resid,
                    'linear_score_3': -1,
                    'linear_score_1': -1,
                    'linear_score_3_segments': -1,
                    'rmsd': -1,
                    'linear_score_local_3': -1,
                    'linear_score_local_1': -1,
                    'linear_score_3_segments_local': -1,    
                    'rmsd_local': -1,
                    'scoreDate': datetime.now().strftime('%Y-%m-%d'),
                    'alt_removed_ligand': alt_lig_removed_dict[lig_key],
                    'alt_removed_holo': alt_removed_holo,
                    'alt_removed_apo': alt_removed_apo,
                    'ligand_contacts': ligand_contacts[lig_key],
                    'holo_full_seq': holo_full_seq,
                    'apo_full_seq': apo_full_seq,
                    'holo_lig_seq': holo_lig_seq,
                    'apo_lig_seq': apo_lig_seq,
                    'linear_score_1': -1,
                    'linear_score_local_1': -1,
                    'lig_around_apo_1': -3,
                    'lig_around_holo_1': -3,
                    'lig_around_apo_3': -3,
                    'lig_around_holo_3': -3,
                    'lig_around_apo_local_1': -3,
                    'lig_around_holo_local_1': -3,
                    'lig_around_apo_local_3': -3,
                    'lig_around_holo_local_3': -3
                }
                holo_apo_score.append(result)
                continue
            min_rmsd = 1000
            best_cutoff = 10
            holo_lig = f'holo_{structureId}_{lig}_{resid}'
            cmd.select(f'apo_{apo}', f"({apo} & polymer & ! hydrogen & ! hetatm)")
            cmd.create(f'apo_{apo}', f'apo_{apo}')
            xyz_base_path = os.path.join(DATA_ROOT, "xyz_files", f"structure_{structureId}")
            cmd.save(f'{xyz_base_path}/apo_{apo}_{structureId}_{holo_lig}.xyz', f'{holo_lig},apo_{apo}')
            for lig_cutoff in range(10, 20):
                if lig_cutoff > 10:
                    cmd.align(apo_selection, holo_selection)
                    if pymolspace[RMSD_name][0] > min_rmsd:
                        cmd.select('sele_best', f'{structureId} & n. CA & polymer within {best_cutoff} of ({structureId} and resid {resid} & ! hydrogen & hetatm)')
                        cmd.align(apo_selection, 'sele_best')
                cmd.select('sele_holo', f'{structureId} & n. CA & polymer within {lig_cutoff} of ({structureId} and resid {resid} & ! hydrogen & hetatm)')
                try:
                    temp = cmd.align(apo_selection, "sele_holo")
                    local_alignment_worked = True
                except Exception:
                    min_rmsd = 1000
                    local_alignment_worked = False
                    pass
                if temp[0] < min_rmsd and temp[1] > 20:
                    min_rmsd = temp[0]
                    best_cutoff = lig_cutoff
            if not local_alignment_worked:
                print(f"Local alignment didn't work for apo {apo} and holo {structureId}")
            cmd.select('sele_holo', f'{structureId} & n. CA & polymer within {best_cutoff} of ({structureId} and resid {resid} & ! hydrogen & hetatm)')
            if local_alignment_worked:
                try:
                    cmd.align(apo_selection, 'sele_holo')
                except Exception:
                    print(f"Local alignment didn't work for apo {apo} and holo {structureId}, for best_cutoff {best_cutoff} and rmsd {min_rmsd}")
            else:
                try:
                    cmd.align(apo_selection, holo_selection)
                except Exception:
                    print(f"Local alignment didn't work AGAIN for apo {apo} and holo {structureId}")

            cmd.select(f'apo_{apo}', f"({apo} & polymer & ! hydrogen & ! hetatm)")
            cmd.create(f'apo_{apo}', f'apo_{apo}')
            xyz_base_path_local = os.path.join(DATA_ROOT, "xyz_files_local", f"structure_{structureId}")
            cmd.save(f'{xyz_base_path_local}/apo_{apo}_{structureId}_{holo_lig}.xyz', f'{holo_lig},apo_{apo}')
            filename_apo = f"{xyz_base_path}/apo_{apo}_{structureId}_{holo_lig}.xyz"
            filename_holo = f"{xyz_base_path}/holo_{structureId}_{holo_lig}.xyz"
            filename_apo_local = f"{xyz_base_path_local}/apo_{apo}_{structureId}_{holo_lig}.xyz"
            filename_holo_local = f"{xyz_base_path_local}/holo_{structureId}_{holo_lig}.xyz"
            resid_name = f'{lig}_{resid}'
            holopdb_chain = structureId
            apopdb_chain = apo
            try:
                linear_score_3, segment_scores_3, lig_around_apo_3, lig_around_holo_3 = get_score(
                    xyz_apo=filename_apo,
                    xyz_holo=filename_holo,
                    apo_id=apopdb_chain,
                    holo_id=holopdb_chain,
                    lig_resid=resid_name,
                    n_lig_splits=3
                )
            except Exception as e:
                print(e)
                linear_score_3 = -3
                segment_scores_3 = [-3,-3,-3]
                lig_around_apo_3 = -3
                lig_around_holo_3 = -3
            try:
                linear_score_1, segment_scores_1, lig_around_apo_1, lig_around_holo_1 = get_score(
                    xyz_apo=filename_apo,
                    xyz_holo=filename_holo,
                    apo_id=apopdb_chain,
                    holo_id=holopdb_chain,
                    lig_resid=resid_name,
                    n_lig_splits=1
                )
            except Exception as e:
                print(e)
                linear_score_1 = -3
                lig_around_apo_1 = -3
                lig_around_holo_1 = -3
                
            try:
                linear_score_local_3, segment_scores_local_3, lig_around_apo_local_3, lig_around_holo_local_3 = get_score(
                    xyz_apo=filename_apo_local,
                    xyz_holo=filename_holo_local,
                    apo_id=apopdb_chain,
                    holo_id=holopdb_chain,
                    lig_resid=resid_name,
                    n_lig_splits=3
                )
            except Exception as e:
                print(e)
                linear_score_local_3 = -3
                segment_scores_local_3 = [-3,-3,-3]
                lig_around_apo_local_3 = -3
                lig_around_holo_local_3 = -3
            try:
                linear_score_local_1, segment_scores_local_1, lig_around_apo_local_1, lig_around_holo_local_1 = get_score(
                    xyz_apo=filename_apo_local,
                    xyz_holo=filename_holo_local,
                    apo_id=apopdb_chain,
                    holo_id=holopdb_chain,
                    lig_resid=resid_name,
                    n_lig_splits=1
                )
            except Exception as e:
                print(e)
                linear_score_local_1 = -3
                lig_around_apo_local_1 = -3
                lig_around_holo_local_1 = -3
            
            round_rmsd = round(pymolspace[RMSD_name][0], 2)
            result = {
                'structureId': structureId,
                'structureId_apo': apo_backup,
                'ligandId': lig, 
                'resid': resid,
                'linear_score_3': linear_score_3,
                'linear_score_3_segments': segment_scores_3,
                'rmsd': round_rmsd,
                'linear_score_local_3': linear_score_local_3,
                'linear_score_3_segments_local': segment_scores_local_3,
                'rmsd_local': min_rmsd,
                'scoreDate': datetime.now().strftime('%Y-%m-%d'),
                'alt_removed_ligand': alt_lig_removed_dict[lig_key],
                'alt_removed_holo': alt_removed_holo,
                'alt_removed_apo': alt_removed_apo,
                'ligand_contacts': ligand_contacts[lig_key],
                'holo_full_seq': holo_full_seq,
                'apo_full_seq': apo_full_seq,
                'holo_lig_seq': holo_lig_seq,
                'apo_lig_seq': apo_lig_seq,
                'linear_score_1': linear_score_1,
                'linear_score_local_1': linear_score_local_1,
                'lig_around_apo_1': lig_around_apo_1,
                'lig_around_holo_1': lig_around_holo_1,
                'lig_around_apo_3': lig_around_apo_3,
                'lig_around_holo_3': lig_around_holo_3,
                'lig_around_apo_local_1': lig_around_apo_local_1,
                'lig_around_holo_local_1': lig_around_holo_local_1,
                'lig_around_apo_local_3': lig_around_apo_local_3,
                'lig_around_holo_local_3': lig_around_holo_local_3
            }
            holo_apo_score.append(result)
    os.remove(session_path)
    cmd.delete(structureId)
    return pd.DataFrame(holo_apo_score)

# -----------------------------------------------------------------------------
# UPDATED PROCESSING FUNCTIONS FOR STRUCTURE-LEVEL CHECKPOINTING
# -----------------------------------------------------------------------------
def process_structure_protein(structureId: str, apo_list: List[str], structure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a single protein structure and its associated apo structures.
    
    Args:
        structureId (str): Structure identifier to process
        apo_list (List[str]): List of associated apo structure identifiers
        structure_df (pd.DataFrame): DataFrame containing structure information
        
    Returns:
        pd.DataFrame: Processed results for the structure
        
    Note:
        Creates necessary directories and manages temporary files
    """
    base_dir = DATA_ROOT
    xyz_dir = os.path.join(base_dir, "xyz_files", f"structure_{structureId}")
    xyz_local_dir = os.path.join(base_dir, "xyz_files_local", f"structure_{structureId}")
    ensure_dir(xyz_dir)
    ensure_dir(xyz_local_dir)
    ligs = list(structure_df['ligandId'].unique())
    structure_df.to_pickle(os.path.join(os.path.dirname(xyz_dir), f'combined_structure_{structureId}.p'))
    ligand_info = structure_df[['ligandId', 'resid']].drop_duplicates().to_dict(orient='records')
    assert len(apo_list) > 0, f"{structureId} | No new apos identified!"
    return pymol_monomer_processing(apo_list, structureId, ligs, ligand_info)

def cleanup_structure_files(structureId: str) -> None:
    """
    Removes temporary files created during structure processing.
    
    Args:
        structureId (str): Structure identifier whose files should be cleaned up
        
    Returns:
        None
        
    Side Effects:
        Removes files from xyz_files and xyz_files_local directories
    """
    base_dir = DATA_ROOT
    xyz_dir = os.path.join(base_dir, "xyz_files", f"structure_{structureId}")
    xyz_local_dir = os.path.join(base_dir, "xyz_files_local", f"structure_{structureId}")
    combined_xyz_file = os.path.join(base_dir, "xyz_files", f"combined_structure_{structureId}.p")
    combined_xyz_local_file = os.path.join(base_dir, "xyz_files_local", f"combined_structure_{structureId}.p")
    if os.path.exists(xyz_dir):
        shutil.rmtree(xyz_dir)
    if os.path.exists(xyz_local_dir):
        shutil.rmtree(xyz_local_dir)
    if os.path.exists(combined_xyz_file):
        os.remove(combined_xyz_file)
    if os.path.exists(combined_xyz_local_file):
        os.remove(combined_xyz_local_file)

def process_single_structure_no_checkpoint(structure_file: str) -> Optional[pd.DataFrame]:
    """
    Processes a single structure file without using checkpointing.
    
    Args:
        structure_file (str): Path to the structure file to process
        
    Returns:
        Optional[pd.DataFrame]: Processed results if successful, None otherwise
        
    Note:
        Used for parallel processing of structures
    """
    structure_df = pd.read_pickle(structure_file)
    structureId = structure_df['structureId'].iloc[0]
    apo_list = list(structure_df[structure_df['scoreDate'].isnull()]['structureId_apo'].unique())
    if not apo_list:
        print(f"No new apo entries for structure {structureId}. Skipping.")
        return None
    result_df = process_structure_protein(structureId, apo_list, structure_df)
    checkpoint_filename = os.path.join(CHECKPOINT_DIR, f"structure_{structureId}.pkl")
    result_df.to_pickle(checkpoint_filename)
    cleanup_structure_files(structureId)
    print(f"Processed structure {structureId} with {len(result_df)} rows.")
    return result_df

def parallel_process_structures(structure_dir: str, n_jobs: int = 16) -> None:
    """
    Processes multiple structures in parallel.
    
    Args:
        structure_dir (str): Directory containing structure files to process
        n_jobs (int): Number of parallel jobs to run (default: 4)
        
    Returns:
        None
        
    Note:
        Uses joblib for parallel processing
    """
    files = [f for f in os.listdir(structure_dir) if f.endswith('.p')]
    structure_files = []
    for fname in sorted(files):
        structure_file = os.path.join(structure_dir, fname)
        try:
            structureId = fname.split('_')[1].split('.')[0]
        except Exception as e:
            print(f"Error extracting structureId from {fname}: {e}")
            continue
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"structure_{structureId}.pkl")
        if os.path.exists(checkpoint_file):
            print(f"Structure {structureId} already processed (checkpoint exists). Skipping.")
            continue
        structure_files.append(structure_file)
    
    if structure_files:
        Parallel(n_jobs=n_jobs)(
            delayed(process_single_structure_no_checkpoint)(sf) for sf in structure_files
        )

def update_master_checkpoint() -> None:
    """
    Updates the master checkpoint file with results from individual structure processing.
    
    Returns:
        None
        
    Side Effects:
        Creates/updates the master checkpoint file
    """
    df_list = []
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.startswith('structure_') and fname.endswith('.pkl'):
            try:
                df = pd.read_pickle(os.path.join(CHECKPOINT_DIR, fname))
                df_list.append(df)
            except Exception as e:
                print(f"Error reading checkpoint file {fname}: {e}")
    if df_list:
        master_df = pd.concat(df_list, ignore_index=True)
        master_df = optimize_dataframe(master_df)
        master_df.to_pickle(MASTER_CHECKPOINT_FILE)
        print(f"Master checkpoint updated with {len(master_df)} total rows.")
    else:
        print("No checkpoint files found to update master checkpoint.")

def split_combined_to_structures(combined_file: str, structure_dir: str) -> None:
    """
    Splits a combined data file into individual structure files.
    
    Args:
        combined_file (str): Path to the combined data file
        structure_dir (str): Directory to store individual structure files
        
    Returns:
        None
        
    Note:
        Only processes structures that haven't been scored yet
    """
    print("Splitting combined.p into structure-specific files for unscored structures...")
    combined = pd.read_pickle(combined_file)
    count = 0
    for structure, group in combined.groupby('structureId'):
        if not group['scoreDate'].isnull().all():
            print(f"Structure {structure} has been scored. Skipping.")
            continue
        structure_file = os.path.join(structure_dir, f"structure_{structure}.p")
        group_reconstructed = pd.DataFrame(group.values, columns=group.columns)
        group_reconstructed.to_pickle(structure_file)
        count += 1
    print(f"Saved {count} structure files in {structure_dir}")

def update_combined_from_structures(structure_dir: str, checkpoint_dir: str, combined_file: str) -> pd.DataFrame:
    """
    Updates the combined data file with results from individual structure processing.
    
    Args:
        structure_dir (str): Directory containing structure files
        checkpoint_dir (str): Directory containing checkpoint files
        combined_file (str): Path to the combined data file
        
    Returns:
        pd.DataFrame: Updated combined DataFrame
        
    Note:
        Merges new results with existing data while preserving previous scores
    """
    updated_structures = []
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith('structure_') and fname.endswith('.pkl'):
            checkpoint_file = os.path.join(checkpoint_dir, fname)
            print(f"Updating from checkpoint file {checkpoint_file} ...")
            structure_df = pd.read_pickle(checkpoint_file)
            updated_structures.append(structure_df)
    if updated_structures:
        new_results = pd.concat(updated_structures, ignore_index=True)
        merge_keys = ['structureId', 'structureId_apo', 'ligandId', 'resid']
        combined = pd.read_pickle(combined_file)
        combined_updated = combined.merge(
            new_results[merge_keys + ['linear_score_3', 'linear_score_3_segments', 'rmsd',
                                       'scoreDate', 'linear_score_local_3', 
                                       'linear_score_3_segments_local', 'rmsd_local',
                                       'alt_removed_ligand', 'alt_removed_holo', 'alt_removed_apo', 
                                       'ligand_contacts', 'holo_full_seq', 'apo_full_seq', 
                                       'holo_lig_seq', 'apo_lig_seq', 'linear_score_1',
                                       'linear_score_local_1', 'lig_around_apo_1', 'lig_around_holo_1',
                                       'lig_around_apo_3', 'lig_around_holo_3', 'lig_around_apo_local_1',
                                       'lig_around_holo_local_1', 'lig_around_apo_local_3', 'lig_around_holo_local_3']],
            on=merge_keys,
            how='left',
            suffixes=('', '_new')
        )
        for col in ['linear_score_3', 'linear_score_3_segments', 'rmsd',
                    'scoreDate', 'linear_score_local_3', 
                    'linear_score_3_segments_local', 'rmsd_local',
                    'alt_removed_ligand', 'alt_removed_holo', 'alt_removed_apo', 
                    'ligand_contacts', 'holo_full_seq', 'apo_full_seq', 
                    'holo_lig_seq', 'apo_lig_seq', 'linear_score_1',
                    'linear_score_local_1', 'lig_around_apo_1', 'lig_around_holo_1',
                    'lig_around_apo_3', 'lig_around_holo_3', 'lig_around_apo_local_1',
                    'lig_around_holo_local_1', 'lig_around_apo_local_3', 'lig_around_holo_local_3']:
            combined_updated[col] = combined_updated[col].combine_first(combined_updated[f"{col}_new"])
            combined_updated.drop(columns=[f"{col}_new"], inplace=True)
        combined_updated = combined_updated.drop_duplicates(subset=["structureId", "resid", "ligandId", "structureId_apo"])
        combined_updated.to_pickle(combined_file)
        print(f"Saved updated combined DataFrame with {len(combined_updated)} rows to {combined_file}")
        return combined_updated
    else:
        print("No updated structure checkpoints found. Returning original combined DataFrame.")
        combined_updated = pd.read_pickle(combined_file)
        combined_updated = combined_updated.drop_duplicates(subset=["structureId", "resid", "ligandId", "structureId_apo"])
        return combined_updated

def save_cluster(structureId: str, structure_dir: str, df: pd.DataFrame) -> None:
    """
    Saves a cluster of related structures to a file.
    
    Args:
        structureId (str): Identifier for the structure cluster
        structure_dir (str): Directory to save the cluster file
        df (pd.DataFrame): DataFrame containing the cluster data
        
    Returns:
        None
    """
    structure_file = os.path.join(structure_dir, f"combined_structure_{structureId}.p")
    df.to_pickle(structure_file)


def update_cif_files(combined: pd.DataFrame) -> None:
    """
    Updates CIF files for structures that have been modified.
    
    Args:
        combined (pd.DataFrame): DataFrame containing structure information
        
    Returns:
        None
        
    Note:
        Uses asynchronous HTTP requests to download updated CIF files
    """
    # Get list of structures that need updating
    files_to_update = [s.split('_')[0] for s in combined[combined['revision_date'] > '2025-02-15 00:00:00+00:00']['structureId'].unique()]
    
    if not files_to_update:
        print("No structures need updating")
        return
        
    print(f"Updating {len(files_to_update)} CIF files...")
    
    # Remove old files
    for structureId in files_to_update:
        file_path = os.path.join(FETCH_PATH, f"{structureId}.cif")
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Create a function to download CIF files
    async def download_cif(session, pdb_id, output_dir):
        output_file = os.path.join(output_dir, f"{pdb_id.lower()}.cif")
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()
                with open(output_file, 'wb') as f:
                    f.write(content)
                print(f"Downloaded: {output_file}")
        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
            # Try once more after a delay
            await asyncio.sleep(5)
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    content = await response.read()
                    with open(output_file, 'wb') as f:
                        f.write(content)
                    print(f"Downloaded: {output_file}")
            except Exception as e:
                print(f"Failed to download {pdb_id} after retry: {e}")
    
    # Download all CIF files
    async def download_all_files():
        async with aiohttp.ClientSession() as session:
            tasks = [download_cif(session, pdb_id, FETCH_PATH) for pdb_id in files_to_update]
            await asyncio.gather(*tasks)
    
    # Save PDB IDs to temporary JSON file (for logging purposes)
    temp_json = os.path.join(FETCH_PATH, "temp_update_ids.json") 
    with open(temp_json, 'w') as f:
        json.dump(files_to_update, f)
    
    # Run the download
    asyncio.run(download_all_files())

    
    # Cleanup temp file
    if os.path.exists(temp_json):
        os.remove(temp_json)
    
    print("CIF file updates completed")

# -----------------------------------------------------------------------------
# MAIN ORCHESTRATION FUNCTION
# -----------------------------------------------------------------------------
def main(db_path: str = '.', n_jobs: int = 16, input_list: bool = False, 
         update_previous: bool = False, max_res: float = 2.5) -> None:
    """
    Main orchestration function for the protein structure analysis pipeline.
    
    Args:
        db_path (str): Path to the database directory (default: '.')
        n_jobs (int): Number of parallel jobs to run (default: 16)
        input_list (bool): Whether to use a predefined list of PDB IDs (default: False)
        update_previous (bool): Whether to update existing data (default: False)
        max_res (float): Maximum resolution threshold in Angstroms (default: 2.5)
        
    Returns:
        None
        
    Note:
        This is the main entry point for the pipeline, coordinating all processing steps
    """
    print("Starting main function...")
    combined_file = os.path.join(db_path, 'combined.p')
    structure_dir = os.path.join(db_path, 'structure_dfs')
    ensure_dir(structure_dir)
    
    backup_file = os.path.join(db_path, 'combined_prev.p')
    
    if update_previous:
        if os.path.exists(combined_file):
            shutil.copy(combined_file, backup_file)
            print("Backed up the previous combined.p to combined_prev.p")
        download_data(input_list=input_list, max_res=max_res, update_previous=True)
        new_combined = pd.read_pickle(combined_file)
        if os.path.exists(backup_file):
            old_combined = pd.read_pickle(backup_file)
            print("Merging backup combined.p with newly downloaded data...")
            update_cols = ['linear_score_3', 'linear_score_3_segments', 'rmsd',
                           'scoreDate', 'linear_score_local_3', 'linear_score_3_segments_local', 'rmsd_local',
                           'alt_removed_ligand', 'alt_removed_holo', 'alt_removed_apo',
                           'ligand_contacts', 'holo_full_seq', 'apo_full_seq', 'holo_lig_seq', 'apo_lig_seq',
                           'linear_score_1', 'linear_score_local_1', 'lig_around_apo_1', 'lig_around_holo_1',
                           'lig_around_apo_3', 'lig_around_holo_3', 'lig_around_apo_local_1',
                           'lig_around_holo_local_1', 'lig_around_apo_local_3', 'lig_around_holo_local_3']
            old_combined_reduced = old_combined[['structureId', 'resid', 'ligandId', 'structureId_apo'] + update_cols]
            merged_combined = pd.merge(new_combined, old_combined_reduced,
                                       on=['structureId', 'resid', 'ligandId', 'structureId_apo'],
                                       how='outer',
                                       suffixes=('', '_old'))
            for col in update_cols:
                merged_combined[col] = merged_combined[col].combine_first(merged_combined[f"{col}_old"])
                merged_combined.drop(columns=[f"{col}_old"], inplace=True)
            merged_combined.to_pickle(combined_file)
            print(f"Merged combined.p now has {len(merged_combined)} rows.")
        else:
            print("No backup file found. Proceeding with newly downloaded data.")
    else:
        if not os.path.exists(combined_file):
            download_data(input_list=input_list, max_res=max_res)
    
    checkpoint_files = [fname for fname in os.listdir(CHECKPOINT_DIR) if fname.startswith('structure_') and fname.endswith('.pkl')]
    if checkpoint_files:
        print("Found existing checkpoint files. Updating master checkpoint and combined.p...")
        update_master_checkpoint()
        combined = update_combined_from_structures(structure_dir, CHECKPOINT_DIR, combined_file)
        for f in checkpoint_files:
            os.remove(os.path.join(CHECKPOINT_DIR, f))
        print("Removed all checkpoint files.")
    else:
        print("No checkpoint files found. Starting from scratch.")
        combined = pd.read_pickle(combined_file)
        print("Loading finished.")
    checkpoint_files_short = [f.split('_')[1].split('.')[0] for f in checkpoint_files]
    
    existing_files = os.listdir(structure_dir)
    for f in existing_files:
        try:
            structureId = f.split('_')[1].split('.')[0]
        except Exception:
            continue
        if structureId in checkpoint_files_short:
            os.remove(os.path.join(structure_dir, f))
    
    existing_structure_ids = set()
    for f in os.listdir(structure_dir):
        try:
            structureId = f.split('_')[1].split('.')[0]
            existing_structure_ids.add(structureId)
        except:
            continue
    unscored_structure_ids = set()
    for structure, group in combined.groupby('structureId'):
        if group['scoreDate'].isnull().all():
            unscored_structure_ids.add(structure)
    missing_structure_ids = unscored_structure_ids - existing_structure_ids
    combined_missing = combined[combined['structureId'].isin(missing_structure_ids)]
    Parallel(n_jobs=n_jobs)(
        delayed(save_cluster)(structureId, structure_dir, group)
        for structureId, group in combined_missing.groupby('structureId')
    )
    update_cif_files_flag = True
    if update_cif_files_flag:
        update_cif_files(combined)
            
    global total_rows_unscored
    total_rows_unscored = len(combined)
    print("Starting parallel processing of structures...")
    parallel_process_structures(structure_dir, n_jobs=n_jobs)
    update_master_checkpoint()
    print("Parallel processing of structures completed.")
    
    combined_updated = update_combined_from_structures(structure_dir, CHECKPOINT_DIR, combined_file)
    print("combined.p updated with checkpoint results from structures.")
    
    for f in os.listdir(structure_dir):
        os.remove(os.path.join(structure_dir, f))
    print("Removed all files in structure_dfs.")
    
    if os.path.exists(MASTER_CHECKPOINT_FILE):
        os.remove(MASTER_CHECKPOINT_FILE)
        print("Removed master_checkpoint.p")

# -----------------------------------------------------------------------------
# CALLING SCRIPT
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=16,
                        help='Number of parallel jobs to run for processing structures')
    parser.add_argument('--input_list', action='store_true',
                        help='Whether to use an input list for downloading data')
    parser.add_argument('--update_previous', action='store_true',
                        help='Force download new data and merge previous combined.p with the new one; then compute scores for unscored rows')
    parser.add_argument('--max_res', type=float, default=2.5,
                        help='Maximum resolution threshold in Å (default: 2.5)')
    parser.add_argument('--custom_folder', type=str, default='data',
                        help='Custom folder to use as the data root (default: data)')
    args = parser.parse_args()

    # Update global data folder settings based on the custom_folder argument.
    DATA_ROOT = args.custom_folder
    monomer_dir = os.path.join(DATA_ROOT, "monomer_calcs")
    STRUCTURE_DFS_DIR = os.path.join(os.path.dirname(__file__), DATA_ROOT, "monomer_calcs", "structure_dfs")
    CHECKPOINT_DIR = os.path.join(monomer_dir, "checkpoints")
    MASTER_CHECKPOINT_FILE = os.path.join(monomer_dir, "master_checkpoint.p")
    for path in [STRUCTURE_DFS_DIR, CHECKPOINT_DIR]:
        ensure_dir(path)

    storage_path = monomer_dir
    rejects_path = os.path.join(DATA_ROOT, "rejects")
    for path in [storage_path, rejects_path]:
        ensure_dir(path)
    main(db_path=storage_path, n_jobs=args.n_jobs, input_list=args.input_list,
         update_previous=args.update_previous, max_res=args.max_res)
