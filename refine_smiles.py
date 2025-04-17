import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger 

from IPython.display import display

from pymol import cmd, util

from joblib import Parallel, delayed
import os
import pickle

RDLogger.DisableLog('rdApp.*')

FETCH_PATH = "/home/users/f/febrerma/scratch/cif_files/"
def create_polymer_selection(holo):
    """
    Create and return the polymer selection name for the holo structure.
    """
    holo_sele = f"polymer and {holo}"
    holo_name = f"{holo}_holo"
    cmd.select("holo_sele", holo_sele)
    cmd.create(holo_name, "holo_sele")
    return holo_name

def create_ligand_selection(holo_name, resid, resname):
    """
    Create and return the ligand selection name using the given residue id and name.
    """
    lig_sel_str = f"not {holo_name} and resid {resid} and resname {resname}"
    cmd.select("lig_sele", lig_sel_str)
    sel_lig = f"lig_{resid}_{resname}"
    cmd.create(sel_lig, "lig_sele")
    return sel_lig

def handle_ligand_alternate_locations(sel_lig, resid, resname):
    """
    Check for and process alternate locations in the ligand.
    If alternate locations are found, keep only the first one.
    """
    lig_alternate_loc = f"not alt '' and resname {resname} and resid {resid}"
    if cmd.count_atoms(lig_alternate_loc) > 0:
        alt_list = []
        cmd.iterate(lig_alternate_loc, "alt_list.append(alt)", space={'alt_list': alt_list})
        unique_altloc = np.unique(alt_list)
        # Keep only the first alternate location
        cmd.remove(f"not alt '' and not alt '{unique_altloc[0]}' and {sel_lig}")

def compute_dist(pymol_model1, pymol_model2):
    """
    Compute and return a list of distances between every pair of atoms in the two models.
    """
    dist_list = []
    for atom1 in pymol_model1.atom:
        for atom2 in pymol_model2.atom:
            dx = atom1.coord[0] - atom2.coord[0]
            dy = atom1.coord[1] - atom2.coord[1]
            dz = atom1.coord[2] - atom2.coord[2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            dist_list.append(dist)
    return np.array(dist_list)

def handle_protein_alternate_locations(holo_name, sele_res, sel_lig, cutoff=4.0):
    """
    For residues within the selection, check for alternate locations and,
    for each residue, keep only the alternate that is closest to the ligand.
    """
    prot_alternate_loc = f"not alt '' and {sele_res}"
    if cmd.count_atoms(prot_alternate_loc) > 0:
        lig_model = cmd.get_model(sel_lig)
        model = cmd.get_model(sele_res)
        unique_residues = set()
        # Build a set of unique residues identified by (chain, resi, resn)
        for atom in model.atom:
            unique_residues.add((atom.chain, atom.resi, atom.resn))
        # Process alternate locations for each unique residue
        for chain, resi, resn in unique_residues:
            res_sel = f"chain {chain} and resi {resi} and resn {resn} and {holo_name}"
            res_alternate_loc = f"not alt '' and {res_sel}"
            if cmd.count_atoms(res_alternate_loc) > 0:
                best_alt = None
                best_count = -1
                alt_list = []
                cmd.iterate(res_alternate_loc, "alt_list.append(alt)", space={'alt_list': alt_list})
                unique_altloc = np.unique(alt_list)

                # Iterate over each alternate location and compute the minimum distance to the ligand
                for alt in unique_altloc:
                    res_alt = f"not alt '' and alt '{alt}' and {res_sel}"
                    res_model = cmd.get_model(res_alt)
                    # Compute all pairwise distances between atoms in this alternate and the ligand
                    distances = compute_dist(res_model, lig_model)
                    alt_count = np.sum(distances <= cutoff)

                    if alt_count > best_count:
                        best_count = alt_count
                        best_alt = alt
                
                # Remove atoms with alternate locations that are not the best one
                cmd.remove(f"not alt '' and not alt '{best_alt}' and {res_sel}")

def compute_mean_sasa(selection):
    """
    Compute and return the mean solvent-accessible surface area (SASA)
    for the given selection.
    """
    sasa_dict = cmd.get_sasa_relative(selection)
    return round(np.mean(list(sasa_dict.values())), 2)   

def compute_ligand_mw(sel_lig):
    """
    Return the molecular weight of the ligand.
    """
    try:
        lig_mw = round(util.compute_mass(sel_lig), 2)
    except Exception as e:
        return None
    return lig_mw

def ligand_properties(holo, resid, resname, index, total_rows):
    """
    Calculate the solvent-accessible surface area (SASA) of residues around a ligand 
    in a given protein structure. Also calculates ligand molecular weight and number of atoms.
    
    Parameters:
      holo (str): The structure identifier in the form 'PDBID_CHAIN'.
      resid (int): The residue ID of the ligand.
      resname (str): The residue name of the ligand.
      
    Returns:
      float: The mean SASA of residues within 5 Å of the ligand.
      int: Number of atoms in the ligand.
      float: Molecular weight of the ligand.
    """
    if index % 1000 == 0:
        print(f"Processing {index} of {total_rows}")
    # Fetch the structure; return error if unsuccessful
    try: 
        cmd.reinitialize()
        cmd.feedback("disable", "all", "output")
        cmd.set('fetch_path', cmd.exp_path(FETCH_PATH), quiet=2)
        cmd.fetch(holo)
    except Exception as e:
        print(f"Error fetching {holo}, {resid}, {resname}, {e}")
        df_results = pd.DataFrame()
        df_results['mean_sasa'] = None
        df_results['lig_atoms'] = None
        df_results['lig_mw'] = None
        df_results['structureId'] = [holo]
        df_results['resid'] = [resid]
        df_results['ligandId'] = [resname]
        return df_results
    
    # Create selections for the polymer and the ligand
    holo_name = create_polymer_selection(holo)
    sel_lig = create_ligand_selection(holo_name, resid, resname)

    if cmd.count_atoms(sel_lig) < 1:    
        holo_name = create_polymer_selection(holo)
        sel_lig = create_ligand_selection(holo_name, resid, resname)
    
    cmd.delete(holo)  # Remove the original fetched structure to avoid duplicates

    # Handle alternate locations for the ligand
    handle_ligand_alternate_locations(sel_lig, resid, resname)
    
    # Check that the ligand selection is not empty
    if cmd.count_atoms(sel_lig) < 1:
        print(holo, resname, resid, "empty lig selection")
        df_results = pd.DataFrame()
        df_results['mean_sasa'] = None
        df_results['lig_atoms'] = None
        df_results['lig_mw'] = None
        df_results['structureId'] = [holo]
        df_results['resid'] = [resid]
        df_results['ligandId'] = [resname]
        return df_results

    # Define the selection for residues within 5 Å of the ligand (i.e., interacting residues)
    sele_res = f"byres {holo_name} & polymer within 4.000 of ({sel_lig})"
    
    # Handle alternate locations for protein residues
    handle_protein_alternate_locations(holo_name, sele_res, sel_lig, cutoff=4.0)
    
    # Check that there are interacting residues
    if cmd.count_atoms(sele_res) < 1:
        print(holo, resid, resname,"no prot closed to lig")
        df_results = pd.DataFrame()
        df_results['mean_sasa'] = None
        df_results['lig_atoms'] = None
        df_results['lig_mw'] = None
        df_results['structureId'] = [holo]
        df_results['resid'] = [resid]
        df_results['ligandId'] = [resname]
        return df_results
    
    sele_res_sasa = f"byres {holo_name} & polymer within 5.000 of ({sel_lig})"
    # Create a named selection for the residues around the ligand
    cmd.select("res_around_lig", sele_res_sasa)
    
    # Compute the mean SASA for the interacting residues
    mean_sasa = compute_mean_sasa("res_around_lig")
    
    # Compute ligand properties: number of atoms and molecular weight
    lig_atoms = cmd.count_atoms(sel_lig)
    lig_mw = compute_ligand_mw(sel_lig)
    #create a dataframe with the results
    df_results = pd.DataFrame()
    df_results['mean_sasa'] = [mean_sasa]
    df_results['lig_atoms'] = [lig_atoms]
    df_results['lig_mw'] = [lig_mw]
    df_results['structureId'] = holo
    df_results['resid'] = resid
    df_results['ligandId'] = resname
    #save the dataframe to a pickle file
    return df_results


def load_data(pickle_path):
    """
    Load the DataFrame from a pickle file.
    
    Args:
        pickle_path (str): Path to the pickle file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_pickle(pickle_path)


def main():
    # Define the steps to report progress
    steps = [
        "Loading data",
        "Preparing ligands",
        "Assigning unique ligand IDs",
        "Merging with original DataFrame",
        "Computing ligand properties",
        "Saving output"
    ]
    
    print("Starting processing...")
    
    # Step 1: Load data
    print(f"[1/6] {steps[0]}")

    columns_4uniqueligandId = [
    'clusterNumber95', 'rcsb_id', 'ligandId', 'resid', 'structureId',
    'ligand_inchi', 'ligand_smiles_openeye', 'ligand_smiles_CACTVS'
    ]

    pickle_path = "combined.p"
    df_light = load_data(pickle_path)

    #df_light = df[columns_4uniqueligandId]
    #del df

    df_light['clusterNumber95'] = df_light['clusterNumber95'].astype('string')
    df_light['rcsb_id'] = df_light['rcsb_id'].astype('string')
    df_light['ligandId'] = df_light['ligandId'].astype('string')
    df_light['resid'] = df_light['resid'].astype('int')
    df_light['structureId'] = df_light['structureId'].astype('string')
    #drop duplicated rows in df_light based on the columns 'structureId', 'resid', 'ligandId'
    df_light = df_light.drop_duplicates(subset=['structureId', 'resid', 'ligandId'])
    #reset the index of df_light
    df_light = df_light.reset_index(drop=True)
    results = Parallel(n_jobs=126)(
        delayed(ligand_properties)(row['structureId'], row['resid'], row['ligandId'], index, len(df_light)) for index, row in df_light.iterrows()
    )
    df_results_list = [df for df in results]
    df_results = pd.concat(df_results_list)
    df_results.to_pickle("sasa_output.pkl")

if __name__ == "__main__":
    main()
