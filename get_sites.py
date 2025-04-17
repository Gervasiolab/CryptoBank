import pymol.cmd as cmd
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import pickle
import os
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

def compute_center_of_geometry(selection):
    """
    Computes the center of geometry (COG) for a given selection in PyMOL.
    Parameters:
        selection (str): The atom selection for which to compute the COG.
    Returns:
        list: The [x, y, z] coordinates of the center of geometry.
    """
    coordinates = cmd.get_coords(f'{selection}', 1)
    if coordinates.size == 0:
        return None
    x, y, z = zip(*coordinates)
    cog = [sum(x) / len(x), sum(y) / len(y), sum(z) / len(z)]
    return cog

def compute_centroids(points, labels):
    """
    Computes centroids for clusters of points.
    Parameters:
        points (np.array): Array of shape (n_points, 3) containing coordinates.
        labels (array-like): Cluster labels for each point.
    Returns:
        np.array: An array of centroids corresponding to each unique label.
    """
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        cluster_points = points[np.where(labels == label)]
        x_mean = np.mean(cluster_points[:, 0])
        y_mean = np.mean(cluster_points[:, 1])
        z_mean = np.mean(cluster_points[:, 2])
        centroids.append([x_mean, y_mean, z_mean])
    return np.array(centroids)

def get_clusters(coords, cutoff=7.0):
    """
    Performs agglomerative clustering on a set of coordinates.
    Parameters:
        coords (np.array): An array of shape (n, 3) with ligand coordinates.
        cutoff (float): The distance threshold to use for clustering.
    Returns:
        tuple: (labels, centroids) where labels is a list of cluster labels and
               centroids is an array of the computed centroids for each cluster.
    """
    if len(coords) == 1:
        labels = [0]
        centroids = coords
        return labels, centroids
    clustering = AgglomerativeClustering(
        metric='euclidean',
        n_clusters=None,
        distance_threshold=cutoff,
        linkage='complete'
    )
    labels = clustering.fit_predict(coords)
    centroids = compute_centroids(coords, labels)
    return labels, centroids

def get_sites_holo(clusterNumber):
    """
    Processes a given clusterNumber: fetches structures, computes the center of 
    geometry for each ligand, and clusters ligands into binding sites using
    agglomerative clustering.
    """

    output_file = f'sites_cluster_{clusterNumber}.p'
    if os.path.exists(output_file):
        print(f"Cluster {clusterNumber} already processed, loading from file.")
        return

    print("Getting sites for clusterNumber:", clusterNumber, "...")
    # Select rows for the current cluster and remove duplicate structure/ligand entries
    to_test = monomers[monomers['clusterNumber95'] == clusterNumber]
    to_test = to_test.drop_duplicates(subset=['structureId', 'ligandId', 'resid'])
    # Retain key columns in the desired order
    to_test_subset = to_test[['structureId', 'resid', 'ligandId']].copy()
    # Initialize the site_coords column
    to_test_subset['site_coords'] = None
    to_test_subset.reset_index(drop=True, inplace=True)
    
    # Process structures in batches for alignment and site coordinate calculation
    fetch_list_original = list(set(to_test_subset["structureId"].tolist()))
    #fetch_list_original = list(set([elem.upper() for elem in to_test_subset["structureId"].tolist()]))
    if not fetch_list_original:
        return to_test_subset
    print("Primary structure in cluster:", fetch_list_original[0])
    
    for i in range(0, len(fetch_list_original), 50):
        print("Processing batch starting at index:", i)
        cmd.reinitialize()
        cmd.feedback("disable", "all", "output")
        cmd.set('fetch_path', '../cif_files')

        # Use the first structure in the list as the reference
        batch = fetch_list_original[i+1:i+1+50]
        try:
            cmd.fetch(fetch_list_original[0])
        except Exception as e:
            print("Problem fetching structure", fetch_list_original[0], ":", e)
            try:
                cmd.fetch(fetch_list_original[0].split('_')[0])
                cmd.select(f"{fetch_list_original[0]}_temp", f"{fetch_list_original[0].split('_')[0]} and chain {fetch_list_original[0].split('_')[1]}")
                cmd.create(f'{fetch_list_original[0]}_temp', f'{fetch_list_original[0]}_temp')
                cmd.delete(fetch_list_original[0].split('_')[0])
                cmd.create(fetch_list_original[0], f'{fetch_list_original[0]}_temp')
                cmd.delete(f'{fetch_list_original[0]}_temp')
            except Exception as e:
                print("Problem fetching structure for the second time", fetch_list_original[0], ":", e)

        temp_list = []


        # Group elements by their lowercase value
        batch_and_original = batch + [fetch_list_original[0]]
        duplicates = [x for x in batch_and_original if sum(y.lower() == x.lower() for y in batch_and_original) > 1]
        #duplicates = duplicates but with everything after the first underscore in lowercase
        low_repeated = list(set([x.split('_')[0] + '_' + x.split('_')[1].lower() for x in duplicates]) )
        print(low_repeated)
        #get a list of elements in batch that have the same with a different case
        for elem in batch:
            #if elem is not the same as fetch_list_original[0] in any case, fetch it
            if elem not in low_repeated:
                try:
                    cmd.fetch(elem, quiet=2)
                except Exception as e:
                    print("Problem fetching structure", elem, "with", fetch_list_original[0], ":", e)
                    print(low_repeated)
            else:
                cmd.fetch(elem.split('_')[0], discrete=1)
                cmd.select(f"{elem}_temp", f"{elem.split('_')[0]} and chain {elem.split('_')[1]}")
                cmd.create(f'{elem}_temp', f'{elem}_temp')
                cmd.delete(elem.split('_')[0])
                temp_list.append(elem)
            #print("Fetched and modified:", elem, "and", fetch_list_original[0])

        for elem in batch:
            try:
                if elem not in low_repeated:
                    cmd.align(elem, fetch_list_original[0])
                else:
                    cmd.align(f'{elem}_temp', fetch_list_original[0])
            except Exception as e:
                print("Problem aligning structure", elem, "with", fetch_list_original[0], ":", e)

        # Process the current batch: compute center of geometry for each ligand
        batch_df = to_test_subset[to_test_subset['structureId'].isin([fetch_list_original[0]] + batch)]
        for index, row in batch_df.iterrows():
            try:
                if row["structureId"] not in low_repeated:
                    cog = compute_center_of_geometry('{} and resi {}* and resn {}'.format(row['structureId'], str(int(row['resid'])), row['ligandId']))
                else:
                    cog = compute_center_of_geometry('{}_temp and resi {}* and resn {}'.format(row['structureId'], str(int(row['resid'])), row['ligandId']))
                cog_round = ['%.3f' % elem for elem in cog]
                cog_float = [float(i) for i in cog_round]
                cog_str = str(cog_float)
                to_test_subset.loc[index, 'site_coords'] = cog_str
            except Exception as e:
                print('Error processing structureId:', row['structureId'], 'resid:', row['resid'], ';', e)

        # Convert the site_coords string into a list of floats when applicable
        to_test_subset['site_coords'] = to_test_subset['site_coords'].apply(
            lambda x: [float(i) for i in x.strip('][').split(', ')] if isinstance(x, str) else x
        )
    
    # Use agglomerative clustering to assign ligand sites
    valid_df = to_test_subset[to_test_subset['site_coords'].apply(lambda x: isinstance(x, list))]
    if not valid_df.empty:
        coords_array = np.array(valid_df['site_coords'].tolist())
        labels, centroids = get_clusters(coords_array, cutoff=7.0)
        centroids_str = [str(centroids[label, :]) for label in labels]
        to_test_subset.loc[valid_df.index, 'new_site'] = labels
        to_test_subset.loc[valid_df.index, 'site_centroid'] = centroids_str
    else:
        to_test_subset['new_site'] = None
        to_test_subset['site_centroid'] = None

    # Record the primary reference structure
    to_test_subset["reference_structureId"] = fetch_list_original[0]
    
    try:
        to_test_subset = to_test_subset.sort_values(by=['new_site'])
        to_test_subset['clusterNumber95'] = clusterNumber
        to_test_reconstructed = pd.DataFrame(to_test_subset.values, columns=to_test_subset.columns)
        pickle.dump(to_test_reconstructed, open(f'sites_cluster_{clusterNumber}.p', 'wb'))
    except Exception as e:
        print("Sorting issue:", e)
        print(to_test_subset)

if __name__ == '__main__':
    # Load the main monomers dataframe
    monomers = pd.read_pickle('combined.p')
    monomers = monomers[monomers['resid'].notna()]
    unique_clusterNumbers = list(monomers.clusterNumber95.unique())
    
    print("Processing clusters in parallel...")
    # Process each cluster in parallel
    results = Parallel(n_jobs=126)(
        delayed(get_sites_holo)(clusterNumber) for clusterNumber in unique_clusterNumbers
    )
    #concatenate all results in the folder get_sites
    #create all_sites_df with the columns Index(['structureId', 'resid', 'ligandId', 'site_coords', 'new_site','site_centroid', 'reference_structureId', 'clusterNumber95']
    dfs = []
    for file in os.listdir('./'):
        if file.startswith('sites_cluster_'):
            df = pd.read_pickle(os.path.join('./', file))
            dfs.append(df)
    all_sites_df = pd.concat(dfs, ignore_index=True)
    # Concatenate all per-cluster dataframes into one
    #all_sites_df = pd.concat(results, ignore_index=True)

    print("Concatenated all cluster dataframes.")
    
    pickle.dump(all_sites_df, open('all_sites_df.p', 'wb'))
    print(all_sites_df.columns)
    print(monomers.columns)
    print("all_sites_df.p has been saved")
    # Merge the sites information with the original combined data on structureId, resid, and ligandId
    #merged = pd.merge(monomers, all_sites_df, on=["structureId", "resid", "ligandId"], how='left')
   # print("Merged sites data with combined.p data.")
    
    # Save the merged dataframe to a new pickle file
    #pickle.dump(merged, open('combined_with_sites.p', 'wb'))
