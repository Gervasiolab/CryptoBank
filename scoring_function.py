import MDAnalysis as mda
from scipy.spatial import distance
import numpy as np
import pickle

def parse_xyz_file(xyz_filepath, apo, holo, lig_id, label):
    """
    Parses the input xyz file and extracts ligand and protein atom positions.
    The xyz files correspond to the aligned holo-apo pairs.

    Parameters:
    ----------
    xyz_filepath : str
        Path to the input xyz file.
    holo : str
        Holo protein identifier.
    apo : str
        Apo protein identifier.
    lig_id : int
        Ligand ID.
    label : str
        Either "holo" or "apo", determines the protein type.

    Returns:
    -------
    tuple:
        (lig_atm_name, lig_atm_pos, prot_atm_name, prot_atm_pos)
    """
    lig_label = f'holo_{holo}_{lig_id}'
    prot_label = f'holo_{holo}' if label == 'holo' else f'apo_{apo}'

    ligand_positions, protein_positions = [], []
    lig_line_number, prot_line_number = -1, -1  # -1 means "not found"

    with open(xyz_filepath, "r") as f:
        for l, line in enumerate(f):
            line = line.strip()  # Remove newline characters

            if line == lig_label:
                lig_line_number = l
                n_ligand = int(previous_line)
            elif line == prot_label:
                prot_line_number = l
                n_prot = int(previous_line)

            # Extract lines for ligand
            if lig_line_number > 0 and lig_line_number < l <= (lig_line_number + n_ligand):
                ligand_positions.append(line)

            # Extract lines for protein
            if prot_line_number > 0 and prot_line_number < l <= (prot_line_number + n_prot):
                protein_positions.append(line)

            previous_line = line

    # Convert string positions into float numpy arrays
    lig_atm_name, lig_atm_pos = zip(*[(line.split()[0], np.array(line.split()[1:], dtype=float)) for line in ligand_positions])
    prot_atm_name, prot_atm_pos = zip(*[(line.split()[0], np.array(line.split()[1:], dtype=float)) for line in protein_positions])

    return list(lig_atm_name), np.array(lig_atm_pos), list(prot_atm_name), np.array(prot_atm_pos)

def create_universe(atom_names, atom_positions):
    """
    Creates an MDAnalysis Universe from atom names and positions.

    Parameters:
    ----------
    atom_names : list of str
        List of atom names.
    atom_positions : np.ndarray
        Array of atom positions.

    Returns:
    -------
    mda.Universe
        MDAnalysis Universe object containing the atoms.
    """
    universe = mda.Universe.empty(len(atom_names), n_residues=1,
                                  atom_resindex=np.zeros(len(atom_names)),
                                  residue_segindex=np.zeros(1),
                                  trajectory=True)
    
    universe.add_TopologyAttr('name', atom_names)
    universe.atoms.positions = atom_positions

    return universe

def split_ligand_into_segments(ligand, n_lig_splits, apo, holo, lig_id):
    """
    Splits a ligand into `n_lig_splits` sections along its longest principal axis.
    If the ligand has fewer than 2 atoms or if all atoms are collinear (degenerate),
    assigns all atoms to the first segment and returns empty segments for the rest.
    """
    ligand_pos = ligand.atoms.positions

    # If there are fewer than 2 atoms, avoid PCA and assign atoms to first segment
    if ligand_pos.shape[0] < 2:
        segments = [np.arange(ligand_pos.shape[0])] + [np.array([], dtype=int)] * (n_lig_splits - 1)
        print(f"Ligand {lig_id} for apo {apo} and holo {holo} has less than 2 atoms. All atoms assigned to first segment.")
        return segments
    
    # Compute principal axes (PCA)
    covariance_matrix = np.cov(ligand_pos.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    longest_axis = eigenvectors[:, np.argsort(eigenvalues)[-1]]  # Largest eigenvalue's eigenvector

    # Project positions onto the longest axis
    projections = np.dot(ligand_pos, longest_axis)
    min_proj, max_proj = np.min(projections), np.max(projections)
    
    # If all projections are identical, return all atoms in the first segment
    if np.isclose(max_proj, min_proj):
        segments = [np.arange(ligand_pos.shape[0])] + [np.array([], dtype=int)] * (n_lig_splits - 1)
        print(f"Ligand {lig_id} for apo {apo} and holo {holo} has all atoms collinear. All atoms assigned to first segment.")
        return segments

    #if either ligand_pos.shape[0] < 2 or np.isclose(max_proj, min_proj) write the apo, holo and lig_id to a file
    if ligand_pos.shape[0] < 2 or np.isclose(max_proj, min_proj):
        with open("degenerate_ligands.txt", "a") as f:
            f.write(f"{apo}, {holo}, {lig_id}\n")

    # Define equal sections along the projection range
    section_size = (max_proj - min_proj) / n_lig_splits
    subsections = []
    for i in range(n_lig_splits):
        lower_bound = min_proj + i * section_size
        if i == n_lig_splits - 1:  # Last section
            subsection_indices = np.where(projections >= lower_bound)[0]
        else:
            upper_bound = min_proj + (i + 1) * section_size
            subsection_indices = np.where((projections >= lower_bound) & (projections < upper_bound))[0]
        subsections.append(subsection_indices)
    
    return subsections



def compute_distance_matrices(ligand_segments, protein):
    """
    Computes Euclidean distance matrices between ligand segments and protein.

    Parameters:
    ----------
    ligand_segments : list of mda.AtomGroup
        List containing ligand subsections (each as an MDAnalysis AtomGroup).
    protein : mda.Universe
        MDAnalysis Universe containing protein atoms.

    Returns:
    -------
    list of np.ndarray
        List of distance matrices (one per ligand subsection).
    """
    dist_matrices = []
    for lig_section in ligand_segments:
        if lig_section.atoms.positions.size == 0 or protein.atoms.positions.size == 0:
            dist = np.empty(0)
        else:
            dist = distance.cdist(lig_section.atoms.positions, protein.atoms.positions, 'euclidean')
        dist_matrices.append(dist)

    return dist_matrices

def return_cutoff_mat(input_mat, min_dist, max_dist):
    """
    Filters values in input_mat to keep only those within the range [min_dist, max_dist].

    Parameters:
    ----------
    input_mat : np.ndarray
        Input matrix containing numerical values.
    min_dist : float
        Lower bound of the distance range.
    max_dist : float
        Upper bound of the distance range.

    Returns:
    -------
    np.ndarray
        A filtered array containing values within [l, u], stored as an object array.
    """
    return np.array(input_mat[(input_mat >= min_dist) & (input_mat <= max_dist)], dtype=object)

def create_distmat(xyz_filepath, apo, holo, lig_id, label, n_lig_splits):
    """
    Computes Euclidean distance matrices between a ligand and a protein, splitting the ligand into
    multiple segments based on the specified number of splits along the ligand's longest axis.

    Parameters:
    ----------
    xyz_filepath : str
        Path to the input xyz file containing the protein-ligand complex.
    holo : str
        Holo protein identifier.
    apo : str
        Apo protein identifier.
    lig_id : int
        Ligand ID, identifying the specific ligand.
    label : str
        String that specifies whether the calculation is for the 'holo' or 'apo' protein state.
    n_lig_splits : int
        Number of segments to split the ligand into along its longest axis for distance matrix calculation.

    Returns:
    -------
    dist_matrices : list
        A list of Euclidean distance matrices, where each matrix corresponds to a ligand segment
        compared with the protein.
    normalization_factor : list
        A list of indices corresponding to valid ligand segments used in the distance matrix calculation.
        This is returned only if the label is "holo".
    """
    # Step 1: Parse the structure file
    lig_names, lig_pos, prot_names, prot_pos = parse_xyz_file(xyz_filepath, apo, holo, lig_id, label)
    # Step 2: Create MDAnalysis Universes
    lig_mda = create_universe(lig_names, lig_pos)
    prot_mda = create_universe(prot_names, prot_pos)

    # Step 3: Split ligand into `n_lig_splits` sections along its longest axis
    sections = split_ligand_into_segments(lig_mda, n_lig_splits, apo, holo, lig_id)
    lig_seg = [lig_mda.atoms[idx] for idx in sections]

    # Step 4: Compute distance matrices
    dist_matrices = compute_distance_matrices(ligand_segments=lig_seg, protein=prot_mda)

    # Step 5: Determine normalization factor
    if label=="holo":
        normalization_factor = [i for i, indices in enumerate(sections) if len(indices) > 0]
        return dist_matrices, normalization_factor
    else:
        return dist_matrices

def _sigmoid(x): 
    """
    Computes the sigmoid of x using a numerically stable implementation.
    
    The sigmoid function is defined as:
        sigmoid(x) = 1 / (1 + exp(-x))
    
    This implementation uses `np.logaddexp` to avoid overflow issues when calculating the exponent.
    
    Parameters:
    ----------
    x : float or np.ndarray
        Input value(s) for which the sigmoid is computed.
    
    Returns:
    -------
    np.ndarray
        The sigmoid of the input value(s).
    """
    return np.exp(-np.logaddexp(0., -x + 10))

def _d_sigmoid(x):
    """
    Computes the derivative of the sigmoid function.

    The derivative of the sigmoid function is given by:
        d(sigmoid(x)) / dx = sigmoid(x) * (1 - sigmoid(x))

    This function computes the sigmoid of `x` and then applies the derivative formula.

    Parameters:
    ----------
    x : float or np.ndarray
        Input value(s) for which the sigmoid derivative is computed.

    Returns:
    -------
    np.ndarray
        The derivative of the sigmoid of the input value(s).
    """
    s = _sigmoid(x)
    return s * (1 - s)

def shell_potential(k, r, shells, min_dist, max_dist):
    """
    Calculates the potential energy and the number of distances within each shell 
    defined by distance bounds for a given set of distances `r` and fitting parameters `k`.
    
    Parameters:
    ----------
    k : np.ndarray
        Array of fitting parameters for each shell.
    r : np.ndarray
        Array of distances for which the potential energy is calculated.
    shells : int
        The number of shells into which the distance space is divided.
    min_dist : float
        Lower bound of the distance range.
    max_dist : float
        Upper bound of the distance range.
    
    Returns:
    -------
    y : np.ndarray
        Array of potential energy values for each shell. Each value is the product of the 
        fitting parameter `k` and the number of distances in that shell.
    dy : np.ndarray
        Array of counts of the distances in each shell.
    """
    
    # Divide the distance space into n shells
    dist_bounds = np.linspace(min_dist, max_dist, shells + 1)
    
    y = []  # List to store the potential values for each shell
    dy = [] # List to store the derivative of the potential
    
    for n in range(shells):
        # Get distances in the current shell
        shell_r = r[(dist_bounds[n] <= r) & (r < dist_bounds[n + 1])]
        
        # Count distances in each bin and multiply by the fitting parameter k
        y.append(k[n] * len(shell_r))
        dy.append(len(shell_r))
    
    return np.array(y), np.array(dy)

def model_predict(model_k, apo_matrices, holo_matrices, valid_segments, lig_splits, shells, min_dist, max_dist):
    """
    Predicts crypticity based on shell potential calculations 
    for apo (unbound) and holo (bound) states.

    Parameters:
    ----------
    model_k : np.ndarray
        Parameter array used in the shell potential calculation.
    apo_matrices : list of np.ndarray
        List of distance matrices for the apo-ligand state.
    holo_matrices : list of np.ndarray
        List of distance matrices for the holo-ligand state.
    valid_segments : list of int
        Indices of ligand segments that contain atoms.
    n_lig_split : int
        Number of ligand segments.
    shells : int
        number of shells considered to calculate the potential.
    min_dist : float
        Lower bound of the distance range.
    max_dist : float
        Upper bound of the distance range.

    Returns:
    -------
    p : np.ndarray
        Mean predicted binding probability.
    p_list : list of np.ndarray
        List of predicted probabilities for each ligand segment.
    eff_p_list : np.ndarray
        Adjusted probabilities scaled by `n_lig_split`.
    """
    v_list = []
    p_list = []
    lig_around_apo = 0
    lig_around_holo = 0
    for i in range(lig_splits):
        # FUNNY NOTE: v = np.zeros([]) creates a scalar, whereas v = np.zeros((1,)) 
        # creates a 1D array. NumPy treats them differently, leading to negligible 
        # numerical differences when computing _sigmoid(v). LOL.
        # v = np.zeros([])
        v = np.zeros((1,))

        apo_start = i * int(2 * shells)
        apo_end = apo_start + int(shells)
        holo_start = apo_end
        holo_end = holo_start + int(shells)

        if i in valid_segments:  # Example: [0,1,2] if all ligand segments contain atoms
            if apo_matrices[i].size != 0:
                pot_apo, d_pot_apo = shell_potential(model_k[apo_start:apo_end], apo_matrices[i], shells, min_dist, max_dist)
                #print("pot_apo for lig_id", i, "is", pot_apo)
                energy_sys_apo = np.sum(pot_apo) / len(apo_matrices[i])
                v += energy_sys_apo
                lig_around_apo = 1

            if holo_matrices[i].size != 0:
                #print("holo_matrices[i] for lig_id", i, "is", holo_matrices[i])
                pot_holo, d_pot_holo = shell_potential(model_k[holo_start:holo_end], holo_matrices[i], shells, min_dist, max_dist)
                #print("pot_holo for lig_id", i, "is", pot_holo)
                energy_sys_holo = np.sum(pot_holo) / len(holo_matrices[i])
                v += energy_sys_holo
                lig_around_holo = 1
            v /= 2
        v_list.append(np.array(v))
        #print("v for lig_id", i, "is", v)
        p_list.append(_sigmoid(v))
    eff_p_list = np.array([x / lig_splits for x in p_list])
    p = np.array([np.mean(p_list)])

    return p, p_list, eff_p_list, v_list, lig_around_apo, lig_around_holo

def make_prediction(predictions):
    """
    Converts probability predictions into binary labels (0 or 1) based on a threshold of 0.5.
    
    Parameters:
    ----------
    predictions : np.ndarray or list
        Array or list of predicted probabilities (values between 0 and 1).
    
    Returns:
    -------
    pred_label : np.ndarray
        Array of binary labels (1 if probability >= 0.5, else 0).
    """
    # Convert probabilities to binary labels based on threshold 0.5
    return np.where(predictions >= 0.5, 1, 0)



#######################################################
###############      PARAMETERS SECTION      #########
#######################################################

# Minimum and maximum radii of the shells (in Angstroms)
min_dist = 0  # Minimum radius of the shell
max_dist = 4  # Maximum radius of the shell

# Optimal hyperparameters for the model
# Number of shells (N) - Defines how many radial bins are used to segment the distance range
n_shells = 4

# Number of ligand segments (S) - Defines how many segments the ligand is divided into
n_lig_splits = 3
# Load the model weights (k) 
weights = np.load('3seg_distmodel_parameters.npy')

#######################################################
###############    END PARAMETERS SECTION    #########
#######################################################



def get_score(xyz_apo, xyz_holo, apo_id, holo_id, lig_resid, n_lig_splits):
    """
    Computes the crypticity score for a ligand-protein binding scenario.

    Parameters:
    ----------
    xyz_apo : str
        Filepath to the apo structure (XYZ format).
    xyz_holo : str
        Filepath to the holo structure (XYZ format).
    holo_id : str
        Identifier for the holo structure.
    apo_id : str
        Identifier for the apo structure.
    lig_resid : int
        Residue ID of the ligand.

    Returns:
    -------
    tuple:
        - score (np.ndarray): Final crypticity score.
        - probabilities (list of np.ndarray): Raw probability values for each ligand section.
    """
    if n_lig_splits == 3:
        weights = np.load('3seg_distmodel_parameters.npy')
    elif n_lig_splits == 1:
        weights = pickle.load(open('weights1seg.pkl', 'rb'))
    else:
        raise ValueError(f"Invalid number of ligand splits: {n_lig_splits}")
    # Compute distance matrices for apo and holo structures
    dist_matrices_apo = create_distmat(
        xyz_filepath=xyz_apo,
        apo=apo_id,  
        holo=holo_id, 
        lig_id=lig_resid, 
        label='apo', 
        n_lig_splits=n_lig_splits
    )
    dist_matrices_holo, valid_lig_splits = create_distmat(
        xyz_filepath=xyz_holo,
        apo=apo_id, 
        holo=holo_id, 
        lig_id=lig_resid, 
        label='holo', 
        n_lig_splits=n_lig_splits
    )
    # Apply cutoff filtering
    dist_matrices_apo = [
        return_cutoff_mat(x, min_dist, max_dist) for x in dist_matrices_apo
    ]
    dist_matrices_holo = [
        return_cutoff_mat(x, min_dist, max_dist) for x in dist_matrices_holo
    ]
    
    # Compute crypticity score
    score, probabilities, eff_p_list, v_list, lig_around_apo, lig_around_holo= model_predict(
        model_k=weights, 
        apo_matrices=dist_matrices_apo, 
        holo_matrices=dist_matrices_holo, 
        valid_segments=valid_lig_splits, 
        lig_splits=n_lig_splits, 
        shells=n_shells, 
        min_dist=min_dist, 
        max_dist=max_dist
    )
    return score.round(2)[0], [x.round(2)[0] for x in probabilities], lig_around_apo, lig_around_holo
