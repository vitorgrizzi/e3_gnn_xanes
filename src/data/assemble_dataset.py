import os
import numpy as np
from ase import Atoms
from ase.io import write
from ase.db import connect

def parse_fdmnes_input(filepath):
    """
    Parses the fdmnes_in.txt file to extract structural data and absorber info.
    
    Returns:
        tuple: (Z_absorber, absorber_index, cell_params, atom_data)
    """
    z_absorber = None
    absorber_idx = None
    cell_params = [] # Stores [a, b, c] and [alpha, beta, gamma]
    atom_data = []   # Stores (Z, x, y, z) tuples
    
    reading_crystal = False
    reading_atoms = False
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        clean_line = line.strip()
        parts = clean_line.split()
        
        if not parts:
            continue
            
        # 1. Extract Z_absorber
        if parts[0] == 'Z_absorber':
            # Usually the value is on the next line
            if i + 1 < len(lines):
                try:
                    z_absorber = int(lines[i+1].strip().split()[0])
                except ValueError:
                    pass

        # 2. Extract Absorber Index
        if parts[0] == 'Absorber':
            # Value can be on same line "Absorber 4" or next line
            if len(parts) > 1:
                try:
                    absorber_idx = int(parts[1]) - 1
                except ValueError:
                    pass
            elif i + 1 < len(lines):
                try:
                    absorber_idx = int(lines[i+1].strip().split()[0]) - 1
                except ValueError:
                    pass

        # 3. Detect Crystal Block
        if parts[0] == 'Crystal':
            reading_crystal = True
            reading_atoms = False # Atoms start after cell dims
            continue

        # 4. Parse Crystal/Atom Data
        if reading_crystal:
            # We expect numeric data. If we hit a keyword, stop.
            try:
                # Try converting the line to floats
                nums = [float(x) for x in parts]
                
                # First two lines after Crystal are cell params
                if len(cell_params) < 2:
                    if len(nums) == 6:
                        cell_params.append(nums[:3])
                        cell_params.append(nums[3:])
                    else:
                        cell_params.append(nums)
                    if len(cell_params) == 2:
                        reading_atoms = True # Next lines should be atoms
                elif reading_atoms:
                    # Expecting [Z, x, y, z]
                    if len(nums) >= 4:
                        atom_data.append(nums[:4])
            except ValueError:
                # We likely hit the next keyword (e.g., "Convolution", "Green"), end of crystal block
                reading_crystal = False
                reading_atoms = False
    
    # When no `absorber` is specified, all atoms with `Z_absorber` are absorbers
    if absorber_idx is None:
        z_numbers = np.array(atom_data, dtype=int)[:, 0]
        absorber_idx = [i for i, Z in enumerate(z_numbers) if Z == z_absorber]

    return z_absorber, absorber_idx, cell_params, atom_data

def save_to_database(dataset, db_path):
    # Connect to database (creates it if it doesn't exist)
    with connect(db_path + '/xanes_dataset.db', append=False) as db: # Append=False ensures we start fresh if the file exists
        
        for atoms in dataset:
            # Although ASE DB can store 'info' dicts automatically, we store in `data`
            # parameter that is designed for array-like objects. 
            spectrum = atoms.info.pop('FDMNES-xanes', None)
            source = atoms.info.pop('source_dir', 'unknown')
            

            # key_value_pairs: Searchable metadata (values must be scalars or strings)
            # data: Heavy arrays (Spectra, Forces), stored as binary
            db.write(atoms, 
                     data={'xanes': spectrum}, 
                     key_value_pairs={'source': source, 
                                      'absorber_z': 26,
                                      'n_atoms': len(atoms)})
            
    print(f"Saved {len(dataset)} structures to SQLite database: {db_path}")

def process_directory_tree(root_dir):
    """
    Recursively searches directories for FDMNES data and compiles an ASE trajectory.
    """
    dataset = []
    
    # Recursively traversing root_dir tree
    for root, dirs, files in os.walk(root_dir):
        
        # 1. Check for convolution file
        conv_file = None
        for file in files:
            if file.endswith('conv.txt'):
                conv_file = file
                break
        
        # If no spectrum found, skip this folder
        if conv_file is None:
            continue
            
        # 2. Check for input file
        input_file_name = 'fdmnes_in.txt' # Adjust if filename varies
        if input_file_name not in files:
            continue
            
        # Paths
        conv_path = os.path.join(root, conv_file)
        input_path = os.path.join(root, input_file_name)
        
        try:
            # --- Extract Spectrum ---
            # Load raw (N, 2) data. Comments usually start with ! or #, adjust if needed.
            spectrum = np.loadtxt(conv_path, skiprows=1)
            
            # --- Extract Structure ---
            z_abs, abs_idx, cell_data, atoms_list = parse_fdmnes_input(input_path)
            
            if not atoms_list:
                print(f"Skipping {root}: Incomplete structure data.")
                continue

            # Convert to numpy arrays
            cell_lengths = cell_data[0] # a, b, c
            cell_angles = cell_data[1]  # alpha (b,c), beta (a,c), gamma (a,b)
            full_cell_par = cell_lengths + cell_angles

            atom_arr = np.array(atoms_list)
            atomic_numbers = atom_arr[:, 0].astype(int)
            scaled_positions = atom_arr[:, 1:4] # Fractional coordinates

            # Create ASE Atoms Object            
            atoms = Atoms(
                numbers=atomic_numbers,
                scaled_positions=scaled_positions,
                cell=full_cell_par,
                pbc=True
            )
            
            # Tagging absorber atom
            tags = np.zeros(len(atoms), dtype=int)
            tags[abs_idx] = 1
            atoms.set_tags(tags)

            # Storing spectrum data
            atoms.info['FDMNES-xanes'] = spectrum
            
            # Use directory name as an ID/Name
            atoms.info['source_dir'] = root
            
            dataset.append(atoms)
            print(f"Processed: {root}")

        except Exception as e:
            print(f"Error processing {root}: {e}")

    # Save Dataset to SQLite database
    if dataset:
        save_to_database(dataset, root_dir)
    else:
        print("\nNo valid data found.")

process_directory_tree('/lcrc/project/Bio_catalysis/lpretzie/fm_catal/convRuns')
