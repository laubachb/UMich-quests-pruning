import os
import glob
import numpy as np
import pandas as pd
from ase.io import read
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Conversion Constants ---
# 1 Hartree = 27.2114 eV
# 1 Angstrom = 1.8897161646320724 Bohr
CONVERSION_FACTOR = 27.2114 * 1.8897161646320724  # ~51.422067

def main(save_file="no_corrected_results.csv"):
    logger.info("Loading ground truth dataset: random_C.xyz")
    dataset = read("random_C.xyz", ':')

    # --- 1. Load and Convert Ground Truth ---
    true_forces_list = []
    for atoms in dataset:
        # Convert Ground Truth from Hartree/Bohr to eV/A
        converted_forces = atoms.get_forces() * CONVERSION_FACTOR
        true_forces_list.append(converted_forces)

    true_forces = np.concatenate(true_forces_list)
    logger.info(f"Loaded {len(dataset)} configs, {len(true_forces)} total atoms.")
    logger.info(f"Applied unit conversion factor (Hartree/Bohr -> eV/A): {CONVERSION_FACTOR:.4f}")

    # --- 2. Map Atom Groups ---
    group_names = sorted(list(set([atoms.info['config_tag'] for atoms in dataset])))
    per_atom_group_labels = []
    for atoms in dataset:
        group_idx = group_names.index(atoms.info['config_tag'])
        per_atom_group_labels.append(np.ones(len(atoms)) * group_idx)
    per_atom_group_labels = np.concatenate(per_atom_group_labels).astype(int)

    # --- 3. Process the Predicted Forces ---
    model_results = []
    
    # We only need the 'Full' arrays, we can mask the groups dynamically
    search_pattern = os.path.join("no_forces", "Full_*.npy")
    full_files = glob.glob(search_pattern)

    if not full_files:
        logger.error("No files matching 'Full_*.npy' found in no_forces/")
        return

    for npy_file in full_files:
        # Parse the filename: "Full_{prune_rate}_{clean_path}.npy"
        basename = os.path.basename(npy_file)
        name_without_ext = basename.replace('.npy', '')
        
        parts = name_without_ext.split('_', 2)
        if len(parts) >= 3:
            prune_rate = parts[1]
            clean_path = parts[2]
        else:
            prune_rate = "Unknown"
            clean_path = name_without_ext

        # Load predictions
        pred_forces = np.load(npy_file)
        
        if pred_forces.shape != true_forces.shape:
            logger.error(f"Shape mismatch for {basename} (Pred: {pred_forces.shape}, True: {true_forces.shape}). Skipping.")
            continue

        # Calculate absolute error (eV/A)
        errors = np.abs(pred_forces - true_forces)

        # Calculate Group MAEs
        for gi, group in enumerate(group_names):
            group_mask = per_atom_group_labels == gi
            mae = np.sum(errors * group_mask[:, None]) / group_mask.sum() / 3
            
            model_results.append({
                'group': group,
                'rate': prune_rate,
                'forces_mae': mae,
                'train_file': 'Unknown (Missing hparams)',
                'ckpt_file': 'Unknown (Missing hparams)',
                'compiled_model': clean_path,
            })

        # Calculate Full MAE
        full_mae = np.mean(errors)
        model_results.append({
            'group': 'Full',
            'rate': prune_rate,
            'forces_mae': full_mae,
            'train_file': 'Unknown (Missing hparams)',
            'ckpt_file': 'Unknown (Missing hparams)',
            'compiled_model': clean_path,
        })
        
        logger.info(f"Processed model {clean_path[:30]}... (Rate: {prune_rate}) | Full MAE: {full_mae:.4f} eV/A")

    # --- 4. Save the Output ---
    df = pd.DataFrame(model_results)
    df.to_csv(save_file, index=False)
    logger.info(f"\nSuccess! Saved {len(model_results)} rows to {save_file}")

if __name__ == '__main__':
    main()