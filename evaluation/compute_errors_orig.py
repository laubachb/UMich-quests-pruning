import argparse
from ase.io import read
from nequip.integrations.ase import NequIPCalculator
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import subprocess
from tqdm import tqdm
import yaml

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(train_folder, save_file):
    dataset, group_names, per_atom_group_labels, selection_masks = load_data()
    checkpoint_files = find_checkpoint_files(train_folder)

    for ckpt in tqdm(checkpoint_files, desc='Evaluating checkpoints'):
        errors, prune_rate, train_file, ckpt_file, compiled_path, pred_forces = evaluate_one_model(train_folder, ckpt, dataset)

        clean_path = compiled_path.replace('/', '_')

        model_results = []
        for gi, group in enumerate(group_names):
            group_mask = per_atom_group_labels == gi

            mae = np.sum(errors*group_mask[:, None])/group_mask.sum()/3  # xyz components

            results_row = {
                'group': group,
                'rate': prune_rate,
                'forces_mae': mae,
                'train_file': train_file,
                'ckpt_file': ckpt_file,
                'compiled_model': compiled_path,
            }

            logger.info(f'{group=} ({group_mask.sum()} atoms), {prune_rate=}, mae={mae:.3f}')
            model_results.append(results_row)
            force_file_name = f'yes_forces/{group}_{prune_rate}_{clean_path}'
            np.save(force_file_name, pred_forces[group_mask])

        model_results.append({
            'group': 'Full',
            'rate': prune_rate,
            'forces_mae': np.mean(errors),
            'train_file': train_file,
            'ckpt_file': ckpt_file,
            'compiled_model': compiled_path,
        })

        force_file_name = f'yes_forces/Full_{prune_rate}_{clean_path}'
        np.save(force_file_name, pred_forces)
        save_model_results(save_file, model_results)
    
def save_model_results(results_file, model_results):
    """Save results for a single model to CSV file (append mode)."""
    if not model_results:
        return
        
    df_new = pd.DataFrame(model_results)
    
    # Check if file exists and has data
    if Path(results_file).exists() and os.path.getsize(results_file) > 0:
        # Append to existing file
        df_new.to_csv(results_file, mode='a', header=False, index=False)
    else:
        # Create new file with header
        df_new.to_csv(results_file, mode='w', header=True, index=False)
    
    logger.info(f"Saved {len(model_results)} results for current model to {results_file}")
    
def load_data():
     # Load the dataset and parse the group labels
    # dataset = read("DS_2km6t3p8hxx2_0.xyz", ':')
    dataset = read("random.xyz", ':')

    for atoms in dataset:
        atoms.info['true_energy'] = atoms.get_potential_energy()
        atoms.arrays['true_forces'] = atoms.get_forces()

    logger.info(f'Dataset: {len(dataset)} configs')

    group_names = sorted(list(set([atoms.info['config_tag'] for atoms in dataset])))
    per_atom_group_labels = []
    for atoms in dataset:
        group_idx = group_names.index(atoms.info['config_tag'])
        per_atom_group_labels.append(np.ones(len(atoms))*group_idx)

    per_atom_group_labels = np.concatenate(per_atom_group_labels)
    per_atom_group_labels = per_atom_group_labels.astype(int)

    # Load the selection masks from the pruned datasets
    prune_rates = np.concatenate([
        np.linspace(0.01, 0.1, 19),
        np.linspace(0.2, 0.9, 8),
    ])

    selection_masks = {}
    ##for rate in prune_rates:
    ##    fname = f'DS_2km6t3p8hxx2_0_{rate:.3f}_yes_clustering.xyz'
    ##    # fname = f'DS_2km6t3p8hxx2_0_{rate:.3f}_no_clustering.xyz'

    ##    masked_dataset = read(fname, ':')
    ##    mask = np.concatenate([
    ##        atoms.arrays['weights'] for atoms in masked_dataset]).astype(int)

    ##    selection_masks[rate] = mask

    ##    logging.debug(f'Expected/observed rate: {rate:.3f}/{mask.mean():.3f}')

    return dataset, group_names, per_atom_group_labels, selection_masks


def find_checkpoint_files(train_folder):
    """Find all checkpoint files (best.ckpt and best-v*.ckpt)."""
    checkpoint_files = []
    
    # Find best.ckpt files
    train_folder_obj = Path(train_folder)
    checkpoint_files.extend(train_folder_obj.glob("**/train_dir/best.ckpt"))
    
    # Find best-v*.ckpt files
    checkpoint_files.extend(train_folder_obj.glob("**/train_dir/best-v*.ckpt"))
    
    logger.info(f"Found {len(checkpoint_files)} checkpoint files")
    for ckpt in checkpoint_files[:5]:  # Show first 5
        logger.debug(f"  {ckpt}")
    
    return sorted(checkpoint_files)
    
def find_hparams_file(ckpt_file):
    """Find the corresponding hparams.yaml file for a checkpoint."""
    train_dir = ckpt_file.parent
    model_basename = ckpt_file.stem
    
    # Determine target version directory
    target_version = None
    if model_basename == "best":
        target_version = "version_0"
    elif model_basename.startswith("best-v"):
        version_num = model_basename.replace("best-v", "")
        target_version = f"version_{version_num}"
    
    # Try target version first
    if target_version:
        hparams_path = train_dir / target_version / "hparams.yaml"
        if hparams_path.exists():
            logger.debug(f"Found hparams.yaml in target version: {hparams_path}")
            return hparams_path
    
    # Fallback: search all version directories
    for version_dir in train_dir.glob("version_*"):
        hparams_path = version_dir / "hparams.yaml"
        if hparams_path.exists():
            logger.debug(f"Found hparams.yaml in: {hparams_path}")
            return hparams_path
    
    logger.warning(f"No hparams.yaml found for {ckpt_file}")
    return None

def evaluate_one_model(train_folder, ckpt_file, dataset):

    # Prepare file names
    rel_path = ckpt_file.relative_to(train_folder)
    parts = list(rel_path.parts[:-1])  # Remove train_dir
    parts.append(ckpt_file.stem)  # Add filename without extension
    model_id = "_".join(parts)

    compiled_path = os.path.join(train_folder, f"compiled_{model_id}.nequip.pth")

    # # Compile the model
    # cmd = [
    #     "nequip-compile",
    #     str(ckpt_file),
    #     str(compiled_path),
    #     "--device", "cuda",
    #     "--mode", "torchscript"
    # ]

    # logger.debug(cmd)
    #         
    # subprocess.run(
    #     cmd,
    #     capture_output=True,
    #     text=True,
    #     check=True
    # )
    
    if os.path.exists(compiled_path):
        logger.info(f"Successfully compiled model: {compiled_path}")
    else:
        logger.error(f"Compilation completed but output file not found: {compiled_path}")
                    
    # except subprocess.CalledProcessError as e:
    #     logger.error(f"Error compiling model {ckpt_file}: {e}")
    #     logger.error(f"STDOUT: {e.stdout}")
    #     logger.error(f"STDERR: {e.stderr}")
    #     return None
    # except Exception as e:
    #     logger.error(f"Unexpected error compiling model {ckpt_file}: {e}")
    #     return None
    
    # Load the model
    calc = NequIPCalculator.from_compiled_model(
        compile_path=str(compiled_path),
        device='cuda',
        species_to_type_name={}
    )
    
    true_forces = []
    pred_forces = []
    for atoms in dataset:
        # Set calculator
        atoms.calc = calc
        
        true_forces.append(atoms.arrays['true_forces'])
        pred_forces.append(atoms.get_forces())

    pred_forces = np.concatenate(pred_forces)
    true_forces = np.concatenate(true_forces)

    if np.any(np.isnan(pred_forces)):
        logger.error("Encountered NaN in predicted forces")

    errors = np.abs(pred_forces - true_forces)

    # Parse hparams.yaml
    hparams_file = find_hparams_file(ckpt_file)
    with open(hparams_file, 'r') as f:
        hparams = yaml.safe_load(f)

    train_file = hparams['info_dict']['data']['train_file_path']
    if 'DS_2km6t3p8hxx2_0.xyz' in train_file:
        prune_rate = 1.00
    else:
        prune_rate = train_file[train_file.find('DS_2km6t3p8hxx2_0_'):]
        prune_rate = prune_rate[:5]

    # Extract prune rate
    # prune_rate = "1.000"  # Default for full dataset
    
    # Pattern: _0.30.xyz
    match = re.search(r'_(\d+\.\d+)_yes_clustering\.xyz$', train_file)
    # match = re.search(r'_(\d+\.\d+)_no_clustering\.xyz$', train_file)
    if match:
        prune_rate = match.group(1)
    else:
        # Pattern: .30.xyz
        match = re.search(r'\.(\d+)\.xyz$', train_file)
        if match:
            prune_rate = f"0.{match.group(1)}"
            
    return errors, prune_rate, train_file, ckpt_file, compiled_path, pred_forces

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-folder',
        type=str,
        help="Folder in which to search for training runs."
    )

    parser.add_argument(
        '--save-file',
        type=str,
        help="CSV file in which to save the test results."
    )

    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    main(args.train_folder, args.save_file)
