from orchestrator.utils.setup_input import setup_orch_modules, read_input
from orchestrator.utils.data_standard import (
    SELECTOR_PROPERTY_MAP,
    SELECTION_MASK_KEY,
)
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
from functools import partial
import itertools

init_dicts = read_input('init_dicts.json')
(
    augmentor,
    descriptor,
    _,
    _,
    _,
    _,
    storage,
    _,
    _,
    workflow,
) = setup_orch_modules(init_dicts)
runtime_args = init_dicts.get('runtime_args')
desc_handle = runtime_args.get('descriptor_dataset')
configs_with_descriptors = storage.get_data(desc_handle)
storage.property_map = storage.get_dataset_property_map(desc_handle) # property_map just needs to not be None for new_properties functionality to work
compute_args={'descriptors_key': f'{descriptor.OUTPUT_KEY}_descriptors'}
new_prop_map = list(SELECTOR_PROPERTY_MAP.values())

# pruning = np.append(np.linspace(0.9, 0.1, 9), 0.05)
pruning = [0.1, 0.05]
num_pruning = len(pruning)

with Pool(num_pruning) as p:
    # pruning variable is amount pruned (so 5% to 95% of dataset remaining)
    full_args_list = zip(
        itertools.repeat(deepcopy(configs_with_descriptors)),
        itertools.repeat('percentage_fps'),
        pruning,
        itertools.repeat(None),
        itertools.repeat(compute_args),
        itertools.repeat(None),
        itertools.repeat(None),
        itertools.repeat(None),
    )
    pool_results = p.starmap(augmentor.simple_prune_dataset, full_args_list)
    
for prune_val, prune_set in zip(pruning, pool_results):
    mask_count = np.sum(np.concatenate([c.get_array(SELECTION_MASK_KEY) for c in prune_set]))
    # save the dataset
    prune_handle = storage.update_data(
        desc_handle,
        prune_set,
        use_orig_property_map=True,
        new_properties={new_prop_map[0]: new_prop_map[1]},
        updated_description=f'ChIMES dataset with QUESTS descriptors pruning value {prune_val}'
    )
    print(f'########## Prune {prune_val} dataset ({mask_count} atoms): {prune_handle} ##########')
    augmentor.logger.info(f'### Prune {prune_val} dataset: {prune_handle} ###')
