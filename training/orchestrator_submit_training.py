from orchestrator.utils.setup_input import setup_orch_modules, read_input
from orchestrator.utils.data_standard import (
    SELECTOR_PROPERTY_MAP,
    SELECTION_MASK_KEY,
)
import numpy as np
from copy import deepcopy

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
    trainer,
    workflow,
) = setup_orch_modules(init_dicts)
runtime_args = init_dicts.get('runtime_args')
desc_handle = runtime_args.get('descriptor_dataset')
training_base =  runtime_args.get('pruned_dataset_base')

training_jobs = []
#for pruning, pid in zip([0.95], [1]):
#range 1,10 will get the 95-20% pruning. _10 is messed up currently. Zip will truncate so can leave pruning array alone
for pruning, pid in zip(np.append(np.append(0.95, np.linspace(0.9, 0.1, 9)), 0.05), range(1,10)):
    # pruning variable is amount pruned (so 90% to 10% of dataset remaining)
    prune_handle = f'{training_base}_{pid}'
    training_job_id = trainer.submit_train(
        storage=storage,
        dataset_list=prune_handle,
        workflow=workflow['workflow_lsf'],
        val_frac=0.0,
        eweight=0,
        fweight=1,
        vweight=0,
        #per_atom_weights=weights,
        apply_mask=True,
        **runtime_args.get('submit_train_args') # set remaining inputs specified in input file
    )
    trainer.logger.info(
        f'########## Prune {pruning} training job ID: {training_job_id} ##########'
    )
    print(
        f'########## Prune {pruning} training job ID: {training_job_id} ##########'
    )
    training_jobs.append(training_job_id)

print(f'All training jobs: {training_jobs}')
trainer.logger.info(f'All training jobs: {training_jobs}')
