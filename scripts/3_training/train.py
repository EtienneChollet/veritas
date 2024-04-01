import torch

# Custom Imports
from veritas.models import Unet
import os

#torch.set_float32_matmul_precision('medium')
#os.environ['CUDA_VISIBLE_DEVICES']='0'


if __name__ == "__main__":
    #torch.set_float32_matmul_precision('medium')

    ##################
    version_n = 6
    data_params = 'complex'
    synth_params = 'complex'
    ##################

    data_params_dict = {'complex': 1,
                        'simple': 2}
    #data_experiment_number = data_params_dict[data_params]
    data_experiment_number = 4
    print(f'Using data from experiment {data_experiment_number}')

    # New unet
    unet = Unet(
        version_n=version_n,
        synth_params=synth_params,
        model_dir='models',
        learning_rate=1e-2 #1e-4,
        )
    
    unet.new(
        nb_levels=4,
        nb_features=[32, 64, 128, 256],
        dropout=0)

    # Load unet (retraining)
    #unet = Unet(
    #    version_n=2564,
    #    synth_params='simple',
    #    model_dir='lets_get_small_vessels-v2',
    #    learning_rate=1e-4
    #    )
    #unet.load(type='last')

    # exp0001 has 368 labels
    # exp0003 has 270 labels
    n_vol = 10
    train_to_val = 0.8
    n_steps = 1e5
    n_gpus = 1
    accum_grad = 1
    batch_size = 1

    n_train = n_vol*train_to_val
    batch_sz_eff = batch_size * n_gpus * accum_grad
    epochs = int((n_steps * batch_sz_eff) // n_train)
    print(f'Training for {epochs} epochs')
    unet.train_it(
        data_experiment_number=data_experiment_number,
        epochs=epochs,
        batch_size=batch_size,
        accumulate_gradient_n_batches=accum_grad,
        subset=n_vol,
        num_workers=3,
        train_to_val=train_to_val
    )