import torch

# Custom Imports
from veritas.models import Unet
import os

#torch.set_float32_matmul_precision('medium')
#os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == "__main__":
    # exp 1000 is best for complex
    # exp 11 is best for simple
    data_experiment_number=256

    # New unet
    unet = Unet(
        version_n=2568,
        synth_params='complex',
        model_dir='lets_get_small_vessels',
        learning_rate=1e-4
        )
    unet.new(
        nb_levels=8,
        nb_features=[32, 64, 128, 256, 512, 512, 512, 512],
        dropout=0)

    # Load unet (retraining)
    #unet = Unet(
    #    version_n=1,
    #    synth_params='complex',
    #    model_dir='lets_get_small_vessels',
    #    learning_rate=1e-4,
    #    )
    #unet.load(type='last')

    n_train = 1000
    n_steps = 1e5
    n_gpus = 1
    accum_grad = 1
    batch_size = 1
    batch_sz_eff = batch_size * n_gpus * accum_grad
    epochs = int((n_steps * batch_sz_eff) // n_train)
    print(f'Training for {epochs} epochs')
    unet.train_it(
        data_experiment_number=data_experiment_number,
        epochs=epochs,
        batch_size=batch_size,
        accumulate_gradient_n_batches=accum_grad,
        subset=n_train
    )