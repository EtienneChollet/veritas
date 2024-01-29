# Custom Imports
from veritas.models import Unet

if __name__ == "__main__":
    import torch
    torch.no_grad()
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    #torch.set_float32_matmul_precision('medium')

    # New unet
    unet = Unet(
        version_n=4,
        device='cuda',
        model_dir='paper_models_context_64',
        )
    unet.new(nb_levels=6, nb_features=[32, 64, 128, 256, 512, 512])

    # Load unet (retraining)
    #unet = Unet(
    #    model_dir='paper_models_context_64',
    #    version_n=3)
    #unet.load(type='last')

    n_train = 1000
    n_steps = 200000

    n_gpus = 4
    accum_grad = 1
    batch_size = 1
    batch_sz_eff = batch_size * n_gpus * accum_grad
    epochs = (n_steps * batch_sz_eff) // n_train

    unet.train_it(
        data_experiment_number=8,
        subset=n_train,
        epochs=epochs,
        batch_size=batch_size,
        texturize_vessels=True,
        z_decay=True,
        i_max=0.99,
        accumulate_gradient_n_batches=accum_grad,
        loader_device='cuda'
    )