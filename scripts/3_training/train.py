# Custom Imports
from veritas.models import Unet


#{
#   "parenchyma": {
#       "nb_classes": 8,
#        "shape": 5
#    },
#    "gamma": [0.5, 2],
#    "z_decay": [32],
#    "noise": [0.2, 0.4]
#}

if __name__ == "__main__":
    import torch
    torch.no_grad()
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    #torch.set_float32_matmul_precision('medium')

    # New unet
    #unet = Unet(version_n=1)
    #unet.new(nb_levels=2, nb_features=[8, 8])
    
    # Load unet (retraining)
    unet = Unet(version_n=2)
    unet.load(type='last')

    unet.train_it(
        data_experiment_number=1,
        subset=1000,
        train_to_val=0.95,
        epochs=1000,
        batch_size=1,
        loader_device='cuda',
        accumulate_gradient_n_batches=1,
        check_val_every_n_epoch=1,
        )