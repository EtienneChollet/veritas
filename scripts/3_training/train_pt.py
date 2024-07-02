import torch
import traceback

from veritas.unet import (
    UNet,
    get_loaders,
    configure_optimizer,
    configure_criterion,
    log_model_graph,
    train_one_epoch,
    validate,
    log_hist,
)
from veritas.utils import save_config_to_json
from veritas.confocal import ConfocalVesselLabelTransform


def train_model(
        model, train_loader, val_loader, num_epochs=25, lr=0.001,
        model_dir='runs/base', weight_decay=1e-5,
        device='cuda'):

    best_vloss = 1.0

    try:
        optimizer = configure_optimizer(model, lr, weight_decay)
        writer = log_model_graph(model_dir, model, train_loader)
        # criterion_ = configure_criterion(_type='dice', weighted=True)
        criterion_ = configure_criterion(_type='dice')
        for epoch in range(num_epochs):
            # model, epoch, writer, loader, opt, criterion
            train_one_epoch(
                model, epoch, writer, train_loader, criterion_, optimizer,
                device)

            best_vloss = validate(
                model, epoch, writer, val_loader, optimizer, criterion_,
                device, best_vloss, model_dir)

            log_hist(model, epoch, writer)

    except Exception as e:
        print("An error occurred:", str(e))
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        writer.close()
        print("Training completed and CUDA memory cleaned up")
    writer.close()
    return model


def main():

    #####################################################
    model_dir = 'runs/base/version_12'
    #####################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    notes = "Same as 11 but no gaussian blurring and increased light range"

    training_config = {
        "num_epochs": 10000,
        "model_dir": model_dir,
        "lr": 0.01,
        'weight_decay': 1e-12,  # 1e-5
    }

    model_config = {
        'n_classes': 3,
        'base_filters': 16,
    }

    data_config = {
        "subset": 100,
        "train_split": 0.8,
        "batch_size": 5,
    }

    transform_config = {
        'vessel_type': 'light',
        'balls_xy': [True, True],
        'gaussian_blurring': [0, 2],
        'bg_class_range': [3, 10],
        'noise_sigma': 0.25,
        'contrast_range': False,
        'max_num_blobs': 50,
        'verbose': False
        }

    # Combine all configurations into one dictionary
    config = {
        "notes": notes,
        "device": str(device),
        "transform": transform_config,
        "training": training_config,
        "model": model_config,
        "data": data_config,
    }

    # Save the configuration to a JSON file
    save_config_to_json(config, model_dir)
    transform = ConfocalVesselLabelTransform(**config['transform'])
    train_loader, val_loader = get_loaders(transform, **config['data'])
    model = UNet(1, **config['model']).to('cuda')
    train_model(model, train_loader, val_loader, **config['training'])


if __name__ == '__main__':
    main()
