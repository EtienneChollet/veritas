import os
import json
import glob
import torch
import traceback
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
from vesselseg.vesselseg import networks, losses
from veritas.unet import (UNet, WarmupLR, ConstantLR, CooldownLR,
                          ChainedLRScheduler, FocalTverskyLoss3D,
                          NoBgFocalTverskyLoss)
from veritas.utils import delete_folder, MatchHistogram
from veritas.confocal import ConfocalVesselLabelTransform, VesselLabelDataset
from veritas.synth import SanityCheckDataset


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save checkpoint to disk."""
    torch.save(state, filename)


class SegmentationDataset(Dataset):
    def __init__(self, label_paths, transform=None):
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx):
        labels = torch.from_numpy(
            nib.load(self.label_paths[idx]).get_fdata()).to(torch.float32)
        x, y = self.transform(labels)
        x = x.unsqueeze(0)  # Add channel dimension
        return x, y


class SanityCheckDataset(Dataset):

    def __init__(self):
        path = ('/autofs/cluster/octdata2/users/epc28/veritas/output/'
                'synthetic_data/sanity_check_synthvols')
        self.x_paths = sorted(glob.glob(f'{path}/x/*'))
        self.y_paths = sorted(glob.glob(f'{path}/y/*'))

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        x = torch.load(self.x_paths[idx]).to('cuda')
        y = torch.load(self.y_paths[idx]).to('cuda')
        return x, y


def worker_init_fn(worker_id):
    """Initialize each worker with its own CUDA context."""
    torch.cuda.init()


def load_data(transform, data_path, subset=-1, batch_size=1, train_split=0.8,
              seed=42):
    # label_paths = f"{data_path}/*label*"
    # label_paths = sorted(glob.glob(label_paths))[:subset]

    dataset = VesselLabelDataset(3, transform=transform)
    # dataset = SanityCheckDataset()
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    print('LOADED DATALOADER')
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False)
    return train_loader, val_loader


def train_model(
        model, train_loader, val_loader, num_epochs=25, warmup_epochs=2,
        begin_cooldown_epoch=500, lr=0.001,
        model_dir='runs/yibei_semantic_yael', weight_decay=1e-5,
        device='cuda'):

    try:
        # optimizer = torch.optim.Adam(
        #    model.parameters(), lr, weight_decay=weight_decay)
        optimizer = torch.optim.NAdam(
            model.parameters(), lr, weight_decay=weight_decay
        )
        criterion_ = losses.DiceLoss(
           weighted=False, activation=torch.nn.Softmax(dim=1))
        # criterion_ = FocalTverskyLoss3D(smooth=0.0000001)
        writer = SummaryWriter(model_dir)
        sample_inputs, _ = next(iter(train_loader))
        writer.add_graph(
            model, sample_inputs.to(next(model.parameters()).device))

        # epoch_steps = len(train_loader)
        # warmup_scheduler = WarmupLR(
        #    optimizer,
        #    warmup_steps=(epoch_steps * warmup_epochs),
        #    start_lr=1e-10, end_lr=lr)
        # constant_scheduler2 = ConstantLR(
        #    optimizer,
        #    constant_steps=(epoch_steps * begin_cooldown_epoch), lr=lr)
        # cooldown_scheduler = CooldownLR(
        #    optimizer,
        #    cooldown_steps=(epoch_steps * 100), start_lr=lr, end_lr=1e-6)
        # scheduler = ChainedLRScheduler(
        #    optimizer,
        #    schedulers=[
        #        warmup_scheduler, constant_scheduler2, cooldown_scheduler])

        best_vloss = 1
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion_(outputs, labels)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                running_loss += loss.item() * inputs.size(0)

                print(
                    f"E-{epoch}, I-{i}, Loss: {loss.item()}",
                    end='\r', flush=True)

                if i % 10 == 0:
                    writer.add_scalar(
                        'training_loss',
                        loss.item(),
                        epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'learning_rate',
                        optimizer.param_groups[0]['lr'],
                        epoch * len(train_loader) + i)

            epoch_loss = running_loss / len(train_loader.dataset)
            writer.add_scalar('epoch_loss', epoch_loss, epoch)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

            # Validation Loop
            running_vloss = 0.0
            model.eval()
            with torch.no_grad():
                for i, (vinputs, vlabels) in enumerate(val_loader):
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                    voutputs = model(vinputs)
                    vloss = criterion_(voutputs, vlabels)
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / len(val_loader)
            writer.add_scalar('val_loss', avg_vloss, epoch)

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                print(f"New best val_loss: {best_vloss}")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=(
                    f'{model_dir}/checkpoints/'
                    f'checkpoint_epoch_{epoch+1}_val-{avg_vloss}.pth.tar')
                    )

            # Add histogram stuff
            # Optionally log parameter histograms and gradients
            if (epoch + 1) % 10 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'{name}.grad', param.grad, epoch)
    except Exception as e:
        print("An error occurred:", str(e))
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        writer.close()
        print("Training completed and CUDA memory cleaned up")
    writer.close()
    return model


def save_config_to_json(config, model_dir, filename="config.json"):
    """Save configuration dictionary to a JSON file.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing all the parameters.
    filename : str
        Name of the file where the JSON data will be saved.

    """
    try:
        delete_folder(model_dir)
    except Exception as e:
        print(f"Warning: {e}")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f'{model_dir}/checkpoints')
    with open(f'{model_dir}/{filename}', 'w') as f:
        json.dump(config, f, indent=4)


def main():

    #####################################################
    model_dir = 'runs/base/version_9'
    #####################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    notes = "Trained using normalization between -1 and 1"

    training_config = {
        "num_epochs": 10000,
        "warmup_epochs": 2,
        "begin_cooldown_epoch": 100,
        "model_dir": model_dir,
        "lr": 0.001,
        'weight_decay': 1e-9,  # 1e-9
    }

    model_config = {
        'n_classes': 3,
        'base_filters': 16,
    }

    data_config = {
        "data_path": ("/autofs/cluster/octdata2/users/epc28/veritas/output/"
                      "synthetic_data/exp0001"),
        "subset": -1,
        "train_split": 0.8,
        "batch_size": 5,
    }

    transform_config = {
        'vessel_type': 'light',
        'light_range': [1.75, 5],
        'balls_range': [1.75, 5],
        'balls_xy': [True, True],
        'gaussian_blurring': [0, 1],
        'bg_class_range': [3, 10],
        'noise_sigma': 1,
        'contrast_range': False,
        'max_num_blobs': 20,
        'verbose': False
        }

    vsynth = {
        'min_difference': 0.1
    }

    # Combine all configurations into one dictionary
    config = {
        "notes": notes,
        "device": str(device),
        "transform": transform_config,
        "training": training_config,
        "model": model_config,
        "data": data_config,
        "synth_config": {
            'vessels': vsynth,
            'parenchyma': transform_config
        }
    }

    # Save the configuration to a JSON file
    save_config_to_json(config, model_dir)
    transform = ConfocalVesselLabelTransform(**config['transform'])
    train_loader, val_loader = load_data(transform, **config['data'])
    model = UNet(1, **config['model']).to('cuda')
    train_model(model, train_loader, val_loader, **config['training'])


if __name__ == '__main__':
    main()
