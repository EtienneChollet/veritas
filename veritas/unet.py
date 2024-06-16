import torch
import traceback
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from vesselseg.vesselseg import losses
from veritas.confocal import VesselLabelDataset
from veritas.utils import save_checkpoint


class DoubleConv3D(nn.Module):
    """
    A module consisting of two sets of 3D convolutions, each followed by
    batch normalization and ELU activation.

    This block is used for feature extraction in 3D convolutional neural
    networks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.

    Attributes
    ----------
    double_conv : nn.Sequential
        The sequential container of two 3D convolutional layers, each followed
        by batch normalization and ELU activation.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the double convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, D, H, W) where N is the batch size,
            C is the number of channels, and D, H, W are the depth, height,
            and width dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor after the double convolution block.
        """
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Simple UNet model for 3D image segmentation.

    Parameters
    ----------
    n_channels : int
        Number of input channels.
    n_classes : int
        Number of output classes.
    base_filters : int, optional
        Number of base filters used in the convolutions. Default is 16.
    """
    def __init__(self, n_channels, n_classes, base_filters=16):
        super(UNet, self).__init__()

        self.inc = DoubleConv3D(n_channels, base_filters)

        self.down1 = DoubleConv3D(base_filters, base_filters * 2)
        self.down2 = DoubleConv3D(base_filters * 2, base_filters * 4)
        self.down3 = DoubleConv3D(base_filters * 4, base_filters*8)
        self.down4 = DoubleConv3D(base_filters * 8, base_filters*16)

        self.up1 = DoubleConv3D(
            base_filters * 16 + base_filters * 8, base_filters * 8)
        self.up2 = DoubleConv3D(
            base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up3 = DoubleConv3D(
            base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up4 = DoubleConv3D(
            base_filters * 2 + base_filters, base_filters)

        self.outc = nn.Conv3d(base_filters, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)

        # First encode
        x2 = F.max_pool3d(x1, 2)
        x2 = self.down1(x2)

        # Second encode
        x3 = F.max_pool3d(x2, 2)
        x3 = self.down2(x3)

        x4 = F.max_pool3d(x3, 2)
        x4 = self.down3(x4)
        # x4 = nn.Dropout3d(0.2)(x4)

        x5 = F.max_pool3d(x4, 2)
        x5 = self.down4(x5)
        # x5 = nn.Dropout3d(0.2)(x5)

        u5 = F.interpolate(
            x5, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([u5, x4], dim=1)
        x = self.up1(x)

        u4 = F.interpolate(
            x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([u4, x3], dim=1)
        x = self.up2(x)

        u3 = F.interpolate(
            x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([u3, x2], dim=1)
        x = self.up3(x)

        u2 = F.interpolate(
            x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([u2, x1], dim=1)
        x = self.up4(x)

        logits = self.outc(x)
        return logits


# TODO : Change name to get_loaders.
def get_loaders(
        transform, subset=-1, batch_size=1, train_split=0.8,
        seed=42):
    """
    Loads and splits data into training and validation sets.

    Parameters
    ----------
    transform : callable
        A function/transform that takes in an image and returns a transformed
        version.
    data_path : str
        Path to the data.
    subset : int, optional
        Number of samples to use. If -1, use the entire dataset. Default is -1.
    batch_size : int, optional
        Number of samples per batch to load. Default is 1.
    train_split : float, optional
        Proportion of the dataset to include in the training split. Default is
        0.8.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    train_loader : DataLoader
        DataLoader for the training set.
    val_loader : DataLoader
        DataLoader for the validation set.
    """
    dataset = VesselLabelDataset(3, transform=transform)

    if subset > 0:
        dataset = torch.utils.data.Subset(dataset, range(subset))

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False
    )

    return train_loader, val_loader


class WarmupCooldownLR:
    """
    Custom learning rate scheduler with warmup, cooldown, and gap phases.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_steps : int
        Number of steps for the warmup phase.
    cooldown_steps : int
        Number of steps for the cooldown phase.
    gap_steps : int
        Number of steps to maintain the maximum learning rate.
    base_lr : float
        Base learning rate.
    max_lr : float
        Maximum learning rate.
    """
    def __init__(self, optimizer: torch.optim, warmup_steps: int,
                 cooldown_steps: int, gap_steps: int, base_lr: float,
                 max_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.gap_steps = gap_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.total_steps = warmup_steps + gap_steps + cooldown_steps
        self.step_n = 0

    def step(self):
        """
        Update the learning rate for the current step according to the warmup,
        gap, and cooldown phases.
        """
        if self.step_n < self.warmup_steps:
            # Warmup phase
            lr = self.base_lr + (self.max_lr - self.base_lr) * (
                self.step_n / self.warmup_steps)
        elif self.step_n < self.warmup_steps + self.gap_steps:
            # Gap phase, maintain max_lr
            lr = self.max_lr
        elif self.step_n < self.total_steps:
            # Cooldown phase
            cooldown_step = self.step_n - self.warmup_steps - self.gap_steps
            lr = self.max_lr - (self.max_lr - self.base_lr) * (
                cooldown_step / self.cooldown_steps)
        else:
            # After cooldown phase, maintain base_lr
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.step_n += 1

    def get_lr(self):
        """
        Get the current learning rate for each parameter group.

        Returns
        -------
        list
            List of learning rates for each parameter group.
        """
        return [
            param_group['lr'] for param_group in self.optimizer.param_groups]


class FocalTverskyLoss3D(nn.Module):
    """
    Focal Tversky Loss for 3D multi-class segmentation tasks.

    Parameters
    ----------
    smooth : float
        Smoothing factor to avoid division by zero errors.
    alpha : float
        Weight for false positives.
    beta : float
        Weight for false negatives.
    gamma : float
        Focusing parameter to emphasize hard examples.

    Attributes
    ----------
    smooth : float
        Smooth factor for numerical stability.
    alpha : float
        Controls the penalty for false positives.
    beta : float
        Controls the penalty for false negatives.
    gamma : float
        Modulates the loss to focus more on difficult examples.

    Examples
    --------
    >>> loss = FocalTverskyLoss3D()
    >>> inputs = torch.randn(10, 3, 64, 256, 256, requires_grad=True)
    >>> targets = torch.empty(10, 64, 256, 256, dtype=torch.long).random_(3)
    >>> output = loss(inputs, targets)
    >>> output.backward()
    """
    def __init__(self, smooth=1e-6, alpha=0.7, beta=0.3, gamma=0.75):
        super(FocalTverskyLoss3D, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute the Focal Tversky loss between `inputs` and the ground truth
        `targets`.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits for each class. Shape (N, C, D, H, W).
        targets : torch.Tensor
            Ground truth labels. Shape (N, D, H, W).

        Returns
        -------
        torch.Tensor
            Calculated loss.

        !!!!FN detections need to be weighted higher than FPs to improve
        recall rate.!!!!
        """
        targets_one_hot = targets
        inputs_soft = F.softmax(inputs, dim=1)

        # True Positives, False Positives & False Negatives
        TP = (inputs_soft * targets_one_hot).sum(dim=(2, 3, 4))
        FP = ((1 - targets_one_hot) * inputs_soft).sum(dim=(2, 3, 4))
        FN = (targets_one_hot * (1 - inputs_soft)).sum(dim=(2, 3, 4))

        Tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky.mean()


class WeightedFocalTverskyLoss(nn.Module):
    """
    Weighted Focal Tversky Loss for multi-class image segmentation.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of weights for each class.
    smooth : float, optional
        Smoothing factor to avoid division by zero. Default is 1e-6.
    alpha : float, optional
        Weight for false positives. Default is 0.7.
    beta : float, optional
        Weight for false negatives. Default is 0.3.
    gamma : float, optional
        Focusing parameter for focal loss. Default is 0.75.
    """
    def __init__(self, weights: torch.Tensor, smooth: float = 1e-6,
                 alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
        super(WeightedFocalTverskyLoss, self).__init__()
        self.weights = weights  # Tensor of weights for each class
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
        """
        Compute the Weighted Focal Tversky Loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits of shape (N, C, D, H, W), where N is the batch
            size, C is the number of classes, and D, H, W are the depth,
            height, and width dimensions.
        targets : torch.Tensor
            Ground truth one-hot encoded labels of shape (N, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        # Ensure weights are on the same device as inputs
        weights = self.weights.to(inputs.device)
        inputs_soft = F.softmax(inputs, dim=1)

        # True Positives, False Positives & False Negatives
        TP = (inputs_soft * targets).sum(dim=(2, 3, 4))
        FP = ((1 - targets) * inputs_soft).sum(dim=(2, 3, 4))
        FN = (targets * (1 - inputs_soft)).sum(dim=(2, 3, 4))

        # Weighted Tversky Index
        weighted_tversky = (
            weights * ((TP + self.smooth) /
                       (TP + self.alpha * FP + self.beta * FN + self.smooth))
        ).sum()

        # Applying the focal component
        focal_tversky = (1 - weighted_tversky) ** self.gamma

        return focal_tversky


class NoBgFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for 3D multi-class segmentation that focuses on the
    foreground classes, excluding the background.

    Parameters
    ----------
    smooth : float, optional
        Smoothing factor to avoid division by zero. Default is 1e-6.
    alpha : float, optional
        Weight for false positives. Default is 0.7.
    beta : float, optional
        Weight for false negatives. Default is 0.3.
    gamma : float, optional
        Focusing parameter for focal loss. Default is 0.75.
    """
    def __init__(self, smooth=1e-6, alpha=0.7, beta=0.3, gamma=0.75):
        super(NoBgFocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute the Focal Tversky Loss, excluding the background class.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits of shape (N, C, D, H, W), where N is the batch
            size, C is the number of classes, and D, H, W are the depth,
            height, and width dimensions.
        targets : torch.Tensor
            Ground truth one-hot encoded labels of shape (N, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        # Apply softmax to the inputs
        inputs_soft = F.softmax(inputs, dim=1)

        # True Positives, False Positives, and False Negatives
        TP = (inputs_soft * targets).sum(dim=(2, 3, 4))
        FP = ((1 - targets) * inputs_soft).sum(dim=(2, 3, 4))
        FN = (targets * (1 - inputs_soft)).sum(dim=(2, 3, 4))

        # Tversky index calculation
        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky.mean()


def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float
                        ) -> torch.optim.NAdam:
    """
    Configures and returns the NAdam optimizer for the given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be optimized.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay (L2 penalty).

    Returns
    -------
    torch.optim.NAdam
        Configured NAdam optimizer.
    """
    return torch.optim.NAdam(
        model.parameters(), lr=lr, weight_decay=weight_decay)


def configure_criterion(weighted: bool = False) -> torch.nn.Module:
    """
    Configures and returns the DiceLoss criterion (loss function).

    Parameters
    ----------
    weighted : bool, optional
        If True, use a weighted version of DiceLoss. Default is False.

    Returns
    -------
    torch.nn.Module
        Configured DiceLoss criterion.
    """
    return losses.DiceLoss(
        weighted=weighted, activation=torch.nn.Softmax(dim=1))


def log_model_graph(model_dir: str, model: torch.nn.Module,
                    train_loader: DataLoader) -> SummaryWriter:
    """
    Logs the model graph to TensorBoard.

    Parameters
    ----------
    model_dir : str
        Directory where the TensorBoard logs will be saved.
    model : torch.nn.Module
        The model to be logged.
    train_loader : DataLoader
        DataLoader for the training data to provide a sample input.

    Returns
    -------
    SummaryWriter
        TensorBoard SummaryWriter object.
    """
    writer = SummaryWriter(model_dir)
    sample_inputs, _ = next(iter(train_loader))
    writer.add_graph(model, sample_inputs.to(next(model.parameters()).device))
    return writer


def log_metrics(writer: SummaryWriter, phase: str, metrics: dict, step: int
                ) -> None:
    """
    Logs training and validation metrics to TensorBoard.

    Parameters
    ----------
    writer : SummaryWriter
        TensorBoard writer object.
    phase : str
        Phase of training (e.g., 'training', 'validation', 'epoch').
    metrics : dict
        Dictionary of metrics to log.
    step : int
        Step (iteration or epoch) at which metrics are logged.
    """
    for key, value in metrics.items():
        writer.add_scalar(f'{phase}_{key}', value, step)


def train_one_epoch(model: torch.nn.Module, epoch: int, writer: SummaryWriter,
                    train_loader: DataLoader, criterion: torch.nn.Module,
                    optimizer: torch.optim.Adam, device: torch.device
                    ) -> float:
    """
    Trains the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    epoch : int
        Current epoch number.
    writer : SummaryWriter
        TensorBoard writer object for logging.
    train_loader : DataLoader
        DataLoader for the training data.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim
        Optimizer.
    device : torch.device
        Device to run the training on (e.g., 'cuda' or 'cpu').
    """
    model.train()
    running_loss = 0.0

    try:
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            print(f"E-{epoch}, I-{i}, Loss: {loss.item()}", end='\r',
                  flush=True)

            if i % 10 == 0:
                log_metrics(writer, 'training', {
                    'loss': loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr']},
                            epoch * len(train_loader) + i)

        epoch_loss = running_loss / len(train_loader.dataset)
        log_metrics(writer, 'epoch', {'loss': epoch_loss}, epoch)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

    except Exception as e:
        print("An error occurred during training:", str(e))
        traceback.print_exc()
        epoch_loss = float('inf')

    return model, optimizer


def validate(model: torch.nn.Module, epoch: int, writer: SummaryWriter,
             val_loader: DataLoader, optimizer: torch.optim.Adam,
             criterion: torch.nn.Module, device: torch.device,
             best_vloss: float, model_dir: str) -> float:
    """
    Validates the model on the validation set.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be validated.
    epoch : int
        Current epoch number.
    writer : SummaryWriter
        TensorBoard writer object for logging.
    val_loader : DataLoader
        DataLoader for the validation data.
    optimizer : Adam
        Optimizer.
    criterion : torch.nn.Module
        Loss function.
    device : torch.device
        Device to run the validation on (e.g., 'cuda' or 'cpu').
    best_vloss : float
        Best validation loss observed so far.
    model_dir : str
        Directory to save the model checkpoints.

    Returns
    -------
    float
        Best validation loss observed so far.
    """
    model.eval()
    running_vloss = 0.0
    try:
        with torch.no_grad():
            for vinputs, vlabels in val_loader:
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
        val_loss = float(running_vloss / len(val_loader))
        writer.add_scalar('val_loss', val_loss, epoch)

        if val_loss < best_vloss:
            best_vloss = val_loss
            print(f"New best val_loss: {best_vloss}")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=(
                f'{model_dir}/checkpoints/'
                f'checkpoint_epoch_{epoch+1}_val-{val_loss}.pth.tar')
                )

    except Exception as e:
        print("An error occurred during validation:", str(e))
        traceback.print_exc()

    return best_vloss


def log_hist(model: torch.nn.Module, epoch: int, writer: SummaryWriter
             ) -> None:
    """
    Logs histograms of model parameters and their gradients to TensorBoard.

    Parameters
    ----------
    model : Module
        The model whose parameters are to be logged.
    epoch : int
        The current epoch number.
    writer : SummaryWriter
        TensorBoard writer object.
    """
    if (epoch + 1) % 10 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}.grad', param.grad, epoch)
