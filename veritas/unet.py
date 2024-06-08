import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """(Convolution3D => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class SelfAttention3D(nn.Module):
    """ Self-attention Layer for 3D data """
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Flatten spatial dimensions
        batch_size, channels, depth, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(batch_size, -1, depth * height * width)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height, width)
        return out

class AttentionGate(nn.Module):
    """
    Attention Gate module.
    Args:
    - F_g: The number of feature channels of the gating signal input.
    - F_l: The number of feature channels of the local input features.
    - F_int: The number of feature channels in the intermediate representation.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Forward pass of the attention gate.
        Args:
        - g: Gating signal, typically the output from the previous layer.
        - x: Local features from the encoder path that need to be refined.
        """
        #print('g.shape:', g.shape)
        #print('x.shape:', x.shape)

        g1 = self.W_g(g)  # Transform gating signal
        x1 = self.W_x(x)  # Transform local features
        psi = self.relu(g1 + x1)  # Additive attention mechanism
        psi = self.psi(psi)  # Activation map

        # Return the gated input as the element-wise multiplication of the input x and sigmoid activation map
        return x * psi


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters=16):
        super(UNet, self).__init__()

        self.inc = DoubleConv3D(n_channels, base_filters)

        self.down1 = DoubleConv3D(base_filters, base_filters * 2)
        self.down2 = DoubleConv3D(base_filters * 2, base_filters * 4)
        self.down3 = DoubleConv3D(base_filters * 4, base_filters*8)
        self.down4 = DoubleConv3D(base_filters * 8, base_filters*16)

        self.up1 = DoubleConv3D(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.up2 = DoubleConv3D(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up3 = DoubleConv3D(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up4 = DoubleConv3D(base_filters * 2 + base_filters, base_filters)

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

        x5 = F.max_pool3d(x4, 2)
        x5 = self.down4(x5)

        u5 = F.interpolate(x5, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([u5, x4], dim=1)
        x = self.up1(x)

        u4 = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([u4, x3], dim=1)
        x = self.up2(x)

        u3 = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([u3, x2], dim=1)
        x = self.up3(x)

        u2 = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([u2, x1], dim=1)
        x = self.up4(x)

        logits = self.outc(x)
        return logits


class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters=16):
        super(AttentionUNet, self).__init__()

        self.inc = DoubleConv3D(n_channels, base_filters)

        self.down1 = DoubleConv3D(base_filters, base_filters * 2)
        self.down2 = DoubleConv3D(base_filters * 2, base_filters * 4)
        self.down3 = DoubleConv3D(base_filters * 4, base_filters*8)
        self.down4 = DoubleConv3D(base_filters * 8, base_filters*16)

        self.up1 = DoubleConv3D(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.up2 = DoubleConv3D(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up3 = DoubleConv3D(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up4 = DoubleConv3D(base_filters * 2 + base_filters, base_filters)

        self.ag1 = AttentionGate(F_g=base_filters*16, F_l=base_filters*8, F_int=base_filters*8)
        self.ag2 = AttentionGate(F_g=base_filters*8, F_l=base_filters*4, F_int=base_filters*4)
        self.ag3 = AttentionGate(F_g=base_filters*4, F_l=base_filters*2, F_int=base_filters*2)
        self.ag4 = AttentionGate(F_g=base_filters*2, F_l=base_filters, F_int=base_filters)

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

        x5 = F.max_pool3d(x4, 2)
        x5 = self.down4(x5)

        u5 = F.interpolate(x5, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.ag1(u5, x4)
        x = torch.cat([x, u5], dim=1)
        x = self.up1(x)

        u4 = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.ag2(u4, x3)
        x = torch.cat([x, u4], dim=1)
        x = self.up2(x)

        u3 = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.ag3(u3, x2)
        x = torch.cat([x, u3], dim=1)
        x = self.up3(x)

        u2 = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.ag4(u2, x1)
        x = torch.cat([x, u2], dim=1)
        x = self.up4(x)

        logits = self.outc(x)
        return logits


class SemanticDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SemanticDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Ensure inputs are softmaxed
        inputs = F.softmax(inputs, dim=1)
        
        # Assuming targets are already one-hot encoded and same shape as inputs
        # inputs and targets shapes: [batch, channels, depth, height, width] (for 3D)
        total_loss = 0
        for class_idx in range(inputs.shape[1]):  # Iterate over channel dimension
            input_flat = inputs[:, class_idx, ...].contiguous().view(-1)
            target_flat = targets[:, class_idx, ...].contiguous().view(-1)
            intersection = (input_flat * target_flat).sum()
            dice_score = (2. * intersection + self.smooth) / \
                         (input_flat.sum() + target_flat.sum() + self.smooth)
            total_loss += 1 - dice_score
        return total_loss / inputs.shape[1]
    

class WarmupLRog:
    def __init__(self, optimizer, warmup_steps, base_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_n = 0

    def step(self):
        if self.step_n < self.warmup_steps:
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.step_n / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.step_n += 1

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
class WarmupCooldownLR:
    def __init__(self, optimizer, warmup_steps, cooldown_steps, gap_steps, base_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.gap_steps = gap_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.total_steps = warmup_steps + gap_steps + cooldown_steps
        self.step_n = 0

    def step(self):
        if self.step_n < self.warmup_steps:
            # Warmup phase
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.step_n / self.warmup_steps)
        elif self.step_n < self.warmup_steps + self.gap_steps:
            # Gap phase, maintain max_lr
            lr = self.max_lr
        elif self.step_n < self.total_steps:
            # Cooldown phase
            cooldown_step = self.step_n - self.warmup_steps - self.gap_steps
            lr = self.max_lr - (self.max_lr - self.base_lr) * (cooldown_step / self.cooldown_steps)
        else:
            # After cooldown phase, maintain base_lr
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.step_n += 1

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class WarmupLR:
    def __init__(self, optimizer, warmup_steps, start_lr, end_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.step_n = 0

    def step(self):
        if self.step_n < self.warmup_steps:
            lr = self.start_lr + (self.end_lr - self.start_lr) * (self.step_n / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.step_n += 1

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def reset(self):
        self.step_n = 0


class ConstantLR:
    def __init__(self, optimizer, constant_steps, lr):
        self.optimizer = optimizer
        self.constant_steps = constant_steps
        self.lr = lr
        self.step_n = 0

    def step(self):
        if self.step_n < self.constant_steps:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        self.step_n += 1

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def reset(self):
        self.step_n = 0

class CooldownLR:
    def __init__(self, optimizer, cooldown_steps, end_lr, start_lr):
        self.optimizer = optimizer
        self.cooldown_steps = cooldown_steps
        self.end_lr = end_lr
        self.start_lr = start_lr
        self.step_n = 0

    def step(self):
        if self.step_n < self.cooldown_steps:
            lr = self.start_lr - (self.start_lr - self.end_lr) * (self.step_n / self.cooldown_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.step_n += 1

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def reset(self):
        self.step_n = 0

class ChainedLRScheduler:
    def __init__(self, optimizer, schedulers):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.current_scheduler_idx = 0
        self.total_steps = sum(scheduler.constant_steps if isinstance(scheduler, ConstantLR) else
                               scheduler.warmup_steps if isinstance(scheduler, WarmupLR) else
                               scheduler.cooldown_steps for scheduler in schedulers)

    def step(self):
        if self.current_scheduler_idx < len(self.schedulers):
            current_scheduler = self.schedulers[self.current_scheduler_idx]
            current_scheduler.step()
            if current_scheduler.step_n >= (current_scheduler.constant_steps if isinstance(current_scheduler, ConstantLR) else
                                            current_scheduler.warmup_steps if isinstance(current_scheduler, WarmupLR) else
                                            current_scheduler.cooldown_steps):
                self.current_scheduler_idx += 1
                if self.current_scheduler_idx < len(self.schedulers):
                    self.schedulers[self.current_scheduler_idx].reset()

    def get_lr(self):
        if self.current_scheduler_idx < len(self.schedulers):
            return self.schedulers[self.current_scheduler_idx].get_lr()
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

# Example usage:
# optimizer = ... # Your optimizer instance
# warmup_scheduler = WarmupLR(optimizer, warmup_steps=1000, base_lr=1e-10, max_lr=0.01)
# constant_scheduler = ConstantLR(optimizer, constant_steps=1000, base_lr=0.01)
# cooldown_scheduler = CooldownLR(optimizer, cooldown_steps=1000, base_lr=1e-10, max_lr=0.01)
# scheduler = ChainedLRScheduler(optimizer, schedulers=[warmup_scheduler, constant_scheduler, cooldown_scheduler])

# for _ in range(scheduler.total_steps):
#     scheduler.step()
#     print(scheduler.get_lr())


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
        Compute the Focal Tversky loss between `inputs` and the ground truth `targets`.

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

        !!!!FN detections need to be weighted higher than FPs to improve recall rate.!!!!
        """
        targets_one_hot = targets
        inputs_soft = F.softmax(inputs, dim=1)

        # True Positives, False Positives & False Negatives
        TP = (inputs_soft * targets_one_hot).sum(dim=(2, 3, 4))
        FP = ((1 - targets_one_hot) * inputs_soft).sum(dim=(2, 3, 4))
        FN = (targets_one_hot * (1 - inputs_soft)).sum(dim=(2, 3, 4))

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky.mean()

# Example usage
#if __name__ == "__main__":
#    N, C, D, H, W = 10, 3, 64, 256, 256
#    inputs = torch.randn(N, C, D, H, W, requires_grad=True)
#    targets = torch.empty(N, D, H, W, dtype=torch.long).random_(3)
#    loss_fn = FocalTverskyLoss3D()
#    loss = loss_fn(inputs, targets)
#    loss.backward()
#    print(f"Calculated Loss: {loss.item()}")


class WeightedFocalTverskyLoss(nn.Module):
    def __init__(self, weights, smooth=1e-6, alpha=0.7, beta=0.3, gamma=0.75):
        super(WeightedFocalTverskyLoss, self).__init__()
        self.weights = weights  # Tensor of weights for each class
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Ensure weights are on the same device as inputs
        weights = self.weights.to(inputs.device)
        targets_one_hot = targets
        inputs_soft = F.softmax(inputs, dim=1)

        # True Positives, False Positives & False Negatives
        TP = (inputs_soft * targets_one_hot).sum(dim=(2, 3, 4))
        FP = ((1 - targets_one_hot) * inputs_soft).sum(dim=(2, 3, 4))
        FN = (targets_one_hot * (1 - inputs_soft)).sum(dim=(2, 3, 4))

        # Weighted Tversky Index
        weighted_tversky = (weights * ((TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth))).sum()
        
        # Applying the focal component
        focal_tversky = (1 - weighted_tversky) ** self.gamma

        return focal_tversky


class NoBgFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for 3D multi-class segmentation that focuses on the
    foreground classes, excluding the background.
    """
    def __init__(self, smooth=1e-6, alpha=0.7, beta=0.3, gamma=0.75):
        super(NoBgFocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Assuming targets are already one-hot encoded and have the shape [N, C, D, H, W]
        # and inputs are [N, C, D, H, W] logits

        #print("Shape of targets:", targets.shape)
        #print("Shape of inputs:", inputs.shape)

        # Apply softmax to the inputs
        inputs_soft = F.softmax(inputs, dim=1)

        # Ensure targets and inputs_soft are aligned in shape if necessary:
        # targets might need to be adjusted if there are any dimension mismatch issues.

        # True Positives, False Positives, and False Negatives
        TP = (inputs_soft * targets).sum(dim=(2, 3, 4))
        FP = ((1 - targets) * inputs_soft).sum(dim=(2, 3, 4))
        FN = (targets * (1 - inputs_soft)).sum(dim=(2, 3, 4))

        # Tversky index calculation
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky.mean()
