import torch
import torch.nn as nn


def circular_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error on unit circle components.
    pred and target shape: (..., 2), where last dim = [sin, cos]
    """
    return torch.mean((pred - target).pow(2))


class CircularMSELoss(nn.Module):
    """Computes MSE on sin & cos components for dihedral angles."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [batch, 2]
        return self.mse(pred, target)


# def cosine_angle_loss(pred: torch.Tensor,
#                       target: torch.Tensor,
#                       eps: float = 1e-8) -> torch.Tensor:
#     # # optional clip to avoid acos > 1 from fp errors
#     # pred   = pred / pred.norm(dim=-1, keepdim=True).clamp(min=eps)
#     # target = target / target.norm(dim=-1, keepdim=True).clamp(min=eps)

#     # dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)  # numeric guard
#     # return (1.0 - dot).mean()
def cosine_angle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dot = (pred * target).sum(dim=-1)
    return (1.0 - dot).mean()





def weighted_cosine_loss(pred, target, weights):
    dots = (pred * target).sum(dim=1)
    return (weights * (1 - dots)).mean()


def von_mises_nll_fixed_kappa(mu, target, kappa=2.0, eps=1e-8):
    kappa = torch.tensor(kappa, device=mu.device, dtype=mu.dtype)
    nll = -kappa * torch.cos(target - mu) + torch.log(2 * torch.pi * torch.special.i0(kappa))
    return nll.mean()
# Model output: mu (radians), Target: target_angle (radians)
def angular_error(mu, target):
    # Returns error in radians, between 0 and pi
    error = torch.atan2(torch.sin(mu - target), torch.cos(mu - target)).abs()
    return error.mean()


def von_mises_nll_per_sample(pred, target, kappa=2.0, eps=1e-6):
    dot = (pred * target).sum(dim=-1).clamp(-1 + eps, 1 - eps)
    return -kappa * dot + torch.log(2 * torch.pi * torch.i0(torch.tensor(kappa, device=pred.device)))

def radial_penalty(pred, target):
    return ((pred.norm(dim=-1) - target.norm(dim=-1))**2)


class CosineAngleLoss(nn.Module):
    """Circular loss using cosine of angular difference."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dot = torch.sum(pred * target, dim=-1)
        return torch.mean(1 - dot)


def angular_error(pred: torch.Tensor, target: torch.Tensor, in_degrees: bool = False) -> torch.Tensor:
    """
    Computes mean absolute angular error.
    pred, target: [batch, 2] sin & cos.
    """
    # recover angles
    pred_angle = torch.atan2(pred[..., 0], pred[..., 1])
    true_angle = torch.atan2(target[..., 0], target[..., 1])
    diff = pred_angle - true_angle
    # wrap to [-pi, pi]
    diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
    err = diff.abs()
    if in_degrees:
        err = err * 180.0 / torch.pi
    return err.mean()


class AngularErrorMetric(nn.Module):
    """Metric for mean absolute angular error."""
    def __init__(self, in_degrees: bool = False):
        super().__init__()
        self.in_degrees = in_degrees

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return angular_error(pred, target, self.in_degrees)



# Example usage:
# loss_fn = CosineAngleLoss()
# metric_fn = AngularErrorMetric(in_degrees=True)
# y_pred = torch.stack([sin_pred, cos_pred], dim=-1)
# y_true = torch.stack([sin_true, cos_true], dim=-1)
# loss = loss_fn(y_pred, y_true)
# mae_deg = metric_fn(y_pred, y_true)
