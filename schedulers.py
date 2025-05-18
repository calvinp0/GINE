import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineRestartsDecay(CosineAnnealingWarmRestarts):
    """
    CosineAnnealingWarmRestarts + multiplicative decay at each restart,
    without the deprecated `verbose` argument.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        decay: float = 0.3,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        # store decay before init
        self.decay = decay

        # initialize parent WITHOUT verbose
        super().__init__(
            optimizer,
            T_0       = T_0,
            T_mult    = T_mult,
            eta_min   = eta_min,
            last_epoch= last_epoch,
        )

    def step(self, epoch=None):
        super().step(epoch)

        # do NOT decay at epoch 0
        if self.T_cur == 0 and self.last_epoch > 0:
            self.base_lrs = [lr * self.decay for lr in self.base_lrs]
            for g, new_base in zip(self.optimizer.param_groups, self.base_lrs):
                g['initial_lr'] = new_base
