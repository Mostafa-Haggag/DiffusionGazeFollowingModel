import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
class LinearWarmup(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs # Maximum number of iterations for linear warmup
        self.max_epochs = max_epochs # Maximum number of iterations
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )
        if self.last_epoch < self.warmup_epochs:
            return [self.optimizer.param_groups[0]["lr"] + (self.base_lrs[0] - self.warmup_start_lr) /
                (self.warmup_epochs - 1),self.base_lrs[1]]
        return self.base_lrs

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + self.last_epoch *
                (self.base_lrs[0] - self.warmup_start_lr) / (self.warmup_epochs - 1),self.base_lrs[1]] 
        self.base_lrs
            
class ConstantLRWithWarmup(_LRScheduler):
    """Implements a constant learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, warmup_epochs=0, last_epoch=-1, verbose=False):
        if warmup_epochs < 0:
            raise ValueError('Invalid value for warmup')
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.last_epoch == 0:
                return [0,self.base_lrs[1]]
            else:
                return [0.0 + (self.last_epoch / self.warmup_epochs) * self.base_lrs[0],self.base_lrs[1]]
        else:
            return self.base_lrs


#################################################################
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs # Maximum number of iterations for linear warmup
        self.max_epochs = max_epochs # Maximum number of iterations
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )
        
        if self.last_epoch == 0:
            return [self.warmup_start_lr,self.base_lrs[1]] 
            # fixing it as it is without changing it 
        if self.last_epoch < self.warmup_epochs:
            return [self.optimizer.param_groups[0]["lr"] + (self.base_lrs[0] - self.warmup_start_lr) /
                (self.warmup_epochs - 1),self.base_lrs[1]]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [self.optimizer.param_groups[0]["lr"]
                + (self.base_lrs[0] - self.eta_min) * (1 - math.cos(math.pi /
                                                           (self.max_epochs - self.warmup_epochs))) / 2,self.base_lrs[1]]
        return [ (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) /
             (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs -
                               1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (self.optimizer.param_groups[0]["lr"] - self.eta_min)
            + self.eta_min
            ,self.base_lrs[1]
        ] 
    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + self.last_epoch *
                (self.base_lrs[0] - self.warmup_start_lr) / (self.warmup_epochs - 1),self.base_lrs[1]] 
        return [self.eta_min
            + 0.5
            * (self.base_lrs[0] - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))),self.base_lrs[1]] 



# warmup + decay as a function
def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """Linear warmup for warmup_steps, optionally with cosine annealing or linear decay to 0 at total_steps."""
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / \
            float(max(1, total_steps - warmup_steps))
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn
