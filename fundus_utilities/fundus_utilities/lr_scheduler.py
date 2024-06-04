import torch

class WarmupLRScheduler():
    def __init__(self, optimizer, warmup_epochs, initial_lr):
        self.epoch = 0
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr

    def step(self):
        if self.epoch <= self.warmup_epochs:
            self.epoch += 1
            curr_lr = (self.epoch / self.warmup_epochs) * self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr

    def finished(self):
        return self.epoch >= self.warmup_epochs

def get_lr_scheduler(type:str, optimizer:torch.optim.Optimizer, len_train_loader:int, lr:float, epochs:int, cosine_len=None, warmup_epochs=None, warmup_lr=None):
    """Create a learning rate scheduler.
        Options: - "constant" or None for no schedule, 
                 - "onecycle" for OneCycleLR, 
                 - "cosine", for CosineAnnealingLR,
                 - "cosine_restart" for CosineAnnealingWarmRestarts.

        If using warmup, note: Discard warmup scheduler when loading from checkpoint.
        See below for sample usage.
    
    Args:
        type (str): Type of scheduler.
        optimizer (torch.optim.Optimizer): Optimizer.
        len_train_loader (int): Length of training data loader (number of batches per epoch)
        lr (float): Learning rate.
        epochs (int): total number of epochs.
        cosine_len (int, optional): Length of cosine annealing period. If set to None, will use
            1/2*epochs. Defaults to None.
        warmup_epochs (int, optional): Number of warmup epochs. Defaults to None.
        warmup_lr (float, optional): Warmup learning rate. If set to None, will use 1/10*lr.
            Defaults to None.

    Returns:
        torch.optim.lr_scheduler: Learning rate scheduler.
        torch.optim.lr_scheduler: Warmup learning rate scheduler.
            In case of no warmup scheduler, it is set to None.
    
    Example:
        Store the <type> in self.cfg.scheduler.
        In the training loop, at the beginning of each epoch, run:
        ```
        if self.warmup_scheduler and not self.warmup_scheduler.finished():
                self.warmup_scheduler.step()
        ```
        And after each batch, run:
        ```
        if self.scheduler and "onecycle" in self.cfg.scheduler:
            if not self.warmup_scheduler or self.warmup_scheduler.finished() and self.warmup_scheduler.epoch != self.current_epoch+1:
                self.scheduler.step()
        ```
        And at the end of each epoch, run:
        ```
        if self.scheduler and not "onecycle" in self.cfg.scheduler:
            if not self.warmup_scheduler or self.warmup_scheduler.finished():
                self.scheduler.step()
        ```
    """
    type = str(type).lower()
    if "none" in type or "constant" in type:
            return None, None
    
    elif "onecycle" in type:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len_train_loader,
        )

    elif "cosine" in type and not "restart" in type:
        if cosine_len is not None:
            period = cosine_len
        else:
            period = epochs/2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(period)
        )

    elif "cosine" in type and "restart" in type:
        if cosine_len is not None:
            length = cosine_len
        else:
            length = epochs/2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=length, T_mult=1
        )
        
    else:
        raise ValueError(f"Invalid scheduler: {type}!")

    if warmup_epochs is not None and warmup_epochs > 0:
        if warmup_lr is None:
            warmup_lr = float(lr)/10
        warmup_scheduler = WarmupLRScheduler(
            optimizer, warmup_epochs, warmup_lr
        )
        return scheduler, warmup_scheduler
    
    else:
        return scheduler, None