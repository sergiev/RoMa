from tqdm import tqdm
from romatch.utils.utils import to_cuda
import romatch
import torch

def log_param_statistics(named_parameters, norm_type = 2):
    named_parameters = list(named_parameters)
    grads = [p.grad for n, p in named_parameters if p.grad is not None]
    if not grads:
        return {}
        
    weight_norms = [p.norm(p=norm_type) for n, p in named_parameters if p.grad is not None]
    names = [n for n,p in named_parameters if p.grad is not None]
    param_norm = torch.stack(weight_norms).norm(p=norm_type)
    device = grads[0].device
    grad_norms = torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads])
    nans_or_infs = torch.isinf(grad_norms) | torch.isnan(grad_norms)
    nan_inf_names = [name for name, naninf in zip(names, nans_or_infs) if naninf]
    total_grad_norm = torch.norm(grad_norms, norm_type)
    if torch.any(nans_or_infs):
        print(f"These params have nan or inf grads: {nan_inf_names}")
    
    return {
        "grad_norm": total_grad_norm.item(),
        "param_norm": param_norm.item()
    }

def train_step(train_batch, model, objective, optimizer, grad_scaler, grad_clip_norm = 1.,**kwargs):
    optimizer.zero_grad()
    out = model(train_batch)
    loss_dict = objective(out, train_batch)
    l = loss_dict['total_loss'] # Извлекаем total_loss для backward
    grad_scaler.scale(l).backward()
    grad_scaler.unscale_(optimizer)
    
    param_stats = log_param_statistics(model.named_parameters())
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm) # what should max norm be?
    grad_scaler.step(optimizer)
    grad_scaler.update()

    param_stats["grad_scale"] = grad_scaler.get_scale()

    if grad_scaler.get_scale() < 1.:
        grad_scaler.update(1.0)
        
    romatch.GLOBAL_STEP = romatch.GLOBAL_STEP + romatch.STEP_SIZE # increment global step
    
    # Объединяем все метрики в один словарь для логирования
    metrics_to_log = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    metrics_to_log.update(param_stats)
    
    return metrics_to_log


def train_k_steps(
    n_0, k, dataloader, model, objective, optimizer, lr_scheduler, grad_scaler, progress_bar=True, grad_clip_norm = 1., warmup = None, ema_model = None, pbar_n_seconds = 1,
):
    all_metrics = []
    for n in tqdm(range(n_0, n_0 + k), disable=(not progress_bar) or romatch.RANK > 0, mininterval=pbar_n_seconds):
        batch = next(dataloader)
        model.train(True)
        batch = to_cuda(batch)
        metrics = train_step(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_scaler=grad_scaler,
            n=n,
            grad_clip_norm = grad_clip_norm,
        )
        if ema_model is not None:
            ema_model.update()
        if warmup is not None:
            with warmup.dampening():
                lr_scheduler.step()
        else:
            lr_scheduler.step()
        
        lr_metrics = {f"lr_group_{grp}": lr for grp, lr in enumerate(lr_scheduler.get_last_lr())}
        metrics.update(lr_metrics)
        all_metrics.append(metrics)
    
    return all_metrics


def train_epoch(
    dataloader=None,
    model=None,
    objective=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
):
    model.train(True)
    print(f"At epoch {epoch}")
    for batch in tqdm(dataloader, mininterval=5.0):
        batch = to_cuda(batch)
        train_step(
            train_batch=batch, model=model, objective=objective, optimizer=optimizer
        )
    lr_scheduler.step()
    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "epoch": epoch,
    }


def train_k_epochs(
    start_epoch, end_epoch, dataloader, model, objective, optimizer, lr_scheduler
):
    for epoch in range(start_epoch, end_epoch + 1):
        train_epoch(
            dataloader=dataloader,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
        )
