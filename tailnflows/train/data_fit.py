from pathlib import Path
from math import ceil
from collections.abc import Iterable
from flow_matching import loss
import torch
from torch import optim
import tqdm
from tailnflows.utils import get_experiment_output_path

from tailnflows.models.flow_matching import cfm_loss

def batch_loader(n: int, batch_size: int) -> Iterable[torch.Tensor]:
    batches = torch.randperm(n).split(batch_size)
    while True:
        for batch_ix in batches:
            yield batch_ix
        batches = torch.randperm(n).split(batch_size)

def train(
    model,
    x_trn,
    x_val,
    x_tst,
    lr=1e-3,
    num_steps=500,
    batch_size=100,
    label="",
    hook=None,
    early_stop_patience=None,
    grad_clip=None,
    optimizer=None,
    lr_scheduler=None,
    preprocess_transformation=None,
    eval_period=1,
    device="cuda",
    model_type=None, # flow matching vs discrete normalizing flow
):
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if lr_scheduler == "cosine_anneal":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, 0)
    
    if lr_scheduler == "cosine_anneal_wr":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50, 
            T_mult=1, 
            eta_min=5e-7,
        )

    if preprocess_transformation is not None:
        x_trn, _ = preprocess_transformation(x_trn)
        x_val, _ = preprocess_transformation(x_val)
        x_pre, pre_process_lad = preprocess_transformation(x_tst)

    ## FLOW MATCHING ##
    # if model_type == "flow_matching":
    #     loss = cfm_loss(model.vf, y.to(self.device), self.q0)

    # training loop data
    loop = tqdm.tqdm(range(num_steps))
    num_evals = ceil(num_steps / eval_period) + 1 # include final eval
    losses = torch.empty(num_steps)
    vlosses = torch.empty(num_evals)
    steps = torch.empty(num_evals, dtype=torch.int)
    hook_data = {}
    tst_loss = torch.tensor(torch.inf)
    best_val_loss = torch.tensor(torch.inf)
    tst_ix = -1

    # data loading
    n = x_trn.shape[0]
    if batch_size is None:
        batch_size = n
    batches = batch_loader(n, batch_size)

    model.to(device)

    for step in loop:
        # mini batch
        model.train()
        getattr(optimizer, 'train', lambda: None)()

        batch_ix = next(batches)
        batch = x_trn[batch_ix, :].to(device)
        optimizer.zero_grad()

        ## normalizing flow loss
        if model_type == "norm_flow":
            trn_loss = -model.log_prob(batch).mean()
        ## flow matching loss
        elif model_type == "flow_matching":
            # Apply inverse tail transform
            y, _ = model.tail_trafo.inverse(batch)
            trn_loss = cfm_loss(model.vf, y.to(device), model.q0)

        # Do backprop and optimizer step
        if ~(torch.isnan(trn_loss) | torch.isinf(trn_loss)):
            trn_loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        losses[step] = trn_loss.detach()

        if step % eval_period == 0 or (step + 1) == num_steps:
            model.eval()
            getattr(optimizer, 'eval', lambda: None)()

            with torch.no_grad():
                eval_number = ceil(step / eval_period)

                if hook is not None:
                    hook(model, hook_data)

                ## normalizing flow loss
                if model_type == "norm_flow":
                    val_loss = -model.log_prob(x_val).mean()
                ## flow matching loss
                elif model_type == "flow_matching":
                    y_val, _ = model.tail_trafo.inverse(x_val)
                    val_loss = cfm_loss(model.vf, y_val.to(device), model.q0)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ## only relevant for normalizing flow models
                    if model_type == "norm_flow":
                        if preprocess_transformation is not None:
                            tst_loss = -(model.log_prob(x_pre) + pre_process_lad).mean()
                        else:
                            tst_loss = -model.log_prob(x_tst).mean()
                        tst_eval = eval_number
                    else:
                        tst_eval = 0

                steps[eval_number] = step
                vlosses[eval_number] = val_loss.detach()

                if (
                    early_stop_patience is not None
                    and step - steps[tst_eval] > early_stop_patience
                ):
                    break
        
        loop.set_postfix(
            {
                "loss train (val)": f"{losses[step]:.2f} ({vlosses[eval_number]:.2f}) {label}: *{tst_loss.detach():.3f} @ {steps[tst_eval]}"
            }
        )
    
    tst_ix = steps[tst_eval]
    val_loss = vlosses[tst_eval]

    return (
        tst_loss.cpu(), 
        val_loss.cpu(), 
        tst_ix.cpu(), 
        losses[:step + 1].cpu(), 
        vlosses[:eval_number + 1].cpu(),  
        steps[:eval_number + 1].cpu(), 
        hook_data
    )


def train_epochs(
    model,
    x_trn,
    x_val,
    x_tst,
    best_model_path,
    lr=1e-4,
    num_epochs=100,
    batch_size=256,
    label="",
    early_stop_patience=None,
    grad_clip=None,
    optimizer=None,
    lr_scheduler=None,
    device="cuda",
):
    """ Simplified training method. Ditches preprocessing, hook and step data.
        Measures val performance every epoch.
        If early stopping is used, best model among the early stopping patience is chosen.
    """
    # Check if path to best model path exists, create otherwise
    if not isinstance(best_model_path, Path):
        best_model_path = Path(best_model_path)
    parent_dir = best_model_path.parent
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)

    parameters = list(model.parameters())
    if optimizer is None:
        optimizer = optim.AdamW(parameters, lr=lr)

    n = x_trn.shape[0]
    num_steps_per_epoch = n // batch_size
    num_steps = num_epochs * num_steps_per_epoch

    # Learning rate schedulers
    if lr_scheduler == "cosine_anneal":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, 0)
    
    if lr_scheduler == "cosine_anneal_wr":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50, 
            T_mult=1, 
            eta_min=5e-7,
        )

    # training loop data
    losses = torch.empty(num_epochs)
    vlosses = torch.empty(num_epochs)
    best_val_loss = torch.tensor(torch.inf)

    loop = tqdm.tqdm(range(num_epochs))

    for epoch in loop:

        model.train()
        getattr(optimizer, 'train', lambda: None)()

        batches = torch.randperm(n).split(batch_size)

        for batch_ix in batches:

            # get data batch
            batch = x_trn[batch_ix, :].to(device)
            optimizer.zero_grad()

            # compute loss and backprop
            trn_loss = -model.log_prob(batch).mean()
            trn_loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        losses[epoch] = trn_loss.detach().cpu()

        model.eval()
        getattr(optimizer, 'eval', lambda: None)()

        with torch.no_grad():

            val_loss = -model.log_prob(x_val).mean()
            vlosses[epoch] = val_loss.detach().cpu()

            # Check val loss for early stopping
            if early_stop_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        print("Early stopping!")
                        vlosses[epoch:] = val_loss.detach().cpu()
                        losses[epoch:] = trn_loss.detach().cpu()
                        model.load_state_dict(torch.load(best_model_path))
                        break

        loop.set_postfix(
            {"loss": f"trn: {losses[epoch]:.2f}, val: {vlosses[epoch]:.2f} ({label} @ epoch {epoch})"}
        )

    # compute test loss for trained model
    tst_loss = -model.log_prob(x_tst).mean()

    return (
        tst_loss.detach().cpu(),
        val_loss.detach().cpu(),
        losses.detach().cpu(), 
        vlosses.detach().cpu(),
    )