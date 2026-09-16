import os
from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.5 # alpha used in plots

def plot_rand_proj_stats(data_true: torch.Tensor, data_synth: dict[str, torch.Tensor], weight_vectors: Optional[list[torch.Tensor]] = None, label: Optional[str] = None):
    """ Project true data and synthetic data in random directions given by weight_vectors and compute statistics on these 1d datasets.
        Then plot the statistics computed with synthetic data against statistics computed with true data.

    Args:
        data_true (torch.Tensor): Data batch sampled from target distribution (shape [batch, features]).
        data_synth (dict[str, torch.Tensor]): Dictionary containing synthetic data batches (shape [batch, features]) of different flow models (given by the key str).
        weight_vectors (list[torch.Tensor], optional): List containing random vectors to project onto (shape [features]). If None (default), 100 weight vectors are sampled independently from U([0,1]^{features}).
        label (str, optional): label for saving plots.

    Computes the following statistics on the 1d datasets:
        - mean
        - standard deviation
        - 5%  quantile
        - 1%  quantile
        - 10% quantile
        - 95% quantile
        - 90% quantile
        - 99% quantile
    """
    stats = [
        "mean",
        "std",
        "q5",
        "q1",
        "q10",
        "q95",
        "q90",
        "q99",
    ]

    models = list(data_synth.keys())

    dim = data_true.shape[-1]
    for synth_batch in data_synth.values():
        assert synth_batch.shape[-1] == dim

    # initialize weight vectors if necessary
    if weight_vectors is None:
        weight_vectors = [torch.rand(dim) for _ in range(100)]

    num_directions = len(weight_vectors)

    # prepare lists to store statistics results
    results = {model: {stat: [] for stat in stats} for model in models}
    results["true"] = {stat: [] for stat in stats}


    for vec in weight_vectors:

        # 2. Project data onto direction given by weight vector
        proj_datasets = {"true": torch.matmul(data_true, vec)}
        for model in models:
            proj_datasets[model] = torch.matmul(data_synth[model], vec)

        # 3. Compute Statistics
        # 3.1. Mean
        results["true"]["mean"].append(torch.mean(proj_datasets["true"]))
        for model in models:
            results[model]["mean"].append(torch.mean(proj_datasets[model]))

        # 3.2. Std
        results["true"]["std"].append(torch.std(proj_datasets["true"]))
        for model in models:
            results[model]["std"].append(torch.std(proj_datasets[model]))

        # 3.3 1%-Quantile
        results["true"]["q1"].append(torch.quantile(proj_datasets["true"], 0.01))
        for model in models:
            results[model]["q1"].append(torch.quantile(proj_datasets[model], 0.01))
        # 3.4 5%-Quantile
        results["true"]["q5"].append(torch.quantile(proj_datasets["true"], 0.05))
        for model in models:
            results[model]["q5"].append(torch.quantile(proj_datasets[model], 0.05))
        # 3.5 10%-Quantile
        results["true"]["q10"].append(torch.quantile(proj_datasets["true"], 0.10))
        for model in models:
            results[model]["q10"].append(torch.quantile(proj_datasets[model], 0.10))
        # 3.6 90%-Quantile
        results["true"]["q90"].append(torch.quantile(proj_datasets["true"], 0.90))
        for model in models:
            results[model]["q90"].append(torch.quantile(proj_datasets[model], 0.90))
        # 3.7 95%-Quantile
        results["true"]["q95"].append(torch.quantile(proj_datasets["true"], 0.95))
        for model in models:
            results[model]["q95"].append(torch.quantile(proj_datasets[model], 0.95))
        # 3.8 99%-Quantile
        results["true"]["q99"].append(torch.quantile(proj_datasets["true"], 0.99))
        for model in models:
            results[model]["q99"].append(torch.quantile(proj_datasets[model], 0.99))

    # Convert to numpy for plotting
    for stat in stats:
        for i in range(num_directions):
            results["true"][stat][i] = results["true"][stat][i].detach().cpu().numpy()
            for model in models:
                results[model][stat][i] = results[model][stat][i].detach().cpu().numpy()

    # 4. Generate plots
    for stat in stats:
        plt.figure(figsize=(10, 6))
        # Collect handles and labels for the legend
        handles = []
        labels = []
        for model in models:
            h = plt.scatter(results["true"][stat], results[model][stat], label=model, alpha=alpha)
            handles.append(h)
            labels.append(model)
        # create diagonal line
        # diag = np.linspace(np.min([np.min(results["true"][stat]), np.min(results[models[0]][stat])]),
        #                    np.max([np.max(results["true"][stat]), np.max(results[models[0]][stat])]))
        diag = np.linspace(np.min(results["true"][stat]),
                           np.max(results["true"][stat]))
        # add legend to the side
        plt.legend(
                handles,
                labels,
                title="Models",
                bbox_to_anchor=(1.05, 1),  # Position outside plot
                loc='upper left',          # Anchor point
                borderaxespad=0,           # No padding between axes and legend
                frameon=False              # Cleaner look without frame
            )
        plt.plot(diag, diag, "--")

        plt.title(stat)
        plt.ticklabel_format(style="sci", scilimits=(0, 0))
        plt.xlabel("data")
        plt.ylabel("flow")
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)

        os.makedirs("plots", exist_ok=True)  # Creates directory if it doesn't exist
        if label is None:
            label = ""
        plt.savefig("plots/" + label + "_random_proj_" + stat + ".pdf", bbox_inches='tight')
        plt.close()