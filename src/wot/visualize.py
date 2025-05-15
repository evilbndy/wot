import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from typing import Callable

from wot.montecarlo import SimulationState


TGetResults = Callable[[SimulationState], int]


fn_purchased_containers: TGetResults = lambda state: state.opened_containers.get("proto", 0) - state.received_containers.get("proto", 0)

def pdf(states: list[SimulationState], fn_get_result: TGetResults, ax: plt.axis = None) -> tuple[plt.figure, plt.axis]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    data = list(map(fn_get_result, states))
    sns.histplot(data, bins=50, kde=True, color="blue", ax=ax, stat="density", alpha=0.3)
    mean_val = np.mean(data)
    ax.axvline(mean_val, color="red", label=f"mean: {mean_val:.2f}", linestyle="--", alpha=0.7)
    ax.legend()
    ax.set_xlabel("Boxes Purchased")
    ax.set_ylabel("Density")
    ax.grid(True)
    return fig, ax


def cdf(states: list[SimulationState], fn_get_result: TGetResults, ax: plt.axis = None) -> tuple[plt.figure, plt.axis]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    data = list(map(fn_get_result, states))

    # Plot the ECDF
    sns.ecdfplot(data, color="blue", ax=ax, stat="percent", alpha=0.3, label="$F_{X}(x)=\\operatorname {P} (X\\leq x)$")
    
    mean_val = np.mean(data)
    
    # Draw lines from axes to F(X) = 0.5 point
    ax.axvline(x=mean_val, color="red", linestyle="--", alpha=0.7)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.7)
    
    # Draw lines from axis to F(X) = 0.95
    percentaile_95 = np.percentile(data, 95)
    ax.axvline(x=percentaile_95, color="green", linestyle="--", alpha=0.7)
    ax.axhline(y=95, color="green", linestyle="--", alpha=0.7)
    
    # Mark the intersection point
    ax.plot(mean_val, 50, 'ro', label=f"mean: {mean_val:.2f}")
    ax.plot(percentaile_95, 95, 'go', label=f"95%: {percentaile_95:.2f}")
    
    ax.legend()
    ax.set_xlabel("Boxes Purchased")
    ax.set_ylabel("Cumulative Density")
    ax.grid(True)
    return fig, ax


def expectation_plot(states: dict[str, list[SimulationState]], fn_get_result: TGetResults, ax: plt.axis = None, yticks: list[int] = None) -> plt.figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure

    target_dfs = {}
    for x, state in states.items():
        data = list(map(fn_get_result, state))
        target_dfs[x] = pd.DataFrame(data)

    target_means = {x: df.mean() for x, df in target_dfs.items()}
    target_stds = {x: df.std() for x, df in target_dfs.items()}
    
    target_means_df = pd.DataFrame(target_means).T
    target_stds_df = pd.DataFrame(target_stds).T
    target_means_df.index = target_means_df.index.astype(int)
    target_stds_df.index = target_stds_df.index.astype(int)
    target_means_df = target_means_df.sort_index()
    target_stds_df = target_stds_df.sort_index()
    for column in target_means_df.columns:
        ax.errorbar(
            target_means_df.index,
            target_means_df[column],
            yerr=target_stds_df[column],
            label=column,
            capsize=3,
            alpha=0.6
        )
    ax.legend()
    ax.grid(True)

    if yticks is not None:
        ax.set_yticks(yticks)
    
    return fig, ax

