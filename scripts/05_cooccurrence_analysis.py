import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DPI = 300

INPUT_DIR = PROJECT_ROOT / "processed_data"
OUTPUT_DIR = PROJECT_ROOT / "processed_data"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = DPI


def phi_coefficient_pvalue(x, y):
    a = ((x == 1) & (y == 1)).sum()
    b = ((x == 1) & (y == 0)).sum()
    c = ((x == 0) & (y == 1)).sum()
    d = ((x == 0) & (y == 0)).sum()

    n = a + b + c + d
    denominator = np.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    if denominator == 0:
        return 0.0, 1.0

    phi = (a * d - b * c) / denominator

    chi2 = n * phi * phi
    p_value = 1.0 - stats.chi2.cdf(chi2, 1)

    return phi, p_value


def main():
    detection_matrix = pd.read_csv(
        INPUT_DIR / "detection_matrix.csv", index_col=0
    )

    species = detection_matrix.columns.tolist()
    n_species = len(species)

    rows = []
    for i in range(n_species):
        for j in range(i + 1, n_species):
            sp1 = species[i]
            sp2 = species[j]
            col1 = detection_matrix[sp1]
            col2 = detection_matrix[sp2]

            cooccur = ((col1 == 1) & (col2 == 1)).sum()
            if cooccur < 3:
                continue

            phi, p = phi_coefficient_pvalue(col1, col2)
            rows.append(
                {
                    "species1": sp1,
                    "species2": sp2,
                    "phi_coefficient": phi,
                    "p_value": p,
                    "significant": p < 0.05,
                }
            )

    pairwise_df = pd.DataFrame(rows)
    pairwise_df.to_csv(
        OUTPUT_DIR / "species_pairwise_phi.csv", index=False
    )
    print(f"Saved species_pairwise_phi.csv with {len(pairwise_df)} pairs")

    sig_df = pairwise_df[pairwise_df["significant"]].copy()
    sig_df.to_csv(OUTPUT_DIR / "significant_pairs.csv", index=False)
    print(f"Saved significant_pairs.csv with {len(sig_df)} significant pairs")

    site_counts = detection_matrix.sum(axis=0)
    species_ge5 = site_counts[site_counts >= 5].index.tolist()
    matrix_ge5 = detection_matrix[species_ge5]

    if matrix_ge5.shape[1] >= 3:
        corr_matrix = pd.DataFrame(
            index=species_ge5, columns=species_ge5, dtype=float
        )
        n_sites = matrix_ge5.shape[0]
        for i, sp1 in enumerate(species_ge5):
            for j, sp2 in enumerate(species_ge5):
                if i == j:
                    corr_matrix.loc[sp1, sp2] = 1.0
                elif i > j:
                    val = corr_matrix.loc[sp2, sp1]
                    corr_matrix.loc[sp1, sp2] = val
                else:
                    phi, _ = phi_coefficient_pvalue(
                        matrix_ge5[sp1], matrix_ge5[sp2]
                    )
                    corr_matrix.loc[sp1, sp2] = phi

        g = sns.clustermap(
            corr_matrix,
            method="average",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5,
            linecolor="gray",
            figsize=(12, 10),
            dendrogram_ratio=0.1,
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
            annot=False,
        )
        g.fig.suptitle(
            "Species Co-occurrence Heatmap (Phi Coefficient)",
            fontsize=14,
            y=1.02,
        )
        g.savefig(
            FIGURE_DIR / "fig3_cooccurrence_heatmap.png",
            dpi=DPI,
            bbox_inches="tight",
        )
        plt.close(g.fig)
        print("Saved fig3_cooccurrence_heatmap.png")

    G = nx.Graph()
    for sp in species:
        G.add_node(sp, site_count=int(site_counts[sp]))

    for _, row in sig_df.iterrows():
        G.add_edge(
            row["species1"],
            row["species2"],
            weight=row["phi_coefficient"],
        )

    pos = nx.spring_layout(G, k=0.6, seed=42, iterations=100)

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("white")

    node_sizes = [
        G.nodes[n]["site_count"] * 80 for n in G.nodes
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color="lightblue",
        edgecolors="gray",
        linewidths=1.0,
        alpha=0.9,
        ax=ax,
    )

    pos_edges = G.edges(data=True)
    pos_edges_list = [
        (u, v) for u, v, d in pos_edges if d["weight"] > 0
    ]
    neg_edges_list = [
        (u, v) for u, v, d in pos_edges if d["weight"] < 0
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=pos_edges_list,
        edge_color="green",
        alpha=0.6,
        width=1.5,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=neg_edges_list,
        edge_color="red",
        alpha=0.6,
        width=1.5,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    ax.set_title(
        "Species Co-occurrence Network\nGreen=Positive, Red=Negative",
        fontsize=14,
        pad=20,
    )
    ax.axis("off")

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="green",
            lw=2,
            label="Positive association (p<0.05)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="red",
            lw=2,
            label="Negative association (p<0.05)",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=10,
        framealpha=0.9,
    )

    fig.savefig(
        FIGURE_DIR / "fig4_cooccurrence_network.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved fig4_cooccurrence_network.png")

    print("Co-occurrence analysis complete!")


if __name__ == "__main__":
    main()
