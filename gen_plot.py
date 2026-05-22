## PLOT
import os
import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# 0. Load Configuration and Shared Data
# ---------------------------------------------------------------------------
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
data_name = config.get("data_name", "cifar10")
print(f"Generating plots for dataset: {data_name}")

assert os.path.exists("results_inferred.pkl"), "Không tìm thấy tệp kết quả results_inferred.pkl. Hãy chạy inference.py trước."
assert os.path.exists("y_loaded.npy"), "Không tìm thấy tệp nhãn y_loaded.npy. Hãy chạy training.py trước."

with open("results_inferred.pkl", "rb") as f:
    results = pickle.load(f)

y = np.load("y_loaded.npy")
y = np.asarray(y, dtype=np.int64)
N_CLUSTERS = len(np.unique(y))

# Thư mục đầu ra tương ứng với dataset
OUTPUT_DIR = data_name
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
MARKER_SIZE = 30
EDGE_WIDTH  = 0.5
ALPHA       = 0.8

# Đỏ - xanh lá - vàng
CLUSTER_COLORS = ["red", "green", "gold"]

# Chỉ dùng 3 shape
SHAPE_LIST = ["o", "s", "^"]
SHAPE_NAMES = {
    "o": "type A",
    "s": "type B",
    "^": "type C",
}

def get_marker_3(label):
    return SHAPE_LIST[int(label) % 3]

def plot_single_pdf(LATENT_DIM):
    Z_2d   = results[LATENT_DIM]["Z_2d"]
    labels = results[LATENT_DIM]["labels"]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor("white")

    unique_y = np.unique(y)

    for lbl in unique_y:
        marker = get_marker_3(lbl)
        mask_y = y == lbl

        for k in range(N_CLUSTERS):
            mask = mask_y & (labels == k)
            if mask.sum() == 0:
                continue

            ax.scatter(
                Z_2d[mask, 0],
                Z_2d[mask, 1],
                marker=marker,
                c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                edgecolors="#333333",
                linewidths=EDGE_WIDTH,
                s=MARKER_SIZE,
                alpha=ALPHA,
            )

    # Chỉ giữ nhãn trục, bỏ tiêu đề rườm rà
    ax.set_xlabel("PC-1", fontsize=11)
    ax.set_ylabel("PC-2", fontsize=11)
    ax.tick_params(labelsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor("#BBBBBB")

    # Legend chỉ giải thích shape
    shape_handles = [
        Line2D(
            [0], [0],
            marker=m,
            linestyle="None",
            markerfacecolor="#AAAAAA",
            markeredgecolor="#333333",
            markeredgewidth=0.8,
            markersize=8,
            label=name
        )
        for m, name in SHAPE_NAMES.items()
    ]

    ax.legend(
        handles=shape_handles,
        fontsize=9,
        title_fontsize=10,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
    )

    plt.tight_layout()

    out_path = os.path.join(
        OUTPUT_DIR,
        f"kmeans_latent_{LATENT_DIM}d_single_8x6.pdf"
    )

    plt.savefig(
        out_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()
    print(f"Saved PDF → {out_path}")

for latent_dim in [32]:
    plot_single_pdf(latent_dim)