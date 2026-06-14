from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "figures" / "results"

METRIC_FILES = {
    "Baseline": PROJECT_ROOT
    / "outputs"
    / "vntraffic"
    / "baseline"
    / "tables"
    / "vntraffic_baseline_metrics.csv",
    "Level 1": PROJECT_ROOT
    / "outputs"
    / "vntraffic"
    / "level1"
    / "tables"
    / "vntraffic_level1_metrics.csv",
    "Level 2": PROJECT_ROOT
    / "outputs"
    / "vntraffic"
    / "level2"
    / "tables"
    / "vntraffic_level2_metrics.csv",
}


def load_metrics():
    rows = []

    for pipeline, path in METRIC_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")

        df = pd.read_csv(path, index_col=0)
        row = df.iloc[0]
        rows.append(
            {
                "pipeline": pipeline,
                "mota": row["mota"] * 100,
                "idf1": row["idf1"] * 100,
                "precision": row["precision"] * 100,
                "recall": row["recall"] * 100,
                "id_switches": int(row["num_switches"]),
            }
        )

    return pd.DataFrame(rows)


def configure_style():
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "serif",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
        }
    )


def save_quality_scores(metrics):
    pipelines = metrics["pipeline"].tolist()
    x = np.arange(len(pipelines))
    width = 0.24

    fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=300)
    ax.bar(x - width, metrics["mota"], width, label="MOTA")
    ax.bar(x, metrics["idf1"], width, label="IDF1")
    ax.bar(x + width, metrics["recall"], width, label="Recall")

    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines)
    ax.set_ylim(0, 70)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mota_idf1_recall.png", bbox_inches="tight")
    plt.close(fig)


def save_id_switches(metrics):
    fig, ax = plt.subplots(figsize=(3.45, 2.15), dpi=300)
    bars = ax.bar(metrics["pipeline"], metrics["id_switches"], width=0.5)

    ax.set_ylabel("ID Switches")
    ax.set_ylim(0, max(metrics["id_switches"]) + 12)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "id_switches.png", bbox_inches="tight")
    plt.close(fig)


def save_precision_recall(metrics):
    fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=300)
    ax.plot(metrics["pipeline"], metrics["precision"], marker="o", linewidth=1.2, label="Precision")
    ax.plot(metrics["pipeline"], metrics["recall"], marker="s", linewidth=1.2, label="Recall")

    ax.set_ylabel("Score (%)")
    ax.set_ylim(40, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="center right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "precision_recall.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    metrics = load_metrics()
    save_quality_scores(metrics)
    save_id_switches(metrics)
    save_precision_recall(metrics)

    print(f"Saved result figures to: {OUTPUT_DIR}")
    for image_path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"- {image_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
