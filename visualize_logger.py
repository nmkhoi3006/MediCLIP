import argparse
import os
import re

import matplotlib.pyplot as plt


def parse_logger(log_path):
    """Read MediCLIP logger.log and extract train loss plus validation metrics."""
    epochs_auroc = []
    image_aurocs = []
    pixel_aurocs = []
    epochs_loss = []
    loss_values = []
    loss_avg_values = []

    auroc_pattern = re.compile(
        r"\): Epoch: (\d+), image auroc: ([0-9.]+)(?:, pixel_auroc: ([0-9.]+))?"
    )
    loss_pattern = re.compile(
        r"Epoch: \[(\d+)/\d+\].*Loss ([0-9.]+) \(([0-9.]+)\)"
    )

    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            auroc_match = auroc_pattern.search(line)
            if auroc_match:
                epochs_auroc.append(int(auroc_match.group(1)))
                image_aurocs.append(float(auroc_match.group(2)))
                pixel_aurocs.append(
                    float(auroc_match.group(3)) if auroc_match.group(3) else None
                )
                continue

            loss_match = loss_pattern.search(line)
            if loss_match:
                epochs_loss.append(int(loss_match.group(1)))
                loss_values.append(float(loss_match.group(2)))
                loss_avg_values.append(float(loss_match.group(3)))

    return {
        "epochs_auroc": epochs_auroc,
        "image_aurocs": image_aurocs,
        "pixel_aurocs": pixel_aurocs,
        "epochs_loss": epochs_loss,
        "loss_values": loss_values,
        "loss_avg_values": loss_avg_values,
    }


def plot_logger(log_path, output_path=None, show=True):
    """Plot metrics parsed from logger.log with matplotlib."""
    metrics = parse_logger(log_path)
    has_pixel_auroc = any(value is not None for value in metrics["pixel_aurocs"])
    panel_count = 2 + int(has_pixel_auroc)

    fig, axes = plt.subplots(1, panel_count, figsize=(7 * panel_count, 5))
    if panel_count == 1:
        axes = [axes]

    axes[0].plot(
        metrics["epochs_auroc"],
        metrics["image_aurocs"],
        marker="o",
        label="Image AUROC",
    )
    axes[0].set_title("Validation Image AUROC")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AUROC")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    loss_axis = axes[1]
    loss_axis.plot(
        metrics["epochs_loss"],
        metrics["loss_values"],
        marker="o",
        label="Loss",
    )
    loss_axis.plot(
        metrics["epochs_loss"],
        metrics["loss_avg_values"],
        marker="s",
        label="Average Loss",
    )
    loss_axis.set_title("Training Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()

    if has_pixel_auroc:
        pixel_epochs = [
            epoch
            for epoch, value in zip(metrics["epochs_auroc"], metrics["pixel_aurocs"])
            if value is not None
        ]
        pixel_values = [value for value in metrics["pixel_aurocs"] if value is not None]
        axes[2].plot(pixel_epochs, pixel_values, marker="o", label="Pixel AUROC")
        axes[2].set_title("Validation Pixel AUROC")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUROC")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Visualize MediCLIP logger.log")
    parser.add_argument("--log_path", type=str, required=True, help="Path to logger.log")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save the generated figure",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Save the figure without opening an interactive window",
    )
    args = parser.parse_args()

    metrics = plot_logger(
        args.log_path,
        output_path=args.output_path,
        show=not args.no_show,
    )

    if metrics["image_aurocs"]:
        best_idx = max(
            range(len(metrics["image_aurocs"])),
            key=lambda index: metrics["image_aurocs"][index],
        )
        print(
            "Best image AUROC: "
            f"{metrics['image_aurocs'][best_idx]:.4f} "
            f"at epoch {metrics['epochs_auroc'][best_idx]}"
        )


if __name__ == "__main__":
    main()
