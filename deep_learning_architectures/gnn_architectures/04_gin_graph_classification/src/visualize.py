from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path | None = None,
) -> None:
    epochs = np.arange(len(history["train_losses"])) + 1

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_losses"], label="Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("GIN Training Loss")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracies"], label="Train accuracy")
    axes[1].plot(
        epochs,
        history["validation_accuracies"],
        label="Validation accuracy",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("GIN Accuracy")
    axes[1].grid()
    axes[1].legend()

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()
