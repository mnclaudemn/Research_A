# main.py
import torch
from model import *
from compute_metrics import *
from dataset import *
from engineA import *
from experiment_logger import *


def main():
    """
    Main orchestration function to run a single deep learning classification 
    experiment tracking unfreezing depths for high-impact publication metrics.
    """
    # --------------------------------------------------
    # 1. Configuration & Initialization
    # --------------------------------------------------
    config = load_config("config.yaml")
    set_seed(config.get("seed", 42))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[DEVICE] Executing experiment on: {device}")

    # --------------------------------------------------
    # 2. Dataset Discovery & Quality Verification
    # --------------------------------------------------
    dataset_root = config["dataset_root"]
    
    # Analyze imbalances and structural health
    splits, train_ds = analyze_dataset(dataset_root)
    
    # Display representative training matrix figures
    show_samples(train_ds, num_classes=5)

    # --------------------------------------------------
    # 3. Data Ingestion Pipeline
    # --------------------------------------------------
    batch_size = config.get("model", {}).get("batch_size", config.get("batch_size", 16))
    image_size = config.get("image_size", 224)

    train_loader, val_loader, test_loader, num_classes = get_loaders(
        dataset_path=dataset_root,
        batch_size=batch_size,
        image_size=image_size
    )

    # --------------------------------------------------
    # 4. Logger & Execution Call
    # --------------------------------------------------
    logger = ExperimentLogger(file_path="experiments.xlsx")

    run_one_experiment(
        model_name=config["model"]["name"],
        n_unfreeze=config["model"]["n_unfreeze"],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        config=config,
        device=device,
        logger=logger
    )


if __name__ == "__main__":
    main()
