def main():

```
config = load_config()

set_seed(config["seed"])

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dataset_root = config["dataset_root"]

splits, train_ds = analyze_dataset(
    dataset_root
)

show_samples(train_ds)

(
    train_loader,
    val_loader,
    test_loader,
    num_classes
) = get_loaders(
    dataset_path=dataset_root,
    batch_size=config["model"]["batch_size"],
    image_size=config["image_size"]
)

logger = ExperimentLogger()

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
```
