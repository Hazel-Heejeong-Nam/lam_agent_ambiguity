import sys
sys.path.insert(0, ".")

from utils.datasets.robomimic import RoboMimicDataset, make_robomimic_loader


def test_dataset():
    dataset = RoboMimicDataset("data/robomimic/lift-img-train.npz")
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"obs shape: {sample['obs'].shape}")
    print(f"obs min/max: {sample['obs'].min():.3f} / {sample['obs'].max():.3f}")
    print(f"actions shape: {sample['actions'].shape}")


def test_loader():
    loader = make_robomimic_loader("data/robomimic/lift-img-train.npz", batch_size=4)
    batch = next(iter(loader))
    print(f"batch obs shape: {batch['obs'].shape}")
    print(f"batch actions shape: {batch['actions'].shape}")


def test_all_tasks():
    tasks = ["lift", "can", "square", "transport"]
    for task in tasks:
        path = f"data/robomimic/{task}-img-train.npz"
        dataset = RoboMimicDataset(path)
        print(f"{task}: {len(dataset)} windows")


if __name__ == "__main__":
    print("=== Dataset ===")
    test_dataset()
    print("\n=== Loader ===")
    test_loader()
    print("\n=== All tasks ===")
    test_all_tasks()