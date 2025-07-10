from transformers import LayoutLMTokenizerFast, LayoutLMConfig
from data import (
    get_funsd_dataset,
    get_sroie_dataset,
    get_rvlcdip_dataset,
    get_loaders,
)

def test_dataset_and_loader(dataset_fn, name="Dataset"):
    print(f"\nTesting {name}...")

    tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
    config = LayoutLMConfig.from_pretrained("microsoft/layoutlm-base-uncased")

    if name == "RVL-CDIP":
        train_ds, val_ds, test_ds = dataset_fn(tokenizer, config)
    else:
        train_ds, val_ds, test_ds = dataset_fn(tokenizer, config, valid_ratio=0.1)

    train_loader, val_loader, test_loader = get_loaders(train_ds, val_ds, test_ds, batch_size=2, num_workers=0)

    for split_name, loader in zip(["Train", "Val", "Test"], [train_loader, val_loader, test_loader]):
        if loader is None:
            print(f"{split_name} loader is None")
            continue
        batch = next(iter(loader))
        print(f"{split_name} loader OK — Batch keys: {list(batch.keys())}, batch size: {len(batch['input_ids'])}")

if __name__ == "__main__":
    test_dataset_and_loader(get_funsd_dataset, name="FUNSD")
    test_dataset_and_loader(get_sroie_dataset, name="SROIE")
    test_dataset_and_loader(get_rvlcdip_dataset, name="RVL-CDIP")
