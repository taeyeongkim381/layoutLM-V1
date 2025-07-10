from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset import FUNSDLayoutLMV1Dataset, SROIELayoutLMV1Dataset, RVLCDIPLayoutLMV1Dataset


def get_funsd_dataset(tokenizer, config, valid_ratio=0.2, seed=42):
    raw_dataset = load_dataset("nielsr/FUNSD_layoutlmv2")  # train, test
    split = raw_dataset["train"].train_test_split(test_size=valid_ratio, seed=seed)
    
    train_dataset = FUNSDLayoutLMV1Dataset(split["train"], tokenizer, config.max_position_embeddings)
    val_dataset = FUNSDLayoutLMV1Dataset(split["test"], tokenizer, config.max_position_embeddings)
    test_dataset = FUNSDLayoutLMV1Dataset(raw_dataset["test"], tokenizer, config.max_position_embeddings)
    
    return train_dataset, val_dataset, test_dataset


def get_sroie_dataset(tokenizer, config, valid_ratio=0.2, seed=42):
    raw_dataset = load_dataset("sizhkhy/SROIE")  # train, test
    split = raw_dataset["train"].train_test_split(test_size=valid_ratio, seed=seed)
    
    train_dataset = SROIELayoutLMV1Dataset(split["train"], tokenizer, config.max_position_embeddings)
    val_dataset = SROIELayoutLMV1Dataset(split["test"], tokenizer, config.max_position_embeddings)
    test_dataset = SROIELayoutLMV1Dataset(raw_dataset["test"], tokenizer, config.max_position_embeddings)
    
    return train_dataset, val_dataset, test_dataset


def get_rvlcdip_dataset(tokenizer, config):
    dataset = load_dataset("aharley/rvl_cdip")  # train, validation, test
    train_dataset = RVLCDIPLayoutLMV1Dataset(dataset["train"], tokenizer, config.max_position_embeddings)
    val_dataset = RVLCDIPLayoutLMV1Dataset(dataset["validation"], tokenizer, config.max_position_embeddings)
    test_dataset = RVLCDIPLayoutLMV1Dataset(dataset["test"], tokenizer, config.max_position_embeddings)
    return train_dataset, val_dataset, test_dataset


def get_loaders(train_dataset, val_dataset, test_dataset, batch_size=4, num_workers=2):
    def make_loader(dataset, shuffle):
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }

        # ⚠ prefetch_factor는 num_workers > 0일 때만 허용됨
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        return DataLoader(**loader_kwargs)

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False) if val_dataset is not None else None
    test_loader = make_loader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader
