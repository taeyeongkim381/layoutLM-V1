from transformers import LayoutLMTokenizerFast, LayoutLMModel, LayoutLMConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from rvlcdip import RVLCDIPLayoutLMV1Dataset

def main():
    dataset = load_dataset("aharley/rvl_cdip")
    tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
    config = LayoutLMConfig.from_pretrained("microsoft/layoutlm-base-uncased")

    train_dataset = RVLCDIPLayoutLMV1Dataset(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings
    )
    train_loader = DataLoader(train_dataset, batch_size=2)

    model = LayoutLMModel(config)
    batch = next(iter(train_loader))

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        bbox=batch["bboxes"]
    )

    print(outputs.last_hidden_state.shape)
    print(batch['image'].shape)
    print(batch['label'].shape)

if __name__ == "__main__":
    main()
    