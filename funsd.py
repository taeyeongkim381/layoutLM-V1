import torch
from torch.utils.data import Dataset


class FUNSDLayoutLMV1Dataset(Dataset):
    """
    bbox는 이미 0~1000으로 normalize되어 있음.
    """
    def __init__(self, dataset, tokenizer, max_length=512, pad_token_label_id=-100):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_label_id = pad_token_label_id
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        words = item["tokens"]
        bboxes = item["bboxes"]
        labels = item["ner_tags"]
        image = item["image"]

        encoding = self.tokenizer(
            words,                                # 입력: 단어 리스트 (예: ["Please", "sign", "here"])
            is_split_into_words=True,             # 입력이 문장이 아니라 단어 단위로 나뉘어져 있음을 명시
            padding="max_length",                 # max_length까지 패딩 (길이가 짧은 입력을 0으로 채움)
            truncation=True,                      # max_length 초과 시 뒤쪽 토큰을 잘라냄 (truncation 적용)
            max_length=self.max_length,           # 토크나이즈된 결과의 최대 길이 (보통 512)
            return_offsets_mapping=True,          # 각 토큰이 원본 단어에서 어느 위치에 해당하는지 오프셋 정보 반환
            return_tensors="pt"                   # 결과를 PyTorch 텐서 형태로 반환 (모델 입력용)
        )

        word_ids = encoding.word_ids(batch_index=0) # 배치의 첫 번째 샘플
        token_bboxes = []
        token_labels = []
        for word_id in word_ids:
            if word_id is None:
                token_bboxes.append([0, 0, 0, 0])
                token_labels.append(self.pad_token_label_id)
            else:
                token_bboxes.append(bboxes[word_id])
                token_labels.append(labels[word_id])

        token_bboxes = torch.tensor(token_bboxes, dtype=torch.long)
        token_labels = torch.tensor(token_labels, dtype=torch.long)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "bboxes": token_bboxes,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": token_labels,
            "image": image,
        }
    