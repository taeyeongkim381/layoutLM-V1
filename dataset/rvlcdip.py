import pytesseract
import torch
from pytesseract import Output
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class RVLCDIPLayoutLMV1Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.to_tensor = ToTensor()

    def ocr_image(self, image):
        ocr_result = pytesseract.image_to_data(image, output_type=Output.DICT)
        words = []
        bboxes = []

        for i in range(len(ocr_result["text"])):
            word = ocr_result["text"][i].strip()
            if word == "":
                continue
            x, y, w, h = (ocr_result["left"][i], ocr_result["top"][i],
                          ocr_result["width"][i], ocr_result["height"][i])
            words.append(word)
            bboxes.append([x, y, x + w, y + h])
        return words, bboxes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]  # PIL Image
        image = image.resize((1000, 1000))  # ✅ 모든 이미지를 1000x1000으로 리사이즈
        image_tensor = self.to_tensor(image)
        label = item["label"]

        words, bboxes = self.ocr_image(image)  # ⬅ 이미 리사이즈된 이미지 기준이므로 normalize 불필요

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        token_bboxes = []
        for word_id in word_ids:
            if word_id is None:
                token_bboxes.append([0, 0, 0, 0])
            else:
                token_bboxes.append(bboxes[word_id])

        pad_len = self.max_length - len(token_bboxes)
        if pad_len > 0:
            token_bboxes.extend([[0, 0, 0, 0]] * pad_len)

        token_bboxes = torch.tensor(token_bboxes, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "bboxes": token_bboxes,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": label,
            "image": image_tensor,
        }
