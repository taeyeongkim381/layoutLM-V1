import pytesseract
import torch
from pytesseract import Output
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class RVLCDIPLayoutLMV1Dataset(Dataset):
    """
    TODO: 서로 이미지 크기가 다른데 어떻게 처리할 예정인지, 테스크 코드 또한 업데이트 필요
    """
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
    
    def normalize_bbox(self, bbox, width, height):
        return [
            int(1000 * bbox[0] / width),
            int(1000 * bbox[1] / height),
            int(1000 * bbox[2] / width),
            int(1000 * bbox[3] / height)
        ]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item["image"] # PIL Image
        image_tensor = self.to_tensor(image)
        label = item["label"]

        words, bboxes = self.ocr_image(image)
        width, height = image.size
        normalized_bboxes = [self.normalize_bbox(b, width, height) for b in bboxes]

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
        for word_id in word_ids:
            if word_id is None:
                token_bboxes.append([0, 0, 0, 0])
            else:
                token_bboxes.append(normalized_bboxes[word_id])

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
    