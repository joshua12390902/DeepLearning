from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
import torch
import re
import html
from transformer.Const import *
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

class SquadSeq2SeqDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_source_len: int = 384,
        max_target_len: int = 64,
        require_target: bool = True,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        self.samples: List[Dict[str, str]] = []
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.require_target = require_target
        self.bos_token = SOS
        self.eos_token = EOS
        self.dataset_tag = self._infer_dataset_tag(path)
        
        suffix = path.suffix.lower()
        if suffix == ".csv":
            ds = load_dataset("csv", data_files=str(path), split="train")
        elif suffix in {".json", ".jsonl"}:
            ds = load_dataset("json", data_files=str(path), split="train")
        else:
            raise ValueError(f"Unsupported dataset format: {path}")
            
        for rec in ds:
            context = self._extract_field(
                rec,
                primary_keys=("context", "dialogue"),
                instance_keys=("selftext_without_tldr", "context", "article"),
            )
            summary = self._extract_field(
                rec,
                primary_keys=("summary", "tldr"),
                instance_keys=("summary", "tldr"),
            )
            if not context or (self.require_target and not summary):
                continue
            identifier = self._extract_id(rec)
            if not identifier:
                identifier = f"sample_{len(self.samples)}"
            self.samples.append({"context": context, "summary": summary, "id": identifier})
            
        if not self.samples:
            raise ValueError(f"No data found in {path}")

    @staticmethod
    def _infer_dataset_tag(path: Path) -> str:
        path_str = str(path).lower()
        if "tifu" in path_str:
            return "[tifu]"
        return ""

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """ [大哥強力清洗版] """
        text = ""
        if isinstance(value, str):
            text = value
        elif isinstance(value, list):
            for item in value:
                normalized = SquadSeq2SeqDataset._normalize_text(item)
                if normalized:
                    text = normalized
                    break
        elif isinstance(value, dict):
            t = value.get("text")
            if isinstance(t, str):
                text = t
        
        if not text:
            return ""

        # 1. HTML 解碼
        text = html.unescape(text)
        # 2. 清除隱形字元
        text = text.replace('\u200b', '')
        # 3. 處理 Markdown 連結
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # 4. 移除 URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 5. 移除格式符號
        text = text.replace("**", "").replace("__", "")
        # 6. 移除 Reddit 標籤
        text = text.replace("[deleted]", "").replace("[removed]", "")
        # 7. 縮減空白
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _extract_field(
        self,
        record: Dict[str, Any],
        primary_keys: Sequence[str],
        instance_keys: Sequence[str],
    ) -> str:
        for key in primary_keys:
            normalized = self._normalize_text(record.get(key))
            if normalized:
                return normalized
        instance = record.get("instance")
        if isinstance(instance, dict):
            for key in instance_keys:
                normalized = self._normalize_text(instance.get(key))
                if normalized:
                    return normalized
        return ""

    @staticmethod
    def _extract_id(record: Dict[str, Any]) -> str:
        def _sanitize(value: Optional[str]) -> str:
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value
            return ""
        record_id = _sanitize(record.get("id"))
        if record_id: return record_id
        for key in ("url", "permalink"):
            value = _sanitize(record.get(key))
            if value:
                if key == "permalink" and value.startswith("/"):
                    return f"https://www.reddit.com{value}"
                return value
        instance = record.get("instance")
        if isinstance(instance, dict):
            inst_id = _sanitize(instance.get("id"))
            if inst_id: return inst_id
            for key in ("url", "permalink"):
                value = _sanitize(instance.get(key))
                if value:
                    if key == "permalink" and value.startswith("/"):
                        return f"https://www.reddit.com{value}"
                    return value
        return ""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.samples[idx]
        
        # 加上 dataset tag
        raw_source = f"{self.dataset_tag} {example['context']}" if self.dataset_tag else example["context"]
        source_text = raw_source
        target_text = example.get("summary", "")
        
        source_tokens = self.tokenizer.encode(
            source_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_len
        )
        
        if not self.require_target:
            target_tokens = None
        else:
            target_tokens = self.tokenizer.encode(
                target_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_target_len
            )
            
        # [標記重點] 回傳這筆資料是不是 TIFU (1=是, 0=否)
        is_tifu = 1 if "tifu" in self.dataset_tag else 0

        return {
            "id": example["id"],
            "input_ids": source_tokens,
            "labels": target_tokens,
            "is_tifu": is_tifu # 傳給 Collator
        }
        
def QACollator(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch provided to collator.")
    src_tokens: List[int] = []
    tgt_tokens: List[int] = []
    src_lens: List[int] = []
    tgt_lens: List[int] = []
    id_s = []
    is_tifu_list = [] # 用來存標籤
    
    for item in batch:
        src = item["input_ids"]
        src_tokens.extend(src)
        src_lens.append(len(src))
        
        tgt = item.get("labels")
        if tgt is not None:
            tgt_tokens.extend(tgt)
            tgt_lens.append(len(tgt))
        else:
            tgt_lens.append(0)
            
        id_s.append(item["id"])
        # 收集 TIFU 標籤
        is_tifu_list.append(item.get("is_tifu", 0))
        
    tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long) if tgt_tokens else torch.tensor([], dtype=torch.long)
    
    return {
        "src": torch.tensor(src_tokens, dtype=torch.long),
        "tgt": tgt_tensor,
        "src_len": torch.tensor(src_lens, dtype=torch.int32),
        "tgt_len": torch.tensor(tgt_lens, dtype=torch.int32),
        "id": id_s,
        "is_tifu": torch.tensor(is_tifu_list, dtype=torch.float32) # 轉成 Tensor
    }

def write_predictions_csv(path: Path, predictions: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "summary"])
        writer.writerows(predictions)