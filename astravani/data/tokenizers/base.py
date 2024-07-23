import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

class Tokenizer(ABC):
    DEFAULT_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    PAD, BLANK, OOV = "<PAD>", "<BLANK>", "<OOV>"

    def __init__(self, vocab_file_or_list: List[str] | str | Path):
        if isinstance(vocab_file_or_list, str) or isinstance(vocab_file_or_list, Path):
            with open(vocab_file_or_list, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
                self.__vocab_size = len(self.vocab)
            logging.info(f"Tokenizer state loaded from {vocab_file_or_list}")
        if isinstance(vocab_file_or_list, list):
            vocab_file_or_list = sorted(list(set(vocab_file_or_list)))
            self.vocab = {token: i for i, token in enumerate(vocab_file_or_list)}
            self.__vocab_size = len(self.vocab)
            logging.info(
                f"Tokenizer initialized from a list of {len(vocab_file_or_list)} elements"
            )

        for unit in Tokenizer.DEFAULT_PUNCTUATION:
            if unit not in self.vocab:
                self.vocab[unit] = self.__vocab_size
                self.__vocab_size += 1

        self.__add_special_tokens()

        logging.info(f"Tokenizer initialized with {self.__vocab_size} tokens")
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __add_special_tokens(self):
        if " " not in self.vocab:
            self.vocab[" "] = self.__vocab_size
            self.__vocab_size += 1
        self.vocab[Tokenizer.PAD] = self.__vocab_size
        self.vocab[Tokenizer.BLANK] = self.__vocab_size + 1
        self.vocab[Tokenizer.OOV] = self.__vocab_size + 2
        self.__vocab_size += 3

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[str]) -> str:
        pass

    @abstractmethod
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        pass

    @abstractmethod
    def batch_decode(self, tokens_list: List[List[int]]) -> List[str]:
        pass

    @property
    def vocab_size(self) -> int:
        return self.__vocab_size

    @property
    def tokens(self) -> List[str]:
        return list(self.vocab.keys())

    @property
    def pad_id(self) -> int:
        return self.vocab[Tokenizer.PAD]

    def save_state(self, file_path: str | Path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        logging.info(f"Tokenizer state saved to {file_path}")
