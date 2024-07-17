from pathlib import Path
from typing import List

import numpy as np

from astravani.data.tokenizers import Tokenizer


class CharacterTokenizer(Tokenizer):
    def __init__(
        self,
        vocab_file_or_list: List[str] | str | Path,
        stress=True,
        instersperse_blanks=True,
        replace_with_oov=False,
        text_preprocessing_func=lambda x: x,
    ):
        super().__init__(vocab_file_or_list=vocab_file_or_list)
        self.stress = stress
        self.instersperse_blanks = instersperse_blanks
        self.replace_with_oov = replace_with_oov

    def encode(self, text: str) -> List[int]:
        codes, space = [], self.vocab[" "]
        text = text.lower() if not self.stress else text
        text = self.text_preprocessing_func(text)
        for char in text:
            if char == " " and len(codes) > 0 and codes[-1] != space:
                codes.append(space)
            elif char in self.vocab:
                codes.append(self.vocab[char])
            elif self.replace_with_oov:
                codes.append(self.vocab[self.OOV])

            if (
                self.instersperse_blanks
                and char != " "
                and len(codes) > 0
                and codes[-1] != space
            ):
                codes.append(self.vocab[self.BLANK])

        return codes

    def decode(self, codes: List[int]) -> str:
        text = ""
        for code in codes:
            if code == self.vocab[" "] and len(text) > 0 and text[-1] != " ":
                text += " "
            elif code in self.vocab_inv and self.vocab_inv[code] not in [
                self.BLANK,
                self.PAD,
            ]:
                text += self.vocab_inv[code]

        return text

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        # Preprocess and vectorize texts
        processed_texts = [
            self.text_preprocessing_func(text.lower() if not self.stress else text)
            for text in texts
        ]
        max_length = max(len(text) for text in processed_texts)
        max_length += int(self.instersperse_blanks) * max_length - 1

        # Initialize numpy arrays for batch processing
        batch_codes = np.full((len(texts), max_length), self.vocab[self.PAD], dtype=int)

        for i, text in enumerate(processed_texts):
            codes = []
            for char in text:
                if char == " " and len(codes) > 0 and codes[-1] != self.vocab[" "]:
                    codes.append(self.vocab[" "])
                elif char in self.vocab:
                    codes.append(self.vocab[char])
                elif self.replace_with_oov:
                    codes.append(self.vocab[self.OOV])

                if (
                    self.instersperse_blanks
                    and char != " "
                    and len(codes) > 0
                    and codes[-1] != self.vocab[" "]
                ):
                    codes.append(self.vocab[self.BLANK])

            batch_codes[i, : len(codes)] = codes

        return batch_codes.tolist()

    def batch_decode(self, codes: List[List[int]]) -> List[str]:
        # Initialize an empty list to store decoded texts
        decoded_texts = []

        # Convert codes to numpy array for batch processing
        codes_array = np.array(codes, dtype=int)

        for code_seq in codes_array:
            text = ""
            for code in code_seq:
                if code == self.vocab[" "] and len(text) > 0 and text[-1] != " ":
                    text += " "
                elif code in self.vocab_inv and self.vocab_inv[code] not in [
                    self.BLANK,
                    self.PAD,
                ]:
                    text += self.vocab_inv[code]
            decoded_texts.append(text)

        return decoded_texts

    def __len__(self) -> int:
        return len(self.vocab)
