import numpy as np


class Encoder:

    def __init__(self, path):
        with open(path) as f:
            text = f.read()

        chars = sorted(set(text))
        self.dictionary = {"[MASK]": 0}
        for i, c in enumerate(chars, start=1):
            self.dictionary[c] = i
        self._inv = {v: k for k, v in self.dictionary.items()}

        max_ord = max((ord(c) for c in chars), default=0)
        self._lookup = np.zeros(max_ord + 1, dtype=np.int32)
        for c, i in self.dictionary.items():
            if c == "[MASK]":
                continue
            self._lookup[ord(c)] = i

    def encode(self, text):
        return [self.dictionary[char] for char in text]

    def encode_array(self, text, chunk_chars=4_000_000):
        out = np.empty(len(text), dtype=np.int32)
        for i in range(0, len(text), chunk_chars):
            block = text[i:i + chunk_chars]
            cps = np.frombuffer(block.encode("utf-32-le"), dtype=np.uint32)
            out[i:i + len(cps)] = self._lookup[cps]
        return out

    def decode(self, encoded) -> list[int]:
        return [self._inv[e] for e in encoded]

    def vocab(self):
        return self.dictionary.keys()


if __name__ == "__main__":
    encoder = Encoder("data/input.txt")

    encoded = encoder.encode("Hello World!")
    print(encoded)
    decoded = encoder.decode(encoded)
    print(decoded)
