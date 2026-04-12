from torch import nn

class Encoder:

    def __init__(self, path):
        with open(path) as f:
            lines = f.readlines()
        
        text = "\n".join(lines)

        self.dictionary = {}

        index = 1
        self.dictionary = {"[MASK]": 0}
        for char in text:
            if self.dictionary.get(char):
                continue
            
            self.dictionary[char] = index
            index += 1 
    
    def encode(self, text):
        return [self.dictionary[char] for char in text]
    
    def decode(self, encoded) -> list[int]:
        inv_dict = {v: k for k, v in self.dictionary.items()}
        return [inv_dict[encoding] for encoding in encoded]
    
    def vocab(self):
        return self.dictionary.keys()


if __name__ == "__main__":
    encoder = Encoder("data/input.txt")

    encoded = encoder.encode("Hello World!")
    print(encoded)
    decoded = encoder.decode(encoded)
    print(decoded)
