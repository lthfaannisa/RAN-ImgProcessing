class HuffmanNode:
    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

def build_huffman_tree(freq_dict):
    nodes = [HuffmanNode(char, freq) for char, freq in freq_dict.items()]

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)

        merged = HuffmanNode(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right

        nodes.append(merged)

    return nodes[0]

def generate_huffman_codes(node, code="", mapping=None):
    if mapping is None:
        mapping = {}

    if node is not None:
        if node.char is not None:
            mapping[node.char] = code
        generate_huffman_codes(node.left, code + "0", mapping)
        generate_huffman_codes(node.right, code + "1", mapping)

    return mapping

def huffman_compress(text):
    freq_dict = {char: text.count(char) for char in set(text)}
    root = build_huffman_tree(freq_dict)
    codes = generate_huffman_codes(root)

    compressed_text = ''.join(codes[char] for char in text)
    return compressed_text, codes

def huffman_decompress(compressed_text, codes):
    reversed_codes = {code: char for char, code in codes.items()}
    current_code = ""
    decompressed_text = ""

    for bit in compressed_text:
        current_code += bit
        if current_code in reversed_codes:
            decompressed_text += reversed_codes[current_code]
            current_code = ""

    return decompressed_text
