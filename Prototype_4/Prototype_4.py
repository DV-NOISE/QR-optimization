import numpy as np
from PIL import Image
import zlib
from docx import Document

# Read data from Word file

def read_word_file(file_path):
    """
    Read all text from a Word (.docx) file and return as bytes.
    """
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    text = '\n'.join(full_text)
    return text.encode('utf-8')  # convert to bytes

# Encoder with Compression

def encode_sparse_compressed(data: bytes, size=160, cell_size=5):
    """
    Encode data into a 160x160 sparse QR-like code with compression.
    """
    print("Step 0: Original data length:", len(data))
    
    # Step 1: Compress data
    compressed = zlib.compress(data)
    print("Step 1: Compressed data length:", len(compressed))
    
    # Step 2: Convert compressed data to bits
    bits = ''.join(f"{b:08b}" for b in compressed)
    n = len(bits)
    print("Step 2: Total bits to encode:", n)
    
    # Step 3: Determine sparse mode (store minority bits)
    ones = [i for i, b in enumerate(bits) if b == '1']
    zeros = [i for i, b in enumerate(bits) if b == '0']
    
    if len(ones) <= len(zeros):
        mode = 0  # store ones
        positions = ones
        print("Step 3: Sparse mode -> store 1s")
    else:
        mode = 1  # store zeros
        positions = zeros
        print("Step 3: Sparse mode -> store 0s")
    
    # Step 4: Metadata (mode + bit length, 16 bits)
    meta_bits = f"{mode:01b}" + f"{n:015b}"
    
    # Step 5: Initialize grid
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Step 6: Store metadata in first row
    for i, bit in enumerate(meta_bits):
        grid[0, i] = 255 if bit == '1' else 0
    
    # Step 7: Store positions of minority bits in grid
    for pos in positions:
        row = (pos // size) + 1  # row 0 reserved for metadata
        col = pos % size
        if row < size:
            grid[row, col] = 255
    
    # Step 8: Convert to image
    img = Image.fromarray(grid).resize((size*cell_size, size*cell_size), Image.NEAREST)
    return img

# Decoder with Decompression

def decode_sparse_compressed(img: Image.Image, size=160):
    """
    Decode sparse QR-like code and decompress to original data.
    """
    # Step 1: Downsample to grid
    grid = img.resize((size, size), Image.NEAREST)
    grid = np.array(grid)
    
    # Step 2: Read metadata
    meta_bits = ''.join('1' if grid[0, i] > 128 else '0' for i in range(16))
    mode = int(meta_bits[0], 2)
    n = int(meta_bits[1:], 2)
    
    # Step 3: Read stored positions
    positions = []
    for r in range(1, size):
        for c in range(size):
            if grid[r, c] > 128:
                pos = (r - 1) * size + c
                positions.append(pos)
    
    # Step 4: Reconstruct bitstring
    bits = ['0'] * n if mode == 0 else ['1'] * n
    if mode == 0:
        for p in positions:
            if p < n:
                bits[p] = '1'
    else:
        for p in positions:
            if p < n:
                bits[p] = '0'
    
    bitstring = ''.join(bits)
    
    # Step 5: Convert bitstring to bytes
    data_bytes = bytearray()
    for i in range(0, len(bitstring), 8):
        byte_bits = bitstring[i:i+8]
        if len(byte_bits) < 8:
            byte_bits = byte_bits.ljust(8, '0')
        data_bytes.append(int(byte_bits, 2))
    
    # Step 6: Decompress to original data
    original_data = zlib.decompress(bytes(data_bytes))
    return original_data

# Main

if __name__ == "__main__":
    # Read from Word file
    file_path = "tstdoc1.docx"  # replace with your file path
    data = read_word_file(file_path)
    print("Original Data Length (bytes):", len(data))
    
    # Encode
    encoded_img = encode_sparse_compressed(data, size=160, cell_size=5)
    encoded_img.save("sparse_160x160_word_compressed.png")
    print("Encoded image saved as 'sparse_160x160_word_compressed.png'")
    
    # Decode
    decoded_data = decode_sparse_compressed(encoded_img, size=160)
    print("Decoded Data matches original:", decoded_data == data)
