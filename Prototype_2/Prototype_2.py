import numpy as np
from PIL import Image

# Encoder

def encode_sparse(data: bytes, size=160, cell_size=5):
    """
    Encode binary data into a sparse QR-like code (160x160 grid).
    
    Args:
        data (bytes): Data to encode.
        size (int): Grid size (size x size).
        cell_size (int): Pixel size of each cell.
    
    Returns:
        PIL.Image: Encoded image.
    """
    # Convert data to bits
    bits = ''.join(f"{byte:08b}" for byte in data)
    n = len(bits)

    # Decide whether to store positions of 1s or 0s
    ones = [i for i, b in enumerate(bits) if b == '1']
    zeros = [i for i, b in enumerate(bits) if b == '0']

    if len(ones) <= len(zeros):
        mode = 0  # store positions of 1s
        positions = ones
    else:
        mode = 1  # store positions of 0s
        positions = zeros

    # Metadata: first row (mode + 15-bit length)
    meta_bits = f"{mode:01b}" + f"{n:015b}"
    grid = np.zeros((size, size), dtype=np.uint8)

    # Fill metadata in first 16 cells of first row
    for i, bit in enumerate(meta_bits):
        grid[0, i] = 255 if bit == '1' else 0

    # Encode positions into the grid
    for pos in positions:
        row = (pos // size) + 1  # start from row 1 (row 0 = metadata)
        col = pos % size
        if row < size:
            grid[row, col] = 255

    # Scale up to an image
    img = Image.fromarray(grid).resize((size*cell_size, size*cell_size), Image.NEAREST)
    return img

# Decoder

def decode_sparse(img: Image.Image, size=160):
    """
    Decode a sparse QR-like code image back to binary data.
    
    Args:
        img (PIL.Image): Encoded image.
        size (int): Grid size (size x size).
    
    Returns:
        bytes: Decoded binary data.
    """
    # Downsample to grid
    grid = img.resize((size, size), Image.NEAREST)
    grid = np.array(grid)

    # Extract metadata (mode + total bit length)
    meta_bits = ''.join('1' if grid[0, i] > 128 else '0' for i in range(16))
    mode = int(meta_bits[0], 2)
    n = int(meta_bits[1:], 2)

    # Read positions
    positions = []
    for r in range(1, size):
        for c in range(size):
            if grid[r, c] > 128:
                pos = (r - 1) * size + c
                positions.append(pos)

    # Reconstruct bitstring
    bits = ['0'] * n if mode == 0 else ['1'] * n
    if mode == 0:
        for p in positions:
            if p < n:
                bits[p] = '1'
    else:
        for p in positions:
            if p < n:
                bits[p] = '0'

    # Convert bitstring to bytes safely
    bitstring = ''.join(bits)
    data = bytearray()
    for i in range(0, len(bitstring), 8):
        byte_bits = bitstring[i:i+8]
        if len(byte_bits) < 8:  # pad last byte if needed
            byte_bits = byte_bits.ljust(8, '0')
        data.append(int(byte_bits, 2))

    return bytes(data)

# Example Usage

if __name__ == "__main__":
    # Example data
    data = b"Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni.Hello, My name is Dev Soni."
    
    # Encode
    encoded_img = encode_sparse(data)
    encoded_img.save("sparse_qr_160x160.png")
    
    # Decode
    decoded_data = decode_sparse(encoded_img)
    print(decoded_data)  # b'Sparse QR 160x160 Prototype!'
