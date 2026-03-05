import numpy as np 
from PIL import Image  

def encode_sparse(data: bytes, size=16, cell_size=20):
    """
    Encode data into a sparse QR-like code.
    
    Args:
    
        data (bytes): The data to encode.
        size (int): Grid size (size x size).
        cell_size (int): Pixel size of each cell.
    
    Returns:
        np.ndarray: Encoded image grid.
    """
    # Convert data into bits
    bits = ''.join(f"{byte:08b}" for byte in data)
    n = len(bits)

    # Decide which to store: positions of 1s or 0s
    ones = [i for i, b in enumerate(bits) if b == '1']
    zeros = [i for i, b in enumerate(bits) if b == '0']
    
    if len(ones) <= len(zeros):
        mode = 0  # store ones
        positions = ones
    else:
        mode = 1  # store zeros
        positions = zeros

    # Reserve first row: [mode bit + length of data in bits (15 bits)]
    meta_bits = f"{mode:01b}" + f"{n:015b}"
    
    # Create grid
    grid = np.zeros((size, size), dtype=np.uint8)

    # Place metadata in first 16 cells of first row
    for i, bit in enumerate(meta_bits):
        grid[0, i] = 255 if bit == '1' else 0

    # Encode positions into grid (row by row)
    for pos in positions:
        row = (pos // size) + 1  # start from row 1 (row 0 is metadata)
        col = pos % size
        if row < size:
            grid[row, col] = 255

    # Scale grid to image
    img = Image.fromarray(grid).resize((size*cell_size, size*cell_size), Image.NEAREST)
    return img


def decode_sparse(img: Image.Image, size=16):
    """
    Decode sparse encoded image back to data.
    
    Args:
        img (Image.Image): Encoded image.
        size (int): Grid size.
    
    Returns:
        bytes: Decoded data.
    """
    # Downsample to grid
    grid = img.resize((size, size), Image.NEAREST)
    grid = np.array(grid)

    # Extract metadata
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
    bits = ['0'] * n
    if mode == 0:  # stored ones
        for p in positions:
            if p < n:
                bits[p] = '1'
    else:  # stored zeros
        bits = ['1'] * n
        for p in positions:
            if p < n:
                bits[p] = '0'

    # Convert to bytes
    bitstring = ''.join(bits)
    data = bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))
    return data


# Test
data = b"Hello"
encoded_img = encode_sparse(data, size=16, cell_size=20)
decoded_data = decode_sparse(encoded_img, size=16)

encoded_img.save("sparse_code.png")
decoded_data = decode_sparse(encoded_img, size=16)
print(decoded_data)
