import numpy as np
from PIL import Image

# Encoder

def encode_sparse(data: bytes, size=16, cell_size=20):
    """
    Encode data into a sparse QR-like code.
    """
    print("Step 0: Input Data:", data)
    
    # Step 1: Convert bytes to bits
    bits = ''.join(f"{byte:08b}" for byte in data)
    n = len(bits)
    print("\nStep 1: Convert bytes to bits")
    print("Bitstring:", bits)
    print("Total bits:", n)
    
    # Step 2: Decide sparse mode (store 1s or 0s)
    ones = [i for i, b in enumerate(bits) if b == '1']
    zeros = [i for i, b in enumerate(bits) if b == '0']
    
    if len(ones) <= len(zeros):
        mode = 0  # store ones
        positions = ones
        print("\nStep 2: Sparse Mode -> store positions of 1s")
    else:
        mode = 1  # store zeros
        positions = zeros
        print("\nStep 2: Sparse Mode -> store positions of 0s")
    
    print("Positions stored:", positions)
    
    # Step 3: Create metadata (mode + length of bitstring)
    meta_bits = f"{mode:01b}" + f"{n:015b}"
    print("\nStep 3: Metadata bits (mode + length):", meta_bits)
    
    # Step 4: Create grid
    grid = np.zeros((size, size), dtype=np.uint8)
    print("\nStep 4: Initialize grid with zeros (black cells)")
    
    # Step 5: Place metadata in first row
    for i, bit in enumerate(meta_bits):
        grid[0, i] = 255 if bit == '1' else 0
    print("Step 5: Place metadata in first row of grid")
    print(grid[0, :16])  # show metadata row
    
    # Step 6: Encode positions into grid
    for pos in positions:
        row = (pos // size) + 1  # +1 because row 0 = metadata
        col = pos % size
        if row < size:
            grid[row, col] = 255
    print("\nStep 6: Mark positions of stored bits in grid")
    print("Grid (numeric values, 0=black, 255=white):")
    print(grid)
    
    # Step 7: Scale to image
    img = Image.fromarray(grid).resize((size*cell_size, size*cell_size), Image.NEAREST)
    print("\nStep 7: Scale grid to image")
    return img

# Decoder with Step-by-Step Explanation

def decode_sparse(img: Image.Image, size=16):
    """
    Decode a sparse QR-like code back to bytes.
    """
    print("\nDecoding Step 1: Downsample image to grid size")
    grid = img.resize((size, size), Image.NEAREST)
    grid = np.array(grid)
    
    # Step 2: Extract metadata
    meta_bits = ''.join('1' if grid[0, i] > 128 else '0' for i in range(16))
    mode = int(meta_bits[0], 2)
    n = int(meta_bits[1:], 2)
    print("Decoding Step 2: Extract metadata")
    print("Mode:", mode, "(0=store 1s, 1=store 0s)")
    print("Total bits:", n)
    
    # Step 3: Read positions of stored bits
    positions = []
    for r in range(1, size):
        for c in range(size):
            if grid[r, c] > 128:
                pos = (r - 1) * size + c
                positions.append(pos)
    print("\nDecoding Step 3: Positions of stored bits:", positions)
    
    # Step 4: Reconstruct bitstring
    bits = ['0'] * n if mode == 0 else ['1'] * n
    if mode == 0:  # stored ones
        for p in positions:
            if p < n:
                bits[p] = '1'
    else:  # stored zeros
        for p in positions:
            if p < n:
                bits[p] = '0'
    bitstring = ''.join(bits)
    print("\nDecoding Step 4: Reconstructed bitstring:", bitstring)
    
    # Step 5: Convert bitstring back to bytes
    data = bytearray()
    for i in range(0, len(bitstring), 8):
        byte_bits = bitstring[i:i+8]
        if len(byte_bits) < 8:
            byte_bits = byte_bits.ljust(8, '0')
        data.append(int(byte_bits, 2))
    print("Decoding Step 5: Converted to bytes:", bytes(data))
    
    return bytes(data)

# Test

data = b"Hello"
print("=== Encoding ===")
encoded_img = encode_sparse(data, size=16, cell_size=20)
encoded_img.save("sparse_code_explained.png")
print("\n=== Decoding ===")
decoded_data = decode_sparse(encoded_img, size=16)
print("\nFinal Decoded Data:", decoded_data)
