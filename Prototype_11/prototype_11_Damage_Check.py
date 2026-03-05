import numpy as np
from PIL import Image
import bz2
import docx
import math
import os
import random
from reedsolo import RSCodec


# MAPPING CONFIGURATION

BITS_TO_VALUE = {
    0: (255, 0, 0),      # 00 -> Red
    1: (0, 255, 0),      # 01 -> Green
    2: (0, 0, 255),      # 10 -> Blue
    3: (255, 255, 255)   # 11 -> White
}
PADDING_COLOR = (0, 0, 0)  # Black for padding
VALUE_TO_BITS = {v: k for k, v in BITS_TO_VALUE.items()}

# Reed–Solomon codec (32 redundancy symbols = correction up to 16 errors)
RS = RSCodec(32)


# DOCX READER

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading docx file: {e}")
        return None


# ENCODER WITH ERROR CORRECTION

def encode_data_quaternary(data_bytes, output_img_path, cell_size=5):
    print("\n--- Starting Quaternary RGBW Encoding (with Reed–Solomon) ---")
    print(f"Original data size: {len(data_bytes) / 1024:.2f} KB")

    # 1. Compress
    compressed = bz2.compress(data_bytes)
    print(f"Compressed size (bz2): {len(compressed) / 1024:.2f} KB")

    # 2. Add error correction
    encoded_with_ecc = RS.encode(compressed)
    print(f"Size after RS error correction: {len(encoded_with_ecc) / 1024:.2f} KB")

    # 3. Convert to quaternary pixels
    pixel_values = []
    for byte in encoded_with_ecc:
        for shift in (6, 4, 2, 0):
            val = (byte >> shift) & 0b11
            pixel_values.append(BITS_TO_VALUE[val])

    # 4. Grid size
    num_pixels = len(pixel_values)
    size = math.ceil(math.sqrt(num_pixels))
    padding_needed = size * size - num_pixels
    padded_values = pixel_values + ([PADDING_COLOR] * padding_needed)

    # 5. Make image
    grid_np = np.array(padded_values, dtype=np.uint8).reshape((size, size, 3))
    img = Image.fromarray(grid_np, mode='RGB')
    upscaled = img.resize((size * cell_size, size * cell_size), Image.NEAREST)

    upscaled.save(output_img_path)
    print(f" Encoded image saved as '{output_img_path}' ({upscaled.size[0]}x{upscaled.size[1]})")

    # return grid size, original length, and encoded length (important for decoding!)
    return size, len(data_bytes), len(encoded_with_ecc)


# DECODER WITH ERROR CORRECTION

def decode_data_quaternary(input_img_path, original_data_len, grid_size, encoded_len):
    print("\n--- Starting Quaternary RGBW Decoding (with Reed–Solomon) ---")

    # 1. Load image
    img = Image.open(input_img_path).convert('RGB')
    img = img.resize((grid_size, grid_size), Image.NEAREST)
    grid = np.array(img, dtype=np.uint8).reshape(-1, 3)

    # 2. Convert pixels back to bytes
    reconstructed_bytes = []
    for i in range(0, encoded_len * 4, 4):  # only decode real data, ignore padding
        chunk = grid[i:i+4]
        if len(chunk) < 4:
            break
        vals = [VALUE_TO_BITS.get(tuple(px), 0) for px in chunk]
        byte = (vals[0] << 6) | (vals[1] << 4) | (vals[2] << 2) | vals[3]
        reconstructed_bytes.append(byte)

    # 3. Reed–Solomon decode
    try:
        corrected = RS.decode(bytes(reconstructed_bytes))[0]
        decompressed = bz2.decompress(corrected)
        return decompressed[:original_data_len]
    except Exception as e:
        print(f" Error during RS decoding or decompression: {e}")
        return None


# DAMAGE IMAGE (noise in blocks)

def damage_image_blocks(input_img_path, output_img_path, num_blocks=10, block_size=20):
    """
    Corrupts random square blocks in the image.
    num_blocks: how many blocks to damage
    block_size: size of each corrupted block
    """
    print(f"\n--- Damaging encoded image (removing {num_blocks} blocks of size {block_size}x{block_size}) ---")

    img = Image.open(input_img_path).convert('RGB')
    grid = np.array(img, dtype=np.uint8)

    h, w, _ = grid.shape
    for _ in range(num_blocks):
        x = random.randint(0, h - block_size)
        y = random.randint(0, w - block_size)

        # Overwrite block with black noise
        grid[x:x+block_size, y:y+block_size] = [0, 0, 0]

    damaged_img = Image.fromarray(grid, mode='RGB')
    damaged_img.save(output_img_path)
    print(f"⚠️ Damaged image saved as '{output_img_path}'")
    return output_img_path


# MAIN

if __name__ == "__main__":
    docx_file = "tstdoc3.docx"
    encoded_img = "encoded_rs.png"
    damaged_img = "encoded_rs_damaged.png"
    recovered_file = "recovered_rs.txt"
    CELL_SIZE = 10

    if not os.path.exists(docx_file):
        print(f"Error: Input file '{docx_file}' not found.")
    else:
        # Encode
        text = extract_text_from_docx(docx_file)
        if text:
            data_bytes = text.encode("utf-8")
            grid_size, original_len, encoded_len = encode_data_quaternary(
                data_bytes, encoded_img, cell_size=CELL_SIZE
            )

            #  Decode (clean) 
            recovered = decode_data_quaternary(encoded_img, original_len, grid_size, encoded_len)
            if recovered:
                print("\n Clean image decode successful!")

            # Damage image (block corruption) 
            damage_image_blocks(encoded_img, damaged_img, num_blocks=19, block_size=40)

            # Decode (damaged) 
            recovered_damaged = decode_data_quaternary(damaged_img, original_len, grid_size, encoded_len)
            if recovered_damaged:
                recovered_text = recovered_damaged.decode("utf-8", errors="ignore")
                if recovered_text == text:
                    print("\n RS succeeded: Damaged image was fully corrected!")
                else:
                    print("\n RS partially succeeded: Some errors remain but text is readable.")
