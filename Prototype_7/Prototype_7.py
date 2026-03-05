import numpy as np
from PIL import Image
import bz2  # <-- Using bz2 instead of zlib
import docx

# HELPER FOR DOCX

def extract_text_from_docx(docx_path):
    """Reads a .docx file and returns its text content as a string."""
    try:
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error reading docx file: {e}")
        return None

#  ENCODER

def encode_data(data_bytes, output_img, size=160, cell_size=5):
    """Encodes a byte string into a sparse image."""
    print(f"Original data size: {len(data_bytes)/1024:.2f} KB")

    # Compress with bz2 for better compression ratio
    compressed = bz2.compress(data_bytes)
    print(f"Compressed size (bz2): {len(compressed)/1024:.2f} KB")

    # Convert to bits
    bits = ''.join(f"{byte:08b}" for byte in compressed)
    n = len(bits)

    # Decide storage mode (1s or 0s)
    ones = [i for i, b in enumerate(bits) if b == '1']
    zeros = [i for i, b in enumerate(bits) if b == '0']

    if len(ones) <= len(zeros):
        mode = 0  # store ones
        positions = ones
    else:
        mode = 1  # store zeros
        positions = zeros

    # Metadata: [mode bit + length of data in bits (31 bits)]
    meta_bits = f"{mode:01b}" + f"{n:031b}"

    # Create grid
    grid = np.zeros((size, size), dtype=np.uint8)

    # Write metadata in first 32 cells
    for i, bit in enumerate(meta_bits):
        grid[0, i] = 255 if bit == '1' else 0

    # Encode positions
    max_positions = size * size - 32
    if len(positions) > max_positions:
        print(f"Warning: Data is too large to fit in the image.")
        print(f"Storing {max_positions} positions out of {len(positions)}.")
        positions = positions[:max_positions]
        
    for pos in positions:
        # We start encoding from cell 32 onwards
        effective_pos = pos + 32
        row = effective_pos // size
        col = effective_pos % size
        
        if row < size:
            grid[row, col] = 255

    # Save image
    img = Image.fromarray(grid).resize((size*cell_size, size*cell_size), Image.NEAREST)
    img.save(output_img)
    print(f"Encoded image saved as {output_img}")


#  DECODER

def decode_data(input_img, size=160):
    """Decodes a sparse image and returns the original data as bytes."""
    # Load image and downscale
    img = Image.open(input_img).resize((size, size), Image.NEAREST)
    # Ensure it's grayscale for simplicity
    grid = np.array(img.convert('L'))

    # Read metadata
    meta_bits = ''.join('1' if grid[0, i] > 128 else '0' for i in range(32))
    mode = int(meta_bits[0], 2)
    n = int(meta_bits[1:], 2)

    # Extract positions
    positions = []
    # Iterate through all cells, starting after metadata
    for i in range(32, size * size):
        row = i // size
        col = i % size
        if grid[row, col] > 128:
            # The position in the bitstream is the cell index minus the metadata length
            pos = i - 32
            positions.append(pos)

    # Reconstruct bitstring
    if mode == 0: # We stored the positions of '1's
        bits = ['0'] * n
        for p in positions:
            if p < n:
                bits[p] = '1'
    else: # We stored the positions of '0's
        bits = ['1'] * n
        for p in positions:
            if p < n:
                bits[p] = '0'

    # Convert bits to bytes
    bitstring = ''.join(bits)
    # The bitstring length might not be a multiple of 8, handle the remainder
    byte_list = []
    for i in range(0, len(bitstring), 8):
        byte = bitstring[i:i+8]
        if len(byte) == 8:
            byte_list.append(int(byte, 2))
            
    data = bytes(byte_list)

    # Decompress and return
    try:
        # Decompress using bz2
        decompressed = bz2.decompress(data)
        return decompressed
    except Exception as e: # bz2 can raise various errors (e.g., OSError)
        print(f"Error decompressing data: {e}")
        return None


# TEST  

# Define file paths
docx_file = "tstdoc2.docx"
image_file = "encoded_text_bz2.png" 
output_text_file = "recovered_text_bz2.txt"

#  1. ENCODE TEXT FROM WORD FILE 
print("--- Starting Encoding Process with BZ2 ---")
# Extract text from the docx file
document_text = extract_text_from_docx(docx_file)

if document_text:
    # Convert the text string to bytes using UTF-8 encoding
    text_as_bytes = document_text.encode('utf-8')
    
    # Encode these bytes into the image
    encode_data(text_as_bytes, image_file, size=160, cell_size=5)

    # 2. DECODE IMAGE BACK INTO TEXT 
    print("\n--- Starting Decoding Process ---")
    # Decode the image to get the original bytes
    recovered_bytes = decode_data(image_file, size=160)

    if recovered_bytes:
        # Convert the bytes back to a string
        recovered_text = recovered_bytes.decode('utf-8')
        
        # Write the recovered text to a .txt file
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(recovered_text)
            
        print(f"Decoded text saved successfully as {output_text_file}")
