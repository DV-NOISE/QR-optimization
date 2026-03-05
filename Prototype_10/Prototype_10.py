import numpy as np
from PIL import Image
import bz2
import docx
import math
import os


# MAPPING CONFIGURATION

# We map each 2-bit value to a distinct, high-contrast RGB color.
BITS_TO_VALUE = {
    0: (255, 0, 0),      # 00 -> Red
    1: (0, 255, 0),      # 01 -> Green
    2: (0, 0, 255),      # 10 -> Blue
    3: (255, 255, 255)   # 11 -> White
}
PADDING_COLOR = (0, 0, 0) # Black for padding

# Create the reverse mapping for the decoder automatically.

VALUE_TO_BITS = {v: k for k, v in BITS_TO_VALUE.items()}



# HELPER: EXTRACT TEXT FROM A .DOCX FILE

def extract_text_from_docx(docx_path):
    """Reads a .docx file and returns its text content as a string."""
    try:
        doc = docx.Document(docx_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading docx file: {e}")
        return None


# QUATERNARY ENCODER

def encode_data_quaternary(data_bytes, output_img_path, cell_size=5):
    """Encodes a byte string into an image using an RGBW (Red, Green, Blue, White) color system."""
    print("--- Starting Quaternary RGBW Encoding Process ---")
    print(f"Original data size: {len(data_bytes) / 1024:.2f} KB")

    # 1. Compress the data using bz2
    compressed_data = bz2.compress(data_bytes)
    print(f"Compressed size (bz2): {len(compressed_data) / 1024:.2f} KB")

    # 2. Convert byte stream to a list of pixel values (now RGB tuples)
    pixel_values = []
    for byte in compressed_data:
        val1 = (byte >> 6) & 0b11
        val2 = (byte >> 4) & 0b11
        val3 = (byte >> 2) & 0b11
        val4 = byte & 0b11
        
        pixel_values.extend([
            BITS_TO_VALUE[val1],
            BITS_TO_VALUE[val2],
            BITS_TO_VALUE[val3],
            BITS_TO_VALUE[val4]
        ])

    # 3. Determine the image grid size
    num_pixels = len(pixel_values)
    size = math.ceil(math.sqrt(num_pixels))
    print(f"Required data grid size: {size}x{size} pixels")

    # 4. Pad the pixel list to make a perfect square
    padding_needed = size * size - num_pixels
    padded_values = pixel_values + ([PADDING_COLOR] * padding_needed)

    # 5. Create the base image grid
    grid_np = np.array(padded_values, dtype=np.uint8)
    # Reshape into a 3-channel (RGB) image grid
    grid = grid_np.reshape((size, size, 3))
    
    # 6. Create final image and upscale it for clarity
    img = Image.fromarray(grid, mode='RGB') # Use 'RGB' mode for color
    
    upscaled_img = img.resize((size * cell_size, size * cell_size), Image.NEAREST)
    
    upscaled_img.save(output_img_path)
    print(f"Encoded image ({upscaled_img.size[0]}x{upscaled_img.size[1]}) saved as '{output_img_path}'")
    return size


# QUATERNARY DECODER

def decode_data_quaternary(input_img_path, original_data_len, grid_size):
    """Decodes an image created with the quaternary RGBW system."""
    print("\n--- Starting Quaternary RGBW Decoding Process ---")
    
    # 1. Load the image, ensure it's in RGB mode, and downscale
    img = Image.open(input_img_path).convert('RGB')
    img = img.resize((grid_size, grid_size), Image.NEAREST)
    grid = np.array(img, dtype=np.uint8)

    # 2. Reshape the grid into a list of pixel tuples
    pixel_values = grid.reshape(-1, 3)

    # 3. Convert pixel tuples back to bytes
    reconstructed_bytes = []
    for i in range(0, len(pixel_values), 4):
        chunk = pixel_values[i:i+4]
        
        # Convert each RGB tuple in the chunk to its 2-bit value
        val1 = VALUE_TO_BITS.get(tuple(chunk[0]), 0)
        val2 = VALUE_TO_BITS.get(tuple(chunk[1]), 0)
        val3 = VALUE_TO_BITS.get(tuple(chunk[2]), 0)
        val4 = VALUE_TO_BITS.get(tuple(chunk[3]), 0)
        
        byte = (val1 << 6) | (val2 << 4) | (val3 << 2) | val4
        reconstructed_bytes.append(byte)
        
    # 4. Decompress the byte stream
    try:
        decompressed_data = bz2.decompress(bytes(reconstructed_bytes))
        return decompressed_data[:original_data_len]
    except Exception as e:
        print(f"Error during decompression: {e}")
        return None


# MAIN

if __name__ == "__main__":
    docx_file = "tstdoc2.docx"
    image_file = "encoded_rgbw.png"
    output_text_file = "recovered_rgbw.txt"
    CELL_SIZE_FOR_ENCODING = 10

    if not os.path.exists(docx_file):
        print(f"Error: The input file '{docx_file}' was not found.")
    else:
        #  1. ENCODE 
        document_text = extract_text_from_docx(docx_file)
        if document_text:
            text_as_bytes = document_text.encode('utf-8')
            original_length = len(text_as_bytes)
            
            data_grid_size = encode_data_quaternary(
                text_as_bytes, 
                image_file, 
                cell_size=CELL_SIZE_FOR_ENCODING
            )

            # 2. DECODE
            recovered_bytes = decode_data_quaternary(
                image_file, 
                original_length, 
                grid_size=data_grid_size
            )

            if recovered_bytes:
                recovered_text = recovered_bytes.decode('utf-8')
                
                # 3. VERIFY AND SAVE 
                if recovered_text == document_text:
                    print("\n Verification successful! Decoded data matches original.")
                    with open(output_text_file, "w", encoding="utf-8") as f:
                        f.write(recovered_text)
                    print(f"Recovered text saved to '{output_text_file}'")
                else:
                    print("\n Verification failed! Decoded data does not match original.")

