import numpy as np
from PIL import Image
import bz2
import docx
import math
import os

# MAPPING CONFIGURATION

# We map each 2-bit value to a distinct, well-separated grayscale value.
BITS_TO_VALUE = {
    0: 0,    # 00 -> Black
    1: 85,   # 01 -> Dark Gray
    2: 170,  # 10 -> Light Gray
    3: 255   # 11 -> White
}

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

def encode_data_quaternary(data_bytes, output_img_path):
    """Encodes a byte string into an image using a 4-level (quaternary) grayscale system."""
    print("--- Starting Quaternary Encoding Process ---")
    print(f"Original data size: {len(data_bytes) / 1024:.2f} KB")

    # 1. Compress the data using bz2
    compressed_data = bz2.compress(data_bytes)
    print(f"Compressed size (bz2): {len(compressed_data) / 1024:.2f} KB")

    # 2. Convert byte stream to a list of pixel values
    pixel_values = []
    for byte in compressed_data:
        # Each byte (8 bits) will be represented by 4 pixels.
        # We extract each 2-bit chunk from the byte.
        
        # Extract bits 7-6 (the first pair)
        val1 = (byte >> 6) & 0b11
        # Extract bits 5-4 (the second pair)
        val2 = (byte >> 4) & 0b11
        # Extract bits 3-2 (the third pair)
        val3 = (byte >> 2) & 0b11
        # Extract bits 1-0 (the last pair)
        val4 = byte & 0b11
        
        # Map these 2-bit values (0, 1, 2, 3) to our grayscale values
        pixel_values.append(BITS_TO_VALUE[val1])
        pixel_values.append(BITS_TO_VALUE[val2])
        pixel_values.append(BITS_TO_VALUE[val3])
        pixel_values.append(BITS_TO_VALUE[val4])

    # 3. Determine the image grid size
    num_pixels = len(pixel_values)
    # Find the smallest square grid that can fit all the pixels
    size = math.ceil(math.sqrt(num_pixels))
    print(f"Required grid size: {size}x{size} pixels")

    # 4. Pad the pixel list to make a perfect square
    padding_needed = size * size - num_pixels
    # We use black (0) for padding
    padded_values = pixel_values + ([0] * padding_needed)

    # 5. Create and save the image
    # Convert the list to a NumPy array
    grid = np.array(padded_values, dtype=np.uint8)
    # Reshape the 1D array into a 2D grid
    grid = grid.reshape((size, size))
    
    # Create an image from the NumPy array
    img = Image.fromarray(grid, mode='L') # 'L' mode for grayscale
    img.save(output_img_path)
    print(f"Encoded image saved as '{output_img_path}'")
    return size # Return the size for the decoder


# QUATERNARY DECODER

def decode_data_quaternary(input_img_path, original_data_len):
    """Decodes an image created with the quaternary system."""
    print("\n--- Starting Quaternary Decoding Process ---")
    
    # 1. Load the image and convert to a NumPy array
    img = Image.open(input_img_path)
    grid = np.array(img, dtype=np.uint8)

    # 2. Flatten the 2D grid back into a 1D list of pixel values
    pixel_values = grid.flatten()

    # 3. Convert pixel values back to bytes
    reconstructed_bytes = []
    # Process the pixel values in chunks of 4
    for i in range(0, len(pixel_values), 4):
        chunk = pixel_values[i:i+4]
        
        # Get the 2-bit value for each pixel in the chunk
        # Use .get() with a default of 0 for robustness against padding
        val1 = VALUE_TO_BITS.get(chunk[0], 0)
        val2 = VALUE_TO_BITS.get(chunk[1], 0)
        val3 = VALUE_TO_BITS.get(chunk[2], 0)
        val4 = VALUE_TO_BITS.get(chunk[3], 0)
        
        # Combine the four 2-bit values back into a single 8-bit byte
        # (val1 << 6) places the first pair at bits 7-6
        # (val2 << 4) places the second pair at bits 5-4 and so on
        byte = (val1 << 6) | (val2 << 4) | (val3 << 2) | val4
        reconstructed_bytes.append(byte)
        
    # 4. Decompress the byte stream
    try:
        decompressed_data = bz2.decompress(bytes(reconstructed_bytes))
        # Important: Truncate to the original length to remove any padding artifacts
        return decompressed_data[:original_data_len]
    except Exception as e:
        print(f"Error during decompression: {e}")
        return None


# MAIN EXECUTION BLOCK

if __name__ == "__main__":
    docx_file = "tstdoc3.docx"
    image_file = "encoded_quaternary.png"
    output_text_file = "recovered_quaternary.txt"

    if not os.path.exists(docx_file):
        print(f"Error: The input file '{docx_file}' was not found.")
    else:
        #  1. ENCODE 
        document_text = extract_text_from_docx(docx_file)
        if document_text:
            text_as_bytes = document_text.encode('utf-8')
            original_length = len(text_as_bytes) # Store original length for the decoder
            
            encode_data_quaternary(text_as_bytes, image_file)

            #  2. DECODE 
            recovered_bytes = decode_data_quaternary(image_file, original_length)

            if recovered_bytes:
                recovered_text = recovered_bytes.decode('utf-8')
                
                #  3. VERIFY AND SAVE 
                if recovered_text == document_text:
                    print("\n Verification successful! Decoded data matches original.")
                    with open(output_text_file, "w", encoding="utf-8") as f:
                        f.write(recovered_text)
                    print(f"Recovered text saved to '{output_text_file}'")
                else:
                    print("\n❌ Verification failed! Decoded data does not match original.")
