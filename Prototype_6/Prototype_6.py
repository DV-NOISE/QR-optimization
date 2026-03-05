import numpy as np
from PIL import Image
import zlib
import lzma
import bz2
import zstandard as zstd  # You may need to run: pip install zstandard
import docx
import textwrap

# HELPER: EXTRACT TEXT FROM A .DOCX FILE

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

# CORE ENCODER

def encode_data(data_bytes, output_img, method='lzma', size=160, cell_size=5):
    """
    Encodes a byte string into a sparse image using the specified compression method.
    
    Args:
        data_bytes (bytes): The raw data to encode.
        output_img (str): Path to save the output image.
        method (str): Compression method ('zlib', 'lzma', 'bz2', 'zstd').
        size (int): The dimension of the square grid for the image.
        cell_size (int): The size of each data cell in the final image.
    """
    print(f"\n Starting Encoding with Method: '{method.upper()}' ")
    
    #  1. Compress Data 
    print(f"[ENCODE STEP 1]: Compressing {len(data_bytes)} bytes...")
    compression_map = {
        'zlib': zlib.compress,
        'lzma': lzma.compress,
        'bz2': bz2.compress,
        'zstd': zstd.ZstdCompressor().compress
    }
    if method not in compression_map:
        raise ValueError(f"Unknown compression method: {method}")
        
    compressed = compression_map[method](data_bytes)
    print(f"  > Compressed size: {len(compressed)} bytes")

    #  2. Convert to Bits 
    bits = ''.join(f"{byte:08b}" for byte in compressed)
    n = len(bits)
    print(f"  > Total bits to encode: {n}")

    #  3. Choose Sparse Storage Mode 
    ones = [i for i, b in enumerate(bits) if b == '1']
    zeros = [i for i, b in enumerate(bits) if b == '0']
    
    mode = 0 if len(ones) <= len(zeros) else 1
    positions = ones if mode == 0 else zeros
    print(f"  > Mode {mode} selected. Storing {len(positions)} positions.")

    #  4. Create Metadata 
    # We create a 32-bit header:
    # [2 bits for method] + [1 bit for sparse mode] + [29 bits for length n]
    method_map = {'zlib': 0, 'lzma': 1, 'bz2': 2, 'zstd': 3}
    method_bits = f"{method_map[method]:02b}"
    mode_bit = f"{mode:01b}"
    length_bits = f"{n:029b}"
    
    if n >= 2**29:
        raise ValueError(f"Data is too large. Bit length {n} exceeds the 29-bit metadata limit.")
        
    meta_bits = method_bits + mode_bit + length_bits
    print(f"[ENCODE STEP 2]: Created 32-bit metadata: {meta_bits}")

    #  5. Create and Populate Image Grid 
    grid = np.zeros((size, size), dtype=np.uint8)

    # Write metadata to the first 32 pixels
    for i, bit in enumerate(meta_bits):
        grid[0, i] = 255 if bit == '1' else 0

    # Write data positions to the rest of the grid
    for pos in positions:
        effective_pos = pos + 32  # Offset to skip metadata
        row = effective_pos // size
        col = effective_pos % size
        if row < size:
            grid[row, col] = 255
            
    print(f"  > Mapped metadata and {len(positions)} data points to image grid.")

    #6. Save Final Image
    img = Image.fromarray(grid).resize((size*cell_size, size*cell_size), Image.NEAREST)
    img.save(output_img)
    print(f" Encoding Complete. Image saved as {output_img} ")


# CORE DECODER

def decode_data(input_img, size=160):
    """
    Decodes a sparse image and returns the original data as bytes,
    automatically detecting the compression method from metadata.
    """
    print(f"\n Starting Decoding for {input_img} ")

    #  1. Load Image and Read Metadata 
    print("[DECODE STEP 1]: Loading image and reading metadata...")
    img = Image.open(input_img).resize((size, size), Image.NEAREST)
    grid = np.array(img.convert('L'))
    
    meta_bits = ''.join('1' if grid[0, i] > 128 else '0' for i in range(32))
    
    method_bits = meta_bits[0:2]
    mode_bit = meta_bits[2:3]
    length_bits = meta_bits[3:32]
    
    method_id = int(method_bits, 2)
    mode = int(mode_bit, 2)
    n = int(length_bits, 2)
    
    method_map = {0: 'zlib', 1: 'lzma', 2: 'bz2', 3: 'zstd'}
    method = method_map.get(method_id)

    if not method:
        raise ValueError(f"Unknown compression method ID: {method_id}")
        
    print(f"  > Metadata Decoded:")
    print(f"    - Compression Method: '{method.upper()}'")
    print(f"    - Sparse Mode: {mode}")
    print(f"    - Original Bitstream Length: {n}")

    #  2. Extract Data Positions 
    positions = []
    for i in range(32, size * size):
        row = i // size
        col = i % size
        if grid[row, col] > 128:
            positions.append(i - 32)
    print(f"  > Found {len(positions)} data positions.")

    #  3. Reconstruct Bitstring 
    bits = ['0'] * n if mode == 0 else ['1'] * n
    for p in positions:
        if p < n:
            bits[p] = '1' if mode == 0 else '0'

    bitstring = "".join(bits)

    #  4. Convert Bits to Bytes 
    byte_list = [int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8) if len(bitstring[i:i+8]) == 8]
    data = bytes(byte_list)
    print(f"  > Reconstructed {len(data)} bytes of compressed data.")

    #  5. Decompress Data 
    print(f"[DECODE STEP 2]: Decompressing data using '{method.upper()}'...")
    decompression_map = {
        'zlib': zlib.decompress,
        'lzma': lzma.decompress,
        'bz2': bz2.decompress,
        'zstd': zstd.ZstdDecompressor().decompress
    }
    
    try:
        decompressed = decompression_map[method](data)
        print(f"  > Decompression successful. Final data size: {len(decompressed)} bytes.")
        print(f" Decoding Complete ")
        return decompressed
    except Exception as e:
        print(f"  > DECOMPRESSION FAILED: {e}")
        return None


# EXECUTION BLOCK

if __name__ == "__main__":
    docx_file = "tstdoc2.docx"
    
    #  1. Get Data 
    print(f"Reading data from '{docx_file}'...")
    document_text = extract_text_from_docx(docx_file)
    
    if document_text:
        text_as_bytes = document_text.encode('utf-8')
        
        #  Loop through all supported compression methods 
        methods_to_test = ['zlib', 'lzma', 'bz2', 'zstd']
        
        for method in methods_to_test:
            print(f"\n{'='*60}")
            print(f"PERFORMING TEST FOR: {method.upper()}")
            print(f"{'='*60}")

            # Define unique filenames for this method
            image_file = f"encoded_text_{method}.png"
            output_text_file = f"recovered_text_{method}.txt"
            
            #  2. Encode 
            encode_data(text_as_bytes, image_file, method=method)
            
            #  3. Decode 
            recovered_bytes = decode_data(image_file)
            
            #  4. Verify 
            if recovered_bytes:
                recovered_text = recovered_bytes.decode('utf-8')
                with open(output_text_file, "w", encoding="utf-8") as f:
                    f.write(recovered_text)
                
                print(f"\n Process for '{method.upper()}' Complete & Verified ")
                print(f"Decoded text saved as '{output_text_file}'")
                
                # Final check to see if the recovered text matches the original
                if document_text == recovered_text:
                    print("✅ Success: The recovered text perfectly matches the original.")
                else:
                    print("❌ Failure: The recovered text does not match the original.")
            else:
                print(f" Process for '{method.upper()}' FAILED during decoding. ")

