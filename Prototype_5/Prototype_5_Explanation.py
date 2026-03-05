import numpy as np
from PIL import Image
import zlib
import docx
import textwrap

# HELPER FUNCTION TO EXTRACT TEXT FROM A .DOCX FILE

def extract_text_from_docx(docx_path):
    """
    Reads a .docx file and returns all of its text content as a single string.
    This function isolates the text, ignoring images, formatting, and other file data.
    """
    try:
        # Open the Word document using the python-docx library.
        doc = docx.Document(docx_path)
        full_text = []
        # A .docx file is made of paragraphs. We loop through each paragraph object.
        for para in doc.paragraphs:
            # We extract the plain text from the paragraph and add it to our list.
            full_text.append(para.text)
        # Join all the collected paragraph texts into a single string,
        # with a newline character between each original paragraph for readability.
        return '\n'.join(full_text)
    except Exception as e:
        # If the file can't be opened or read, print an error and return nothing.
        print(f"Error reading docx file: {e}")
        return None

#  CORE ENCODING LOGIC

def encode_data(data_bytes, output_img, size=160, cell_size=5):
    """
    Encodes a given set of bytes into a black and white image.
    This process involves compression, conversion to bits, and then mapping
    the bit information to pixel locations on a grid.
    """
    #  Part A: Data Preparation
    print("\n[ENCODE STEP 1]: Compressing data with zlib...")
    print(f"  > Original data size: {len(data_bytes)} bytes")

    # 1. COMPRESS DATA: Use zlib compression to reduce the size of the input bytes.
    #    This is crucial for fitting more data into the limited space of the image.
    compressed = zlib.compress(data_bytes)
    print(f"  > Compressed data size: {len(compressed)} bytes")
    print(f"  > Sample of compressed data (first 32 bytes): {compressed[:32].hex(' ')}")


    # 2. CONVERT TO BITS: Transform the compressed bytes into a single, long string of '1's and '0's.
    #    Each byte becomes 8 characters (e.g., b'\x0f' becomes '00001111').
    print("\n[ENCODE STEP 2]: Converting compressed bytes to a bitstring...")
    bits = ''.join(f"{byte:08b}" for byte in compressed)
    n = len(bits) # 'n' is the total length of our bitstream.
    print(f"  > Total bits to encode: {n}")
    print(f"  > Bitstring preview (first 64 bits): {bits[:64]}")

    # 3. CHOOSE STORAGE MODE (SPARSE ENCODING): This is a space-saving trick.
    #    Instead of storing every bit, we only store the positions of the least common bit.
    #    If there are fewer '1's than '0's, we store the locations of '1's. Otherwise, we store '0's.
    print("\n[ENCODE STEP 3]: Choosing sparse storage mode...")
    ones = [i for i, b in enumerate(bits) if b == '1']
    zeros = [i for i, b in enumerate(bits) if b == '0']
    
    print(f"  > Count of '1's: {len(ones)}")
    print(f"  > Count of '0's: {len(zeros)}")

    if len(ones) <= len(zeros):
        mode = 0  # Mode 0 means we are storing the positions of '1's.
        positions = ones
        print("  > Mode 0 selected: Storing positions of '1's.")
    else:
        mode = 1  # Mode 1 means we are storing the positions of '0's.
        positions = zeros
        print("  > Mode 1 selected: Storing positions of '0's.")
    print(f"  > Number of positions to store: {len(positions)}")

    # 4. CREATE METADATA: The decoder needs to know two things: the storage mode (0 or 1)
    #    and the original total length of the bitstream ('n').
    #    We create a 32-bit header: [1 bit for mode] + [31 bits for length n].
    print("\n[ENCODE STEP 4]: Creating 32-bit metadata header...")
    meta_bits = f"{mode:01b}" + f"{n:031b}"
    print(f"  > Mode bit: {meta_bits[0]}")
    print(f"  > Length bits: {meta_bits[1:]} (Value: {n})")
    print(f"  > Full metadata header: {meta_bits}")


    # Image Creation

    # 5. CREATE THE IMAGE GRID: We create a 2D array (a grid) of the specified size using NumPy.
    # It's initialized with all zeros, which corresponds to black pixels.
    print("\n[ENCODE STEP 5]: Generating image grid and mapping data...")
    grid = np.zeros((size, size), dtype=np.uint8)

    # 6. WRITE METADATA TO IMAGE: We place the 32-bit metadata into the first 32 pixels of the first row. A '1' becomes a white pixel (value 255) and a '0' remains black (value 0).
    for i, bit in enumerate(meta_bits):
        grid[0, i] = 255 if bit == '1' else 0
    print("  > Metadata written to pixels [0, 0] through [0, 31].")

    # 7. WRITE DATA POSITIONS TO IMAGE: Now we mark the positions of our data bits.
    #    A 'position' is just a number (e.g., the 500th bit). We need to map this number to a 2D coordinate (row, col) on our grid.
    #    We add 32 to the position to ensure we start writing *after* the metadata block.
    for pos in positions:
        effective_pos = pos + 32  # Offset to skip the 32 metadata pixels.
        row = effective_pos // size # Integer division gives the row number.
        col = effective_pos % size  # The remainder gives the column number.
        
        # As long as we are within the image boundaries, mark the pixel as white.
        if row < size:
            grid[row, col] = 255
    print(f"  > {len(positions)} data positions marked as white pixels on the grid.")


    # 8. SAVE THE FINAL IMAGE: Convert the NumPy array into an actual image file using Pillow (PIL).
    #    We resize it by 'cell_size' to make the individual pixels larger and easier to see/scan.
    #    'Image.NEAREST' ensures the pixels stay sharp and square, not blurry.
    print("\n[ENCODE STEP 6]: Saving the final image...")
    img = Image.fromarray(grid).resize((size*cell_size, size*cell_size), Image.NEAREST)
    img.save(output_img)
    print(f"  > Encoded image saved as {output_img}")


# STEP 3: CORE DECODING LOGIC

def decode_data(input_img, size=160):
    """
    Decodes a sparse image and returns the original data as bytes.
    This process reverses the encoding steps: reading pixels, reconstructing
    the bitstream, decompressing, and returning the original bytes.
    """
    # Image Processing 
    # 1. LOAD AND PREPARE IMAGE: Open the image file and resize it back down to its
    #    original grid dimensions (e.g., 160x160).
    #    '.convert('L')' converts it to grayscale, which simplifies pixel reading.
    print("\n[DECODE STEP 1]: Loading and reading pixel data from image...")
    img = Image.open(input_img).resize((size, size), Image.NEAREST)
    grid = np.array(img.convert('L'))
    print(f"  > Image loaded and resized to {size}x{size} grid.")

    # 2. READ METADATA: Read the first 32 pixels of the first row to get the metadata.
    #    If a pixel's value is > 128 (more white than black), we read it as a '1'.
    print("\n[DECODE STEP 2]: Reading metadata from first 32 pixels...")
    meta_bits = ''.join('1' if grid[0, i] > 128 else '0' for i in range(32))
    mode = int(meta_bits[0], 2)      # The first bit is the mode.
    n = int(meta_bits[1:], 2)        # The next 31 bits are the original bitstream length.
    print(f"  > Read metadata bits: {meta_bits}")
    print(f"  > Decoded Mode: {mode}")
    print(f"  > Decoded Bitstream Length (n): {n}")


    # 3. EXTRACT DATA POSITIONS: Loop through the rest of the image grid (starting from pixel 32)
    #    to find all the white pixels. For each white pixel, we calculate its original
    #    linear position in the bitstream.
    print("\n[DECODE STEP 3]: Scanning grid for white pixels to extract data positions...")
    positions = []
    for i in range(32, size * size): # Start scanning after the metadata.
        row = i // size
        col = i % size
        if grid[row, col] > 128:
            # The position in the bitstream is the pixel index minus the metadata length.
            pos = i - 32
            positions.append(pos)
    print(f"  > Found {len(positions)} data positions.")

    # Data Reconstruction

    # 4. RECONSTRUCT THE BITSTRING: Now we rebuild the original string of '1's and '0's.
    #    We know the total length 'n' and the storage 'mode' from the metadata.
    print("\n[DECODE STEP 4]: Reconstructing original bitstring based on mode and positions...")
    if mode == 0:
        # Mode 0 means we stored the positions of '1's.
        # So, we create a list of all '0's...
        bits = ['0'] * n
        # ...and then place '1's at the positions we found.
        for p in positions:
            if p < n: bits[p] = '1'
        print("  > Mode 0 detected. Rebuilt bitstring by placing '1's into a field of '0's.")
    else: # mode == 1
        # Mode 1 means we stored the positions of '0's.
        # So, we create a list of all '1's...
        bits = ['1'] * n
        # ...and then place '0's at the positions we found.
        for p in positions:
            if p < n: bits[p] = '0'
        print("  > Mode 1 detected. Rebuilt bitstring by placing '0's into a field of '1's.")

    bitstring = ''.join(bits)
    print(f"  > Reconstructed bitstring preview (first 64 bits): {bitstring[:64]}")

    # 5. CONVERT BITS TO BYTES: Join the list of bits into a single string.
    #    Then, process this string in 8-character chunks, converting each chunk
    #    back into an integer, and then into a byte.
    print("\n[DECODE STEP 5]: Converting bitstring back into bytes...")
    byte_list = []
    for i in range(0, len(bitstring), 8):
        byte_chunk = bitstring[i:i+8]
        if len(byte_chunk) == 8: # Ensure we don't process a partial byte at the end.
            byte_list.append(int(byte_chunk, 2))
    data = bytes(byte_list)
    print(f"  > Reconstructed {len(data)} bytes of compressed data.")
    print(f"  > Sample of reconstructed data (first 32 bytes): {data[:32].hex(' ')}")


    # 6. DECOMPRESS AND RETURN: Use zlib to decompress the byte data, which
    #    reverses the compression from the encoding step.
    print("\n[DECODE STEP 6]: Decompressing data with zlib to get original bytes...")
    try:
        decompressed = zlib.decompress(data)
        # Return the final, original bytes.
        print(f"  > Decompression successful. Final data size: {len(decompressed)} bytes.")
        return decompressed
    except zlib.error as e:
        # If decompression fails, the data was likely corrupted.
        print(f"  > Error decompressing data: {e}")
        return None


# STEP 4: EXECUTION BLOCK

# Define file paths for input and output.
docx_file = "tstdoc1.docx"
image_file = "encoded_text.png"
output_text_file = "recovered_text_proto_5.txt"

#  ENCODING PROCESS 
print("--- Starting Encoding Process ---")
# 1. Use the helper function to get plain text from the Word document.
document_text = extract_text_from_docx(docx_file)

if document_text:
    print(f"Successfully extracted {len(document_text)} characters from '{docx_file}'.")
    # 2. Convert the text string to bytes. Computers work with bytes, not characters.
    #    UTF-8 is a standard encoding that handles almost all characters and symbols.
    text_as_bytes = document_text.encode('utf-8')
    
    # 3. Call the main encoding function to perform the conversion.
    encode_data(text_as_bytes, image_file, size=160, cell_size=5)

    # DECODING PROCESS 
    print("\n\n--- Starting Decoding Process ---")
    # 4. Call the main decoding function to read the image and extract the bytes.
    recovered_bytes = decode_data(image_file, size=160)

    if recovered_bytes:
        # 5. Convert the recovered bytes back into a human-readable string using UTF-8.
        recovered_text = recovered_bytes.decode('utf-8')
        
        # 6. Save the recovered text to a new file to verify the process worked.
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(recovered_text)
            
        print(f"\n--- Process Complete ---")
        print(f"Decoded text saved successfully as {output_text_file}")
        # Display a preview of the recovered text
        print("\nRecovered Text Preview (first 200 characters):")
        print("-------------------------------------------------")
        print(textwrap.shorten(recovered_text, width=200, placeholder="..."))
        print("-------------------------------------------------")

