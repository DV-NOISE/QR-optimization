import os
import glob
import zlib
import lzma
import bz2
import zstandard as zstd
import docx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# HELPER: EXTRACT TEXT FROM A .DOCX FILE

def extract_text_from_docx(docx_path):
    """Reads a .docx file and returns its text content as a string."""
    try:
        doc = docx.Document(docx_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"  - Could not read {os.path.basename(docx_path)}: {e}")
        return None

# ANALYSIS FUNCTION

def analyze_file_compression(file_path):
    """
    Analyzes a single docx file by compressing its content with multiple algorithms.
    Returns a dictionary with original size and detailed metrics for each algorithm.
    """
    print(f"-> Analyzing '{os.path.basename(file_path)}'...")
    document_text = extract_text_from_docx(file_path)
    
    if not document_text:
        return None
        
    text_as_bytes = document_text.encode('utf-8')
    original_size = len(text_as_bytes)
    
    if original_size == 0:
        print("  - File is empty. Skipping.")
        return None

    # Define compressors
    cctx = zstd.ZstdCompressor()
    
    # Prepare results dictionary and define algorithms to run
    file_metrics = {}
    algorithms_to_run = {
        'zlib': zlib.compress,
        'lzma': lzma.compress,
        'bz2': bz2.compress,
        'zstd': cctx.compress
    }
    
    print(f"  > Original size: {original_size} bytes")

    # Run all compression algorithms and analyze bit counts
    for name, compressor in algorithms_to_run.items():
        compressed_bytes = compressor(text_as_bytes)
        compressed_size = len(compressed_bytes)

        bits = ''.join(f"{byte:08b}" for byte in compressed_bytes)
        total_bits = len(bits)
        
        print(f"    - {name.upper()}:")
        print(f"      - Compressed Size: {compressed_size} bytes")

        diff_percent = 0
        if total_bits > 0:
            ones_count = bits.count('1')
            zeros_count = total_bits - ones_count
            difference = abs(ones_count - zeros_count)
            diff_percent = difference / total_bits
            
            print(f"      - Bit Counts: Ones={ones_count}, Zeros={zeros_count}")
            print(f"      - Difference: {difference} ({diff_percent * 100:.2f}%)")

        file_metrics[name] = {
            'size': compressed_size,
            'diff_percent': diff_percent
        }
    
    return {'original_size': original_size, 'metrics': file_metrics}

# PLOTTING FUNCTION

def plot_compression_results(results):
    """
    Generates and saves a bar chart comparing the compression results.
    """
    print("\n[STEP 3]: Generating comparison plot...")
    
    # Prepare data for plotting
    filenames = [os.path.basename(f) for f in results.keys()]
    algorithms = ['zstd', 'lzma', 'bz2', 'zlib'] # Order for plotting
    
    # Group compressed sizes by algorithm
    data = {alg: [] for alg in algorithms}
    for file in results.keys():
        for alg in algorithms:
            data[alg].append(results[file]['metrics'][alg]['size'])
            
    # Set up the plot
    x = np.arange(len(filenames))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(15, 8))

    # Create bars for each algorithm
    for algorithm, sizes in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, sizes, width, label=algorithm)
        ax.bar_label(rects, padding=3, rotation=90, fmt='d') # 'd' for integer format
        multiplier += 1

    # Add text, title, and labels
    ax.set_ylabel('Compressed Size (Bytes)')
    ax.set_title('Compression Algorithm Performance Across DOCX Files', fontsize=16)
    ax.set_xticks(x + width * 1.5, filenames, rotation=45, ha='right')
    ax.legend(loc='upper right', ncols=len(algorithms))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis limit to give space for labels
    all_sizes = [size for sizes in data.values() for size in sizes]
    ax.set_ylim(0, max(all_sizes) * 1.15) 

    plt.tight_layout()
    
    # Save the plot
    output_filename = "compression_comparison.png"
    plt.savefig(output_filename)
    print(f"  > Plot saved as '{output_filename}'")
    plt.show()


# MAIN EXECUTION BLOCK

if __name__ == "__main__":
    # 1. Find Files 
    docs_folder = "test_docs"
    print(f"[STEP 1]: Searching for .docx files in the '{docs_folder}' folder...")
    
    if not os.path.exists(docs_folder):
        print(f"  > ERROR: The folder '{docs_folder}' was not found.")
        print("  > Please create it and place your .docx files inside.")
    else:
        docx_files = glob.glob(os.path.join(docs_folder, "*.docx"))
        
        if not docx_files:
            print(f"  > No .docx files were found in '{docs_folder}'.")
        else:
            # 2. Analyze Files 
            print(f"\n[STEP 2]: Found {len(docx_files)} files. Starting analysis...")
            all_results = {}
            best_performers = []

            for file_path in docx_files:
                result = analyze_file_compression(file_path)
                if result:
                    all_results[file_path] = result
                    
                    # NEW SCORING LOGIC
                    metrics = result['metrics']
                    
                    # Find max values for normalization
                    max_size = max(m['size'] for m in metrics.values())
                    max_diff = max(m['diff_percent'] for m in metrics.values())
                    
                    if max_size == 0 or max_diff == 0: continue

                    scores = {}
                    print("  > Scoring (lower is better):")
                    for name, m in metrics.items():
                        # Normalize both metrics to be between 0 and 1
                        norm_size = m['size'] / max_size
                        norm_diff = m['diff_percent'] / max_diff
                        
                        # Score: Lower size is good (low norm_size). Higher diff is good (low 1 - norm_diff).
                        # We give equal weight to both factors.
                        score = norm_size + (1 - norm_diff)
                        scores[name] = score
                        print(f"    - {name.upper()}: {score:.4f}")
                        
                    winner = min(scores, key=scores.get)
                    # END NEW SCORING LOGIC
                    
                    best_performers.append(winner)
                    print(f"  > Best for this file (balanced score): {winner.upper()}")

            #  3. Plot Results
            if all_results:
                plot_compression_results(all_results)
                
                # 4. Declare Overall Winner 
                print("\n[STEP 4]: Calculating overall winner...")
                if best_performers:
                    # Count the wins for each algorithm
                    win_counts = Counter(best_performers)
                    overall_winner = win_counts.most_common(1)[0][0]
                    
                    print("  > Win counts (based on balanced score):")
                    for alg, count in win_counts.items():
                        print(f"    - {alg.upper()}: {count} win(s)")
                        
                    print(f"\nOverall Best Performer: {overall_winner.upper()}")
                else:
                    print("\nNo data was successfully analyzed. Cannot generate plot.")

