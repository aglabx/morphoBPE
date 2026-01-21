import argparse
import glob
import os
import concurrent.futures
import subprocess
from pathlib import Path
from tqdm import tqdm
import logging

def process_file(input_file, vocab_size, base_output_dir):
    input_path = Path(input_file)
    
    # Check if file is empty
    if input_path.stat().st_size == 0:
        logging.warning(f"Skipping empty file: {input_file}")
        return input_file, False, "File is empty"
        
    # Create output filename by replacing .tfdf.tsv with .json
    output_name = input_path.name.replace('.tfdf.tsv', '.json')
    output_path = Path(base_output_dir) / output_name
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Construct command
    cmd = [
        'python3', 'bpe.py',
        '--output-file', str(output_path),
        '--vocab-size', str(vocab_size),
        str(input_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return input_file, True, "Success"
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}\nStderr: {e.stderr}\nStdout: {e.stdout}"
        logging.error(f"Error processing {input_file}:\n{error_msg}")
        return input_file, False, error_msg

def main():
    parser = argparse.ArgumentParser(description='Run BPE training in parallel on multiple files')
    parser.add_argument('input_dir', help='Directory containing .step1.tfdf.tsv files')
    parser.add_argument('output_dir', help='Directory for output JSON files')
    parser.add_argument('--vocab-size', type=int, default=16384, help='Vocabulary size')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--log-file', default='bpe_parallel.log', help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    
    # Find all matching files
    pattern = os.path.join(args.input_dir, "**/*.step1.tfdf.tsv")
    input_files = glob.glob(pattern, recursive=True)
    
    # Filter out empty files before processing
    input_files = [f for f in input_files if Path(f).stat().st_size > 0]
    
    if not input_files:
        print(f"No non-empty .step1.tfdf.tsv files found in {args.input_dir}")
        return
    
    print(f"Found {len(input_files)} non-empty files to process")
    
    # Process files in parallel
    failed_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_file, f, args.vocab_size, args.output_dir)
            for f in input_files
        ]
        
        # Show progress
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in concurrent.futures.as_completed(futures):
                input_file, success, message = future.result()
                status = "✓" if success else "✗"
                if not success:
                    failed_files.append((input_file, message))
                pbar.set_postfix({'current': f"{status} {Path(input_file).name}"})
                pbar.update(1)
    
    # Report failed files
    if failed_files:
        logging.error("\nFailed files:")
        for file, message in failed_files:
            logging.error(f"\n{file}:\n{message}")
        logging.error(f"\nTotal failed files: {len(failed_files)}")

if __name__ == "__main__":
    main()
