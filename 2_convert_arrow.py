import os
import argparse
from datasets import Dataset

def convert_arrow_to_txt(input_folder):
    """Recursively finds all .arrow files in subdirectories and converts them to .txt files."""
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".arrow"):
                arrow_path = os.path.join(root, file_name)
                txt_path = os.path.splitext(arrow_path)[0] + ".txt"
                
                print(f"Processing: {arrow_path} -> {txt_path}")
                
                try:
                    table = Dataset.from_file(arrow_path).to_pandas()
                    
                    if "text" in table.columns:
                        with open(txt_path, "w", encoding="utf-8") as outfile:
                            for text in table["text"]:
                                outfile.write(text + "\n\n\n")
                        print(f"Saved: {txt_path}")
                    else:
                        print(f"Warning: 'text' column not found in {arrow_path}")
                except Exception as e:
                    print(f"Error processing {arrow_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .arrow datasets to .txt files.")
    parser.add_argument("input_folder", help="Path to the folder containing .arrow files")
    args = parser.parse_args()
    
    convert_arrow_to_txt(args.input_folder)
