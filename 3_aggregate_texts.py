import os
import argparse

def merge_text_files(input_folder):
    """Merges all text files from second-level subdirectories into a single file."""
    for language_folder in os.listdir(input_folder):
        lang_path = os.path.join(input_folder, language_folder)
        
        if os.path.isdir(lang_path):
            output_file = os.path.join(lang_path, "all_texts.txt")
            
            with open(output_file, "w", encoding="utf-8") as outfile:
                for root, _, files in os.walk(lang_path):
                    for file in files:
                        if file.endswith(".txt") and not file.startswith("all_texts"):
                            file_path = os.path.join(root, file)
                            
                            with open(file_path, "r", encoding="utf-8") as infile:
                                outfile.write(infile.read() + "\n\n")
            
            print(f"Merged texts into: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge text files from second-level subdirectories.")
    parser.add_argument("input_folder", help="Path to the uralic_datasets folder")
    args = parser.parse_args()
    
    merge_text_files(args.input_folder)
