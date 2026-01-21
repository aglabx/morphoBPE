import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def train_bpe(input_file, words_output=None, save_path=None):
    with open(input_file) as fh:
        data = fh.readlines()
    data = [x.split()[0].strip().lower() for x in data]
    print(f"Total entries: {len(data)}")
    data = list(set(data))
    print(f"Unique entries: {len(data)}")

    if words_output is None:
        words_output = input_file + ".words"

    with open(words_output, "w") as fw:
        fw.write(" ".join(data))

    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(special_tokens=[], show_progress=True, vocab_size=32000)
    tokenizer.train(files=[words_output], trainer=trainer)
    
    if save_path:
        print(f"Saving tokenizer to {save_path}")
        tokenizer.save(save_path)
    return tokenizer

def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer on input text file')
    parser.add_argument('input_file', help='Input TSV file path')
    parser.add_argument('--words-output', help='Output file for preprocessed words (optional)')
    parser.add_argument('--save-path', help='Path to save the trained tokenizer (optional)', default='tokenizer.json')
    
    args = parser.parse_args()
    train_bpe(args.input_file, args.words_output, args.save_path)

if __name__ == "__main__":
    main()
