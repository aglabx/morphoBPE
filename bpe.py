from collections import Counter, defaultdict
from tqdm import tqdm
import argparse
import json

def get_next(token2chars, pairs, tokens, merges):
    if not pairs:
        raise ValueError("No more valid pairs available for merging")
    new_token = max(token2chars.keys()) + 1
    pair, freq = pairs.most_common(1)[0]
    del pairs[pair]
    token2chars[new_token] = f"{token2chars[pair[0]]}{token2chars[pair[1]]}"
    tokens.append((token2chars[new_token], new_token))
    merges.append((pair[0], pair[1]))
    return pair, freq, new_token

def merge(input_data, pair, freq, new_token, pairs, pairs_positions, tfs):
    # Накапливаем изменения
    pairs_changes = Counter()  # Счетчик для изменений в парах
    new_positions = defaultdict(list)  # Новые позиции для пар
    
    for (i,j) in pairs_positions[pair]:
        left = input_data[i]
        right = input_data[j]
        if left is None or right is None:
            continue
        prev = None
        next = None
        k = 1
        while i - k >= 0:
            prev = input_data[i-k]
            if prev != None:
                break
            k += 1
        l = 1
        while j + l < len(input_data):
            next = input_data[j+l]
            if next != None:
                break
            l += 1
        
        # Накапливаем изменения вместо прямого изменения счетчиков
        if prev is not None and prev != 32:
            pairs_changes[(prev, left)] -= 1
            pairs_changes[(prev, new_token)] += 1
            new_positions[(prev, new_token)].append((i-k, i))
            
        if next is not None and next != 32:
            pairs_changes[(right, next)] -= 1
            pairs_changes[(new_token, next)] += 1
            new_positions[(new_token, next)].append((i, j+l))
            
        input_data[j] = None
        input_data[i] = new_token
        tfs[new_token] += 1

    # Применяем изменения разом
    for p, change in pairs_changes.items():
        pairs[p] += change
        if pairs[p] <= 0:
            del pairs[p]
            
    # Обновляем позиции
    del pairs_positions[pair]
    for p, positions in new_positions.items():
        pairs_positions[p].extend(positions)

def train_bpe(input_file, vocab_size=None, min_freq=None):
    with open(input_file) as fh:
        data = fh.readlines()
    data = [x.split()[0].strip().lower() for x in data]
    print(f"Total lines: {len(data)}")
    data = list(set(data))
    print(f"Unique words: {len(data)}")

    input_data = data[::]
    input_data = [ord(x) for x in " ".join(input_data)]

    new_words = []
    merges = []
    tfs = Counter()
    pairs = Counter()
    pairs_positions = defaultdict(list)
    
    for i in tqdm(range(len(input_data)-1)):
        tfs[input_data[i]] += 1
        if input_data[i] == 32 or input_data[i+1] == 32:
            continue
        pair = (input_data[i], input_data[i+1])
        pairs[pair] += 1
        pairs_positions[pair].append((i, i+1))
    tfs[input_data[-1]] += 1

    token2chars = {}
    tokens = []
    for token in tfs:
        tokens.append((chr(token), token))
        token2chars[token] = chr(token)

    # Calculate target size for progress bar
    target_size = vocab_size if vocab_size else float('inf')
    pbar = tqdm(total=target_size, desc="Building vocabulary")
    
    while pairs:  # Only continue if there are pairs to merge
        try:
            pair, freq, new_token = get_next(token2chars, pairs, tokens, merges)
            
            if vocab_size and len(merges) >= vocab_size:
                break
            if min_freq and freq < min_freq:
                break
                
            merge(input_data, pair, freq, new_token, pairs, pairs_positions, tfs)
            pbar.update(1)
            pbar.set_postfix({'freq': freq})
        except ValueError as e:
            print(f"Training stopped: {e}")
            break
    
    pbar.close()
    
    if len(merges) == 0:
        print("Warning: No merges were performed. Check if the input data contains valid pairs.")
        
    return merges, token2chars, tokens, tfs

def format_vocab_for_hf(token2chars, tokens, merges, tfs):
    output = {}
    
    # Create vocabulary section
    vocab = {}
    # Add special tokens first

    # Add regular tokens
    for token_str, token_id in tokens:
        vocab[token_str] = token_id  # Offset by 4 due to special tokens
    
    # Create merges section
    merges_list = []
    for left, right in merges:
        merges_list.append(f"{token2chars[left]} {token2chars[right]}")
    
    # Create frequency section
    freq = {token2chars[token_id]: count for token_id, count in tfs.items()}
    
    # Combine all sections
    output["vocab"] = vocab
    output["merges"] = merges_list
    output["freq"] = freq
    
    return output

def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('input_file', help='Path to input TSV file')
    parser.add_argument('--output-file', help='Path to output vocabulary JSON file', default='vocab.json')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--vocab-size', type=int, help='Maximum vocabulary size')
    group.add_argument('--min-freq', type=int, help='Minimum pair frequency')
    
    args = parser.parse_args()
    
    merges, token2chars, tokens, tfs = train_bpe(
        args.input_file,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq
    )
    
    # Format and save vocabulary with additional information
    output_data = format_vocab_for_hf(token2chars, tokens, merges, tfs)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nFinal vocabulary size: {len(merges)}")
    print(f"Vocabulary saved to: {args.output_file}")
    print("Done!")

if __name__ == "__main__":
    main()