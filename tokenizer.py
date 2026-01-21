import json
from typing import List, Dict, Tuple
import re

class Tokenizer:
    def __init__(self, vocab_file: str):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            
        self.vocab = vocab_data['vocab']
        self.merges = vocab_data['merges']
        self.freq = vocab_data['freq']
        
        # Create reverse vocab for decoding
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Convert merges to tuples of strings
        self.bpe_ranks = {
            tuple(merge.split()): i for i, merge in enumerate(self.merges)
        }
    
    def get_pairs(self, word: List[str]) -> set:
        """Get all adjacent pairs in a word."""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs
    
    def encode_word(self, word: str) -> List[str]:
        """Encode a single word using BPE."""
        word = list(word)
        pairs = self.get_pairs(word)

        while pairs:
            pair = min(pairs, key=lambda x: self.bpe_ranks.get(x, float('inf')))
            if pair not in self.bpe_ranks:
                break

            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            pairs = self.get_pairs(word)

        return word
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        tokens = []
        # Normalize text and split into words
        words = text.lower().strip().split()
        
        for word in words:
            # Add space before each word except the first one
            if tokens:
                tokens.append(self.vocab[' '])
            
            subwords = self.encode_word(word)
            for subword in subwords:
                if subword in self.vocab:
                    tokens.append(self.vocab[subword])
                else:
                    # Handle unknown tokens character by character
                    for char in subword:
                        tokens.append(self.vocab.get(char, self.vocab['�']))
                        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token ids back to text."""
        return ''.join(self.reverse_vocab.get(token, '�') for token in tokens)
    
    def get_token_texts(self, tokens: List[int]) -> str:
        """Get space-separated text representation of tokens."""
        return ' '.join(self.reverse_vocab.get(token, '�') for token in tokens)
    
    def get_merge_history(self, token: str) -> List[tuple]:
        """Get the merge history for a token showing how it was formed."""
        if len(token) <= 1:
            return []
        
        history = []
        for merge_pair in self.merges:
            first, second = merge_pair.split()
            if first + second == token:
                history.append((first, second))
                return history + self.get_merge_history(first) + self.get_merge_history(second)
        return []
    
    def build_token_tree(self, token: str) -> dict:
        """Build a recursive tree structure for a token."""
        if len(token) <= 1:
            return {"token": token, "children": []}
            
        for merge_pair in self.merges:
            first, second = merge_pair.split()
            if first + second == token:
                return {
                    "token": token,
                    "children": [
                        self.build_token_tree(first),
                        self.build_token_tree(second)
                    ]
                }
        
        # If no merge rule found, treat as individual characters
        return {"token": token, "children": []}
    
    def get_tokenization_tree(self, word: str) -> List[dict]:
        """Get the complete tokenization tree for a word."""
        subwords = self.encode_word(word)
        return [self.build_token_tree(subword) for subword in subwords]
    
    def print_token_tree(self, tree: dict, level: int = 0):
        """Pretty print the token tree."""
        indent = "  " * level
        print(f"{indent}{tree['token']}")
        for child in tree['children']:
            self.print_token_tree(child, level + 1)

if __name__ == "__main__":
    # Example usage
    tokenizer = Tokenizer('vocab.json')
    
    # Test text
    text = "Hello"
    print(f"\nAnalyzing word: {text}")
    
    # Get and print token tree
    print("\nTokenization tree:")
    trees = tokenizer.get_tokenization_tree(text.lower())
    for tree in trees:
        tokenizer.print_token_tree(tree)
    
    # Regular tokenization
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    token_texts = tokenizer.get_token_texts(tokens)
    
    print(f"\nTokens: {tokens}")
    print(f"Token texts: {token_texts}")
    print(f"Decoded: {decoded}")
