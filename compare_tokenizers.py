from tokenizer import Tokenizer
from typing import Dict, List, Tuple, Set
import argparse
from collections import defaultdict

class TokenizerComparator:
    def __init__(self, tokenizer1: Tokenizer, tokenizer2: Tokenizer):
        self.t1 = tokenizer1
        self.t2 = tokenizer2
        
    def get_merge_chain(self, token: str, tokenizer: Tokenizer) -> List[Tuple[str, str]]:
        """Get the complete chain of merges that led to this token."""
        chain = []
        current = token
        while len(current) > 1:
            for merge in tokenizer.merges:
                first, second = merge.split()
                if first + second == current:
                    chain.append((first, second))
                    current = first
                    break
            else:
                break
        return chain
    
    def compare_word(self, word: str) -> dict:
        """Compare tokenization of a single word between two tokenizers."""
        # Get trees from both tokenizers
        trees1 = self.t1.get_tokenization_tree(word)
        trees2 = self.t2.get_tokenization_tree(word)
        
        result = {
            'word': word,
            'tokens1': [tree['token'] for tree in trees1],
            'tokens2': [tree['token'] for tree in trees2],
            'common_merges': [],
            'different_merges': {
                'tokenizer1': [],
                'tokenizer2': []
            }
        }
        
        # Analyze each token
        for token in set(result['tokens1'] + result['tokens2']):
            chain1 = self.get_merge_chain(token, self.t1)
            chain2 = self.get_merge_chain(token, self.t2)
            
            # Find common merges
            common = set(chain1) & set(chain2)
            result['common_merges'].extend(common)
            
            # Find different merges
            diff1 = set(chain1) - common
            diff2 = set(chain2) - common
            
            if diff1:
                result['different_merges']['tokenizer1'].extend(diff1)
            if diff2:
                result['different_merges']['tokenizer2'].extend(diff2)
                
        return result
    
    def print_comparison(self, result: dict):
        """Pretty print the comparison results."""
        print(f"\nAnalyzing word: {result['word']}")
        print("\nTokenization:")
        print(f"Tokenizer 1: {' '.join(result['tokens1'])}")
        print(f"Tokenizer 2: {' '.join(result['tokens2'])}")
        
        print("\nCommon merge operations:")
        for merge in result['common_merges']:
            print(f"  {merge[0]} + {merge[1]}")
            
        print("\nDifferent merge operations:")
        print("Tokenizer 1 unique merges:")
        for merge in result['different_merges']['tokenizer1']:
            print(f"  {merge[0]} + {merge[1]}")
            
        print("Tokenizer 2 unique merges:")
        for merge in result['different_merges']['tokenizer2']:
            print(f"  {merge[0]} + {merge[1]}")
            
    def visualize_trees(self, word: str):
        """Visualize tokenization trees side by side."""
        trees1 = self.t1.get_tokenization_tree(word)
        trees2 = self.t2.get_tokenization_tree(word)
        
        print(f"\nTokenization trees for '{word}':")
        print("\nTokenizer 1:")
        for tree in trees1:
            self.t1.print_token_tree(tree)
            
        print("\nTokenizer 2:")
        for tree in trees2:
            self.t2.print_token_tree(tree)

def main():
    parser = argparse.ArgumentParser(description='Compare two BPE tokenizers')
    parser.add_argument('vocab1', help='Path to first vocabulary file')
    parser.add_argument('vocab2', help='Path to second vocabulary file')
    parser.add_argument('words', nargs='+', help='Words to analyze')
    
    args = parser.parse_args()
    
    # Initialize tokenizers
    tokenizer1 = Tokenizer(args.vocab1)
    tokenizer2 = Tokenizer(args.vocab2)
    
    comparator = TokenizerComparator(tokenizer1, tokenizer2)
    
    # Analyze each word
    for word in args.words:
        result = comparator.compare_word(word.lower())
        comparator.print_comparison(result)
        comparator.visualize_trees(word.lower())

if __name__ == "__main__":
    main()
