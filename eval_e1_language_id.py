#!/usr/bin/env python3
"""
E1 Evaluation: Simple Language Identification via Token Count

Hypothesis: Native words produce fewer, longer tokens.
            Foreign words produce more, shorter tokens.

Method: For each word, tokenize with all language tokenizers.
        Predict language = argmin(token_count)

Metrics: Accuracy, Macro-F1, Confusion matrix
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm

# Import tokenizer from local module
from tokenizer import Tokenizer


def load_tokenizers(tokenizer_dir: str, script: str = "cyrillic",
                    lang_filter: List[str] = None) -> Dict[str, Tokenizer]:
    """Load all tokenizers for a given script."""
    tokenizers = {}
    pattern = f"*.{script}.step1.json"

    tokenizer_files = list(Path(tokenizer_dir).glob(pattern))
    print(f"Found {len(tokenizer_files)} {script} tokenizer files")

    for tfile in tqdm(tokenizer_files, desc=f"Loading {script} tokenizers"):
        # Extract language code from filename
        # wikipedia_uk_all_nopic_2024-05.cyrillic.step1.json -> uk
        fname = tfile.name
        parts = fname.split("_")
        if len(parts) >= 2:
            lang_code = parts[1]

            # Filter languages if specified
            if lang_filter and lang_code not in lang_filter:
                continue

            try:
                tokenizers[lang_code] = Tokenizer(str(tfile))
            except Exception as e:
                print(f"Warning: Could not load {tfile}: {e}")

    print(f"Loaded {len(tokenizers)} tokenizers")
    return tokenizers


def load_test_words(tfdf_dir: str, script: str, lang_code: str,
                    n_words: int = 1000, min_length: int = 5) -> List[str]:
    """Load test words from tfdf file for a language."""
    pattern = f"wikipedia_{lang_code}_*.{script}.step1.tfdf.tsv"
    tfdf_files = list(Path(tfdf_dir).glob(pattern))

    if not tfdf_files:
        return []

    words = []
    tfdf_file = tfdf_files[0]

    try:
        with open(tfdf_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    word = parts[0].lower()
                    # Filter by length and alphabetic
                    if len(word) >= min_length and word.isalpha():
                        words.append(word)
    except Exception as e:
        print(f"Warning: Could not read {tfdf_file}: {e}")
        return []

    # Sample random words
    if len(words) > n_words:
        words = random.sample(words, n_words)

    return words


def tokenize_word(word: str, tokenizer: Tokenizer) -> int:
    """Get number of tokens for a word."""
    try:
        tokens = tokenizer.encode_word(word)
        return len(tokens)
    except:
        return float('inf')


def predict_language(word: str, tokenizers: Dict[str, Tokenizer]) -> Tuple[str, Dict[str, int]]:
    """Predict language for a word based on minimum token count."""
    token_counts = {}

    for lang, tok in tokenizers.items():
        token_counts[lang] = tokenize_word(word, tok)

    # Predict = language with minimum token count
    predicted = min(token_counts, key=token_counts.get)

    return predicted, token_counts


def evaluate(tokenizers: Dict[str, Tokenizer],
             test_data: Dict[str, List[str]],
             output_file: str = None) -> Dict:
    """Run E1 evaluation."""

    results = {
        'correct': 0,
        'total': 0,
        'per_language': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'confusion': defaultdict(lambda: defaultdict(int)),
        'examples': []
    }

    for true_lang, words in tqdm(test_data.items(), desc="Evaluating languages"):
        if true_lang not in tokenizers:
            print(f"Skipping {true_lang}: no tokenizer")
            continue

        for word in tqdm(words, desc=f"  {true_lang}", leave=False):
            predicted, token_counts = predict_language(word, tokenizers)

            results['total'] += 1
            results['per_language'][true_lang]['total'] += 1
            results['confusion'][true_lang][predicted] += 1

            if predicted == true_lang:
                results['correct'] += 1
                results['per_language'][true_lang]['correct'] += 1
            else:
                # Store some error examples
                if len(results['examples']) < 100:
                    results['examples'].append({
                        'word': word,
                        'true': true_lang,
                        'predicted': predicted,
                        'true_tokens': token_counts.get(true_lang, -1),
                        'pred_tokens': token_counts.get(predicted, -1)
                    })

    # Calculate metrics
    if results['total'] > 0:
        results['accuracy'] = results['correct'] / results['total']
    else:
        results['accuracy'] = 0

    # Per-language accuracy
    for lang in results['per_language']:
        stats = results['per_language'][lang]
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total']
        else:
            stats['accuracy'] = 0

    # Macro F1
    precisions = []
    recalls = []
    for lang in test_data.keys():
        if lang not in tokenizers:
            continue
        tp = results['confusion'][lang][lang]
        fp = sum(results['confusion'][other][lang] for other in test_data.keys() if other != lang and other in tokenizers)
        fn = sum(results['confusion'][lang][other] for other in tokenizers.keys() if other != lang)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    results['macro_f1'] = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    # Save results
    if output_file:
        # Convert defaultdicts to regular dicts for JSON
        output = {
            'accuracy': results['accuracy'],
            'macro_f1': results['macro_f1'],
            'correct': results['correct'],
            'total': results['total'],
            'per_language': {k: dict(v) for k, v in results['per_language'].items()},
            'confusion': {k: dict(v) for k, v in results['confusion'].items()},
            'examples': results['examples'][:20]  # Save first 20 examples
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")

    return results


def print_results(results: Dict, top_n: int = 10):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("E1 EVALUATION RESULTS: Language Identification")
    print("="*60)

    print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Macro F1 Score:   {results['macro_f1']:.4f}")

    # Per-language results (sorted by accuracy)
    lang_results = [(lang, stats['accuracy'], stats['total'])
                    for lang, stats in results['per_language'].items()]
    lang_results.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} languages by accuracy:")
    print("-"*40)
    for lang, acc, total in lang_results[:top_n]:
        print(f"  {lang:10s}: {acc:.4f} ({total} words)")

    print(f"\nBottom {top_n} languages by accuracy:")
    print("-"*40)
    for lang, acc, total in lang_results[-top_n:]:
        print(f"  {lang:10s}: {acc:.4f} ({total} words)")

    # Error examples
    if results.get('examples'):
        print(f"\nError examples:")
        print("-"*60)
        for ex in results['examples'][:5]:
            print(f"  '{ex['word']}': {ex['true']} -> {ex['predicted']} "
                  f"(tokens: {ex['true_tokens']} vs {ex['pred_tokens']})")


def main():
    parser = argparse.ArgumentParser(description='E1 Evaluation: Language ID via token count')
    parser.add_argument('--tokenizer-dir', required=True,
                        help='Directory with tokenizer JSON files')
    parser.add_argument('--tfdf-dir', default=None,
                        help='Directory with tfdf files (default: same as tokenizer-dir)')
    parser.add_argument('--script', default='cyrillic', choices=['cyrillic', 'latin'],
                        help='Script type (default: cyrillic)')
    parser.add_argument('--languages', nargs='+', default=None,
                        help='Filter to specific language codes')
    parser.add_argument('--n-words', type=int, default=500,
                        help='Number of test words per language (default: 500)')
    parser.add_argument('--min-word-length', type=int, default=5,
                        help='Minimum word length (default: 5)')
    parser.add_argument('--output', default='e1_results.json',
                        help='Output JSON file (default: e1_results.json)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    random.seed(args.seed)

    tfdf_dir = args.tfdf_dir or args.tokenizer_dir

    # Define language subsets for evaluation
    CYRILLIC_SLAVIC = ['ru', 'uk', 'be', 'bg', 'sr', 'mk', 'cu']
    CYRILLIC_TURKIC = ['kk', 'ky', 'tt', 'ba', 'cv', 'sah']
    CYRILLIC_ALL = None  # Use all available

    # Use provided filter or default
    lang_filter = args.languages
    if lang_filter is None:
        print(f"No language filter specified, using ALL available {args.script} languages")
        lang_filter = None  # Will load all available

    # Load tokenizers
    print(f"\nLoading {args.script} tokenizers...")
    tokenizers = load_tokenizers(args.tokenizer_dir, args.script, lang_filter)

    if not tokenizers:
        print("No tokenizers loaded. Exiting.")
        return

    # Load test data
    print(f"\nLoading test words ({args.n_words} per language)...")
    test_data = {}
    for lang in tqdm(tokenizers.keys(), desc="Loading test data"):
        words = load_test_words(tfdf_dir, args.script, lang,
                               args.n_words, args.min_word_length)
        if words:
            test_data[lang] = words
            print(f"  {lang}: {len(words)} words")

    if not test_data:
        print("No test data loaded. Exiting.")
        return

    # Run evaluation
    print(f"\nRunning E1 evaluation...")
    results = evaluate(tokenizers, test_data, args.output)

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
