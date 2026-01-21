#!/usr/bin/env python3
"""
E1 Baseline: fastText Language Identification

Compare our BPE-based language ID with fastText (SOTA baseline).
Uses the same test words as eval_e1_language_id.py for fair comparison.

Requirements:
    pip install fasttext

    # Download model (one-time):
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

FASTTEXT_AVAILABLE = False
fasttext = None

try:
    import fasttext as ft_module
    fasttext = ft_module
    FASTTEXT_AVAILABLE = True
except ImportError:
    print("Warning: fasttext not installed. Run: pip install fasttext")


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
                    if len(word) >= min_length and word.isalpha():
                        words.append(word)
    except Exception as e:
        print(f"Warning: Could not read {tfdf_file}: {e}")
        return []

    if len(words) > n_words:
        words = random.sample(words, n_words)

    return words


def evaluate_fasttext(model, test_data: Dict[str, List[str]],
                      lang_mapping: Optional[Dict[str, str]] = None) -> Dict:
    """
    Evaluate fastText on test data.

    Args:
        model: fastText model
        test_data: {lang_code: [words]}
        lang_mapping: map our lang codes to fastText codes if different
    """
    # Default mapping (fastText uses __label__xx format)
    if lang_mapping is None:
        lang_mapping = {
            'ru': 'ru', 'uk': 'uk', 'be': 'be',
            'bg': 'bg', 'sr': 'sr', 'mk': 'mk',
            'cu': 'cu',  # Old Church Slavonic - may not be in fastText
        }

    results = {
        'correct': 0,
        'total': 0,
        'per_language': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'confusion': defaultdict(lambda: defaultdict(int)),
        'examples': [],
        'not_in_fasttext': set()
    }

    for true_lang, words in tqdm(test_data.items(), desc="Evaluating fastText"):
        ft_true_lang = lang_mapping.get(true_lang, true_lang)

        for word in words:
            # fastText prediction
            predictions = model.predict(word, k=1)
            # Returns (('__label__xx',), (probability,))
            predicted_label = predictions[0][0].replace('__label__', '')
            confidence = predictions[1][0]

            # Map back to our lang codes
            predicted = predicted_label

            results['total'] += 1
            results['per_language'][true_lang]['total'] += 1
            results['confusion'][true_lang][predicted] += 1

            if predicted == ft_true_lang:
                results['correct'] += 1
                results['per_language'][true_lang]['correct'] += 1
            else:
                if len(results['examples']) < 100:
                    results['examples'].append({
                        'word': word,
                        'true': true_lang,
                        'predicted': predicted,
                        'confidence': float(confidence)
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
        ft_lang = lang_mapping.get(lang, lang)
        tp = results['confusion'][lang].get(ft_lang, 0)
        fp = sum(results['confusion'][other].get(ft_lang, 0)
                 for other in test_data.keys() if other != lang)
        fn = sum(count for pred, count in results['confusion'][lang].items()
                 if pred != ft_lang)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    results['macro_f1'] = (2 * avg_precision * avg_recall /
                          (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0)

    return results


def print_comparison(our_results: Dict, fasttext_results: Dict):
    """Print side-by-side comparison."""
    print("\n" + "="*70)
    print("COMPARISON: BPE Token Count vs fastText (SOTA)")
    print("="*70)

    print(f"\n{'Metric':<25} {'BPE (ours)':<20} {'fastText':<20}")
    print("-"*70)
    print(f"{'Overall Accuracy':<25} {our_results.get('accuracy', 0):.4f} ({our_results.get('correct', 0)}/{our_results.get('total', 0)})".ljust(45) +
          f"{fasttext_results['accuracy']:.4f} ({fasttext_results['correct']}/{fasttext_results['total']})")
    print(f"{'Macro F1':<25} {our_results.get('macro_f1', 0):.4f}".ljust(45) +
          f"{fasttext_results['macro_f1']:.4f}")

    print(f"\n{'Language':<15} {'BPE Acc':<15} {'fastText Acc':<15} {'Winner':<15}")
    print("-"*60)

    all_langs = set(our_results.get('per_language', {}).keys()) | set(fasttext_results['per_language'].keys())
    for lang in sorted(all_langs):
        our_acc = our_results.get('per_language', {}).get(lang, {}).get('accuracy', 0)
        ft_acc = fasttext_results['per_language'].get(lang, {}).get('accuracy', 0)

        if our_acc > ft_acc:
            winner = "BPE ✓"
        elif ft_acc > our_acc:
            winner = "fastText ✓"
        else:
            winner = "tie"

        print(f"{lang:<15} {our_acc:.4f}".ljust(30) + f"{ft_acc:.4f}".ljust(15) + f"{winner}")

    print("\n" + "="*70)
    if our_results.get('accuracy', 0) > fasttext_results['accuracy']:
        print("RESULT: BPE outperforms fastText on this task!")
    elif fasttext_results['accuracy'] > our_results.get('accuracy', 0):
        diff = fasttext_results['accuracy'] - our_results.get('accuracy', 0)
        print(f"RESULT: fastText wins by {diff:.1%}")
        print("NOTE: Our method uses ONLY tokenization, no training on labeled data!")
    else:
        print("RESULT: Methods are comparable")


def main():
    parser = argparse.ArgumentParser(description='E1 Baseline: fastText comparison')
    parser.add_argument('--model', required=True,
                        help='Path to fastText lid.176.bin model')
    parser.add_argument('--tfdf-dir', required=True,
                        help='Directory with tfdf files')
    parser.add_argument('--script', default='cyrillic', choices=['cyrillic', 'latin'],
                        help='Script type (default: cyrillic)')
    parser.add_argument('--languages', nargs='+', default=None,
                        help='Language codes to test')
    parser.add_argument('--n-words', type=int, default=500,
                        help='Number of test words per language')
    parser.add_argument('--min-word-length', type=int, default=5,
                        help='Minimum word length')
    parser.add_argument('--our-results', default=None,
                        help='Path to our E1 results JSON for comparison')
    parser.add_argument('--output', default='e1_fasttext_results.json',
                        help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    if not FASTTEXT_AVAILABLE:
        print("ERROR: fasttext not installed. Run: pip install fasttext")
        return

    random.seed(args.seed)

    # Default to Slavic languages
    if args.languages is None:
        args.languages = ['ru', 'uk', 'be', 'bg', 'sr', 'mk']

    # Load fastText model
    print(f"Loading fastText model from {args.model}...")
    if fasttext is None:
        print("ERROR: fasttext module not loaded")
        return
    model = fasttext.load_model(args.model)
    print("Model loaded.")

    # Load test data
    print(f"\nLoading test words ({args.n_words} per language)...")
    test_data = {}
    for lang in tqdm(args.languages, desc="Loading test data"):
        words = load_test_words(args.tfdf_dir, args.script, lang,
                               args.n_words, args.min_word_length)
        if words:
            test_data[lang] = words
            print(f"  {lang}: {len(words)} words")

    if not test_data:
        print("No test data loaded. Exiting.")
        return

    # Evaluate fastText
    print("\nRunning fastText evaluation...")
    ft_results = evaluate_fasttext(model, test_data)

    # Save results
    output = {
        'model': 'fasttext_lid.176.bin',
        'accuracy': ft_results['accuracy'],
        'macro_f1': ft_results['macro_f1'],
        'correct': ft_results['correct'],
        'total': ft_results['total'],
        'per_language': {k: dict(v) for k, v in ft_results['per_language'].items()},
        'confusion': {k: dict(v) for k, v in ft_results['confusion'].items()},
        'examples': ft_results['examples'][:20]
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print results
    print("\n" + "="*60)
    print("FASTTEXT RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {ft_results['accuracy']:.4f} ({ft_results['correct']}/{ft_results['total']})")
    print(f"Macro F1 Score:   {ft_results['macro_f1']:.4f}")

    print(f"\nPer-language accuracy:")
    print("-"*40)
    for lang, stats in sorted(ft_results['per_language'].items(),
                              key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"  {lang:10s}: {stats['accuracy']:.4f} ({stats['total']} words)")

    # Compare with our results if provided
    if args.our_results and os.path.exists(args.our_results):
        print(f"\nLoading our results from {args.our_results}...")
        with open(args.our_results, 'r') as f:
            our_results = json.load(f)
        print_comparison(our_results, ft_results)


if __name__ == "__main__":
    main()
