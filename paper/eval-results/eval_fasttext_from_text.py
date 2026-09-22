#!/usr/bin/env python3
"""
Adapted E1 fastText baseline.

Reads *.step1.wiki.cpp.text files directly (no TF-DF intermediate),
samples 500 test words per language (min 5 chars, lowercase, alpha-only),
runs fastText lid.176.bin, reports per-language accuracy + overall.

Usage:
    python3 eval_fasttext_from_text.py --text-dir /mnt/.../texts_wiki_cpp \\
                                        --model /mnt/.../lid.176.bin \\
                                        --n-words 500 --min-len 5 \\
                                        --output-json fasttext_e1.json
"""
import argparse, json, os, random, re, sys
from collections import defaultdict
from pathlib import Path

random.seed(42)

def parse_langcode(filename: str):
    """wikipedia_<code>_all_nopic_YYYY-MM.step1.wiki.cpp.text -> <code>"""
    m = re.match(r"wikipedia_([a-zA-Z0-9\-]+)_all_nopic", filename)
    return m.group(1) if m else None

def load_test_words(text_file: Path, n_words: int, min_len: int, max_scan_lines: int = 200000):
    """Reservoir-sample n_words distinct alpha lowercase words with len>=min_len."""
    seen = set()
    picks = []
    line_count = 0
    try:
        with open(text_file, encoding='utf-8', errors='replace') as f:
            for line in f:
                line_count += 1
                if line_count > max_scan_lines:
                    break
                for w in line.split():
                    w = w.lower()
                    if len(w) < min_len:
                        continue
                    if not w.isalpha():
                        continue
                    if w in seen:
                        continue
                    seen.add(w)
                    picks.append(w)
                if len(picks) >= n_words * 3:
                    break
    except Exception as e:
        print(f"[warn] {text_file.name}: {e}", file=sys.stderr)
        return []
    if len(picks) > n_words:
        picks = random.sample(picks, n_words)
    return picks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text-dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--n-words', type=int, default=500)
    ap.add_argument('--min-len', type=int, default=5)
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--pattern', default='wikipedia_*.step1.wiki.cpp.text')
    args = ap.parse_args()

    import fasttext
    fasttext.FastText.eprint = lambda x: None  # silence C++ warnings
    model = fasttext.load_model(args.model)
    print(f"[ok] loaded fastText model: {args.model}", file=sys.stderr)

    text_files = sorted(Path(args.text_dir).glob(args.pattern))
    print(f"[ok] found {len(text_files)} text files under {args.text_dir}", file=sys.stderr)

    per_lang = {}
    covered = 0
    total_correct = 0
    total_pred = 0
    covered_correct = 0
    covered_pred = 0

    for i, tf in enumerate(text_files, 1):
        code = parse_langcode(tf.name)
        if not code:
            continue
        words = load_test_words(tf, args.n_words, args.min_len)
        if not words:
            per_lang[code] = {'n': 0, 'correct': 0, 'covered': False, 'accuracy': None}
            continue
        # fastText predict expects a single line; do batch to speed up
        labels, _ = model.predict(words, k=1)
        preds = [lab[0].replace('__label__', '') for lab in labels]
        correct = sum(1 for p in preds if p == code)
        per_lang[code] = {
            'n': len(words),
            'correct': correct,
            'accuracy': correct / len(words),
            'covered': True,
            'top_pred_counts': {},
        }
        # tally top confusions
        cnt = defaultdict(int)
        for p in preds:
            cnt[p] += 1
        top = sorted(cnt.items(), key=lambda x: -x[1])[:3]
        per_lang[code]['top_pred_counts'] = dict(top)
        total_pred += len(words); total_correct += correct
        # Coverage: fastText's lid.176 lists these codes as supported. We treat any prediction != empty as "covered by model" if the language code appears in its output alphabet at least once across the corpus. Simpler: covered if fastText ever predicts code for its own words with prob > 0 (we take: at least one of its own predictions == code).
        if correct > 0 or code in [p for p in preds]:
            covered += 1
            covered_pred += len(words); covered_correct += correct
        if i % 20 == 0:
            print(f"  [{i:3d}/{len(text_files)}] {code}: {per_lang[code]['accuracy']:.3f}", file=sys.stderr)

    overall = total_correct / total_pred if total_pred else 0.0
    overall_covered = covered_correct / covered_pred if covered_pred else 0.0
    summary = {
        'model': args.model,
        'text_dir': args.text_dir,
        'n_langs_seen': len(per_lang),
        'n_langs_covered_by_fasttext': covered,
        'overall_accuracy_all_langs': overall,
        'overall_accuracy_on_covered_langs': overall_covered,
        'per_language': per_lang,
    }
    with open(args.output_json, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {args.output_json}", file=sys.stderr)
    print(f"  Overall accuracy across {len(per_lang)} langs: {overall*100:.2f}%", file=sys.stderr)
    print(f"  Coverage: {covered}/{len(per_lang)} langs where fastText assigned correct label at least once", file=sys.stderr)
    print(f"  Accuracy on covered subset: {overall_covered*100:.2f}%", file=sys.stderr)

if __name__ == '__main__':
    main()
