# morphoBPE

Subword-based comparative linguistics toolkit for analyzing 242+ languages using BPE tokenization.

## Paper

This repository contains the implementation for:

**"Subword-Based Comparative Linguistics across 242 Languages Using Wikipedia Glottosets"**

- Authors: Iaroslav Chelombitko, Mika Hämäläinen, Aleksey Komissarov
- Venues: ACL 2025 (submitted)
- Lab Journal: [aglabx/labjournal](https://github.com/aglabx/labjournal)

## Features

- **BPE Training**: Word-only BPE tokenizer (no space tokens) with position tracking
- **Tokenization Trees**: Hierarchical visualization of subword decomposition
- **Cross-Language Comparison**: Merge graph analysis between language-specific tokenizers
- **Script-Level Analysis**: Combined tokenizers for Latin (205 languages) and Cyrillic (37 languages)

## Components

### Core BPE

| File | Description |
|------|-------------|
| `bpe.py` | Python BPE trainer with vocab/min-freq modes |
| `tokenizer.py` | Tokenizer with merge tree visualization |
| `bpes/bpe.cpp` | C++ BPE implementation |
| `bpes/bpe_sa.cpp` | Suffix-array optimized BPE |

### Comparison Tools

| File | Description |
|------|-------------|
| `compare_merge_structures.py` | Merge graph analysis with networkx/graphviz |
| `compare_tokenizers.py` | Cross-language tokenizer comparison |

### Data Pipeline

| File | Description |
|------|-------------|
| `1_download_uralic_cc.py` | Download Uralic languages from Common Crawl |
| `2_convert_arrow.py` | Convert to Arrow format |
| `3_aggregate_texts.py` | Aggregate texts by language |
| `from_text_to_tfdf.py` | Extract TF-DF from text |

### Preprocessing (C++)

| File | Description |
|------|-------------|
| `tf_df.cpp` | Fast TF-DF extraction |
| `clean_text.cpp` | Text cleaning and normalization |

## Usage

### Train BPE tokenizer

```bash
# With vocabulary size limit
python bpe.py input.tsv --vocab-size 4096 --output-file vocab.json

# With minimum frequency threshold
python bpe.py input.tsv --min-freq 2 --output-file vocab.json
```

### Tokenize and visualize

```python
from tokenizer import Tokenizer

tok = Tokenizer('vocab.json')

# Get tokenization tree
trees = tok.get_tokenization_tree("промисловість")
for tree in trees:
    tok.print_token_tree(tree)
```

### Compare tokenizers

```bash
python compare_merge_structures.py vocab_uk.json vocab_ru.json --output-dir analysis/
```

## Datasets

- Wikipedia dumps (320 languages): [dumps.wikimedia.org/kiwix/zim/wikipedia/](https://dumps.wikimedia.org/kiwix/zim/wikipedia/)
- Processed glottosets: Hugging Face (link TBD)

## License

MIT
