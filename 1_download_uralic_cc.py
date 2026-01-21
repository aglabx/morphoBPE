DOWNLOAD_DATASET = True
CONVERT_TO_TEXT_DATASET = False
uralic_languages = ["kca", "mns", "mrj", "mhr", "myv", "mdf", "koi", "kpv", "udm", "smn", "sjd", "smj", "sme", "sms", "sma", "enf", "nio", "sel", "yrk", "yux", "ykg", "ekk", "fin", "liv", "krl", "olo", "vep", "hun"]  # Add more if needed
dataset_name = "HuggingFaceFW/fineweb-2"
cc_arrow_uralic = "/mnt/projects/users/ichelombitko/uralic_datasets"
сс_uralic = "/mnt/projects/users/ichelombitko/texts_uralic"
wiki_uralic = "/mnt/projects/users/ichelombitko/texts"

# cc_arrow_uralic = "/media/eternus1/nfs/projects/users/ichelombitko/uralic_datasets"
# сс_uralic = "/media/eternus1/nfs/projects/users/ichelombitko/texts_uralic"
# wiki_uralic = "/media/eternus1/nfs/projects/users/ichelombitko/texts"

import os
from datasets import load_dataset

if DOWNLOAD_DATASET:

    os.makedirs(cc_arrow_uralic, exist_ok=True)

    total_languages = len(uralic_languages)  # Всего языков

    for i, lang_code in enumerate(uralic_languages, start=1):
        print(f"[{i}/{total_languages}] Downloading language: {lang_code}")
        
        
        potential_code_langs = [
            lang_code + "_Latn", lang_code + "_Latn_removed", 
            lang_code + "_Cyrl", lang_code + "_Cyrl_removed"
        ]
        dataset = None
        
        for j, lang_variant in enumerate(potential_code_langs, start=1):
            lang_dir = os.path.join(cc_arrow_uralic, lang_variant)
            os.makedirs(lang_dir, exist_ok=True)
            if os.path.exists(lang_dir) and os.listdir(lang_dir):
                print(f"    [{j}/{len(potential_code_langs)}] {lang_variant} already exists. Skipping.")
                continue
            
            try:
                print(f"    [{j}/{len(potential_code_langs)}] Loading dataset for {lang_variant}...")
                dataset = load_dataset(dataset_name, lang_variant, split=None)  # Загружаем весь датасет
            except ValueError:
                print(f"    [{j}/{len(potential_code_langs)}] {lang_variant} did not exist.")
                continue

            dataset.save_to_disk(lang_dir)
            print(f"    [{j}/{len(potential_code_langs)}] Saved {lang_variant} dataset to {lang_dir}")

        print(f"[{i}/{total_languages}] Finished processing {lang_code}\n")

    print("✅ All downloads completed!")
