"""
QnA ko genuinely 3 baar corpus mein daalne ke liye: 3 alag symlink-folders
banata hai (QnA_copy1/2/3), har ek mein asli QnA files ke symlink, par
HAR FOLDER MEIN ALAG naam se (file1.jsonl -> copy1__file1.jsonl waghera)
taaki absolute path teeno copies mein ALAG ho -- bulk_corpus.py ka
path_to_source (jo str(p) absolute-path se key banata hai) in teeno ko
collapse na kar sake, jaisa "same folder, teen DATASETS entries" wale
tareeqe mein hua tha.

Colab mein chalao, phir DATASETS mein teen entries daalo (neeche
print hoti hain) -- bas ab har entry ka target folder ALAG hai.
"""

import os
from pathlib import Path

QNA_SRC = "/content/drive/MyDrive/Aetherion_GamaX1/data/QnA"  # apna path daalo
DATA_DIR = "/content/gamax1_data_v2"  # notebook wale DATA_DIR jaisa hi
N_COPIES = 3


def main():
    src = Path(QNA_SRC)
    if not src.is_dir():
        raise FileNotFoundError(f"QnA source nahi mila: {src}")
    files = sorted(
        p for p in src.rglob("*")
        if p.is_file() and p.suffix.lower() in (".txt", ".json", ".jsonl")
    )
    if not files:
        raise FileNotFoundError(f"{src} mein koi .txt/.json/.jsonl nahi mili")

    print(f"Source: {len(files)} files in {src}\n")
    for i in range(1, N_COPIES + 1):
        dst = Path(DATA_DIR) / f"QnA_copy{i}"
        dst.mkdir(parents=True, exist_ok=True)
        linked = 0
        for f in files:
            # Prefix se naam unique banaya -- isse absolute path bhi alag ho
            # jata hai, chahe original filename same rahe.
            link = dst / f"copy{i}__{f.name}"
            if not (link.exists() or link.is_symlink()):
                os.symlink(f, link)
                linked += 1
        print(f"QnA_copy{i}: {linked} naye links -> {dst}")

    print("\nAb DATASETS mein ye entries daalo (bulk_corpus_v2.py ki QnA_repeat* entries ki jagah):")
    print('    "Q&A": "QnA",')
    for i in range(1, N_COPIES + 1):
        print(f'    "Q&A_copy{i}": "QnA_copy{i}",')
    print("\nAur DEFAULT_SOURCE_FORMATS mein bhi sabko SOURCE_FORMAT_USER_ASSISTANT do.")


if __name__ == "__main__":
    main()
