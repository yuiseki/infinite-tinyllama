import lit_llama.packed_dataset as packed_dataset
from lit_llama import Tokenizer, HFTokenizer
from datasets import load_dataset
import numpy as np

from pathlib import Path
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

sample_ids = ["izumi-lab/wikinews-ja-20230728", "izumi-lab/wikinews-en-20230728", "if001/aozorabunko-clean-sin"]


def format_number(num):
    if abs(num) >= 10**12:  # Trillion
        return "{:.2f}T".format(num / 10**12)
    elif abs(num) >= 10**9:  # Billion
        return "{:.2f}B".format(num / 10**9)
    elif abs(num) >= 10**6:  # Million
        return "{:.2f}M".format(num / 10**6)
    else:
        return str(num)


def prepare_for_dataset(
    dataset_ids: list[str],
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)
    # tokenizer = Tokenizer(tokenizer_path)
    tokenizer = HFTokenizer(model_path=tokenizer_path)
    total_token_cnt = 0
    for dataset_id in dataset_ids:
        token_cnt = 0
        print(f"Processing {dataset_ids}")
        prefix = dataset_id.split("/")[-1]
        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.bos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )
        ds = load_dataset(dataset_id)
        ds = ds["train"]

        if "aozora" in dataset_id:
            for v in ds["text"]:
                text_ids = tokenizer.encode(v)
                token_cnt += len(text_ids)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
        else:
            for v in ds:
                text_ids = tokenizer.encode(v["text"])
                token_cnt += len(text_ids)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
        builder.write_reminder()
        print("tokens ", format_number(token_cnt))
        total_token_cnt += token_cnt
    print("total tokens", format_number(total_token_cnt))


def prepare(
    destination_path: Path = Path("/data/YuisekinAI_data"),
    # 2048 block size + 1 for causal (from LLama), 1024 blocks
    chunk_size: int = 2049 * 1024,
) -> None:
    prepare_for_dataset(
        dataset_ids=dataset_ids,
        tokenizer_path=tokenizer_path,
        destination_path=destination_path,
        chunk_size=chunk_size,
    )


if __name__ == "__main__":
    prepare()
