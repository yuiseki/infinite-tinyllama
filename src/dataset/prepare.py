import os
import time
from pathlib import Path

from litgpt import HFTokenizer
from litgpt.data.prepare_starcoder import DataChunkRecipe
from litdata.processing.data_processor import DataProcessor

from datasets.load import load_dataset

import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

dataset_list = [
    {"id": "wikimedia/wikipedia", "config": "20231101.en"},
    {"id": "wikimedia/wikipedia", "config": "20231101.ja"},
    {"id": "CohereForAI/aya_dataset", "config": "en"},
    {"id": "CohereForAI/aya_dataset", "config": "ja"},
]


def format_number(num):
    if abs(num) >= 10**12:  # Trillion
        return "{:.2f}T".format(num / 10**12)
    elif abs(num) >= 10**9:  # Billion
        return "{:.2f}B".format(num / 10**9)
    elif abs(num) >= 10**6:  # Million
        return "{:.2f}M".format(num / 10**6)
    else:
        return str(num)


class YuisekinAIDataRecipe(DataChunkRecipe):
    def __init__(self, tokenizer: HFTokenizer, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer
        self.total_token_cnt = 0

    def prepare_item(self):
        for dataset_data in dataset_list:
            print("start...", dataset_data["id"], dataset_data["config"])
            dataset_id = dataset_data["id"]
            dataset_config = dataset_data["config"]
            if dataset_config is not None:
                dataset = load_dataset(dataset_id, dataset_config)
            else:
                dataset = load_dataset(dataset_id)
            ds = dataset["train"]
            print("ds", ds)
        if "aya" in dataset_id:
            for v in ds["inputs"]:
                text_ids = self.tokenizer.encode(v, bos=False, eos=True)
                self.total_token_cnt += len(text_ids)
                yield text_ids
        else:
            for v in ds:
                text_ids = self.tokenizer.encode(v["text"], bos=False, eos=True)
                self.total_token_cnt += len(text_ids)
                yield text_ids


def prepare_for_dataset(
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = HFTokenizer(tokenizer_path)
    data_recipe = YuisekinAIDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
    data_processor = DataProcessor(
        input_dir=None,
        output_dir=str(destination_path),
        fast_dev_run=True,
        num_workers=os.cpu_count(),
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


def prepare(
    destination_path: Path = Path("/data/YuisekinAI_data"),
    # 2048 block size + 1 for causal (from LLama), 1024 blocks
    chunk_size: int = 2049 * 1024,
) -> None:
    tokenizer_path = Path("./tmp/tokenizer.json")
    prepare_for_dataset(
        tokenizer_path=tokenizer_path,
        destination_path=destination_path,
        chunk_size=chunk_size,
    )


if __name__ == "__main__":
    prepare()
