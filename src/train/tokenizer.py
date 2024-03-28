# https://zenn.dev/if001/articles/87bbe893411fa1
import datasets
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

dataset_list = [
    {"id": "wikimedia/wikipedia", "config": "20231101.en"},
    {"id": "wikimedia/wikipedia", "config": "20231101.ja"},
    {"id": "CohereForAI/aya_dataset", "config": "en"},
    {"id": "CohereForAI/aya_dataset", "config": "ja"},
]


def init_tokenizer():
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.UnicodeScripts()
    tokenizer.decoder = decoders.BPEDecoder()
    return tokenizer


def train(tokenizer, trainer):
    def ds_yielder():
        for dataset_data in dataset_list:
            print("start...", dataset_data["id"], dataset_data["config"])
            dataset_id = dataset_data["id"]
            dataset_config = dataset_data["config"]
            if dataset_config is not None:
                dataset = datasets.load_dataset(dataset_id, dataset_config)
            else:
                dataset = datasets.load_dataset(dataset_id)
            ds = dataset["train"]
            print("ds", ds)
            # ds = ds.select(range(0, 100))
            if "aya" in dataset_id:
                for v in ds["inputs"]:
                    yield v
            else:
                for v in ds:
                    yield v["text"]

    tokenizer.train_from_iterator(ds_yielder(), trainer=trainer)
    return tokenizer


def main():
    save_path = "./tmp/tokenizer.json"
    vocab_size = 32000

    tokenizer = init_tokenizer()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>"],
        unk_token="<UNK>",
    )
    tokenizer = train(tokenizer, trainer)
    tokenizer.save(save_path)
    print(f"save... {save_path}")


if __name__ == "__main__":
    main()
