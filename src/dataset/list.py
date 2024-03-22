
# Print all the available datasets
from huggingface_hub import list_datasets

dataset_ids = [dataset.id for dataset in list_datasets()]

print(len(dataset_ids))
