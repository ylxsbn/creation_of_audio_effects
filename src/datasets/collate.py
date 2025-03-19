import torch


def collate_fn(dataset_items: list[dict]):
    result_batch = {}

    result_batch["input_audio"] = torch.vstack(
        [elem["input_audio"] for elem in dataset_items]
    )
    
    result_batch["output_audio"] = torch.vstack(
        [elem["output_audio"] for elem in dataset_items]
    )

    return result_batch
