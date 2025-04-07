import torch


def collate_fn(dataset_items: list[dict]):
    return {"input_audio": [item["input_audio"] for item in dataset_items],
            "output_audio": [item["output_audio"] for item in dataset_items],
            "input_path": [item["input_path"] for item in dataset_items],
            "output_path": [item["output_path"] for item in dataset_items],
            "input_spec": torch.vstack([item["input_spec"] for item in dataset_items]).unsqueeze(1),
            "output_spec": torch.vstack([item["output_spec"] for item in dataset_items]).unsqueeze(1),
            }
