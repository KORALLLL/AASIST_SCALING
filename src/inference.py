import argparse

import torch, torch.nn as nn

from model import Model


def load_without_ssl(model: nn.Module, path: str, strict: bool = True) -> None:
    """
    Load weights that were saved with `save_without_ssl`.

    `strict=True`  → will raise if a non-ssl parameter is missing
    `strict=False` → ignores keys that are absent/present only in the file.
    """
    stored = torch.load(path, map_location="cpu")       # 1. read file
    model_sd = model.state_dict()                       # 2. current weights

    # 3. Update only the keys we stored
    model_sd.update(stored)

    # 4. Push back into the network
    model.load_state_dict(model_sd, strict=strict)


def main(args):
    model = Model(args=None)
    load_without_ssl(model, args.weigths_path)

    mock_data = torch.randn([1, 64600])
    result = model.forward(mock_data)
    print(model.norm_master.bias)

    print("Success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weigths_path",
        type=str,
        default="../model.pt"
    )
    args = parser.parse_args()
    main(args)