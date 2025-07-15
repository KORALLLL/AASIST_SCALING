import argparse

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model
from utils import Dataset_ASVspoof5_eval, load_config, genSpoof_list
from metrics import compute_eer


def load_without_ssl(model: nn.Module, path: str) -> None:
    stored = torch.load(path, map_location="cpu")
    model_sd = model.state_dict()
    model_sd.update(stored)
    model.load_state_dict(model_sd, strict=True)


def main(args):
    model = Model(args=None)
    # load_without_ssl(model, args.weigths_path)
    checkpoint = torch.load("/home/kirill/AASIST_SCALING/weights/best_check.pth", map_location='cpu', weights_only=False)

    # Load state for all components
    model.load_state_dict(checkpoint['model'])
    model = model.to("cuda:0")

    data_args = load_config()

    d_label_eval, file_eval = genSpoof_list(dir_meta=args.eval_meta_path, is_train=False, is_eval=False)

    eval_set = Dataset_ASVspoof5_eval(args=data_args, list_IDs=file_eval, labels=d_label_eval, base_dir=args.eval_base_dir,
                                       algo=data_args.algo, is_train=False)

    # eval_dataloader = DataLoader(eval_set, batch_size=54, num_workers=20, shuffle=False)

    scores, labels = [], []
    counter = 0
    with torch.inference_mode():
        for idx in tqdm(range(len(eval_set))):
            inputs, target = eval_set[idx]
            pred = model.forward(inputs.to('cuda:0'))[:, 1]
            pred = pred.mean()
            scores.append(pred.unsqueeze(0))
            labels.append(target)
            counter += 1
            # if counter == 20: break

    # print(scores)
    scores_np = torch.cat(scores).cpu().numpy()
    # print(labels)
    lables_np = torch.Tensor(labels).cpu().numpy()
    bonafide_scores = scores_np[lables_np == 1]
    spoof_scores = scores_np[lables_np == 0]

    eer, _, _, _ = compute_eer(bonafide_scores, spoof_scores)
    print(eer * 100)

    print("Success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weigths_path",
        type=str,
        default="../model.pt"
    )
    parser.add_argument(
        "--eval_meta_path",
        type=str,
        default='/mnt/datasets/asvspoof5/last_eval/ASVspoof5.eval.track_1.tsv'
        # default='/mnt/datasets/asvspoof5/ASVSpoof5.train.tsv'
        # default='/mnt/datasets/asvspoof5/ASVspoof5.dev.tsv'
    )
    parser.add_argument(
        "--eval_base_dir",
        type=str,
        default="/mnt/datasets/asvspoof5/last_eval/flac_E_eval/"
        # default="/mnt/datasets/asvspoof5/flac_T/"
        # default="/mnt/datasets/asvspoof5/flac_D/"
    )
    args = parser.parse_args()
    main(args)