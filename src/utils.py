from pathlib import Path
from tqdm import tqdm
import torch
import random
import numpy as np
from metrics import compute_eer, calculate_CLLR, compute_mindcf
from torch.utils.data import Dataset
import librosa
from torch import Tensor
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
from types import SimpleNamespace
import yaml
import os


def seed_everything(seed: int = 42):
    """
    Устанавливает seed для всех генераторов случайных чисел

    Args:
        seed: Seed для генераторов случайных чисел
    """

    # Базовые Python генераторы
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Настройки для детерминированности
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Для новых версий PyTorch
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)

        # Переменные окружения для CUDA
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Дополнительные переменные окружения
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Настройки для многопоточности
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Дополнительные библиотеки
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass

    try:
        from accelerate.utils import set_seed
        set_seed(seed)
    except ImportError:
        pass

    print(f'Global seed={seed}')


def get_all_file_paths(root_dir, file_ext=None):
    """
    Recursively finds all files within a given root directory, optionally filtering by extension.
    Handles directory paths with multiple intermediate directories.

    Args:
        root_dir (str): The root directory path to start the search from.
        file_ext (str, optional): The file extension to filter by (e.g., ".txt", ".wav").
                        If None, all file types are returned.

    Returns:
        list: A list of strings, where each string is a full file path.
             Returns an empty list if no files are found or an error occurs.
              Prints informative error messages.
    """
    all_file_paths = []
    root_path = Path(root_dir)

    if not root_path.exists():
        return all_file_paths  # return empty list

    if not root_path.is_dir():
        return all_file_paths  # return empty list

    try:
        for file_path in tqdm(root_path.rglob("*")):
            if file_path.is_file():
                if file_ext is None or file_path.suffix.lower() == file_ext.lower():
                    all_file_paths.append(str(file_path))
    except OSError as e:
        raise e

    return all_file_paths


def pad_random(x: torch.Tensor, max_len: int = 64600):
    while x.ndim > 1:
        x = x.squeeze(0)
    x_len = x.shape[0]
    if x_len > max_len:
        start = random.randint(0, x_len - max_len)
        return x[start:start + max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat(num_repeats)[:max_len]
    return padded_x


def produce_evaluation_file(
        data_loader,
        model,
        accelerator,
        window_slicing=False
) -> None:
    """Perform evaluation"""
    model.eval()
    label_list, score_list = [], []
    for audio, label in tqdm(data_loader):
        try:
            with torch.no_grad():
                if window_slicing:
                    batch_out = model(audio.squeeze(0)).clone()
                    batch_score = (audio[:, 1].mean()).data.cpu().numpy().ravel()
                else:
                    batch_out = model(audio)
                    # print(batch_out)
                    # print(label)
                    batch_out, label = accelerator.gather_for_metrics(
                        (batch_out.clone().detach(), label.clone().detach()))
                    batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            # add outputs
            label_list.extend(label.tolist())
            score_list.extend(batch_score.tolist())
            # break
        except Exception as e:
            print(e)
            continue

    return score_list, label_list


def calculate_minDCF_EER_CLLR(score_list, label_list):
    Pspoof = 0.05
    dcf_cost_model = {
        'Pspoof': Pspoof,
        'Cmiss': 1,
        'Cfa': 10,
    }

    score_list, label_list = np.array(score_list).astype(np.float64), np.array(label_list)

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = score_list[label_list == 1]
    spoof_cm = score_list[label_list == 0]

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_cm, frr, far, thresholds = compute_eer(bona_cm, spoof_cm)  # [0]
    cllr_cm = calculate_CLLR(bona_cm, spoof_cm)
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])

    return minDCF_cm, eer_cm, cllr_cm


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, key, _, _, _, _, _, _, label, _ = line.strip().split()

            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif (is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, _, _, _, _, label, _ = line.strip().split()

            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


from augmentations import AudioAugmentor


class Dataset_ASVspoof5_train(Dataset):

    def __init__(self, args,
                 list_IDs, labels, base_dir,
                 algo,
                 is_train: bool = False
                 ):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)
           is_train      : bool, True
        '''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = 64600
        self.is_train = is_train

        self.augmentor = None
        if self.is_train:
            self.augmentor = AudioAugmentor(sample_rate=16000, initial_p=0.4, initial_intensity=1.0)
            print("Тренировочный датасет создан. Аугментатор активирован.")
        else:
            print("Валидационный/тестовый датасет создан. Аугментации отключены.")

    def __len__(self):
        return len(self.list_IDs)

    def update_augmentation_params(self, new_p, new_intensity_coeff):
        """Прокси-метод для обновления параметров вложенного аугментатора."""
        if self.is_train and self.augmentor is not None:
            self.augmentor.update_parameters(new_p, new_intensity_coeff)

    def __getitem__(self, index):

        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + utt_id + '.flac', sr=16000)

        if self.is_train and self.augmentor is not None:
            X_tensor = torch.from_numpy(X).float()

            X_aug_tensor = self.augmentor(X_tensor)

            X_aug_np = X_aug_tensor.numpy()

            Y = process_Rawboost_feature(X_aug_np, fs, self.args, self.algo)
        else:
            Y = process_Rawboost_feature(X, fs, self.args, self.algo)

        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]

        return x_inp, target


def process_Rawboost_feature(feature, sr, args, algo):
    # Data process by Convolutive noise (1st algo)
    if algo == 1:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
                                     args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG,
                                     args.maxG, sr)

        # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

        # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

        # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

        # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                         args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                         args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return feature


def load_config(path="aug_config.yaml"):
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)


def make_windows(
    x: np.ndarray,
    window_size: int = 64_600,
    step_size: int = 16_150,
) -> np.ndarray:

    T = x.shape[0]

    if T <= window_size:
        pad_len = window_size - T
        x = np.pad(x, (0, pad_len), mode="constant")
        return x[None, :]

    n_full = 1 + (T - window_size) // step_size

    last_start = n_full * step_size
    need_tail = last_start < T
    n_windows = n_full + int(need_tail)

    windows = []
    for i in range(min(n_full, 32)):
        s = i * step_size
        windows.append(x[s:s + window_size])

    if need_tail:
        tail = x[last_start:]
        pad_len = window_size - tail.shape[0]
        tail = np.pad(tail, (0, pad_len), mode="constant")
        windows.append(tail)

    batch = np.stack(windows, axis=0)
    # print(windows)
    return batch


class Dataset_ASVspoof5_eval(Dataset):
    def __init__(self, args,
                 list_IDs, labels, base_dir,
                 algo,
                 is_train: bool = False
                 ):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)
           is_train      : bool, True
        '''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = 64600
        self.is_train = is_train

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + utt_id + '.flac', sr=16000)

        X_pad = make_windows(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]

        return x_inp, target
