import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torch import nn


def accuracy(preds, labels):
    print(preds, labels)
    accuracy = torch.sum(preds == labels).item() / preds.size(0) * 100
    return accuracy


def compute_det_curve(bonafide_scores, spoof_scores):
    """
    function, that comuputes FRR and FAR with their thresholds

    args:
        bonafide_scores: score for bonafide speech
        spoof_scores: score for spoofed speech
    output:
        frr: false rejection rate
        far: false acceptance rate
        threshlods: thresholds for frr and far
    todo:
        rewrite to torch
        create tests
    """
    # number of scores
    n_scores = bonafide_scores.size + spoof_scores.size

    # bona fide scores and spoof scores
    all_scores = np.concatenate((bonafide_scores, spoof_scores))

    # label of bona fide score is 1
    # label of spoof score is 0
    labels = np.concatenate((np.ones(bonafide_scores.size), np.zeros(spoof_scores.size)))

    # indexes of sorted scores in all scores
    indices = np.argsort(all_scores, kind='mergesort')
    # sort labels based on scores
    labels = labels[indices]

    # Compute false rejection and false acceptance rates

    # tar cumulative value
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = spoof_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / bonafide_scores.size))

    # false acceptance rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / spoof_scores.size))

    # Thresholds are the sorted scores
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, frr, far, thresholds


@torch.inference_mode
def produce_evaluation_file(data_loader,
                            model,
                            device,
                            loss_fn,
                            save_path,
                            trial_path,
                            random=False,
                            dropout=0):
    """
    Create file, that need to give in function calculcate_t-DCF_EER
    args:
        data_loader: loader, that gives batch to model
        model: model, that calculate what we need
        device: device for data, model
        save_path: path where file shoud be saved
        trial_path: path from LA CM protocols
    todo:
        this function must return result: tensor of uid, src, key, score
    """

    # turning model into evaluation mode
    model.eval()

    # read file ASVspoof2019.LA.cm.<dev/train/eval>.trl.txt
    with open(trial_path, "r") as file_trial:
        trial_lines = file_trial.readlines()

    # list of utterance id and list of score for appropiate uid
    fname_list = []
    score_list = []
    current_loss = 0
    # inference
    for batch_x, utt_id, batch_y in data_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        with torch.no_grad():
            # first is hidden layer, second is result
            classes, batch_out = model.forward(batch_x, random=random, dropout=dropout)
            # 1 - for bonafide speech class
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            if classes.shape[-1] == 54:
                loss = loss_fn(batch_out, batch_y)
            else:
                loss = loss_fn(classes, batch_y)
            current_loss += loss.item() / len(data_loader)

        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
    assert len(trial_lines) == len(fname_list) == len(score_list)

    # saving results
    with open(save_path, "w") as fh:

        # fn - uid, sco - score, trl - trial_lines
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            # format: utterance id - type of spoof attack - key - score
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))

    return current_loss


@torch.inference_mode
def produce_submit_file(data_loader,
                        model,
                        device,
                        save_path):
    """
    Create file, that need to give in function calculcate_t-DCF_EER
    args:
        data_loader: loader, that gives batch to model
        model: model, that calculate what we need
        device: device for data, model
        save_path: path where file shoud be saved
    """

    # turning model into evaluation mode
    model.eval()

    # list of utterance id and list of score for appropiate uid
    fname_list = []
    score_list = []
    # inference
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            # first is hidden layer, second is result
            batch_out = model(batch_x.squeeze(0))
            # 1 - for bonafide speech class
            batch_score = (batch_out[:, 1].mean()).data.cpu().numpy().ravel()

        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
    assert len(fname_list) == len(score_list)

    # saving results
    with open(save_path, "w") as fh:
        for fn, sco in zip(fname_list, score_list):
            if ".wav" in fn:
                fn = fn.replace(".wav", "")
            fh.write("{} {}\n".format(fn, sco))
    df = pd.read_csv(save_path, sep=" ", names=["ID", "score"])
    df.to_csv(save_path, index=False)
    print("Scores saved to {}".format(save_path))

    return 0


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_thresholds):
    """
    Calculate false alarm rate and miss rate for asv scores

    args:
        tar_asv: scores for asv targets
        non_asv: scores for asv nontargets
        spoof_asv: scores for asv spoofed
        asv_threshold: threshold for asv EER between targets and non_targets
    returns:
        Pfa_asv: false alarm rate for asv
        Pmiss_asv: false miss rate for asv
        Pmiss_spoof_asv: rate of rejection spoofs in asv
    todo:
        rewrite to torch
    """
    Pfa_asv = sum(non_asv >= asv_thresholds) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_thresholds) / tar_asv.size

    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_thresholds) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv,
                 Pmiss_asv, Pmiss_spoof_asv, cost_model):
    """
    This function computes min t-DCF value

    args:
        bonafide_score_cm: score for bonafide speech from CM system
        spoof_score_cm: score for spoofed speech from CM systn
        Pfa_asv: false alarm rate from asv system
        Pmiss_asv: miss rate from asv sustem
        Pmiss_spoof_asv: miss rate for spoof utterance from asv system
        cost_model: dict of parameters for t-DCF
    output:
        t-DCF: computed value
        CM_threshold: threshold for EER between Pmiss_cm and Pfa_cm
    todo:
        rewrite to torch
    """

    # obtain miss and false alarm rate of cm
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm
    )

    # Constants
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
        cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv

    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # normalized t-DCF
    tDCFnorm = tDCF / np.minimum(C1, C2)

    return tDCFnorm, CM_thresholds


def compute_mindcf(frr, far, thresholds, Pspoof, Cmiss, Cfa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds

    p_target = 1 - Pspoof
    for i in range(0, len(frr)):
        # Weighted sum of false negative and false positive errors.
        c_det = Cmiss * frr[i] * p_target + Cfa * far[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(Cmiss * p_target, Cfa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def calculate_CLLR(target_llrs, nontarget_llrs):
    """
    Calculate the CLLR of the scores.

    Parameters:
    target_llrs (list or numpy array): Log-likelihood ratios for target trials.
    nontarget_llrs (list or numpy array): Log-likelihood ratios for non-target trials.

    Returns:
    float: The calculated CLLR value.
    """
    def negative_log_sigmoid(lodds):
        """
        Calculate the negative log of the sigmoid function.

        Parameters:
        lodds (numpy array): Log-odds values.

        Returns:
        numpy array: The negative log of the sigmoid values.
        """
        return np.log1p(np.exp(-lodds))

    # Convert the input lists to numpy arrays if they are not already
    target_llrs = np.array(target_llrs)
    nontarget_llrs = np.array(nontarget_llrs)

    # Calculate the CLLR value
    cllr = 0.5 * (np.mean(negative_log_sigmoid(target_llrs)) + np.mean(negative_log_sigmoid(-nontarget_llrs))) / np.log(2)

    return cllr


def calculate_eer_tdcf(cm_scores_file, asv_score_file, output_file, printout=True):
    """
    Function cimputes tdcf, eer for CM sustem, and also compute
    EER of each type of attack and write them into file
    args:
        cm_scores_file: file from produce_evaluation file
        asv_score_file: file from organizers
        ouput_file: file where information of each type of attack for eval dataset will be
        printout: print this file or not
    output:
        EER * 100: percentage of equal error rate for CM system
        min_tDCF: value of t-DCF for CM system
    todo:
        rewrite into torch
        return array instead of create file
    """
    # cm data from file
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)

    # type of spoof attack
    cm_sources = cm_data[:, 1]

    # spoof or bonafide speech
    cm_keys = cm_data[:, 2]

    # score for utterance
    cm_scores = cm_data[:, 3].astype(np.float64)

    # score for bonafide speech
    bona_cm = cm_scores[cm_keys == 'bonafide']

    # score for spoofed utterance
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # equal error rate
    EER, _ = compute_eer(bona_cm, spoof_cm)

    # fix parameters for t-DCF
    cost_model = {
        'Pspoof': 0.05,
        'Ptar': 0.9405,
        'Pnon': 0.0095,
        'Cmiss': 1,
        'Cfa': 10,  # !!!!!!!!!!
        'Cmiss_asv': 1,
        'Cfa_asv': 10,
        'Cmiss_cm': 1,
        'Cfa_cm': 10,
    }

    # load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)

    # keys: target, non-target, spoof
    asv_keys = asv_data[:, 1]

    # score for each utterance
    asv_scores = asv_data[:, 2].astype(np.float64)

    # target, non-target and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # EER of the standalone systems and fix ASV operation point to
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)

    # generate attack types from A07 to A19
    attack_types = [f'A{_id:02d}' for _id in range(7, 20)]

    # compute eer for each type of attack
    if printout:
        spoof_cm_breakdown = {
            attack_type: cm_scores[cm_sources == attack_type]
            for attack_type in attack_types
        }

        eer_cm_breakdown = {
            attack_type: compute_eer(bona_cm, spoof_cm_breakdown[attack_type])[0]
            for attack_type in attack_types
        }
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = obtain_asv_error_rates(
        tar_asv,
        non_asv,
        spoof_asv,
        asv_threshold
    )

    # Compute t-DCF
    tDCF_curve, CM_thresholds = compute_tDCF(
        bona_cm,
        spoof_cm,
        Pfa_asv,
        Pmiss_asv,
        Pmiss_spoof_asv,
        cost_model
    )

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]
    # write results into file
    if printout:
        with open(output_file, 'w') as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write("""\tEER\t\t= {:8.9f} %
            (Equal error rate for countermeasure)\n""".format(EER * 100)
                        )
            f_res.write('\nTANDEM\n')
            f_res.write('\tmin-tDCF\t\t= {:8.9f}\n'.format(min_tDCF))
            f_res.write('\nBREAKDOWN CM SYSTEM\n')
            for attack_type in attack_types:
                _eer = eer_cm_breakdown[attack_type] * 100
                f_res.write(
                    f'\tEER {attack_type}\t\t= {_eer:8.9f} % (Equal error rate for {attack_type})\n'
                )
        os.system(f"cat {output_file}")
    return EER * 100, min_tDCF


def evaluate_EER_file(ref_df, pred_df, output_file):
    """

        :param ref_df: csv file with columns: uttid, label
        :param pred_df: csv file with columns: uttid, score
        :return: err
        """

    ref_df = pd.read_csv(ref_df, header=None, names=["_", "uttid", "___", "__", "label"], sep=" ")
    ref_df = ref_df.sort_values("uttid")

    pred_df = pd.read_csv(pred_df, header=None, names=["uttid", "_", "__", "scores"], sep=" ")
    pred_df = pred_df.sort_values("uttid")
    if not ref_df["uttid"].equals(pred_df["uttid"]):
        raise ValueError("The 'uttid' columns in the reference and prediction files do not match.")

    pos_scores = pred_df["scores"][ref_df["label"] == "bonafide"]
    neg_scores = pred_df["scores"][ref_df["label"] == "spoof"]

    eer, _ = compute_eer(pos_scores, neg_scores)
    with open(output_file, "w") as f:
        f.write(f"EER: {eer}")
    return eer * 100


def evaluate_EER(ref_df, pred_df):
    """

    :param ref_df: csv file with columns: uttid, label
    :param pred_df: csv file with columns: uttid, score
    :return: err
    """

    ref_df = pd.read_csv(ref_df, header=None, names=["uttid", "label"], sep=" ")
    ref_df = ref_df.sort_values("uttid")

    pred_df = pd.read_csv(pred_df, header=None, names=["uttid", "scores"], sep=" ")
    pred_df = pred_df.sort_values("uttid")

    if not ref_df["uttid"].equals(pred_df["uttid"]):
        raise ValueError("The 'uttid' columns in the reference and prediction files do not match.")

    pos_scores = pred_df["scores"][ref_df["label"] == 1]
    neg_scores = pred_df["scores"][ref_df["label"] == 0]

    eer, _ = compute_eer(pos_scores, neg_scores)
    return eer * 100
