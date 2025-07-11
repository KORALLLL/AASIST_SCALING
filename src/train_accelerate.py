import os
import random
import numpy as np
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import wandb

from utils import genSpoof_list, Dataset_ASVspoof5_train, load_config
from model import Model
from metrics import compute_eer

warnings.filterwarnings('ignore')


# --- Hybrid Loss Function Components ---

class FocalLoss(nn.Module):
    """
    Focal Loss implementation to address class imbalance.
    It down-weights the loss assigned to well-classified examples, focusing training on hard negatives.
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            # Apply alpha weighting
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha) if isinstance(self.alpha, float) else \
            self.alpha[targets]
            focal_loss = alpha_t.to(device=inputs.device) * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HybridLoss(nn.Module):
    """
    A loss function that smoothly transitions from Cross-Entropy to Focal Loss.
    The transition is triggered when the validation EER drops below a specified threshold,
    allowing the model to switch focus to harder examples once it has learned the basics.
    """

    def __init__(self, eer_threshold: float, transition_epochs: int, gamma: float = 2.0, alpha: float = None):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, reduction='mean')
        self.eer_threshold = eer_threshold
        self.transition_epochs = max(1, transition_epochs)  # Ensure at least one epoch for transition
        self.weight = 0.0  # Weight for Focal Loss, starts at 0
        self.transition_start_epoch = -1  # -1 indicates transition has not started

    def update_weight(self, current_epoch: int, best_eval_eer: float):
        """
        Updates the interpolation weight between CE and Focal Loss based on EER and epoch.
        """
        # Trigger the transition if EER threshold is met for the first time
        if self.transition_start_epoch == -1 and best_eval_eer < self.eer_threshold:
            print(
                f"\n[HybridLoss] EER {best_eval_eer:.2f}% < {self.eer_threshold}%. Starting transition to Focal Loss.")
            self.transition_start_epoch = current_epoch

        # Linearly increase the weight towards Focal Loss during the transition period
        if self.transition_start_epoch != -1:
            progress = (current_epoch - self.transition_start_epoch) / self.transition_epochs
            self.weight = min(1.0, progress)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the interpolated loss.
        """
        if self.weight == 0.0:
            return self.cross_entropy(inputs, targets)
        if self.weight == 1.0:
            return self.focal_loss(inputs, targets)

        loss_ce = self.cross_entropy(inputs, targets)
        loss_fl = self.focal_loss(inputs, targets)
        return (1.0 - self.weight) * loss_ce + self.weight * loss_fl


# --- Utilities for Reproducibility and Metric Calculation ---

def initialize_deterministic_environment(random_state: int):
    """
    Sets all necessary seeds for reproducibility across runs.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)
    print(f"All major seeds set to {random_state}.")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("CUDA configured for deterministic operation.")

    def deterministic_worker_init(worker_identifier):
        """
        Ensures DataLoader workers are also deterministic.
        """
        worker_seed = random_state + worker_identifier
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return deterministic_worker_init


@torch.inference_mode()
def calculate_eer(dataloader, model, accelerator, description=""):
    """
    Computes the Equal Error Rate (EER) for a given model and dataloader.
    Handles distributed evaluation by gathering results from all processes.
    """
    model.eval()
    all_scores, all_labels = [], []
    progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=description)

    for inputs, labels in progress_bar:
        outputs = model(inputs)
        scores = outputs[:, 1]  # Assuming bonafide is class 1
        all_scores.append(scores)
        all_labels.append(labels)

    # Gather scores and labels from all distributed processes
    gathered_scores = accelerator.gather_for_metrics(torch.cat(all_scores))
    gathered_labels = accelerator.gather_for_metrics(torch.cat(all_labels))

    eer = 0.0
    # EER calculation is only performed on the main process
    if accelerator.is_main_process:
        scores_np = gathered_scores.cpu().numpy()
        labels_np = gathered_labels.cpu().numpy()
        bonafide_scores = scores_np[labels_np == 1]
        spoof_scores = scores_np[labels_np == 0]

        if bonafide_scores.size > 0 and spoof_scores.size > 0:
            eer_val, _, _, _ = compute_eer(bonafide_scores, spoof_scores)
            eer = eer_val * 100  # Convert to percentage

    return eer, gathered_scores, gathered_labels


# --- Main Training Function ---

def training_function(train_set, eval_set, worker_init_fn, resume_from_checkpoint=False, checkpoint_path=None):
    # Initialize Accelerator for distributed training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # --- Training and Dynamic Difficulty Parameters ---
    batch_size, eval_batch_size = 48, 32
    learning_rate, num_epochs = 1e-4, 20

    # Option to skip validation for the first few epochs to speed up initial training
    SKIP_EVAL_ON_START = True
    SKIP_EVAL_EPOCHS = 2

    # Dynamic augmentation parameters (start, max)
    AUG_PROB_START, AUG_PROB_MAX = 0.5, 0.9
    AUG_INTENSITY_START, AUG_INTENSITY_MAX = 1, 1.8

    # Dynamic loss parameters
    EER_THRESHOLD_FOR_TRANSITION = 8.0  # EER to trigger HybridLoss transition
    LOSS_TRANSITION_EPOCHS = 5  # Epochs for the loss to fully transition
    AUG_TRANSITION_EPOCHS = 10  # Epochs for augmentations to reach max intensity

    # --- Setup Logging (WandB) ---
    if accelerator.is_main_process:
        wandb.init(project="asv-spoof-accelerate-demo",
                   config={"learning_rate": learning_rate, "batch_size": batch_size, "epochs": num_epochs},
                   resume="allow")

    # --- Model, Optimizer, Scheduler, and Loss Initialization ---
    model = Model(args=None)
    optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=300)
    criterion = HybridLoss(eer_threshold=EER_THRESHOLD_FOR_TRANSITION, transition_epochs=LOSS_TRANSITION_EPOCHS,
                           gamma=2.0)

    start_epoch = 0
    best_eval_eer = float('inf')
    best_checkpoint_path = None
    current_aug_p = AUG_PROB_START
    current_aug_intensity = AUG_INTENSITY_START

    # --- Checkpoint Resuming Logic ---
    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        accelerator.print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Load state for all components
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if 'eer' in checkpoint: best_eval_eer = checkpoint['eer']
        if 'best_checkpoint_path' in checkpoint: best_checkpoint_path = checkpoint['best_checkpoint_path']

        # Restore dynamic difficulty state
        if 'aug_p' in checkpoint and 'aug_intensity' in checkpoint:
            current_aug_p = checkpoint['aug_p']
            current_aug_intensity = checkpoint['aug_intensity']
        if 'loss_transition_start_epoch' in checkpoint:
            unwrapped_criterion = criterion
            unwrapped_criterion.transition_start_epoch = checkpoint['loss_transition_start_epoch']
            unwrapped_criterion.update_weight(start_epoch - 1, best_eval_eer)

        accelerator.print(f"Checkpoint loaded. Training will start from epoch {start_epoch}.")
        accelerator.print(
            f"Restored state: EER={best_eval_eer:.2f}%, AugP={current_aug_p:.3f}, AugI={current_aug_intensity:.3f}")
    elif resume_from_checkpoint:
        accelerator.print(f"Warning: Checkpoint not found at {checkpoint_path}. Starting from scratch.")
    else:
        accelerator.print("Starting training from scratch.")

    # --- DataLoaders ---
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    eval_dataloader = DataLoader(eval_set, batch_size=eval_batch_size, shuffle=False, worker_init_fn=worker_init_fn)

    # Initialize augmentation parameters in the dataset
    if hasattr(train_dataloader.dataset, 'update_augmentation_params'):
        train_dataloader.dataset.update_augmentation_params(current_aug_p, current_aug_intensity)

    # --- Prepare all components with Accelerator ---
    # This handles device placement, model wrapping (DDP), etc.
    model, optimizer, scheduler, train_dataloader, eval_dataloader, criterion = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, eval_dataloader, criterion
    )

    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=100)

    # --- Main Training Loop ---
    accelerator.print("Starting training loop")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process,
                            desc=f"Epoch {epoch + 1}/{num_epochs}")

        for inputs, labels in progress_bar:
            # Standard training step
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        # --- Validation Step ---
        if not SKIP_EVAL_ON_START or epoch >= SKIP_EVAL_EPOCHS:
            eval_eer, gathered_scores, gathered_labels = calculate_eer(eval_dataloader, model, accelerator,
                                                                       description="Calculating Eval EER")
            torch.save(gathered_scores, f'gathered_scores_after_{epoch}_epoch.pt')
            torch.save(gathered_labels, f'gathered_labels_after_{epoch}_epoch.pt')
        else:
            eval_eer = float('inf')  # Set to infinity to prevent model saving
            if accelerator.is_local_main_process:
                accelerator.print(f"Epoch {epoch + 1}: Skipping validation as configured.")

        scheduler.step()

        # --- Save Last Checkpoint ---
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            last_checkpoint_path = "last_checkpoint.pth"
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_criterion = accelerator.unwrap_model(criterion)

            # Save the best EER seen so far, unless current eval was skipped
            eer_to_save = min(best_eval_eer, eval_eer) if eval_eer != float('inf') else best_eval_eer

            checkpoint = {
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "eer": eer_to_save,
                "best_checkpoint_path": best_checkpoint_path,
                "aug_p": current_aug_p,
                "aug_intensity": current_aug_intensity,
                "loss_transition_start_epoch": unwrapped_criterion.transition_start_epoch
            }
            accelerator.print('Saved last checkpoint')
            torch.save(checkpoint, last_checkpoint_path)
            if wandb.run: wandb.save(last_checkpoint_path)

        # --- Dynamic Difficulty Scaling Update ---
        unwrapped_criterion = accelerator.unwrap_model(criterion)

        # Update loss weight on main process first
        if accelerator.is_main_process:
            unwrapped_criterion.update_weight(epoch, best_eval_eer)

        # Broadcast the transition start epoch to all processes to keep them in sync
        transition_start_epoch_tensor = torch.tensor([unwrapped_criterion.transition_start_epoch], dtype=torch.float32,
                                                     device=accelerator.device)
        dist.broadcast(tensor=transition_start_epoch_tensor, src=0)
        unwrapped_criterion.transition_start_epoch = int(transition_start_epoch_tensor.item())

        # All processes update their weights based on the synchronized state
        unwrapped_criterion.update_weight(epoch, best_eval_eer)
        current_focal_weight = unwrapped_criterion.weight

        # Update augmentation parameters if loss transition has started
        if unwrapped_criterion.transition_start_epoch != -1:
            epochs_since_start = epoch - unwrapped_criterion.transition_start_epoch
            aug_progress = min(1.0, epochs_since_start / AUG_TRANSITION_EPOCHS)
            current_aug_p = AUG_PROB_START + (AUG_PROB_MAX - AUG_PROB_START) * aug_progress
            current_aug_intensity = AUG_INTENSITY_START + (AUG_INTENSITY_MAX - AUG_INTENSITY_START) * aug_progress

        # Apply updated augmentation params to the dataset
        if hasattr(train_dataloader.dataset, 'update_augmentation_params'):
            train_dataloader.dataset.update_augmentation_params(current_aug_p, current_aug_intensity)

        # --- Logging ---
        current_lr = scheduler.get_last_lr()[0]
        eer_str = f"{eval_eer:.2f}%" if eval_eer != float('inf') else "Skipped"
        accelerator.print(
            f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | Eval EER: {eer_str} | "
            f"LR: {current_lr:.6f} | Focal W: {current_focal_weight:.2f} | "
            f"Aug P: {current_aug_p:.2f} | Aug Int: {current_aug_intensity:.2f}"
        )

        # --- Save Best Checkpoint ---
        if accelerator.is_main_process:
            # This condition is naturally false if eval was skipped (eval_eer = inf)
            if eval_eer < best_eval_eer:
                best_eval_eer = eval_eer
                accelerator.print(f"New best EER: {best_eval_eer:.2f}%. Saving best checkpoint...")

                # Remove old best checkpoint to save space
                if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
                    try:
                        os.remove(best_checkpoint_path)
                    except OSError as e:
                        print(f"Error removing old checkpoint {best_checkpoint_path}: {e}")

                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
                new_best_checkpoint_path = f"best_checkpoint_eer_{best_eval_eer:.2f}_{current_time}.pth"
                best_checkpoint = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "eer": best_eval_eer,
                    "best_checkpoint_path": new_best_checkpoint_path,
                    "aug_p": current_aug_p,
                    "aug_intensity": current_aug_intensity,
                    "loss_transition_start_epoch": accelerator.unwrap_model(criterion).transition_start_epoch
                }
                torch.save(best_checkpoint, new_best_checkpoint_path)
                best_checkpoint_path = new_best_checkpoint_path
                if wandb.run: wandb.save(best_checkpoint_path)

            # Log metrics to WandB
            if wandb.run:
                log_data = {
                    "epoch": epoch + 1, "train/avg_loss": avg_loss,
                    "learning_rate": current_lr, "loss/focal_weight": current_focal_weight,
                    "augmentation/probability": current_aug_p, "augmentation/intensity": current_aug_intensity
                }
                if eval_eer != float('inf'):
                    log_data["eval/eer"] = eval_eer

                wandb.log(log_data)

    accelerator.print("Training finished")


def evaluate_checkpoint_only(eval_set, checkpoint_path="last_checkpoint.pth", eval_batch_size=32, worker_init_fn=None):
    """
    Loads a model from a checkpoint and runs evaluation only.
    """
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(project="asv-spoof-accelerate-demo", name="eval_only", resume="allow")

    model = Model(args=None)

    if not os.path.exists(checkpoint_path):
        accelerator.print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    accelerator.print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    epoch_loaded = checkpoint.get('epoch', '?')
    accelerator.print(f"Model loaded from {checkpoint_path}, trained for {epoch_loaded} epochs.")

    eval_dataloader = DataLoader(eval_set, batch_size=eval_batch_size, worker_init_fn=worker_init_fn)

    # Prepare model and dataloader for evaluation
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()

    # Calculate and print EER
    eval_eer, gathered_scores, gathered_labels = calculate_eer(eval_dataloader, model, accelerator,
                                                               description="Evaluating EER (eval-only)")
    accelerator.print(f"[EVAL-ONLY] EER: {eval_eer:.2f}%")

    # Save scores and labels for further analysis
    torch.save(gathered_scores, f'gathered_scores_after_{epoch_loaded}_epoch.pt')
    torch.save(gathered_labels, f'gathered_labels_after_{epoch_loaded}_epoch.pt')

    # Log results to WandB
    if accelerator.is_main_process and wandb.run:
        wandb.log({
            "eval/eer": eval_eer,
            "epoch": epoch_loaded,
            "eval/checkpoint": os.path.basename(checkpoint_path)
        })
        wandb.finish()


if __name__ == '__main__':
    # --- Script Configuration ---
    # Set to True to run only evaluation on a checkpoint
    EVAL_ONLY = False
    # Set to True to resume training from a checkpoint
    RESUME_TRAINING = True
    # Path to the checkpoint file for resuming or evaluation
    CHECKPOINT_PATH = 'last_checkpoint.pth'
    # Seed for reproducibility
    SEED = 42

    worker_init_fn = initialize_deterministic_environment(random_state=SEED)
    args = load_config()

    # --- Data Preparation ---
    # NOTE: Update paths to your dataset metadata and audio files
    train_meta_path = 'dataset/files/ASVspoof5.train.tsv'
    eval_meta_path = '/home/user2/dataset/files/ASVspoof5.eval.track_1.tsv'

    d_label_train, file_train = genSpoof_list(dir_meta=train_meta_path, is_train=True, is_eval=False)
    d_label_eval, file_eval = genSpoof_list(dir_meta=eval_meta_path, is_train=False, is_eval=False)

    train_base_dir = '/home/user2/dataset/files/flac_T/'
    # Adjust evaluation data path based on the dataset split (Dev or Eval)
    eval_base_dir = '/home/user2/dataset/files/flac_D/'

    train_set = Dataset_ASVspoof5_train(args, list_IDs=file_train, labels=d_label_train, base_dir=train_base_dir,
                                        algo=args.algo, is_train=True)
    eval_set = Dataset_ASVspoof5_train(args, list_IDs=file_eval, labels=d_label_eval, base_dir=eval_base_dir,
                                       algo=args.algo, is_train=False)

    # --- Run Mode Selection ---
    if EVAL_ONLY:
        evaluate_checkpoint_only(
            eval_set,
            checkpoint_path=CHECKPOINT_PATH,
            eval_batch_size=32,
            worker_init_fn=worker_init_fn
        )
    else:
        training_function(
            train_set,
            eval_set,
            worker_init_fn,
            resume_from_checkpoint=RESUME_TRAINING,
            checkpoint_path=CHECKPOINT_PATH
        )