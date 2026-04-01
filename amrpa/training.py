"""
AMRPA Training Utilities
--------------------------
Dataset, training loop, evaluation — ported from original script.

Compatible with preprocessed .pt files from the original HotpotQA pipeline.
These files were tokenized with RoBERTa tokenizer for span extraction.
"""

import torch
import torch.nn as nn
import numpy as np
import string
import re
import gc
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from .config import AMRPAConfig


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class PreprocessedQADataset(torch.utils.data.Dataset):
    """
    Loads preprocessed .pt files from original HotpotQA tokenization pipeline.
    Compatible with train_processed.pt and val_processed.pt.
    """

    def __init__(self, processed_file: str):
        print(f"Loading: {processed_file}")
        self.data = torch.load(processed_file)
        print(f"  ✓ {len(self.data)} samples loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        return {
            'input_ids':       torch.tensor(item['input_ids'],       dtype=torch.long),
            'attention_mask':  torch.tensor(item['attention_mask'],  dtype=torch.long),
            'start_positions': torch.tensor(item['start_positions'], dtype=torch.long),
            'end_positions':   torch.tensor(item['end_positions'],   dtype=torch.long),
            'answer_text':     item.get('answer_text', "")
        }


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens  = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common    = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def get_best_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    max_answer_length: int = 30
) -> Tuple[int, int]:
    start_idx = torch.argmax(start_logits).item()
    end_idx   = torch.argmax(end_logits).item()
    if end_idx < start_idx:
        end_idx = start_idx
    if end_idx - start_idx + 1 > max_answer_length:
        end_idx = start_idx + max_answer_length - 1
    return start_idx, end_idx


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER SETUP
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, config: AMRPAConfig, lr: float):
    """
    Differential learning rates — same as original script:
        RoBERTa unfrozen layers: lr * 0.5
        AMRPA params:            lr * 3
        QA head:                 lr * 5
    """
    amrpa_params        = []
    qa_head_params      = []
    roberta_params      = []

    amrpa_param_names = ['mlp_alpha', 'w_mem', 'proj_attention', 'gamma_g', 'bias_g']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(n in name for n in amrpa_param_names):
            amrpa_params.append(param)
        elif 'qa_outputs' in name:
            qa_head_params.append(param)
        else:
            roberta_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': roberta_params,  'lr': lr * 0.5},
        {'params': amrpa_params,    'lr': lr * 3.0},
        {'params': qa_head_params,  'lr': lr * 5.0},
    ], weight_decay=config.dropout * 0.025)  # scaled weight decay

    print(f"  Optimizer groups:")
    print(f"    RoBERTa unfrozen: {len(roberta_params)} params, lr={lr*0.5}")
    print(f"    AMRPA:            {len(amrpa_params)} params, lr={lr*3.0}")
    print(f"    QA head:          {len(qa_head_params)} params, lr={lr*5.0}")

    return optimizer


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    config: AMRPAConfig,
    epoch: int
) -> Tuple[float, Dict]:
    """
    One training epoch. Ported from original script with clean library calls.
    """
    model.train()
    total_loss  = 0.0
    all_metrics = defaultdict(list)

    loss_fct = nn.CrossEntropyLoss(
        ignore_index=-1,
        label_smoothing=config.label_smoothing
    )
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [train]")

    for batch_idx, batch in enumerate(pbar):
        input_ids        = batch['input_ids'].to(device)
        attention_mask   = batch['attention_mask'].to(device)
        start_positions  = batch['start_positions'].to(device)
        end_positions    = batch['end_positions'].to(device)

        start_logits, end_logits, metrics = model(
            input_ids, attention_mask, return_metrics=True
        )

        # QA span loss
        start_loss = loss_fct(start_logits, start_positions)
        end_loss   = loss_fct(end_logits,   end_positions)
        qa_loss    = (start_loss + end_loss) / 2.0

        # AMRPA regularization losses
        diversity_loss    = 0.0
        gate_reg          = 0.0

        if metrics:
            alpha_div  = metrics.get('alpha_diversity',  torch.zeros(1)).mean().item()
            gate_mean  = metrics.get('gate_impact',      torch.zeros(1)).mean().item()

            if alpha_div < 0.15:
                diversity_loss += (0.15 - alpha_div) ** 2
            if gate_mean < 0.1:
                gate_reg += (0.1 - gate_mean) ** 2
            elif gate_mean > 0.95:
                gate_reg += (gate_mean - 0.95) ** 2

        loss = (qa_loss
                + config.diversity_weight * diversity_loss
                + config.gate_reg_weight  * gate_reg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # Aggregate metrics
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    all_metrics[key].extend(value.detach().cpu().tolist())

        # Progress bar
        postfix = {'loss': f'{loss.item():.4f}'}
        if all_metrics.get('gate_impact'):
            postfix['gate'] = f'{np.mean(all_metrics["gate_impact"][-32:]):.3f}'
        if all_metrics.get('alpha_diversity'):
            postfix['α_div'] = f'{np.mean(all_metrics["alpha_diversity"][-32:]):.3f}'
        pbar.set_postfix(postfix)

        # First batch mechanism check
        if batch_idx == 0 and epoch == 0 and metrics:
            print(f"\n{'='*50}")
            print("Mechanism Check (Epoch 1, Batch 1):")
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.mean().item():.4f} (±{v.std().item():.4f})")
            print(f"{'='*50}\n")

        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss        = total_loss / len(dataloader)
    avg_metrics     = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    print(f"\n{'─'*50}")
    print("Training Epoch Summary:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'─'*50}")

    return avg_loss, avg_metrics


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model,
    dataloader,
    tokenizer,
    device: torch.device
) -> Tuple[float, float, float, Dict]:
    """
    Full evaluation with EM, F1, and AMRPA metrics.
    Returns: avg_loss, avg_em, avg_f1, avg_metrics
    """
    model.eval()
    total_loss   = 0.0
    all_em       = []
    all_f1       = []
    all_metrics  = defaultdict(list)
    loss_fct     = nn.CrossEntropyLoss(ignore_index=-1)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions   = batch['end_positions'].to(device)
            answer_texts    = batch['answer_text']

            start_logits, end_logits, metrics = model(
                input_ids, attention_mask, return_metrics=True
            )

            start_loss = loss_fct(start_logits, start_positions)
            end_loss   = loss_fct(end_logits,   end_positions)
            loss       = (start_loss + end_loss) / 2.0
            total_loss += loss.item()

            if metrics:
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        all_metrics[k].extend(v.detach().cpu().tolist())

            for i in range(input_ids.size(0)):
                s, e        = get_best_span(start_logits[i], end_logits[i])
                pred_tokens = input_ids[i][s:e+1]
                pred        = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                gold        = answer_texts[i]
                all_em.append(compute_exact_match(pred, gold))
                all_f1.append(compute_f1(pred, gold))

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'em':   f'{np.mean(all_em):.3f}',
                'f1':   f'{np.mean(all_f1):.3f}'
            })

    avg_loss    = total_loss / len(dataloader)
    avg_em      = float(np.mean(all_em))   if all_em  else 0.0
    avg_f1      = float(np.mean(all_f1))   if all_f1  else 0.0
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    print(f"\n{'─'*50}")
    print("Validation Summary:")
    print(f"  EM: {avg_em:.4f}  F1: {avg_f1:.4f}")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'─'*50}")

    return avg_loss, avg_em, avg_f1, avg_metrics
