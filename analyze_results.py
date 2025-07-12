import os
import pandas as pd
import torch
import pickle as pkl
import json
import torch.nn as nn
import numpy as np
import tqdm
import traceback
import pdb
import csv
from enum import Enum
from typing import List, Dict, Any, Tuple
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer, util

from utils import (
    compute_semantic_clusters, 
    compute_semantic_entropy, 
    compute_semantic_entropy_new, 
    prepare_results, 
    prepare_results_new
)
from args import Args


class CorrectnessMetric(Enum):
    """Enumeration for correctness metrics."""
    ROUGE_L = "rougeL"
    BLEURT = "bleurt"
    ROUGE_1 = "rouge1"


class RunKey(Enum):
    """Enumeration for run keys."""
    DOLA_SDLG = "dola_sdlg"
    BASELINE = "baseline"
    SDLG = "sdlg"


class Constants:
    """Named constants to replace magic numbers."""
    
    # Dataset and model constants
    NUM_INSTANCES = 817
    NUM_TOTAL_GENERATIONS = 10
    MIN_GENERATIONS = 2
    DEFAULT_AUROC_VALUE = 11.0
    
    # Device configurations
    DEVICE_LLM = 'cuda:0'
    DEVICE_DEBERTA = 'cuda:0'
     
    # Path configurations
    BASE_PATH = '/notebooks/clarification/SDLG/modeified_sdlg/result/truthful-mistralv1-7b/'
    BASE_PATH_CLUSTER = '/notebooks/clarification/SDLG/modeified_sdlg/result/truthful-mistralv1-7b/'
    BASE_PATH_FINAL = '/notebooks/clarification/SDLG/modeified_sdlg/'
    
    # Thresholds
    CORRECTNESS_THRESHOLDS = [
        -0.05, -0.1, -0.15, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, 
        0, 0.03, 0.07, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5
    ]
    
    # Metric keys
    AUROC_KEYS_BASELINE = ["normalised_semantic_entropy", "unnormalised_semantic_entropy"]
    AUROC_KEYS = [
        "normalised_semantic_entropy", "unnormalised_semantic_entropy",
        'entropy_final', 'jsd', 'jsd_min', 'jsd_max', 'kl', 'kl_min', 'kl_max',
        'entropy_mixture', 'entropy_mixture_min', 'entropy_mixture_max',
        'mutual_info', 'mutual_info_min', 'mutual_info_max',
        'epkl', 'epkl_min', 'epkl_max', 'cv', 'cv_min', 'cv_max',
        'bhattacharyya', 'bhattacharyya_min', 'bhattacharyya_max',
        'tvd', 'tvd_min', 'tvd_max'
    ]
    
    # CSV header
    CSV_HEADER = [
        "run_key", "num_gens", "correctness_threshold",
        "auroc_norm_sem_ent", "auroc_unnorm_sem_ent",
        "auroc_ours_norm_sem_ent", "auroc_ours_unnorm_sem_ent",
        'entropy_final', 'jsd', 'jsd_min', 'jsd_max', 'kl', 'kl_min', 'kl_max',
        'entropy_mixture', 'entropy_mixture_min', 'entropy_mixture_max',
        'mutual_info', 'mutual_info_min', 'mutual_info_max',
        'epkl', 'epkl_min', 'epkl_max', 'cv', 'cv_min', 'cv_max',
        'bhattacharyya', 'bhattacharyya_min', 'bhattacharyya_max',
        'tvd', 'tvd_min', 'tvd_max',
        "auroc_kuhn_norm_ent", "auroc_kuhn_unnorm_ent"
    ]


def setup_environment() -> None:
    """Configure environment variables."""
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def initialize_csv_file(filepath: Path) -> None:
    """Initialize CSV file with header if it doesn't exist."""
    if not filepath.exists():
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(Constants.CSV_HEADER)


def clean_results_dict(results_dict: Dict[str, Any], run_key: str) -> None:
    """Clean results dictionary by removing unnecessary keys."""
    keys_to_remove = [
        'unfiltered_premature_logits',
        'unfiltered_final_logits', 
        'tokens_kept_count',
        'tokens_kept_percentage'
    ]
    
    for entropy_dict in results_dict[run_key]['epistem_entropies']:
        for key in keys_to_remove:
            entropy_dict.pop(key, None)


def create_boolean_mask(num_gens: int, total_generations: int) -> torch.Tensor:
    """Create boolean mask for generation filtering."""
    if num_gens < total_generations:
        mask_values = [True] * num_gens + [False] * (total_generations - num_gens)
    else:
        mask_values = [True] * total_generations
    return torch.tensor(mask_values)


def create_run_mask(run_key: str, results_dict: Dict[str, Any], boolean_mask: torch.Tensor) -> torch.Tensor:
    """Create appropriate mask based on run key."""
    if run_key in (RunKey.SDLG.value, RunKey.DOLA_SDLG.value):
        mask_values = [1] + [gen['token_likelihood'] for gen in results_dict[run_key]['generations'][1:]]
        mask = torch.tensor(mask_values)
        assert torch.all(mask > 0) and torch.all(mask[1:] < 1), f"mask: {mask}"
        return mask
    elif run_key == RunKey.BASELINE.value:
        return boolean_mask
    else:
        raise ValueError(f"Unknown run key: {run_key}")


def filter_generations_and_likelihoods(
    results_dict: Dict[str, Any], 
    run_key: str, 
    boolean_mask: torch.Tensor
) -> Tuple[List[Any], List[Any]]:
    """Filter generations and likelihoods based on boolean mask."""
    considered_generations = []
    considered_likelihoods = []
    
    for idx, included in enumerate(boolean_mask):
        if included:
            considered_generations.append(results_dict[run_key]['generations'][idx])
            considered_likelihoods.append(results_dict[run_key]['likelihoods'][idx])
    
    return considered_generations, considered_likelihoods


def process_semantic_pairs(
    results_dict: Dict[str, Any], 
    run_key: str, 
    model_type: str, 
    boolean_mask: torch.Tensor, 
    compute_cleaned: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process semantic pairs with proper masking."""
    semantic_pairs_data = results_dict[run_key][f'semantic_pairs_{model_type}']
    
    if semantic_pairs_data['semantic_pairs'].shape[0] > 1:
        semantic_pairs = semantic_pairs_data['semantic_pairs'][boolean_mask, :][:, boolean_mask]
        cleaned_semantic_pairs = None
        if compute_cleaned:
            cleaned_semantic_pairs = semantic_pairs_data['cleaned_semantic_pairs'][boolean_mask, :][:, boolean_mask]
    else:
        assert boolean_mask.item() == True, f"mask: {boolean_mask}"
        semantic_pairs = semantic_pairs_data['semantic_pairs']
        cleaned_semantic_pairs = None
        if compute_cleaned:
            cleaned_semantic_pairs = semantic_pairs_data['cleaned_semantic_pairs']
    
    semantic_pairs = semantic_pairs & semantic_pairs.T
    assert np.array_equal(semantic_pairs, semantic_pairs.T)
    
    if compute_cleaned and cleaned_semantic_pairs is not None:
        cleaned_semantic_pairs = cleaned_semantic_pairs & cleaned_semantic_pairs.T
        assert np.array_equal(cleaned_semantic_pairs, cleaned_semantic_pairs.T)
    
    return semantic_pairs, cleaned_semantic_pairs


def compute_weights(run_key: str, boolean_mask: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute weights for entropy calculations."""
    if run_key == RunKey.BASELINE.value:
        return boolean_mask[boolean_mask].to(torch.float32)
    else:
        return torch.nn.functional.normalize(mask[boolean_mask].to(torch.float32), p=1, dim=0)


def compute_all_entropies(
    weights: torch.Tensor,
    my_weights: torch.Tensor,
    considered_likelihoods: List[Any],
    semantic_difference: Dict[str, Any],
    epistem_entropies: List[Dict[str, Any]],
    compute_cleaned: bool
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Compute all entropy types."""
    
    entropy_kuhn = compute_semantic_entropy(
        weights=weights,
        mc_estimate_over_clusters=True,
        neg_log_likelihoods=considered_likelihoods,
        semantic_difference=semantic_difference,
        epistem_entropy_list=epistem_entropies,
        compute_cleaned=compute_cleaned
    )
    
    entropy_standard = compute_semantic_entropy(
        weights=weights,
        mc_estimate_over_clusters=False,
        neg_log_likelihoods=considered_likelihoods,
        semantic_difference=semantic_difference,
        epistem_entropy_list=epistem_entropies,
        compute_cleaned=compute_cleaned
    )
    
    entropy_ours = compute_semantic_entropy_new(
        weights=my_weights,
        mc_estimate_over_clusters=False,
        neg_log_likelihoods=considered_likelihoods,
        semantic_difference=semantic_difference,
        epistem_entropy_list=epistem_entropies,
        compute_cleaned=compute_cleaned
    )
    
    return entropy_standard, entropy_ours, entropy_kuhn


def compute_auroc_scores(
    correct_labels: torch.Tensor,
    semantic_entropies: List[Dict[str, float]],
    semantic_entropies_ours: List[Dict[str, float]],
    semantic_entropies_kuhn: List[Dict[str, float]]
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Compute AUROC scores for all entropy types."""
    
    aurocs = {}
    for key in Constants.AUROC_KEYS_BASELINE:
        aurocs[key] = roc_auc_score(
            correct_labels,
            [entropy_dict[key] for entropy_dict in semantic_entropies]
        )
    
    aurocs_ours = {}
    for key in Constants.AUROC_KEYS:
        aurocs_ours[key] = roc_auc_score(
            correct_labels,
            [entropy_dict[key] for entropy_dict in semantic_entropies_ours]
        )
    
    aurocs_kuhn = {}
    for key in Constants.AUROC_KEYS_BASELINE:
        aurocs_kuhn[key] = roc_auc_score(
            correct_labels,
            [entropy_dict[key] for entropy_dict in semantic_entropies_kuhn]
        )
    
    return aurocs, aurocs_ours, aurocs_kuhn


def write_results_to_csv(
    csv_filepath: Path,
    run_key: str,
    num_gens: int,
    correctness_threshold: float,
    aurocs: Dict[str, float],
    aurocs_ours: Dict[str, float],
    aurocs_kuhn: Dict[str, float]
) -> None:
    """Write results to CSV file."""
    row_data = [
        run_key,
        num_gens,
        correctness_threshold,
        aurocs.get("normalised_semantic_entropy", Constants.DEFAULT_AUROC_VALUE),
        aurocs.get("unnormalised_semantic_entropy", Constants.DEFAULT_AUROC_VALUE),
        aurocs_ours.get("normalised_semantic_entropy", Constants.DEFAULT_AUROC_VALUE),
        aurocs_ours.get("unnormalised_semantic_entropy", Constants.DEFAULT_AUROC_VALUE),
    ]
    
    for key in Constants.AUROC_KEYS[2:]:
        row_data.append(aurocs_ours.get(key, Constants.DEFAULT_AUROC_VALUE))
    
    row_data.extend([
        aurocs_kuhn.get("normalised_semantic_entropy", Constants.DEFAULT_AUROC_VALUE),
        aurocs_kuhn.get("unnormalised_semantic_entropy", Constants.DEFAULT_AUROC_VALUE),
    ])
    
    with open(csv_filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)


def main() -> None:
    """Main execution function."""
    setup_environment()
    
    args = Args()
    removed_sample_ids = []
    model_type = args.deberta_model
    print(f"Model type: {model_type}")
    
    correctness_metric = CorrectnessMetric.BLEURT.value
    run_keys = [RunKey.DOLA_SDLG.value]
    compute_cleaned = args.compute_cleaned
    
    csv_filepath = Path(Constants.BASE_PATH_FINAL) / 'auroc_llama_truthful.csv'
    initialize_csv_file(csv_filepath)
    
    for run_key in run_keys:
        list_results_dict, list_bleurt, dataset_size = prepare_results(
            num_samples=Constants.NUM_INSTANCES,
            run_key=run_key,
            metric=correctness_metric,
            start_sample_id=0,
            base_path=Constants.BASE_PATH_CLUSTER
        )
        
        print(f'BLEURT range: max={max(list_bleurt)}, min={min(list_bleurt)}')
        
        with open(Path(Constants.BASE_PATH) / f'{run_key}_label_llama_truthful.pkl', 'wb') as f:
            pkl.dump(list_bleurt, f)
        
        for results_dict in list_results_dict:
            clean_results_dict(results_dict, run_key)
        
        for num_gens in tqdm.tqdm(range(Constants.MIN_GENERATIONS, Constants.NUM_TOTAL_GENERATIONS + 1)):
            all_semantic_entropies = []
            all_semantic_entropies_kuhn = []
            all_semantic_entropies_ours = []
            list_num_semantic_clusters = []
            list_num_generations = []
            
            for i, results_dict in enumerate(list_results_dict):
                boolean_mask = create_boolean_mask(num_gens, len(results_dict[run_key]['generations']))
                mask = create_run_mask(run_key, results_dict, boolean_mask)
                
                list_num_generations.append(torch.sum(boolean_mask).item())
                
                considered_generations, considered_likelihoods = filter_generations_and_likelihoods(
                    results_dict, run_key, boolean_mask
                )
                
                semantic_pairs, cleaned_semantic_pairs = process_semantic_pairs(
                    results_dict, run_key, model_type, boolean_mask, compute_cleaned
                )
                
                semantic_difference = compute_semantic_clusters(
                    generations=considered_generations,
                    cleaned_semantic_pairs=cleaned_semantic_pairs if compute_cleaned else None,
                    semantic_pairs=semantic_pairs,
                    compute_cleaned=compute_cleaned
                )
                
                list_num_semantic_clusters.append(
                    torch.unique(semantic_difference["semantic_clusters"]).shape[0]
                )
                
                weights = compute_weights(run_key, boolean_mask, mask)
                my_weights = results_dict[run_key]['seq_level_impo'][boolean_mask].to('cpu')
                
                entropy_standard, entropy_ours, entropy_kuhn = compute_all_entropies(
                    weights, my_weights, considered_likelihoods, semantic_difference,
                    results_dict['sdlg']['epistem_entropies'], compute_cleaned
                )
                
                all_semantic_entropies.append(entropy_standard)
                all_semantic_entropies_ours.append(entropy_ours)
                all_semantic_entropies_kuhn.append(entropy_kuhn)
            
            all_semantic_entropies_combined = all_semantic_entropies_ours + all_semantic_entropies_kuhn
            
            with open(Path(Constants.BASE_PATH) / f'{run_key}_entropies_llama_truthful.pkl', 'wb') as f:
                pkl.dump(all_semantic_entropies_combined, f)
            
            for correctness_threshold in Constants.CORRECTNESS_THRESHOLDS:
                try:
                    correct_labels = torch.logical_not(torch.tensor(list_bleurt) >= correctness_threshold)
                    
                    aurocs, aurocs_ours, aurocs_kuhn = compute_auroc_scores(
                        correct_labels,
                        all_semantic_entropies,
                        all_semantic_entropies_ours,
                        all_semantic_entropies_kuhn
                    )
                    
                    write_results_to_csv(
                        csv_filepath, run_key, num_gens, correctness_threshold,
                        aurocs, aurocs_ours, aurocs_kuhn
                    )
                    
                except Exception as e:
                    print(f"Error processing threshold {correctness_threshold}: {e}")
                    traceback.print_exc()
        
        del list_results_dict, list_bleurt


if __name__ == "__main__":
    main()
