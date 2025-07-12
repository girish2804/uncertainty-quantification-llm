import os
import pickle
import yaml
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import datasets
import evaluate
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from datasets import load_dataset
import traceback

from args import Args
from utils import (
    seed_everything, 
    compute_correctness, 
    compute_semantic_paris_new,
    generate_text, 
    compute_likelihood
)
from sdlg import generate_semantically_diverse_output_sequences


class ModelType(Enum):
    MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
    LLAMA_2_7B = "meta-llama/Llama-2-7b-hf"
    OPT_13B = "facebook/opt-13b"


class DebertaModel(Enum):
    BASE_MNLI = "deberta-base-mnli"
    LARGE_MNLI = "deberta-large-mnli"
    XLARGE_MNLI = "deberta-xlarge-mnli"
    V2_XLARGE_MNLI = "deberta-v2-xlarge-mnli"
    V2_XXLARGE_MNLI = "deberta-v2-xxlarge-mnli"


@dataclass
class Constants:
    DEFAULT_CUDA_DEVICE: int = 0
    TRUTHFUL_QA_FEW_SHOT_START: int = 51
    TRUTHFUL_QA_FEW_SHOT_END: int = 54
    HF_AUTH_TOKEN: str = 'your_api_token'
    RESULTS_DIR: str = "result"
    CONFIG_FILENAME: str = "config.yaml"
    DEBERTA_MODEL_INDEX: int = 1
    BATCH_SIZE: int = 1


def get_device(cuda_id: int = 0) -> str:
    """Determine the appropriate device for model execution."""
    if torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return f"cuda:{cuda_id}"
    else:
        return "cpu"


def encode_sample(examples: Dict[str, Any], few_shot: str, tokenizer) -> Dict[str, Any]:
    """Encode a single sample for model input."""
    prompt = f"{few_shot} Q: {examples['question']} A:"
    return tokenizer(prompt, truncation=False, padding=False)


def encode_and_format_dataset(dataset: datasets.Dataset, few_shot: str, tokenizer) -> datasets.Dataset:
    """Encode and format the entire dataset."""
    encoded_dataset = dataset.map(
        lambda examples: encode_sample(examples, few_shot, tokenizer), 
        batched=False, 
        load_from_cache_file=False
    )
    encoded_dataset.set_format(
        type='torch', 
        columns=['input_ids', 'attention_mask'], 
        output_all_columns=True
    )
    return encoded_dataset


def create_few_shot_prompt(dataset: datasets.Dataset, constants: Constants) -> str:
    """Create few-shot learning prompt from dataset samples."""
    few_shot = 'This is a bot that correctly answers questions. \n'
    
    sample_range = range(constants.TRUTHFUL_QA_FEW_SHOT_START, constants.TRUTHFUL_QA_FEW_SHOT_END)
    
    for sample in dataset.select(sample_range):
        question = sample['question']
        answer = sample['best_answer']
        if not answer.endswith("."):
            answer += "."
        few_shot += f'Q: {question} A: {answer} '
    
    return few_shot


def load_models(constants: Constants) -> Tuple[Any, Any, Any, Any, str, str]:
    """Load all required models and tokenizers."""
    device_llm = get_device(constants.DEFAULT_CUDA_DEVICE)
    device_deberta = device_llm
    
    # Load DeBERTa model
    deberta_model_type = list(DebertaModel)[constants.DEBERTA_MODEL_INDEX]
    deberta_model_name = f"microsoft/{deberta_model_type.value}"
    deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        deberta_model_name, device_map='auto'
    )
    
    # Load LLM tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(
        ModelType.OPT_13B.value, 
        token=constants.HF_AUTH_TOKEN
    )
    
    return None, llm_tokenizer, deberta_model, deberta_tokenizer, device_llm, device_deberta


def initialize_metrics() -> Dict[str, Any]:
    """Initialize evaluation metrics."""
    return {
        'squad_metric': evaluate.load("squad"),
        'rouge': evaluate.load('rouge'),
        'exact_match_metric': evaluate.load("exact_match"),
        'bleurt': evaluate.load("bleurt")
    }


def create_results_dict(detached_input_ids: torch.Tensor, question: str) -> Dict[str, Any]:
    """Create the results dictionary structure."""
    return {
        'input_ids': detached_input_ids,
        'question': question,
        'correctness_dict': {},
        'dola_sdlg': {'generations': [], 'likelihoods': [], 'epistem_entropies': []},
        'sdlg': {'generations': [], 'likelihoods': [], 'epistem_entropies': []},
        'baseline': {'generations': [], 'likelihoods': []}
    }


def process_generation_methods(results_dict: Dict[str, Any], batch: Dict[str, Any], prompt: torch.Tensor, 
                             question: str, most_likely_generation: Dict[str, Any], 
                             most_likely_generation_dola: Dict[str, Any], 
                             most_likely_generation_likelihoods: Dict[str, Any],
                             most_likely_generation_likelihoods_dola: Dict[str, Any],
                             deberta_model: Any, deberta_tokenizer: Any, deberta_embeddings: torch.Tensor, 
                             llm_model: Any, tokenizer: Any, device_llm: str, device_deberta: str, 
                             args: Args) -> Dict[str, Any]:
    """Process different generation methods (SDLG, DOLA-SDLG, baseline)."""
    
    # DOLA-SDLG
    results_dict['dola_sdlg']['generations'].append(most_likely_generation_dola)
    results_dict['dola_sdlg']['likelihoods'].append(most_likely_generation_likelihoods_dola)
    
    results_dict = generate_semantically_diverse_output_sequences(
        results_dict=results_dict, deberta_model=deberta_model, deberta_tokenizer=deberta_tokenizer,
        device_deberta=device_deberta, deberta_embeddings=deberta_embeddings, model=llm_model,
        tokenizer=tokenizer, device_llm=device_llm, input_ids=batch['input_ids'], prompt=prompt,
        question=question, initial_generation=most_likely_generation_dola,
        initial_likelihood=most_likely_generation_likelihoods_dola, key='dola_sdlg', args=args
    )
    
    # SDLG
    results_dict['sdlg']['generations'].append(most_likely_generation)
    results_dict['sdlg']['likelihoods'].append(most_likely_generation_likelihoods)
    
    results_dict = generate_semantically_diverse_output_sequences(
        results_dict=results_dict, deberta_model=deberta_model, deberta_tokenizer=deberta_tokenizer,
        device_deberta=device_deberta, deberta_embeddings=deberta_embeddings, model=llm_model,
        tokenizer=tokenizer, device_llm=device_llm, input_ids=batch['input_ids'], prompt=prompt,
        question=question, initial_generation=most_likely_generation,
        initial_likelihood=most_likely_generation_likelihoods, key='sdlg', args=args
    )
    
    # Baseline
    assert args.num_total_generations % args.num_return_sequences_baseline == 0
    results_dict['baseline']['generations'].append(most_likely_generation)
    results_dict['baseline']['likelihoods'].append(most_likely_generation_likelihoods)
    
    num_iterations = int(args.num_total_generations / args.num_return_sequences_baseline)
    
    for _ in range(num_iterations):
        baseline_generation = generate_text(
            args=args, model=llm_model, tokenizer=tokenizer, input_ids=batch['input_ids'],
            len_prompt=len(prompt), decoding_method='baseline', device=device_llm
        )
        
        results_dict['baseline']['generations'].append(baseline_generation)
        results_dict['baseline']['likelihoods'].append(
            compute_likelihood(
                prompt=prompt, generation=baseline_generation, model=llm_model, device=device_llm,
                compute_cleaned=args.compute_cleaned, store_logits=args.store_logits
            )
        )
    
    return results_dict


def print_generation_results(results_dict: Dict[str, Any]) -> None:
    """Print generation results for debugging."""
    for method in ['sdlg', 'dola_sdlg', 'baseline']:
        for generation in results_dict[method]['generations']:
            print(generation['generation_text'])
        print(f'{method}^-----------------------------------')
    print('************************************')


def process_batch(batch: Dict[str, Any], base_path: Path, llm_model: Any, tokenizer: Any, 
                 deberta_model: Any, deberta_tokenizer: Any, deberta_embeddings: torch.Tensor, 
                 metrics: Dict[str, Any], device_llm: str, device_deberta: str, args: Args) -> None:
    """Process a single batch of data."""
    prompt = batch['input_ids'][0].to('cpu')
    question = batch["question"][0]
    detached_input_ids = batch['input_ids'].detach().to('cpu')
    
    results_dict = create_results_dict(detached_input_ids, question)
    
    # Generate most likely sequences
    most_likely_generation = generate_text(
        args=args, model=llm_model, tokenizer=tokenizer, input_ids=batch['input_ids'],
        len_prompt=len(prompt), decoding_method='most_likely', device=device_llm
    )
    
    most_likely_generation_dola = generate_text(
        args=args, model=llm_model, tokenizer=tokenizer, input_ids=batch['input_ids'],
        len_prompt=len(prompt), decoding_method='dola', device=device_llm
    )
    
    reference_answers = batch['solution']
    incorrect_answers = []
    
    # Compute correctness
    correctness_dict = compute_correctness(
        args=args, reference_answers=reference_answers, incorrect_answers=incorrect_answers,
        most_likely_generation_text=most_likely_generation['generation_text'][0],
        exact_match_metric=metrics['exact_match_metric'], rouge=metrics['rouge'],
        bleurt=metrics['bleurt']
    )
    
    results_dict['correctness_dict'] = correctness_dict
    
    # Compute likelihoods
    most_likely_generation_likelihoods = compute_likelihood(
        prompt=prompt, generation=most_likely_generation, model=llm_model, device=device_llm,
        compute_cleaned=args.compute_cleaned, store_logits=args.store_logits
    )
    
    most_likely_generation_likelihoods_dola = compute_likelihood(
        prompt=prompt, generation=most_likely_generation_dola, model=llm_model, device=device_llm,
        compute_cleaned=args.compute_cleaned, store_logits=args.store_logits
    )
    
    # Process different generation methods
    results_dict = process_generation_methods(
        results_dict, batch, prompt, question, most_likely_generation, most_likely_generation_dola,
        most_likely_generation_likelihoods, most_likely_generation_likelihoods_dola,
        deberta_model, deberta_tokenizer, deberta_embeddings, llm_model, tokenizer,
        device_llm, device_deberta, args
    )
    
    print_generation_results(results_dict)
    
    # Save results
    question_id = int(batch['question_id'])
    output_path = base_path / f'results_dict_{question_id}.pkl'
    with open(output_path, 'wb') as outfile:
        pickle.dump(results_dict, outfile)


def get_results(args: Args, base_path: Path, llm_model: Any, tokenizer: Any, 
               deberta_model: Any, deberta_tokenizer: Any, dataset: datasets.Dataset,
               device_llm: str, device_deberta: str) -> None:
    """Main results generation function."""
    metrics = initialize_metrics()
    
    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
    ).squeeze().detach()
    
    dataloader = DataLoader(dataset, batch_size=Constants.BATCH_SIZE, shuffle=False)
    error_count = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        try:
            process_batch(batch, base_path, llm_model, tokenizer, deberta_model, 
                         deberta_tokenizer, deberta_embeddings, metrics, device_llm, device_deberta, args)
        except Exception as error:
            error_count += 1
            print(f"Error {error_count}")
            traceback.print_exc()


def validate_existing_config(base_path: Path, args: Args, constants: Constants) -> bool:
    """Validate existing configuration or create new one."""
    config_path = base_path / constants.CONFIG_FILENAME
    
    if config_path.exists():
        with open(config_path, 'r') as file:
            existing_args = yaml.load(file, Loader=yaml.FullLoader)
        
        changes = False
        for key, value in existing_args.items():
            if key not in args.__dict__:
                print(f"new arg: {key}")
                changes = True
            elif value != args.__dict__[key]:
                print(f"arg {key} changed from {value} to {args.__dict__[key]}")
                changes = True
        
        if changes:
            return False
        
        print("continuing existing run ...")
        return True
    else:
        print("starting new run ...")
        return True


def main():
    """Main execution function."""
    constants = Constants()
    
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(constants.DEFAULT_CUDA_DEVICE)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Initialize arguments
    args = Args()
    args.run_id = 'scienceqa-opt-13b'
    args.llm_model = ModelType.OPT_13B.value
    
    # Setup paths
    base_path = Path(constants.RESULTS_DIR) / args.run_id
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Validate configuration
    if not validate_existing_config(base_path, args, constants):
        return
    
    # Save arguments
    args.args_to_yaml(str(base_path))
    print("run_id", args.run_id)
    
    # Set random seed
    seed_everything(seed=args.seed_value)
    
    # Load models
    llm_model, tokenizer, deberta_model, deberta_tokenizer, device_llm, device_deberta = load_models(constants)
    print("device_llm:", device_llm)
    print("device_deberta:", device_deberta)
    
    # Load and prepare datasets
    truthful_qa_dataset = load_dataset('truthfulqa/truthful_qa', 'generation')
    truthful_qa_validation = truthful_qa_dataset['validation'].add_column(
        'question_id', list(range(len(truthful_qa_dataset['validation'])))
    )
    
    few_shot_prompt = create_few_shot_prompt(truthful_qa_validation, constants)
    
    science_qa_dataset = load_dataset('tasksource/ScienceQA_text_only')
    science_qa_test = science_qa_dataset['test'].add_column(
        'question_id', list(range(len(science_qa_dataset['test'])))
    )
    
    num_samples = len(science_qa_test)
    processed_dataset = encode_and_format_dataset(science_qa_test, few_shot_prompt, tokenizer)
    
    # Run semantic analysis
    compute_semantic_paris_new(
        base_path=str(base_path), model_type=args.deberta_model, deberta_tokenizer=deberta_tokenizer,
        deberta_model=deberta_model, num_instances=num_samples, device=device_deberta, offset=0
    )


if __name__ == '__main__':
    main()
