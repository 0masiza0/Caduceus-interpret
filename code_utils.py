from collections import defaultdict
import numpy as np

def sum_of_windows_np(arr):
    # Создаем 2D массив с окнами размером 5xN, где каждый столбец - это окно
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=7)
    
    # Шаг 2: суммируем каждое окно по строкам (оси 1) и возвращаем результат
    window_sums = np.sum(windows[::1, :], axis=1)
    
    return window_sums
    
def top_attr_score(all_scores, all_seqs, k, n_top):

    counts = defaultdict(int)
    score_sums = defaultdict(float)
    
    for scores, seq in zip(all_scores, all_seqs):
        sums = sum_of_windows_np(scores)
        for i, score in enumerate(sums):
            kmer = seq[i:i+k]
            counts[kmer] += 1
            score_sums[kmer] += score
            
    importances = defaultdict(float)
    for key in counts:
        importances[key] = score_sums[key] / counts[key]
        
    result = [(k, v, counts[k]) for k, v in sorted(importances.items(), key=lambda item: item[1])][-n_top:][::-1]
    print(f'Top-{n_top} most important {k}mers:')
    #print(result[0])
    return result
    
    
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch.nn as nn
import torch

def load_model(path=None):
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    #tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForMaskedLM.from_pretrained(model_name, 
                                                        trust_remote_code=True)
    masked_lm_model_path = model_name
    #tokenizer = AutoTokenizer.from_pretrained(masked_lm_model_path)
    masked_lm_model = AutoModelForMaskedLM.from_pretrained(masked_lm_model_path)
    num_labels = 2
    
    classification_model = AutoModelForSequenceClassification.from_pretrained(masked_lm_model_path, num_labels=num_labels)
    
    classification_model.score = nn.Sequential(classification_model.score, nn.Softmax(dim=-1))

    if path is not None:
        classification_model.load_state_dict(torch.load(path, weights_only=True))
        classification_model.eval()
    return classification_model#, tokenizer
    
from datasets import load_dataset, get_dataset_config_names
from transformers import DataCollatorWithPadding

def preprocess_function(tokenizer, examples):
  # just truncate right, but for some tasks symmetric truncation from left and right is more reasonable
  # set max_length to 128 tokens to make experiments faster
    return tokenizer(examples["sequence"], truncation=True, max_length=128)

def load_gue(add_special_tokens=False):
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    masked_lm_model_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(masked_lm_model_path)
    
    config_names = get_dataset_config_names("leannmlindsey/GUE")
    dataset = load_dataset("leannmlindsey/GUE", name="prom_core_all")
    
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["sequence"], 
            #truncation=True, 
            #max_length=128,
            add_special_tokens=add_special_tokens
        ),
        batched=True
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenizer, dataset, tokenized_dataset, data_collator
