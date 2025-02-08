from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import numpy as np
import os
from transformers import OPTForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.cluster import KMeans
from kneed import KneeLocator
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_opt_classifier(model_name):
    """Load the specified OPT model and tokenizer."""
    model = OPTForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
def get_optimal_clusters(attention_scores):
    """ Determines optimal clusters for attention heads using the Elbow Method. """
    num_heads = len(attention_scores)
    if num_heads <= 10:
        return num_heads
    max_clusters = max(num_heads - int(0.2 * num_heads), num_heads * 4 // 5)
    errors = []
    cluster_range = range(1, max_clusters + 1)
    for num_clusters in cluster_range:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        kmeans.fit(attention_scores.reshape(-1, 1))
        errors.append(kmeans.inertia_)
    elbow = KneeLocator(cluster_range, errors, curve="convex", direction="decreasing")
    return max(num_heads - int(0.2 * num_heads), elbow.elbow if elbow.elbow else max_clusters)

def get_model_size(model, path="temp_model.pth"):
    import torch
    """Calculate model size in MB."""
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size_mb
def enforce_head_constraint(num_heads, embed_dim):
    """ Adjusts number of heads to ensure divisibility with embedding dimension. """
    while embed_dim % num_heads != 0:
        num_heads -= 1
    return num_heads

def load_opt_classifier():
    model = OPTForSequenceClassification.from_pretrained("facebook/opt-350m", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    return model, tokenizer

def get_attention_scores(model, input_ids):
    """Extracts attention scores from the model while ensuring correct dimensions."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)  # ✅ Move input IDs to GPU
    model.to(device)
    
    attention_scores = {}

    with torch.no_grad():
        outputs = model(input_ids)

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            for layer_idx, attn in enumerate(outputs.attentions):
                attn = attn.cpu().numpy()  # ✅ Move to CPU
                if attn.ndim == 4:  # Expected shape: (batch_size, num_heads, seq_length, seq_length)
                    attn = np.mean(attn, axis=(0, 2, 3))  # ✅ Average across heads and sequences
                elif attn.ndim == 3:  # Unexpected case, still averaging
                    attn = np.mean(attn, axis=(0, 2))
                elif attn.ndim == 2:  # More unexpected cases
                    attn = np.mean(attn, axis=0)
                
                attention_scores[layer_idx] = attn  # ✅ Store correctly processed attention scores

        else:
            return {"error": "Model does not return attention scores. Check model architecture."}

    return attention_scores


def cluster_heads(attention_scores, num_clusters):
    """ Clusters attention heads while ensuring correct shape. """
    num_heads = len(attention_scores)

    if num_heads <= 10:
        return list(range(num_heads))

    attention_scores = np.array(attention_scores).reshape(-1, 1)  # ✅ Flatten for clustering

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    kmeans.fit(attention_scores)

    labels = kmeans.labels_
    cluster_representatives = []

    for cluster_idx in range(num_clusters):
        indices = np.where(labels == cluster_idx)[0]
        if len(indices) > 0:
            keep_count = max(1, len(indices) * 5 // 10)  # ✅ Pruning 50% of heads per cluster
            cluster_representatives.extend(indices[:keep_count])

    return sorted(cluster_representatives)

def prune_attention_heads(model, clustered_heads):
    """ Prunes attention heads while ensuring correct embedding dimensions. """
    for layer_idx, heads_to_keep in enumerate(clustered_heads):
        attn_layer = model.model.decoder.layers[layer_idx].self_attn

        # ✅ Ensure valid number of heads per layer
        original_num_heads = attn_layer.num_heads
        new_num_heads = enforce_head_constraint(len(heads_to_keep), attn_layer.embed_dim)

        # ✅ Update number of heads
        attn_layer.num_heads = new_num_heads

        # ✅ Ensure Q, K, V projections match new number of heads
        head_dim = attn_layer.embed_dim // original_num_heads
        new_embed_dim = new_num_heads * head_dim

        attn_layer.q_proj = nn.Linear(attn_layer.embed_dim, new_embed_dim, bias=False)
        attn_layer.k_proj = nn.Linear(attn_layer.embed_dim, new_embed_dim, bias=False)
        attn_layer.v_proj = nn.Linear(attn_layer.embed_dim, new_embed_dim, bias=False)

        # ✅ Ensure output projection layer matches new size
        attn_layer.out_proj = nn.Linear(new_embed_dim, attn_layer.embed_dim, bias=False)

    return model

def divide_layers_by_sensitivity(sensitivities):
    """ Splits layers into 3 groups (High, Medium, Low) based on sensitivity scores. """
    sorted_layers = sorted(sensitivities, key=sensitivities.get, reverse=True)
    num_layers = len(sorted_layers)
    high = sorted_layers[: int(num_layers * 0.2)]
    medium = sorted_layers[int(num_layers * 0.2) : int(num_layers * 0.7)]
    low = sorted_layers[int(num_layers * 0.7) :]
    return high, medium, low

def apply_mixed_precision(model, medium, low):
    """ Applies mixed precision quantization to the model. """
    for layer_idx in medium + low:
        for param in model.model.decoder.layers[layer_idx].parameters():
            param.data = param.data.half()
    model.half()
    return model

def compute_sensitivity(attention_scores):
    cleaned_scores = {
        layer_idx: np.mean(np.abs(np.array(scores, dtype=np.float32)))
        for layer_idx, scores in attention_scores.items()
        if isinstance(scores, (list, np.ndarray))
    }
    return cleaned_scores

# -------------------- Evaluation -------------------- #
def evaluate_model(model, tokenizer):
    """ Evaluates model accuracy on PIQA dataset (multiple-choice task). """
    dataset = load_dataset("piqa", split="validation[:100]")
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for example in dataset:
            prompt = example["goal"]
            choices = [example["sol1"], example["sol2"]]
            inputs = tokenizer([prompt + " " + choice for choice in choices], return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            logits = outputs.logits.squeeze()
            predicted_choice = torch.argmax(logits).item()
            correct += (predicted_choice == example["label"])
            total += 1

    return (correct / total) * 100

# -------------------- Main Execution -------------------- #
def get_model_size(model, path="temp_model.pth"):
    """ Saves model temporarily and checks disk size. """
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)  # Convert bytes to MB
    os.remove(path)  # ✅ Clean up after measurement
    return size_mb

def evaluate_model(model, tokenizer, dataset_name):
    """Evaluates model accuracy on the given dataset."""
    dataset_mapping = {
        "sst2": ("glue", "sst2", "sentence"),
        "piqa": ("piqa", "train", "goal"),
        "rte": ("glue", "rte", "sentence1"),
    }

    if dataset_name not in dataset_mapping:
        return {"error": f"Unsupported dataset: {dataset_name}"}

    dataset_source, dataset_subset, text_key = dataset_mapping[dataset_name]

    try:
        dataset = load_dataset(dataset_source, dataset_subset, split="train", trust_remote_code=True)
    except Exception as e:
        return {"error": f"Error loading dataset: {str(e)}"}

    model.eval()
    correct, total = 0, 0
    start_time = time.time()

    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(example[text_key], return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions.item() == example["label"])
            total += 1

    end_time = time.time()
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    latency = end_time - start_time
    return accuracy, latency

def apply_pruning(model):
    """Simulates pruning by reducing the model size."""
    for param in model.parameters():
        param.data = param.data * (param.data > 0.01).float()
    return model
def chai_quant_enhancement(chai_base_model,tokenizer,dataset_name):
    # ✅ Save & Reload to Apply Pruning
    # ✅ Measure Model Size After Pruning
    # size_chai_base = get_model_size(chai_base_model)

    # # ✅ Evaluate Model After CHAI-Base
    # accuracy_after_chai_base = evaluate_model(chai_base_model, tokenizer,dataset_name)
    # accuracy_drop_chai_base = accuracy_before - accuracy_after_chai_base

    # # ✅ Count Clustered Heads After CHAI-Base
    # total_heads_after_chai_base = sum(len(heads) for heads in clustered_heads)

    # print(f"📦 Model Size After CHAI-Base (Pruning): {size_chai_base:.2f} MB")
    # print(f"🎯 Accuracy After CHAI-Base: {accuracy_after_chai_base:.2f}%")
    # print(f"📉 Accuracy Drop (CHAI-Base): {accuracy_drop_chai_base:.2f}%")
    # print(f"🔢 Clustered Heads After CHAI-Base: {total_heads_after_chai_base}\n")
    input_ids = torch.randint(0, 50256, (1, 32))
    attention_scores = get_attention_scores(chai_base_model, input_ids)
    sensitivities = compute_sensitivity(attention_scores)
    high, medium, low = divide_layers_by_sensitivity(sensitivities)

    # ✅ Apply Mixed Precision Quantization (CHAI-Quant)
    print("\n🚀 Applying Mixed Precision Quantization (CHAI-Quant)...")
    chai_quant_model = apply_mixed_precision(chai_base_model, medium, low)
    return chai_quant_model
