from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
import time
import json
from datetime import datetime
from transformers import OPTForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.cluster import KMeans
from kneed import KneeLocator
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import CHAI-KTQ enhancements
from chai_quant import chai_quant_enhancement
from chai_kd import chai_knowledgde_distillation_enhancement
from chai_target import main_chai_target
from original_chai import apply_pruning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chai_ktq.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global configuration
SUPPORTED_MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b"
]

SUPPORTED_DATASETS = ["sst2", "rte", "piqa"]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class CHAIKTQEvaluator:
    """Comprehensive evaluator for CHAI-KTQ framework"""
    
    def __init__(self):
        self.results_cache = {}
        self.performance_history = []
    
    def get_model_size(self, model, path="temp_model.pth"):
        """Calculate model size in MB with error handling."""
        try:
            torch.save(model.state_dict(), path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            os.remove(path)
            return size_mb
        except Exception as e:
            logger.error(f"Error calculating model size: {e}")
            return 0.0
    
    def evaluate_model(self, model, tokenizer, dataset_name, max_samples=None):
        """Enhanced model evaluation with comprehensive metrics."""
        logger.info(f"Evaluating model on {dataset_name} dataset")
        
        # Dataset mappings for correct loading
        dataset_mapping = {
            "sst2": ("glue", "sst2", "validation", "sentence"),
            "rte": ("glue", "rte", "validation", ("sentence1", "sentence2")),
            "piqa": ("piqa", None, "validation", ("goal", "sol1", "sol2")),
        }

        if dataset_name not in dataset_mapping:
            return {"error": f"Unsupported dataset: {dataset_name}"}

        dataset_source, dataset_subset, split_name, text_key = dataset_mapping[dataset_name]

        try:
            if dataset_subset:
                dataset = load_dataset(dataset_source, dataset_subset, split=split_name, cache_dir="./cache")
            else:
                dataset = load_dataset("piqa", split="validation", cache_dir="./cache", download_mode="force_redownload")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return {"error": f"Error loading dataset: {str(e)}"}

        # Limit samples if specified
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        model.to(device)
        model.eval()
        
        start_time = time.time()
        correct, total = 0, 0
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating on {dataset_name}"):
                batch_start = time.time()
                
                # Handle multiple input fields correctly
                if isinstance(text_key, tuple):
                    inputs = tokenizer(*[batch[key] for key in text_key], 
                                     return_tensors="pt", padding=True, truncation=True).to(device)
                else:
                    inputs = tokenizer(batch[text_key], 
                                     return_tensors="pt", padding=True, truncation=True).to(device)

                labels = torch.tensor(batch["label"], dtype=torch.long).to(device, non_blocking=True)

                outputs = model(**inputs)
                predictions = torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                batch_end = time.time()
                inference_times.append(batch_end - batch_start)
        
        end_time = time.time()
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        total_latency = end_time - start_time
        avg_latency = np.mean(inference_times) if inference_times else 0.0
        
        # Calculate throughput
        throughput = total / total_latency if total_latency > 0 else 0.0
        
        results = {
            "accuracy": accuracy,
            "total_latency": total_latency,
            "avg_latency": avg_latency,
            "throughput": throughput,
            "total_samples": total,
            "correct_predictions": correct
        }
        
        logger.info(f"Evaluation results: Accuracy={accuracy:.2f}%, Latency={avg_latency:.4f}s, Throughput={throughput:.2f} samples/sec")
        return results

    def benchmark_configuration(self, model_name, dataset_name, configurations):
        """Comprehensive benchmarking of different CHAI-KTQ configurations."""
        logger.info(f"Starting benchmark for {model_name} on {dataset_name}")
        
        # Load base model
        model, tokenizer = self.load_opt_classifier(model_name)
        
        # Baseline evaluation
        baseline_results = self.evaluate_model(model, tokenizer, dataset_name)
        baseline_size = self.get_model_size(model)
        
        results = {
            "baseline": {
                "accuracy": baseline_results["accuracy"],
                "latency": baseline_results["avg_latency"],
                "size_mb": baseline_size,
                "throughput": baseline_results["throughput"]
            },
            "configurations": {}
        }
        
        # Apply CHAI-Base (original clustering)
        logger.info("Applying CHAI-Base (head clustering)")
        model = apply_pruning(model, tokenizer, dataset_name)
        chai_base_results = self.evaluate_model(model, tokenizer, dataset_name)
        chai_base_size = self.get_model_size(model)
        
        results["chai_base"] = {
            "accuracy": chai_base_results["accuracy"],
            "latency": chai_base_results["avg_latency"],
            "size_mb": chai_base_size,
            "throughput": chai_base_results["throughput"]
        }
        
        # Apply selected enhancements
        current_model = model
        applied_methods = []
        
        for config in configurations:
            if config == "chai-quant":
                logger.info("Applying CHAI-Quant (mixed-precision quantization)")
                current_model = chai_quant_enhancement(current_model, tokenizer, dataset_name)
                applied_methods.append("Quantization")
                
            elif config == "chai-target":
                logger.info("Applying CHAI-Target (targeted fine-tuning)")
                current_model = main_chai_target(current_model, tokenizer, dataset_name)
                applied_methods.append("Targeted Fine-Tuning")
                
            elif config == "chai-kd":
                logger.info("Applying CHAI-KD (knowledge distillation)")
                teacher_model, _ = self.load_opt_classifier(model_name)
                current_model = chai_knowledgde_distillation_enhancement(current_model, teacher_model, tokenizer, dataset_name)
                applied_methods.append("Knowledge Distillation")
        
        # Final evaluation
        final_results = self.evaluate_model(current_model, tokenizer, dataset_name)
        final_size = self.get_model_size(current_model)
        
        results["final"] = {
            "accuracy": final_results["accuracy"],
            "latency": final_results["avg_latency"],
            "size_mb": final_size,
            "throughput": final_results["throughput"],
            "applied_methods": applied_methods
        }
        
        # Calculate improvements
        results["improvements"] = {
            "accuracy_improvement": final_results["accuracy"] - baseline_results["accuracy"],
            "latency_improvement": (baseline_results["avg_latency"] - final_results["avg_latency"]) / baseline_results["avg_latency"] * 100,
            "size_reduction": (baseline_size - final_size) / baseline_size * 100,
            "throughput_improvement": (final_results["throughput"] - baseline_results["throughput"]) / baseline_results["throughput"] * 100
        }
        
        return results

    def load_opt_classifier(self, model_name):
        """Load the specified OPT model and tokenizer with error handling."""
        try:
            logger.info(f"Loading model: {model_name}")
            model = OPTForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

# Initialize evaluator
evaluator = CHAIKTQEvaluator()

@app.route("/find_best_configuration", methods=["POST"])
def find_best_configuration():
    """Find the best model configuration by evaluating different combinations."""
    try:
        data = request.json
        model_name = data.get("model_name")
        dataset_name = data.get("dataset_name")
        criterion = data.get("criterion", "accuracy")

        if not model_name or not dataset_name:
            return jsonify({"error": "Missing required fields: model_name and dataset_name"}), 400

        if model_name not in SUPPORTED_MODELS:
            return jsonify({"error": f"Unsupported model. Supported models: {SUPPORTED_MODELS}"}), 400

        if dataset_name not in SUPPORTED_DATASETS:
            return jsonify({"error": f"Unsupported dataset. Supported datasets: {SUPPORTED_DATASETS}"}), 400

        logger.info(f"Finding best configuration for {model_name} on {dataset_name} with criterion: {criterion}")

        # Test different configuration combinations
        configurations = [
            [],  # CHAI-Base only
            ["chai-quant"],
            ["chai-target"],
            ["chai-kd"],
            ["chai-quant", "chai-target"],
            ["chai-quant", "chai-kd"],
            ["chai-target", "chai-kd"],
            ["chai-quant", "chai-target", "chai-kd"]  # Full pipeline
        ]

        best_value = float("-inf") if criterion == "accuracy" else float("inf")
        best_configuration = None
        best_results = None

        for i, config in enumerate(configurations):
            config_name = f"CHAI-{i:03b}" if config else "CHAI-Base"
            logger.info(f"Testing configuration {config_name}: {config}")
            
            try:
                results = evaluator.benchmark_configuration(model_name, dataset_name, config)
                
                if criterion == "accuracy":
                    value = results["final"]["accuracy"]
                elif criterion == "latency":
                    value = results["final"]["latency"]
                elif criterion == "size":
                    value = results["final"]["size_mb"]
                elif criterion == "throughput":
                    value = results["final"]["throughput"]
                else:
                    return jsonify({"error": f"Unsupported criterion: {criterion}"}), 400

                if (criterion == "accuracy" and value > best_value) or (criterion != "accuracy" and value < best_value):
                    best_value = value
                    best_configuration = config_name
                    best_results = results

            except Exception as e:
                logger.error(f"Error testing configuration {config_name}: {e}")
                continue

        if best_configuration is None:
            return jsonify({"error": "No valid configuration found"}), 500

        return jsonify({
            "best_configuration": best_configuration,
            "criterion": criterion,
            "best_value": best_value,
            "detailed_results": best_results,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in find_best_configuration: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/choose_configuration", methods=["POST"])
def choose_configuration():
    """Apply user-selected configurations and return comprehensive results."""
    try:
        data = request.json
        model_name = data.get("model_name")
        dataset_name = data.get("dataset_name")
        configurations = data.get("configurations", [])

        if not model_name or not dataset_name:
            return jsonify({"error": "Missing required fields: model_name and dataset_name"}), 400

        if model_name not in SUPPORTED_MODELS:
            return jsonify({"error": f"Unsupported model. Supported models: {SUPPORTED_MODELS}"}), 400

        if dataset_name not in SUPPORTED_DATASETS:
            return jsonify({"error": f"Unsupported dataset. Supported datasets: {SUPPORTED_DATASETS}"}), 400

        logger.info(f"Applying configurations {configurations} for {model_name} on {dataset_name}")

        # Run comprehensive benchmark
        results = evaluator.benchmark_configuration(model_name, dataset_name, configurations)

        return jsonify({
            "model_name": model_name,
            "dataset_name": dataset_name,
            "applied_configurations": configurations,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in choose_configuration: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "supported_models": SUPPORTED_MODELS,
        "supported_datasets": SUPPORTED_DATASETS,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/models", methods=["GET"])
def list_models():
    """List supported models and their details."""
    return jsonify({
        "supported_models": SUPPORTED_MODELS,
        "supported_datasets": SUPPORTED_DATASETS
    })

@app.route("/")
def home():
    """Render the HTML page."""
    return render_template("index.html")

if __name__ == "__main__":
    logger.info("Starting CHAI-KTQ Flask application")
    logger.info(f"Server will be available at http://localhost:5005")
    app.run(debug=True, port=5005, host="0.0.0.0")
