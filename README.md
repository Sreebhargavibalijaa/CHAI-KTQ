# CHAI-KTQ: A Novel Framework for Efficient and Scalable Large Language Models built in colloboration with FAIR - META Research Team

## 📋 Overview

CHAI-KTQ is an enhanced framework that builds upon the original CHAI (Clustering Heads for Attention-based Inference) research, introducing three key innovations: **Quantization (CHAI-Quant)**, **Targeted Fine-Tuning (CHAI-Target)**, and **Knowledge Distillation (CHAI-KD)**. This unified pipeline achieves significant improvements in memory efficiency, latency, and accuracy while maintaining the core benefits of the original CHAI approach.

## 🚀 Key Enhancements Over Original CHAI

### Original CHAI Framework
- **Runtime Head Clustering**: Dynamically clusters attention heads during inference
- **Memory Reduction**: ~15% reduction in memory usage
- **No Full Model Retraining**: Preserves original model weights

### CHAI-KTQ Enhancements

#### 1. **CHAI-Quant (Mixed-Precision Quantization)**
- **Post-clustering quantization** to further reduce KV cache size
- **Sensitivity-aware layer division** (High/Medium/Low sensitivity)
- **Mixed-precision application** to medium and low sensitivity layers
- **Memory reduction**: Additional 42.8% beyond original CHAI

#### 2. **CHAI-Target (Targeted Fine-Tuning)**
- **Sensitivity-aware fine-tuning** on top 30% most sensitive layers
- **Improved robustness** without full model retraining
- **Layer-specific optimization** based on attention score analysis
- **Accuracy improvement**: 22-24% increase over baseline CHAI

#### 3. **CHAI-KD (Knowledge Distillation)**
- **Structured knowledge distillation** from large teacher models
- **Temperature-scaled distillation** with configurable parameters
- **Minimal accuracy loss** with significant model compression
- **Inference speed**: 3× improvement (3000 inferences/sec for 125M models)

## 📊 Performance Improvements

### Benchmark Results (OPT-350M Model)

| Metric | Original CHAI | CHAI-KTQ | Improvement |
|--------|---------------|----------|-------------|
| **Memory Usage** | ~15% reduction | 57.8% reduction | +42.8% |
| **Latency** | 6.4ms | 1.99ms | 69% faster |
| **SST2 Accuracy** | 52% | 74-76% | +22-24% |
| **Inference Speed** | 1000 inf/sec | 3000 inf/sec | 3× faster |

### Full Pipeline Performance (111 Configuration)
- **76% accuracy** on SST2 dataset
- **50% reduction** in latency and memory
- **No full model retraining** required
- **Maintains CHAI's original strengths**

## 🏗️ Architecture

```
Original Model
     ↓
CHAI-Base (Head Clustering & Pruning)
     ↓
CHAI-Quant (Mixed-Precision Quantization)
     ↓
CHAI-Target (Targeted Fine-Tuning)
     ↓
CHAI-KD (Knowledge Distillation)
     ↓
Optimized CHAI-KTQ Model
```

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install torch transformers datasets scikit-learn kneed tqdm flask
```

### Quick Start
```bash
cd Src
python app.py
```

The application will be available at `http://localhost:5005`

## 📁 Project Structure

```
Src/
├── app.py                 # Main Flask application with API endpoints
├── original_chai.py       # Original CHAI implementation (head clustering)
├── chai_quant.py         # CHAI-Quant enhancement (mixed-precision quantization)
├── chai_target.py        # CHAI-Target enhancement (targeted fine-tuning)
├── chai_kd.py           # CHAI-KD enhancement (knowledge distillation)
└── results.ipynb        # Jupyter notebook for analysis and visualization
```

## 🔧 API Endpoints

### 1. Find Best Configuration
```bash
POST /find_best_configuration
{
    "model_name": "facebook/opt-350m",
    "dataset_name": "sst2",
    "criterion": "accuracy"
}
```

### 2. Apply Custom Configuration
```bash
POST /choose_configuration
{
    "model_name": "facebook/opt-350m",
    "dataset_name": "sst2",
    "configurations": ["chai-quant", "chai-target", "chai-kd"]
}
```

## 🎯 Usage Examples

### Basic Usage
```python
from chai_quant import chai_quant_enhancement
from chai_target import main_chai_target
from chai_kd import chai_knowledgde_distillation_enhancement
from original_chai import apply_pruning

# Load model
model, tokenizer = load_opt_classifier("facebook/opt-350m")

# Apply CHAI-Base (original clustering)
model = apply_pruning(model, tokenizer, "sst2")

# Apply CHAI-Quant
model = chai_quant_enhancement(model, tokenizer, "sst2")

# Apply CHAI-Target
model = main_chai_target(model, tokenizer, "sst2")

# Apply CHAI-KD
teacher_model, _ = load_opt_classifier("facebook/opt-350m")
model = chai_knowledgde_distillation_enhancement(model, teacher_model, tokenizer, "sst2")
```

### Advanced Configuration
```python
# Custom parameters for CHAI-Target
model = main_chai_target(
    model, 
    tokenizer, 
    "sst2",
    epochs=5,
    batch_size=32,
    learning_rate=1e-5
)
```

## 📈 Supported Models & Datasets

### Models
- **OPT-125M**: Lightweight model for fast inference
- **OPT-350M**: Balanced model for accuracy/speed trade-off
- **Extensible**: Framework supports other transformer models

### Datasets
- **SST2**: Stanford Sentiment Treebank (sentiment analysis)
- **RTE**: Recognizing Textual Entailment (natural language inference)
- **PIQA**: Physical Intelligence Question Answering (commonsense reasoning)

## 🔬 Technical Details

### Head Clustering Algorithm
- **Elbow Method**: Automatic determination of optimal cluster count
- **K-means Clustering**: Groups similar attention heads
- **50% Pruning**: Removes redundant heads per cluster

### Sensitivity Analysis
- **Attention Score Analysis**: Computes layer sensitivity based on attention patterns
- **Top 30% Selection**: Targets most sensitive layers for fine-tuning
- **Three-tier Division**: High/Medium/Low sensitivity layers for quantization

### Knowledge Distillation
- **Temperature Scaling**: Configurable temperature (default: 2.0)
- **Alpha Balancing**: Weighted combination of distillation and task loss
- **Structured Distillation**: Preserves model architecture while compressing knowledge

## 📊 Evaluation Metrics

The framework provides comprehensive evaluation across multiple dimensions:

1. **Accuracy**: Task-specific performance on target datasets
2. **Latency**: Inference time per sample
3. **Memory Usage**: Model size and memory footprint
4. **Throughput**: Inferences per second
5. **Trade-off Analysis**: Multi-dimensional performance visualization

## 🎨 Visualization Features

- **Comparative Heatmaps**: Performance comparison across configurations
- **Radar Charts**: Multi-dimensional metric visualization
- **3D Trade-off Plots**: Accuracy-Latency-Memory optimization space
- **Progress Tracking**: Real-time monitoring of optimization steps

## 🔍 Research Contributions

This work extends the original CHAI research with:

1. **Novel Quantization Strategy**: Mixed-precision approach based on layer sensitivity
2. **Targeted Fine-tuning**: Layer-specific optimization without full retraining
3. **Enhanced Knowledge Distillation**: Structured compression with minimal accuracy loss
4. **Unified Pipeline**: Seamless integration of multiple optimization techniques
5. **Comprehensive Evaluation**: Multi-dimensional performance analysis

## 📚 References

- **Original CHAI Paper**: [CHAI: Clustering Heads for Attention-based Inference](https://arxiv.org/pdf/2403.08058)
- **CHAI-KTQ Research**: Currently under review at Transactions on Machine Learning Research (TMLR)
- **Kaggle Repository**: [CHAI-KTQ Research](https://www.kaggle.com/chai-ktq-research)

## 🤝 Acknowledgments

We extend our gratitude to the original CHAI research team at Facebook Research for laying the groundwork that made this extension possible. Their innovative approach to runtime head clustering provided the foundation for our enhancements.

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🐛 Issues & Contributions

For bug reports, feature requests, or contributions, please open an issue or submit a pull request on the project repository.

---

**Note**: This implementation is based on research currently under review. Results may vary based on hardware, model versions, and specific use cases. 
