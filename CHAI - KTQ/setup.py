#!/usr/bin/env python3
"""
CHAI-KTQ Setup Script
=====================

This script sets up the CHAI-KTQ framework with all necessary dependencies
and configurations for optimal performance.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_banner():
    """Print the CHAI-KTQ banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    CHAI-KTQ Framework                        ║
    ║              Efficient and Scalable LLMs                     ║
    ║                                                              ║
    ║  🚀 Quantization + Targeted Fine-tuning + Knowledge Distillation ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    dependencies = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "scikit-learn>=1.3.0",
        "kneed>=0.8.3",
        "tqdm>=4.65.0",
        "flask>=2.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    print("✅ All dependencies installed successfully")
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "cache",
        "logs",
        "models",
        "results",
        "Src/templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}/")

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  Checking hardware...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU detected. CPU will be used (slower performance)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed. GPU check skipped.")
        return False

def download_models():
    """Download pre-trained models"""
    print("\n🤖 Downloading pre-trained models...")
    
    models = [
        "facebook/opt-125m",
        "facebook/opt-350m"
    ]
    
    try:
        from transformers import AutoTokenizer, OPTForSequenceClassification
        
        for model_name in models:
            print(f"Downloading {model_name}...")
            try:
                # Download tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.save_pretrained(f"models/{model_name.split('/')[-1]}")
                
                # Download model
                model = OPTForSequenceClassification.from_pretrained(model_name, num_labels=2)
                model.save_pretrained(f"models/{model_name.split('/')[-1]}")
                
                print(f"✅ {model_name} downloaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to download {model_name}: {e}")
                print("Models will be downloaded automatically when first used")
                
    except ImportError:
        print("⚠️  Transformers not installed. Model download skipped.")

def create_config_file():
    """Create configuration file"""
    print("\n⚙️  Creating configuration file...")
    
    config_content = """# CHAI-KTQ Configuration File

[General]
# Enable/disable logging
enable_logging = true
log_level = INFO
log_file = logs/chai_ktq.log

# Model settings
default_model = facebook/opt-350m
default_dataset = sst2
batch_size = 16
max_length = 512

[Optimization]
# CHAI-Quant settings
quantization_enabled = true
mixed_precision = true

# CHAI-Target settings
targeted_fine_tuning = true
sensitivity_threshold = 0.3
learning_rate = 2e-5
epochs = 3

# CHAI-KD settings
knowledge_distillation = true
temperature = 2.0
alpha = 0.5

[Evaluation]
# Evaluation settings
evaluation_samples = 1000
metrics = ["accuracy", "latency", "memory", "throughput"]

[API]
# Flask API settings
host = 0.0.0.0
port = 5005
debug = true
"""
    
    with open("config.ini", "w") as f:
        f.write(config_content)
    
    print("✅ Configuration file created: config.ini")

def run_tests():
    """Run basic tests"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        import torch
        import transformers
        import datasets
        import sklearn
        print("✅ All core libraries imported successfully")
        
        # Test model loading
        from transformers import AutoTokenizer, OPTForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        model = OPTForSequenceClassification.from_pretrained("facebook/opt-125m", num_labels=2)
        print("✅ Model loading test passed")
        
        # Test Flask app
        import sys
        sys.path.append('Src')
        from app import app
        print("✅ Flask app import test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 CHAI-KTQ Setup Complete!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Start the application:")
    print("   cd Src")
    print("   python app.py")
    print("\n2. Open your browser and go to:")
    print("   http://localhost:5005")
    print("\n3. Run the analysis notebook:")
    print("   jupyter notebook results.ipynb")
    print("\n4. Check the documentation:")
    print("   README.md")
    
    print("\n🔧 Available Endpoints:")
    print("   • GET  /health - Health check")
    print("   • GET  /models - List supported models")
    print("   • POST /find_best_configuration - Find optimal configuration")
    print("   • POST /choose_configuration - Apply custom configuration")
    
    print("\n📊 Key Features:")
    print("   • CHAI-Quant: Mixed-precision quantization")
    print("   • CHAI-Target: Targeted fine-tuning")
    print("   • CHAI-KD: Knowledge distillation")
    print("   • Comprehensive evaluation metrics")
    print("   • Interactive web interface")
    
    print("\n🚀 Performance Improvements:")
    print("   • 57.8% memory reduction")
    print("   • 69% latency improvement")
    print("   • 3× throughput increase")
    print("   • 76% accuracy on SST2")
    
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check GPU
    check_gpu()
    
    # Download models (optional)
    download_models()
    
    # Create config file
    create_config_file()
    
    # Run tests
    if run_tests():
        print("✅ All tests passed")
    else:
        print("⚠️  Some tests failed, but setup can continue")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 