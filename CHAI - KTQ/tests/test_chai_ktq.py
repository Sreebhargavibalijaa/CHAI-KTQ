#!/usr/bin/env python3
"""
CHAI-KTQ Test Suite
===================

Comprehensive tests for the CHAI-KTQ framework components.
"""

import unittest
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add Src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "Src"))

class TestCHAIKTQFramework(unittest.TestCase):
    """Test suite for CHAI-KTQ framework"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("Setting up CHAI-KTQ test environment...")
        
        # Set device
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {cls.device}")
        
        # Import modules
        try:
            from original_chai import apply_pruning, get_attention_scores, cluster_heads
            from chai_quant import chai_quant_enhancement, compute_sensitivity
            from chai_target import main_chai_target, get_attention_scores as target_get_attention_scores
            from chai_kd import chai_knowledgde_distillation_enhancement
            from app import CHAIKTQEvaluator
            
            cls.original_chai = apply_pruning
            cls.get_attention_scores = get_attention_scores
            cls.cluster_heads = cluster_heads
            cls.chai_quant_enhancement = chai_quant_enhancement
            cls.compute_sensitivity = compute_sensitivity
            cls.main_chai_target = main_chai_target
            cls.chai_kd_enhancement = chai_knowledgde_distillation_enhancement
            cls.CHAIKTQEvaluator = CHAIKTQEvaluator
            
            print("✅ All modules imported successfully")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            raise
    
    def setUp(self):
        """Set up for each test"""
        self.evaluator = self.CHAIKTQEvaluator()
    
    def test_01_model_loading(self):
        """Test model loading functionality"""
        print("\n🧪 Testing model loading...")
        
        try:
            model, tokenizer = self.evaluator.load_opt_classifier("facebook/opt-125m")
            
            self.assertIsNotNone(model)
            self.assertIsNotNone(tokenizer)
            self.assertEqual(model.config.num_labels, 2)
            
            print("✅ Model loading test passed")
            
        except Exception as e:
            self.fail(f"Model loading failed: {e}")
    
    def test_02_attention_scores_extraction(self):
        """Test attention scores extraction"""
        print("\n🧪 Testing attention scores extraction...")
        
        try:
            model, tokenizer = self.evaluator.load_opt_classifier("facebook/opt-125m")
            input_ids = torch.randint(0, 50256, (1, 32))
            
            attention_scores = self.get_attention_scores(model, input_ids)
            
            self.assertIsInstance(attention_scores, dict)
            self.assertGreater(len(attention_scores), 0)
            
            print("✅ Attention scores extraction test passed")
            
        except Exception as e:
            self.fail(f"Attention scores extraction failed: {e}")
    
    def test_03_head_clustering(self):
        """Test head clustering functionality"""
        print("\n🧪 Testing head clustering...")
        
        try:
            # Create dummy attention scores
            attention_scores = np.random.rand(12)  # 12 attention heads
            
            clustered_heads = self.cluster_heads(attention_scores, 6)
            
            self.assertIsInstance(clustered_heads, list)
            self.assertGreater(len(clustered_heads), 0)
            self.assertLessEqual(len(clustered_heads), len(attention_scores))
            
            print("✅ Head clustering test passed")
            
        except Exception as e:
            self.fail(f"Head clustering failed: {e}")
    
    def test_04_sensitivity_computation(self):
        """Test sensitivity computation"""
        print("\n🧪 Testing sensitivity computation...")
        
        try:
            # Create dummy attention scores
            attention_scores = {
                0: np.random.rand(12),
                1: np.random.rand(12),
                2: np.random.rand(12)
            }
            
            sensitivities = self.compute_sensitivity(attention_scores)
            
            self.assertIsInstance(sensitivities, dict)
            self.assertEqual(len(sensitivities), len(attention_scores))
            
            for layer_idx, sensitivity in sensitivities.items():
                self.assertIsInstance(sensitivity, (int, float))
                self.assertGreaterEqual(sensitivity, 0)
            
            print("✅ Sensitivity computation test passed")
            
        except Exception as e:
            self.fail(f"Sensitivity computation failed: {e}")
    
    def test_05_model_size_calculation(self):
        """Test model size calculation"""
        print("\n🧪 Testing model size calculation...")
        
        try:
            model, _ = self.evaluator.load_opt_classifier("facebook/opt-125m")
            
            size_mb = self.evaluator.get_model_size(model)
            
            self.assertIsInstance(size_mb, float)
            self.assertGreater(size_mb, 0)
            
            print(f"✅ Model size calculation test passed: {size_mb:.2f} MB")
            
        except Exception as e:
            self.fail(f"Model size calculation failed: {e}")
    
    def test_06_chai_base_pruning(self):
        """Test CHAI-Base pruning functionality"""
        print("\n🧪 Testing CHAI-Base pruning...")
        
        try:
            model, tokenizer = self.evaluator.load_opt_classifier("facebook/opt-125m")
            original_size = self.evaluator.get_model_size(model)
            
            # Apply pruning
            pruned_model = self.original_chai(model, tokenizer, "sst2")
            
            self.assertIsNotNone(pruned_model)
            
            # Check if model size is reduced (optional, as pruning might not always reduce size)
            pruned_size = self.evaluator.get_model_size(pruned_model)
            print(f"Original size: {original_size:.2f} MB, Pruned size: {pruned_size:.2f} MB")
            
            print("✅ CHAI-Base pruning test passed")
            
        except Exception as e:
            self.fail(f"CHAI-Base pruning failed: {e}")
    
    def test_07_chai_quant_enhancement(self):
        """Test CHAI-Quant enhancement"""
        print("\n🧪 Testing CHAI-Quant enhancement...")
        
        try:
            model, tokenizer = self.evaluator.load_opt_classifier("facebook/opt-125m")
            
            # Apply quantization
            quantized_model = self.chai_quant_enhancement(model, tokenizer, "sst2")
            
            self.assertIsNotNone(quantized_model)
            
            # Check if model is quantized (should be in half precision)
            if hasattr(quantized_model, 'dtype'):
                self.assertEqual(quantized_model.dtype, torch.float16)
            
            print("✅ CHAI-Quant enhancement test passed")
            
        except Exception as e:
            self.fail(f"CHAI-Quant enhancement failed: {e}")
    
    def test_08_chai_target_enhancement(self):
        """Test CHAI-Target enhancement"""
        print("\n🧪 Testing CHAI-Target enhancement...")
        
        try:
            model, tokenizer = self.evaluator.load_opt_classifier("facebook/opt-125m")
            
            # Apply targeted fine-tuning (with reduced epochs for testing)
            targeted_model = self.main_chai_target(model, tokenizer, "sst2", epochs=1)
            
            self.assertIsNotNone(targeted_model)
            
            print("✅ CHAI-Target enhancement test passed")
            
        except Exception as e:
            self.fail(f"CHAI-Target enhancement failed: {e}")
    
    def test_09_chai_kd_enhancement(self):
        """Test CHAI-KD enhancement"""
        print("\n🧪 Testing CHAI-KD enhancement...")
        
        try:
            student_model, tokenizer = self.evaluator.load_opt_classifier("facebook/opt-125m")
            teacher_model, _ = self.evaluator.load_opt_classifier("facebook/opt-125m")
            
            # Apply knowledge distillation
            distilled_model = self.chai_kd_enhancement(student_model, teacher_model, tokenizer, "sst2")
            
            self.assertIsNotNone(distilled_model)
            
            print("✅ CHAI-KD enhancement test passed")
            
        except Exception as e:
            self.fail(f"CHAI-KD enhancement failed: {e}")
    
    def test_10_evaluator_functionality(self):
        """Test evaluator functionality"""
        print("\n🧪 Testing evaluator functionality...")
        
        try:
            model, tokenizer = self.evaluator.load_opt_classifier("facebook/opt-125m")
            
            # Test evaluation with limited samples
            results = self.evaluator.evaluate_model(model, tokenizer, "sst2", max_samples=10)
            
            self.assertIsInstance(results, dict)
            self.assertIn('accuracy', results)
            self.assertIn('latency', results)
            self.assertIn('throughput', results)
            
            self.assertGreaterEqual(results['accuracy'], 0)
            self.assertLessEqual(results['accuracy'], 100)
            self.assertGreater(results['latency'], 0)
            self.assertGreater(results['throughput'], 0)
            
            print(f"✅ Evaluator test passed - Accuracy: {results['accuracy']:.2f}%, Latency: {results['latency']:.4f}s")
            
        except Exception as e:
            self.fail(f"Evaluator functionality failed: {e}")
    
    def test_11_benchmark_configuration(self):
        """Test benchmark configuration functionality"""
        print("\n🧪 Testing benchmark configuration...")
        
        try:
            # Test with minimal configuration
            results = self.evaluator.benchmark_configuration(
                "facebook/opt-125m", 
                "sst2", 
                ["chai-quant"]
            )
            
            self.assertIsInstance(results, dict)
            self.assertIn('baseline', results)
            self.assertIn('chai_base', results)
            self.assertIn('final', results)
            self.assertIn('improvements', results)
            
            print("✅ Benchmark configuration test passed")
            
        except Exception as e:
            self.fail(f"Benchmark configuration failed: {e}")
    
    def test_12_full_pipeline(self):
        """Test full CHAI-KTQ pipeline"""
        print("\n🧪 Testing full CHAI-KTQ pipeline...")
        
        try:
            # Test full pipeline with all enhancements
            results = self.evaluator.benchmark_configuration(
                "facebook/opt-125m", 
                "sst2", 
                ["chai-quant", "chai-target", "chai-kd"]
            )
            
            self.assertIsInstance(results, dict)
            self.assertIn('final', results)
            self.assertIn('applied_methods', results['final'])
            
            applied_methods = results['final']['applied_methods']
            self.assertIn('Quantization', applied_methods)
            self.assertIn('Targeted Fine-Tuning', applied_methods)
            self.assertIn('Knowledge Distillation', applied_methods)
            
            print("✅ Full pipeline test passed")
            
        except Exception as e:
            self.fail(f"Full pipeline failed: {e}")
    
    def test_13_error_handling(self):
        """Test error handling"""
        print("\n🧪 Testing error handling...")
        
        try:
            # Test with invalid model name
            with self.assertRaises(Exception):
                self.evaluator.load_opt_classifier("invalid/model")
            
            # Test with invalid dataset
            model, tokenizer = self.evaluator.load_opt_classifier("facebook/opt-125m")
            results = self.evaluator.evaluate_model(model, tokenizer, "invalid_dataset")
            self.assertIn('error', results)
            
            print("✅ Error handling test passed")
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")
    
    def test_14_performance_metrics(self):
        """Test performance metrics calculation"""
        print("\n🧪 Testing performance metrics...")
        
        try:
            model, tokenizer = self.evaluator.load_opt_classifier("facebook/opt-125m")
            
            # Test baseline evaluation
            baseline_results = self.evaluator.evaluate_model(model, tokenizer, "sst2", max_samples=5)
            
            # Test with one enhancement
            enhanced_model = self.chai_quant_enhancement(model, tokenizer, "sst2")
            enhanced_results = self.evaluator.evaluate_model(enhanced_model, tokenizer, "sst2", max_samples=5)
            
            # Verify metrics are reasonable
            for metric in ['accuracy', 'latency', 'throughput']:
                self.assertIn(metric, baseline_results)
                self.assertIn(metric, enhanced_results)
                self.assertGreaterEqual(baseline_results[metric], 0)
                self.assertGreaterEqual(enhanced_results[metric], 0)
            
            print("✅ Performance metrics test passed")
            
        except Exception as e:
            self.fail(f"Performance metrics test failed: {e}")


def run_tests():
    """Run all tests"""
    print("🚀 Starting CHAI-KTQ Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCHAIKTQFramework)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed!")
        return True
    else:
        print("\n⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 