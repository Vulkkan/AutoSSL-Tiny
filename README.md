## AutoSSL-Tiny: 
### Ultra-Lightweight Neural Architecture Search for Self-Supervised Learning

AutoSSL-Tiny is a computationally efficient neural architecture search (NAS) framework designed for self-supervised learning on resource-constrained devices. It discovers optimal micro-architectures under strict computational budgets (<1M parameters, <100M FLOPs) using differentiable architecture search (DARTS) with bilevel optimization.

### ✨ Key Features

1. Ultra-Lightweight Design: Target <1M parameters and <100M FLOPs for edge deployment
2. Differentiable Architecture Search: End-to-end trainable NAS with continuous relaxation
3. Self-Supervised Learning: Barlow Twins objective for representation learning without labels
4. Memory-Efficient: Gradient accumulation and minimal data loading overhead
5. Comprehensive Evaluation: Benchmarks against MobileNetV2/V3, EfficientNet-B0
6. Data Efficiency: Optimized for low-data regimes (1-20% labeled data)

### 📊 Performance Highlights

| Model                  | Parameters | FLOPs | CIFAR-10 (10% data) | CIFAR-100 (500 samples) |
|------------------------|------------|-------|-----------------------|---------------------------|
| AutoSSL-Tiny           | ~0.8M      | ~80M  | ~85%                  | ~45%                      |
| MobileNetV2 (0.35x)    | ~0.9M      | ~95M  | ~82%                  | ~40%                      |
| MobileNetV3-Small      | ~1.1M      | ~110M | ~84%                  | ~42%                      |
| EfficientNet-B0 (Tiny) | ~1.2M      | ~120M | ~86%                  | ~44%                      |


### 🏗️ Architecture Components

#### Core Building Blocks

```python
# Lightweight Squeeze-and-Excitation
class SqueezeExcite(nn.Module):
    """Channel attention with minimal parameters"""
    
# Depthwise Separable Convolution  
class TinyDSConv(nn.Module):
    """Efficient conv with optional SE attention"""
    
# Inverted Residual Block  
class TinyIRB(nn.Module):
    """MobileNetV2-style block with expansion factors"""
    
# Differentiable Search Operations
TINY_OPS = {
    'ds_conv': TinyDSConv,
    'ds_conv_se': TinyDSConv with SE,
    'irb_e3': TinyIRB (3× expansion),
    'irb_e6': TinyIRB (6× expansion),
    'se': Standalone SE,
    'identity': Skip connection,
    'zero': Zero operation
}
```

#### Search Space Cell

```python
class Cell(nn.Module):
    """DARTS search cell with 2 nodes × 2 edges"""
    # Mixed operations per edge
    # Continuous architecture parameters (α)
    # Normal and reduction cell variants
```

#### Searchable Model

```python
class DARTSAutoSSL(nn.Module):
    """Main searchable architecture with bilevel optimization"""
    # Stem convolution
    # Stack of search cells
    # Continuous architecture parameters
    # Projection head for SSL
```

### Training Pipeline

#### 1. Architecture Search (Self-Supervised):

```python
# Bilevel optimization: weights on train, architecture on validation
ssl_model = train_darts_ssl(epochs=20)
```

#### 2. Evaluation Model Creation:

```python
from autossl_tiny import create_autossl_eval_model

eval_model = create_autossl_eval_model(ssl_model, num_classes=10)
```

#### 3. Downstream Fine-tuning:

```python
from autossl_tiny import finetune_classifier

classifier, accuracy = finetune_classifier(
    ssl_model,
    data_fraction=0.1,  # Use only 10% labeled data
    epochs=50
)
```

### 📈 Comprehensive Evaluation

#### Data Efficiency Analysis

```python
from autossl_tiny import PaperModelEvaluator

evaluator = PaperModelEvaluator(
    data_fractions=[0.01, 0.05, 0.1, 0.2],
    epochs=20
)

# Compare against baselines
results = evaluator.evaluate_cifar10_data_efficiency(ssl_model)

# Generate plots and tables
evaluator.plot_comprehensive_results(results)
```

#### Extreme Low-Data Regimes

```python
# CIFAR-100 with only 200-1000 samples
cifar100_results = evaluator.get_cifar100_extreme_low_data(
    n_samples_list=[200, 500, 1000]
)
```

### 🔧 Advanced Configuration

#### Model Scaling

```python
# Adjust model capacity
model = DARTSAutoSSL(
    C=5,           # Base channels (smaller = more efficient)
    num_cells=4    # Number of search cells
)

# Expand channels for higher capacity
model.expand_channels(width_multiplier=1.5)
```

#### Custom Search Space

```python
# Define custom operations
CUSTOM_OPS = {
    'my_conv': MyCustomConv,
    'my_attention': MyAttentionBlock,
}

# Use in search
cell = Cell(C_prev=16, C=16, op_names=list(CUSTOM_OPS.keys()))
```

#### Training Options

```python
# Gradient accumulation for memory efficiency
train_darts_ssl(
    epochs=30,
    accum_steps=8,      # Effective batch size = batch_size × accum_steps
    batch_size=16,
    subset_size=5000,   # Subset of unlabeled data
    lam=0.005,          # Barlow Twins λ parameter
)
```

### 🎯 Design Principles

1. Minimalism: Every operation optimized for parameter/FLOP efficiency
2. Differentiability: End-to-end trainable architecture search
3. Self-Supervision: Learn from unlabeled data for data efficiency
4. Practicality: Real-world constraints (memory, computation, data scarcity)

### 📄 License

This project is licensed under the CC license. 

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

### 📧 Contact

For questions or feedback, please open an issue or contact:

· victoriano.3996@gmail.com 
· https://github.com/vulkkan/autossl-tiny

---

AutoSSL-Tiny: Making architecture search accessible for resource-constrained environments.

