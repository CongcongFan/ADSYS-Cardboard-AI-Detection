# Memory Optimization Guide for LoRA Operations

This guide explains the memory-efficient solutions for handling LoRA merge operations when encountering memory allocation errors.

## Problem

You may encounter this error during LoRA merge:
```
RuntimeError: [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 4345298944 bytes.
```

## Solutions

### 1. Memory-Efficient Merge (Recommended)

Use the new memory-efficient merge script:

```bash
python 02_merge_lora_memory_efficient.py
```

**Features:**
- Gradient checkpointing to reduce memory usage
- Sequential layer processing
- Aggressive memory cleanup between operations
- Automatic fallback to CPU mode if GPU memory insufficient
- Memory usage monitoring and logging

**Options:**
- `--use-cpu`: Force CPU-only processing (slower but uses less GPU memory)
- `--force-merge`: Bypass memory checks and attempt merge anyway

### 2. Direct LoRA Inference (Fastest Setup)

Skip the merge entirely and use LoRA adapter directly:

```bash
python 06_direct_lora_inference.py
```

**Benefits:**
- No merge required - uses much less memory
- Faster setup time
- Can run on systems with limited memory
- Supports interactive chat and single-prompt inference

**Usage Examples:**
```bash
# Interactive chat mode
python 06_direct_lora_inference.py

# Single prompt with image
python 06_direct_lora_inference.py --prompt "Analyze this cardboard" --image image.jpg

# Test model functionality
python 06_direct_lora_inference.py --test-model
```

### 3. System Memory Analysis

Check your system's capability before starting:

```bash
python 07_memory_check.py --quick-check
```

**Full Analysis:**
```bash
python 07_memory_check.py
```

**Features:**
- RAM, GPU, and disk space analysis
- Specific recommendations for your system
- Memory stress testing
- Real-time memory monitoring

## Updated Setup Script

The main setup script now includes automatic memory analysis:

```bash
# Automatic mode (analyzes memory and chooses best approach)
python run_full_setup.py

# Force direct inference mode
python run_full_setup.py --use-direct-inference

# Force memory-efficient merge
python run_full_setup.py --use-memory-efficient-merge --use-cpu

# Skip memory analysis
python run_full_setup.py --skip-memory-check
```

## Memory Requirements

### LoRA Merge
- **Minimum:** 8GB RAM, 4GB GPU memory
- **Recommended:** 16GB RAM, 8GB GPU memory
- **Disk:** 20-50GB free space

### Direct Inference
- **Minimum:** 4GB RAM, 2GB GPU memory
- **Recommended:** 8GB RAM, 4GB GPU memory
- **Disk:** 10-20GB free space

## Troubleshooting

### Memory Error During Merge
1. Try memory-efficient merge with CPU mode:
   ```bash
   python 02_merge_lora_memory_efficient.py --use-cpu
   ```

2. If still failing, use direct inference:
   ```bash
   python 06_direct_lora_inference.py
   ```

### Low System Memory
1. Close other applications
2. Use direct inference mode
3. Enable CPU mode for all operations

### GPU Out of Memory
1. Use `--use-cpu` flag
2. Close GPU-intensive applications
3. Consider direct inference mode

## New Command Line Options

### run_full_setup.py
- `--skip-memory-check`: Skip automatic memory analysis
- `--use-direct-inference`: Use direct LoRA inference instead of merge
- `--use-memory-efficient-merge`: Use memory-efficient merge script
- `--force-merge`: Force merge even with low memory warnings
- `--auto-continue`: Skip user prompts for automation

### Memory-Efficient Scripts
- `02_merge_lora_memory_efficient.py`: Memory-optimized merge
- `06_direct_lora_inference.py`: Direct LoRA inference engine
- `07_memory_check.py`: System analysis and monitoring

## Windows-Specific Notes

- Scripts automatically handle Windows paths and Unicode issues
- Memory monitoring works with Windows Task Manager integration
- CPU mode is often more reliable on Windows systems
- Consider using PowerShell or Command Prompt for better Unicode support

## Performance Comparison

| Method | Setup Time | Memory Usage | Disk Usage | Ollama Ready |
|--------|------------|--------------|------------|--------------|
| Original Merge | 20-30 min | High | High | Yes |
| Memory-Efficient Merge | 25-35 min | Medium | High | Yes |
| Direct Inference | 2-5 min | Low | Low | No* |

*Direct inference doesn't create Ollama-compatible files but provides immediate model usage

## Best Practices

1. **Always run memory check first** to understand your system's capabilities
2. **Use direct inference for development** and testing
3. **Use memory-efficient merge for production** deployment
4. **Monitor memory usage** during operations
5. **Close other applications** before running memory-intensive operations

## Getting Help

If you continue to experience memory issues:

1. Run comprehensive system analysis:
   ```bash
   python 07_memory_check.py --stress-test
   ```

2. Try the most conservative approach:
   ```bash
   python 06_direct_lora_inference.py --use-cpu
   ```

3. Check the generated recommendations from memory analysis

## Summary

The memory optimization features provide multiple pathways to use your fine-tuned LoRA model:

- **Best for immediate use:** Direct LoRA inference
- **Best for deployment:** Memory-efficient merge
- **Best for analysis:** Memory check and monitoring

Choose the approach that best fits your system capabilities and use case requirements.