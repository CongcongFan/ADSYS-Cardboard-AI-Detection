---
license: cc-by-4.0
task_categories:
- image-classification
- visual-question-answering
tags:
- cardboard
- quality-control
- computer-vision
- manufacturing
- vision-language
size_categories:
- n<1K
language:
- en
pretty_name: "Cardboard Quality Control Dataset"
---

# Cardboard Quality Control Dataset

## Dataset Description

This dataset contains cardboard bundle images with quality assessments for training vision-language models to perform quality control in manufacturing environments.

### Dataset Summary

- **Total samples**: 168
- **Task**: Binary classification of cardboard bundles (Pass/Fail)
- **Format**: Vision-language conversation format
- **Image resolution**: 640x640 pixels
- **Labels**: Pass (bundle appears flat), Fail (bundle appears warped)

### Dataset Splits

| Split | Samples | Pass | Fail |
|-------|---------|------|------|
| Train | 117 | 75 | 42 |
| Validation | 16 | 14 | 2 |
| Test | 35 | 25 | 10 |

## Dataset Structure

### Data Fields

- `image`: PIL Image of cardboard bundle (640x640 RGB)
- `conversations`: List of conversation turns between user and assistant
- `filename`: Original image filename
- `label`: Binary label ("Pass" or "Fail")
- `reason`: Text explanation of the assessment

### Data Format

Each sample contains a conversation where:
- **User**: "Analyze this cardboard bundle image. Is the bundle flat or warped? Provide your assessment."
- **Assistant**: Provides assessment based on the label and reason

Example conversation:
```python
{
  "conversations": [
    {
      "role": "user",
      "content": "Analyze this cardboard bundle image. Is the bundle flat or warped? Provide your assessment."
    },
    {
      "role": "assistant", 
      "content": "The bundle appears flat. Bundle appears flat"
    }
  ]
}
```

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("cardboard-qc-dataset")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"] 
test_data = dataset["test"]

# Example usage
sample = train_data[0]
image = sample["image"]  # PIL Image
conversation = sample["conversations"]  # List of messages
label = sample["label"]  # "Pass" or "Fail"
```

## Dataset Creation

This dataset was created from:
- **Images**: Overlay images from cardboard bundle photography
- **Labels**: Quality control assessments in CSV format
- **Processing**: Automated conversation generation for vision-language training

### Data Collection

- Images captured from cardboard manufacturing environment
- Quality assessments performed by human experts
- Labels indicate whether bundles meet flatness requirements for packaging

### Annotation Process

- Binary classification: Pass (flat) vs Fail (warped)
- Expert reasoning provided for each assessment
- Conversation format generated automatically for model training

## Intended Use

### Primary Use Cases

1. **Quality Control Automation**: Train models to automatically assess cardboard bundle quality
2. **Vision-Language Model Training**: Fine-tune multimodal models for manufacturing inspection
3. **Computer Vision Research**: Benchmark for industrial quality control tasks

### Limitations

- Limited to cardboard bundle assessment
- Binary classification only (no severity grading)
- Single camera angle/lighting condition
- Small dataset size (168 samples)

## Ethics and Bias

- Dataset represents specific manufacturing environment
- May not generalize to different lighting/camera setups
- No personal or sensitive information included
- Focus on industrial/manufacturing use case

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{cardboard_qc_2024,
  title={Cardboard Quality Control Dataset},
  author={ADSYS Manufacturing},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/cardboard-qc-dataset}
}
```

## License

This dataset is released under CC BY 4.0 License.

## Contact

For questions about this dataset, please open an issue in the repository.
