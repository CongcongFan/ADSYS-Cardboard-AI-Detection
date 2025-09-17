#!/usr/bin/env python3
"""
Evaluation script for the fine-tuned Qwen2.5-VL cardboard QC model
Tests the model on validation and test sets and provides performance metrics.
"""

import torch
from unsloth import FastVisionModel
from datasets import load_dataset
from transformers import TextStreamer
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
from tqdm import tqdm

def load_finetuned_model(model_path="cardboard_qc_lora"):
    """Load the fine-tuned model."""
    print(f"🔧 Loading fine-tuned model from {model_path}...")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def predict_single_sample(model, tokenizer, image, instruction="Analyze this cardboard bundle image. Is the bundle flat or warped? Provide your assessment."):
    """Make a prediction on a single image."""
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            temperature=0.1,  # Lower temperature for consistent evaluation
            min_p=0.05,
            do_sample=True
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after the prompt)
    prompt_len = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    generated_text = response[prompt_len:].strip()
    
    return generated_text

def extract_prediction(response_text):
    """
    Extract Pass/Fail prediction from the model's response.
    Uses keyword matching to determine the prediction.
    """
    response_lower = response_text.lower()
    
    # Look for clear indicators
    if 'pass' in response_lower and 'fail' not in response_lower:
        return "Pass"
    elif 'fail' in response_lower and 'pass' not in response_lower:
        return "Fail"
    elif 'flat' in response_lower and 'warp' not in response_lower:
        return "Pass"
    elif 'warp' in response_lower and 'flat' not in response_lower:
        return "Fail"
    elif 'good' in response_lower or 'acceptable' in response_lower:
        return "Pass"
    elif 'bad' in response_lower or 'unacceptable' in response_lower:
        return "Fail"
    else:
        # If unclear, check for more indicators
        pass_indicators = ['appear flat', 'looks flat', 'seems flat', 'is flat', 'quality good']
        fail_indicators = ['appear warped', 'looks warped', 'seems warped', 'is warped', 'quality poor']
        
        pass_count = sum(1 for indicator in pass_indicators if indicator in response_lower)
        fail_count = sum(1 for indicator in fail_indicators if indicator in response_lower)
        
        if pass_count > fail_count:
            return "Pass"
        elif fail_count > pass_count:
            return "Fail"
        else:
            return "Unknown"  # Ambiguous response

def evaluate_dataset(model, tokenizer, dataset, split_name="validation", max_samples=None):
    """Evaluate the model on a dataset split."""
    print(f"\n📊 Evaluating on {split_name} set...")
    
    split_data = dataset[split_name]
    
    if max_samples:
        split_data = split_data.select(range(min(max_samples, len(split_data))))
    
    predictions = []
    true_labels = []
    detailed_results = []
    
    print(f"Processing {len(split_data)} samples...")
    
    for i, sample in enumerate(tqdm(split_data)):
        try:
            # Get prediction
            response = predict_single_sample(model, tokenizer, sample['image'])
            predicted_label = extract_prediction(response)
            true_label = sample['label']
            
            predictions.append(predicted_label)
            true_labels.append(true_label)
            
            # Store detailed result
            detailed_results.append({
                'filename': sample['filename'],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'model_response': response,
                'correct': predicted_label == true_label
            })
            
            # Print some examples
            if i < 3:
                print(f"\n📝 Example {i+1}:")
                print(f"   File: {sample['filename']}")
                print(f"   True: {true_label}")
                print(f"   Predicted: {predicted_label}")
                print(f"   Response: {response[:100]}...")
                
        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
            predictions.append("Error")
            true_labels.append(sample['label'])
    
    return predictions, true_labels, detailed_results

def calculate_metrics(predictions, true_labels):
    """Calculate evaluation metrics."""
    # Filter out unknown/error predictions for main metrics
    valid_indices = [i for i, pred in enumerate(predictions) if pred in ["Pass", "Fail"]]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]
    
    if not valid_predictions:
        print("❌ No valid predictions found!")
        return {}
    
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Valid predictions: {len(valid_predictions)}/{len(predictions)}")
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Classification report
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(valid_true_labels, valid_predictions))
    
    # Confusion matrix
    print(f"\n🔍 Confusion Matrix:")
    cm = confusion_matrix(valid_true_labels, valid_predictions)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'valid_predictions': len(valid_predictions),
        'total_predictions': len(predictions),
        'classification_report': classification_report(valid_true_labels, valid_predictions, output_dict=True),
        'confusion_matrix': cm.tolist()
    }

def save_evaluation_results(results, metrics, filename="evaluation_results.json"):
    """Save evaluation results to a file."""
    output_data = {
        'metrics': metrics,
        'detailed_results': results,
        'summary': {
            'total_samples': len(results),
            'correct_predictions': sum(1 for r in results if r['correct']),
            'accuracy': metrics.get('accuracy', 0)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to {filename}")

def main():
    """Main evaluation function."""
    print("🧪 Cardboard QC Model Evaluation")
    print("=" * 50)
    
    # Load the fine-tuned model
    try:
        model, tokenizer = load_finetuned_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you have run the fine-tuning script first!")
        return
    
    # Load the dataset
    print("\n📊 Loading dataset...")
    dataset = load_dataset("Cong2612/cardboard-qc-dataset")
    
    # Evaluate on validation set
    val_predictions, val_true_labels, val_results = evaluate_dataset(
        model, tokenizer, dataset, "validation"
    )
    
    val_metrics = calculate_metrics(val_predictions, val_true_labels)
    save_evaluation_results(val_results, val_metrics, "validation_results.json")
    
    # Evaluate on test set
    print(f"\n" + "="*50)
    test_predictions, test_true_labels, test_results = evaluate_dataset(
        model, tokenizer, dataset, "test"
    )
    
    test_metrics = calculate_metrics(test_predictions, test_true_labels)
    save_evaluation_results(test_results, test_metrics, "test_results.json")
    
    # Summary
    print(f"\n🎯 Final Summary:")
    print(f"=" * 50)
    print(f"Validation Accuracy: {val_metrics.get('accuracy', 0):.3f}")
    print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.3f}")
    
    # Show some error cases
    print(f"\n❌ Sample Incorrect Predictions:")
    incorrect_cases = [r for r in test_results if not r['correct']][:3]
    for i, case in enumerate(incorrect_cases):
        print(f"{i+1}. {case['filename']}")
        print(f"   True: {case['true_label']}, Predicted: {case['predicted_label']}")
        print(f"   Response: {case['model_response'][:80]}...")
    
    print(f"\n✅ Evaluation complete!")
    print(f"📁 Results saved to validation_results.json and test_results.json")

if __name__ == "__main__":
    main()