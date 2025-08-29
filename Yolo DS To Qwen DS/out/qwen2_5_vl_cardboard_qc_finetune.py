#!/usr/bin/env python3
"""
Qwen2.5-VL (7B) Fine-tuning for Cardboard Quality Control
Modified from Unsloth's original notebook to work with cardboard QC dataset.

This script fine-tunes Qwen2.5-VL-7B on the cardboard quality control dataset
for automated quality assessment of cardboard bundles.
"""

# Installation (uncomment if needed)
# import os, re
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     # For Colab
#     import torch; v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
#     xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
#     !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
#     !pip install --no-deps unsloth

from unsloth import FastVisionModel
import torch

print("🔧 Setting up Qwen2.5-VL for Cardboard Quality Control Fine-tuning")
print("=" * 60)

# Load the base model
print("📥 Loading Qwen2.5-VL-7B model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use
    use_gradient_checkpointing = "unsloth", # For memory efficiency
)

# Add LoRA adapters for efficient fine-tuning
print("🔧 Adding LoRA adapters...")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,  # Fine-tune vision layers for cardboard images
    finetune_language_layers   = True,  # Fine-tune language for QC assessment
    finetune_attention_modules = True,  # Fine-tune attention layers
    finetune_mlp_modules       = True,  # Fine-tune MLP layers

    r = 16,           # LoRA rank - balance between accuracy and efficiency
    lora_alpha = 16,  # LoRA alpha - recommended to match r
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

print("✅ Model setup complete!")

# Load cardboard QC dataset
print("\n📊 Loading Cardboard Quality Control Dataset...")
from datasets import load_dataset

# Load your uploaded dataset
dataset = load_dataset("Cong2612/cardboard-qc-dataset")
print(f"✅ Dataset loaded successfully!")
print(f"   Train samples: {len(dataset['train'])}")
print(f"   Validation samples: {len(dataset['validation'])}")
print(f"   Test samples: {len(dataset['test'])}")

# Display example
example = dataset['train'][0]
print(f"\n📝 Example data point:")
print(f"   Filename: {example['filename']}")
print(f"   Label: {example['label']}")
print(f"   Reason: {example['reason']}")
print(f"   Image size: {example['image'].size}")
print(f"   Conversations: {len(example['conversations'])} messages")

# Data conversion function for cardboard QC
def convert_cardboard_to_conversation(sample):
    """
    Convert cardboard QC data to Unsloth conversation format.
    Input conversations are already in the right format, just need to restructure.
    """
    # Extract conversations from your dataset format
    conversations = sample["conversations"]
    
    # Convert to Unsloth format
    messages = []
    for msg in conversations:
        if msg["role"] == "user":
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": msg["content"]},
                    {"type": "image", "image": sample["image"]}
                ]
            })
        else:  # assistant
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": msg["content"]}
                ]
            })
    
    return {"messages": messages}

# Convert the training dataset
print("\n🔄 Converting dataset to training format...")
train_dataset = dataset['train']
converted_train = [convert_cardboard_to_conversation(sample) for sample in train_dataset]

# Also convert validation set for potential evaluation
val_dataset = dataset['validation']
converted_val = [convert_cardboard_to_conversation(sample) for sample in val_dataset]

print(f"✅ Dataset conversion complete!")
print(f"   Converted train samples: {len(converted_train)}")
print(f"   Converted validation samples: {len(converted_val)}")

# Display converted example
print(f"\n📋 Converted conversation example:")
print(f"   Messages: {len(converted_train[0]['messages'])}")
for i, msg in enumerate(converted_train[0]['messages']):
    print(f"   {i+1}. {msg['role']}: {len(msg['content'])} content items")
    if msg['content'][0]['type'] == 'text':
        print(f"      Text: {msg['content'][0]['text'][:100]}...")

# Test model before fine-tuning
print("\n🧪 Testing model before fine-tuning...")
FastVisionModel.for_inference(model)

# Test with a sample from the dataset
test_sample = train_dataset[0]
test_image = test_sample["image"]
test_instruction = "Analyze this cardboard bundle image. Is the bundle flat or warped? Provide your assessment."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": test_instruction}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

print("🔍 Model response before training:")
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs, 
    streamer=text_streamer, 
    max_new_tokens=128,
    use_cache=True, 
    temperature=1.5, 
    min_p=0.1
)

# Setup training
print("\n🏋️ Setting up training...")
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

# Training configuration optimized for cardboard QC
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_train,
    args=SFTConfig(
        per_device_train_batch_size=2,     # Adjust based on your GPU memory
        gradient_accumulation_steps=4,      # Effective batch size = 2 * 4 = 8
        warmup_steps=10,                   # Warm-up steps
        # max_steps=60,                    # Use this for quick testing
        num_train_epochs=3,                # Full training epochs for production
        learning_rate=2e-4,                # Learning rate
        logging_steps=5,                   # Log every 5 steps
        optim="adamw_8bit",                # 8-bit optimizer for memory efficiency
        weight_decay=0.01,                 # Regularization
        lr_scheduler_type="linear",        # Learning rate schedule
        seed=3407,                         # Reproducibility
        output_dir="outputs/cardboard_qc", # Output directory
        report_to="none",                  # Set to "wandb" if using Weights & Biases
        save_steps=50,                     # Save checkpoint every 50 steps
        save_total_limit=2,                # Keep only 2 checkpoints
        evaluation_strategy="no",          # Set to "steps" if you want evaluation
        # eval_steps=25,                   # Evaluate every 25 steps
        
        # Required for vision fine-tuning
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    ),
)

# Show memory stats
print("\n💾 Memory Statistics:")
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU: {gpu_stats.name}")
print(f"Max memory: {max_memory} GB")
print(f"Reserved memory: {start_gpu_memory} GB")

# Start training
print("\n🚀 Starting training...")
print("⏰ This will take some time depending on your hardware...")
trainer_stats = trainer.train()

# Show final stats
print("\n📈 Training Complete!")
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
print(f"Training time: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")
print(f"Peak memory: {used_memory} GB")
print(f"Peak memory for training: {used_memory_for_lora} GB")
print(f"Peak memory %: {used_percentage}%")
print(f"Training memory %: {lora_percentage}%")

# Test model after fine-tuning
print("\n🧪 Testing fine-tuned model...")
FastVisionModel.for_inference(model)

# Test with the same sample
inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

print("🔍 Model response after fine-tuning:")
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs, 
    streamer=text_streamer, 
    max_new_tokens=128,
    use_cache=True, 
    temperature=1.5, 
    min_p=0.1
)

# Save the model
print("\n💾 Saving fine-tuned model...")
model.save_pretrained("cardboard_qc_lora")
tokenizer.save_pretrained("cardboard_qc_lora")
print("✅ LoRA adapters saved to 'cardboard_qc_lora'")

# Optional: Push to Hugging Face Hub
push_to_hub = False  # Set to True if you want to upload
if push_to_hub:
    hub_model_name = "your_username/qwen2.5-vl-cardboard-qc"  # Change this
    hf_token = "your_token_here"  # Add your token
    
    print(f"\n📤 Pushing model to Hugging Face Hub: {hub_model_name}")
    model.push_to_hub(hub_model_name, token=hf_token)
    tokenizer.push_to_hub(hub_model_name, token=hf_token)
    print("✅ Model uploaded to Hugging Face Hub!")

# Save to 16bit for production deployment (optional)
save_16bit = False  # Set to True if you want 16bit model
if save_16bit:
    print("\n💾 Saving 16-bit merged model for production...")
    model.save_pretrained_merged("cardboard_qc_16bit", tokenizer)
    print("✅ 16-bit model saved to 'cardboard_qc_16bit'")

print("\n🎉 Fine-tuning Complete!")
print("=" * 60)
print("Your Qwen2.5-VL model has been fine-tuned for cardboard quality control!")
print("\n📁 Saved files:")
print("   • cardboard_qc_lora/ - LoRA adapters (small, efficient)")
if save_16bit:
    print("   • cardboard_qc_16bit/ - Full 16-bit model (for production)")

print("\n🚀 Next steps:")
print("   1. Test the model on validation data")
print("   2. Evaluate performance metrics")
print("   3. Deploy for production quality control")
print("   4. Monitor performance and retrain as needed")

print("\n📖 To load the fine-tuned model later:")
print("""
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="cardboard_qc_lora",
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)
""")