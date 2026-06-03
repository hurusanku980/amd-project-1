import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3-8b-hf")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    # ROCm GPU check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=32, target_modules=["q_proj","v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("tatsu-lab/alpaca", split="train[:10000]")

    def tokenize(ex):
        return tokenizer(ex["instruction"] + "\n" + ex["input"] + "\n" + ex["output"],
                        truncation=True, max_length=512, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="./output", num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr, fp16=True, gradient_checkpointing=True,
        logging_steps=50, save_strategy="epoch", optim="adamw_torch"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained("./output/final")

if __name__ == "__main__":
    main()
