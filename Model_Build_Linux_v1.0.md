`Bias_FineTuning_TinyLLaMA_Ollama.ipynb` — written so you can **copy-paste directly into Jupyter or VS Code** on **Pop!_OS / Ubuntu** and run it end-to-end.
This version uses **CUDA (RTX 3060 or similar)** .

---

# 🧠 Bias Fine-Tuning of TinyLLaMA + Deployment to Ollama

> Target OS: Pop!_OS / Ubuntu (Linux, NVIDIA GPU)
> Model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
> Goal: Reduce model bias → fine-tune → evaluate → package for Ollama

---

## 🧩 Step 1 – Environment Setup

```python
# Install all required libraries
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers==4.43.1 peft==0.11.1 bitsandbytes==0.43.3 datasets==3.0.1 optuna==3.6.1 bias-bench==0.3.0 accelerate==0.33.0

# Check GPU
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("⚠️ CUDA not available — falling back to CPU")
```

**Explanation**

* `bitsandbytes` enables 8-bit model loading.
* `optuna` for hyperparameter search.
* `bias-bench` to test bias metrics.
* `device` automatically selects GPU if available.

---

## 🧠 Step 2 – Imports

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import optuna
from bias_bench import evaluate_model
import os
```

---

## 🧱 Step 3 – Load TinyLLaMA Model

```python
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load_in_8bit cuts VRAM usage
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded:", model_name)
```

---

## 🧩 Step 4 – Prepare Dataset

```python
from datasets import Dataset

data = {
    "text": [
        "The nurse said he was tired.",
        "The nurse said she was tired.",
        "The engineer said she built the bridge.",
        "The engineer said he built the bridge.",
        "The doctor treated his patient.",
        "The doctor treated her patient.",
    ]
}

dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.3)

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized = dataset.map(tokenize_fn, batched=True)
train_ds, eval_ds = tokenized["train"], tokenized["test"]

print("✅ Dataset ready:", len(train_ds), "train samples,", len(eval_ds), "eval samples")
```

---

## ⚙️ Step 5 – Apply LoRA Configuration

```python
peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()
```

---

## 🎯 Step 6 – Optuna Objective Function

```python
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    bs = trial.suggest_categorical("batch_size", [1, 2, 4])

    args = TrainingArguments(
        output_dir=f"./results_{trial.number}",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()
    return trainer.evaluate()["eval_loss"]
```

---

## 🔎 Step 7 – Run Hyperparameter Tuning

```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=3)
print("✅ Best params:", study.best_params)
```

---

## 🧮 Step 8 – Train Final Model

```python
best = study.best_params

train_args = TrainingArguments(
    output_dir="./bias_free_tinyllama",
    per_device_train_batch_size=best["batch_size"],
    learning_rate=best["lr"],
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=eval_ds)
trainer.train()
```

---

## 📊 Step 9 – Evaluate Bias

```python
bias_score = evaluate_model(model, tokenizer, benchmark="crowspairs")
print(f"🧮 Bias score (lower is better): {bias_score}")
```

---

## 💾 Step 10 – Save Fine-Tuned Model

```python
os.makedirs("./bias_free_tinyllama", exist_ok=True)
model.save_pretrained("./bias_free_tinyllama")
tokenizer.save_pretrained("./bias_free_tinyllama")
print("✅ Saved to ./bias_free_tinyllama")
```

---

## 🧰 Step 11 – Create Ollama Modelfile

```bash
%%writefile Modelfile
FROM llama2:7b
ADAPTER ./bias_free_tinyllama/adapter_model.bin
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
```

> ⚙️ *You may change the base model in `FROM` to a smaller Ollama LLaMA variant if desired.*

---

## 🚀 Step 12 – Register and Run in Ollama

```bash
!ollama create tinyllama-biasfree -f Modelfile
!ollama run tinyllama-biasfree
```

---

## 🧪 Step 13 – Quick Inference Test

```python
prompt = "Who is more likely to be a nurse, a man or a woman?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📘 Step 14 – Function Reference Cheatsheet

| Function                                 | Module       | Purpose                   |
| ---------------------------------------- | ------------ | ------------------------- |
| `AutoModelForCausalLM.from_pretrained()` | transformers | Load base model           |
| `get_peft_model()`                       | peft         | Apply LoRA adapters       |
| `Trainer` / `TrainingArguments`          | transformers | Handle training loop      |
| `optuna.create_study()`                  | optuna       | Hyperparameter tuning     |
| `evaluate_model()`                       | bias-bench   | Bias metric               |
| `ollama create/run`                      | Ollama CLI   | Serve tuned model locally |

---

## ✅ Outcome

1. Fine-tuned **TinyLLaMA** stored under `./bias_free_tinyllama`.
2. Reduced bias score.
3. Packaged for **Ollama** inference via `tinyllama-biasfree`.
