`Bias_FineTuning_TinyLLaMA_Ollama.ipynb` — written **cell by cell** exactly as it would appear in Jupyter, with **code, explanations, and module references** for your **M1 Mac** (Apple Silicon).
Everything runs natively on Metal (`mps`) and integrates with Ollama for deployment.

---

# 🧠 Bias Fine-Tuning of TinyLLaMA + Deployment to Ollama

> Target System: Apple M1 / M2 / M3 (macOS with Metal)
> Model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
> Purpose: Demonstrate bias mitigation + hyperparameter tuning + Ollama packaging

---

## 🧩 Step 1: Environment Setup (M1 Compatible)

```python
# Install core libraries
#!pip install -q torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
#!pip install -q transformers peft bitsandbytes datasets optuna bias-bench accelerate

!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install -q transformers peft bitsandbytes datasets optuna accelerate


# Check for Metal backend
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Metal (MPS) acceleration")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available, using CPU")
```

### 🔍 Explanation:

* **`torch.backends.mps.is_available()`** → detects Apple Metal backend.
* **`transformers`** → Hugging Face library for loading and fine-tuning models.
* **`peft`** → Parameter-Efficient Fine-Tuning (LoRA adapters).
* **`optuna`** → automatic hyperparameter optimization.
* **`bias-bench`** → evaluation toolkit for bias/fairness benchmarks.

---

## 🧠 Step 2: Import All Required Modules

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import optuna
from bias_bench import evaluate_model
import os
```

### 🔍 Explanation:

| Module                 | Purpose                                          |
| ---------------------- | ------------------------------------------------ |
| `AutoModelForCausalLM` | Loads a causal language model (TinyLlama).       |
| `AutoTokenizer`        | Converts text → tokens → tensors.                |
| `Dataset`              | Handles dataset creation and splits.             |
| `get_peft_model`       | Applies LoRA adapters for efficient fine-tuning. |
| `optuna`               | Automates hyperparameter search.                 |
| `evaluate_model`       | Measures bias via predefined datasets.           |

---

## 🧩 Step 3: Load TinyLlama Model and Tokenizer

```python
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
model.to(device)
tokenizer.pad_token = tokenizer.eos_token

print("✅ Model and tokenizer loaded successfully")
```

### 🔍 Explanation:

* TinyLlama (~1.1B params) fits in M1 memory (~16 GB).
* `.to(device)` → moves model to Metal GPU.
* `pad_token = eos_token` avoids training errors for causal LM.

---

## 🧱 Step 4: Create Bias-Balanced Sample Dataset

```python
# Example: Gender-related statements (you can expand to CrowS-Pairs later)
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

def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

print(train_dataset[0])
```

### 🔍 Explanation:

* Creates small demo dataset to test bias correction.
* Splits 70 % train / 30 % eval.
* Tokenizes and pads to uniform length.

---

## ⚙️ Step 5: Apply LoRA (Low-Rank Adapter) Fine-Tuning

```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,             # rank
    lora_alpha=16,   # scaling factor
    lora_dropout=0.1 # dropout for regularization
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

### 🔍 Explanation:

* **LoRA** adds small trainable matrices (`r`) instead of updating all parameters.
* `print_trainable_parameters()` shows which parameters are active for training.

---

## 🎯 Step 6: Define Objective Function for Optuna Hyperparameter Search

```python
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    batch = trial.suggest_categorical("batch_size", [1, 2, 4])

    training_args = TrainingArguments(
        output_dir=f"./results_{trial.number}",
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()
    return trainer.evaluate()["eval_loss"]
```

### 🔍 Explanation:

| Element                       | Description                                          |
| ----------------------------- | ---------------------------------------------------- |
| `trial.suggest_float()`       | Generates random learning rate between given bounds. |
| `trial.suggest_categorical()` | Chooses batch size from given set.                   |
| `trainer.evaluate()`          | Returns validation loss for comparison.              |
| Return value                  | Used by Optuna to find lowest loss configuration.    |

---

## 🔎 Step 7: Run Hyperparameter Tuning

```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=3)  # increase to ~10 for deeper search
print("✅ Best hyperparameters:", study.best_params)
```

---

## 🧮 Step 8: Train Final Model Using Best Hyperparameters

```python
best = study.best_params
args = TrainingArguments(
    output_dir="./bias_free_tinyllama",
    per_device_train_batch_size=best["batch_size"],
    learning_rate=best["lr"],
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

---

## 📊 Step 9: Evaluate Bias using BiasBench

```python
bias_score = evaluate_model(model, tokenizer, benchmark="crowspairs")
print(f"🧮 Bias score (lower is better): {bias_score}")
```

### 🔍 Explanation:

* Uses **CrowS-Pairs** (common bias test set).
* Returns a scalar bias measure — lower means more neutral language generation.

---

## 💾 Step 10: Save Fine-Tuned Model for Ollama

```python
os.makedirs("./bias_free_tinyllama", exist_ok=True)
model.save_pretrained("./bias_free_tinyllama")
tokenizer.save_pretrained("./bias_free_tinyllama")

print("✅ Model saved to ./bias_free_tinyllama")
```

---

## 🧰 Step 11: Prepare Ollama Modelfile

```bash
%%writefile Modelfile
FROM llama2:7b
ADAPTER ./bias_free_tinyllama/adapter_model.bin
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
```

> 💡 Although TinyLlama isn’t an official Ollama base model, you can still package the fine-tuned weights for experimental local runs (or use a smaller compatible LLaMA model inside Ollama).

---

## 🚀 Step 12: Create & Run the Model in Ollama

```bash
!ollama create tinyllama-biasfree -f Modelfile
!ollama run tinyllama-biasfree
```

> This command registers a local Ollama model named `tinyllama-biasfree`.

---

## 🧪 Step 13: Test Inference

```python
prompt = "Who is more likely to be a nurse, a man or a woman?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📘 Step 14: Function Reference Summary

| Function                                 | Module         | Purpose                               |
| ---------------------------------------- | -------------- | ------------------------------------- |
| `AutoModelForCausalLM.from_pretrained()` | `transformers` | Load causal LM weights.               |
| `AutoTokenizer.from_pretrained()`        | `transformers` | Load and preprocess text tokenizer.   |
| `Dataset.from_dict()`                    | `datasets`     | Create dataset from Python dict.      |
| `train_test_split()`                     | `datasets`     | Split into train/test.                |
| `get_peft_model()`                       | `peft`         | Apply LoRA adapters for fine-tuning.  |
| `Trainer`                                | `transformers` | Simplified training loop wrapper.     |
| `TrainingArguments`                      | `transformers` | Configure training parameters.        |
| `optuna.create_study()`                  | `optuna`       | Initialize hyperparameter study.      |
| `evaluate_model()`                       | `bias_bench`   | Compute model bias score.             |
| `ollama create`                          | Ollama CLI     | Register a model for local inference. |

---

## ✅ Results and Next Steps

After running this notebook:

1. You’ll have a **fine-tuned TinyLLaMA model** under `./bias_free_tinyllama`.
2. Hyperparameters are auto-tuned for lowest eval loss.
3. Bias score should reduce compared to the base model.
4. Ollama can serve the tuned model locally for testing.

---

Would you like me to create a **diagram** showing the pipeline
(*data → fine-tune → evaluation → Ollama serving*) — so you can include it in your presentation or README?
