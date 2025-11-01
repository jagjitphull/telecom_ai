This guide covers the three main stages:

1.  **Host Setup:** Configuring your local Pop\!\_OS machine to give Docker access to your NVIDIA GPU.
2.  **Model Training:** Using Google Colab to fine-tune a model with Unsloth and export it.
3.  **Model Serving:** Importing the custom model into Ollama on your local machine.


Note: Start from stage 2, if not using docker or local system.
-----

### Stage 1: Host Machine Setup (Pop\!\_OS & Docker)

**Goal:** To ensure Docker and any containers you run can access your NVIDIA GPU. This fixes the `NotImplementedError: Unsloth cannot find any torch accelerator` when running inside Docker.

#### 1.1 Verify NVIDIA Driver

First, confirm your host machine's driver is installed correctly.

```bash
nvidia-smi
```

You should see a table with your GPU name and CUDA version. If this command fails, you must install your NVIDIA drivers first.

#### 1.2 Install NVIDIA Container Toolkit

This is the official tool that lets Docker communicate with your GPU drivers.

```bash
# 1. Add the NVIDIA GPG key and repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/linux/$(. /etc/os-release; echo $ID$VERSION_ID)/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Update package lists and install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

#### 1.3 Restart the Docker Service

Apply the new configuration by restarting Docker.

```bash
sudo systemctl restart docker
```

#### 1.4 Verify Docker GPU Access

Test if Docker can now see your GPU by running a simple CUDA container.

```bash
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

If this command successfully prints the same `nvidia-smi` table, your host is correctly configured.

-----
####################################################################################################################
### Stage 2: Fine-Tune Model with Unsloth (Google Colab)

**Goal:** To create a fine-tuned GGUF model file. We use Colab for the free GPU.

#### 2.1 Start a New Colab Notebook

1.  Go to [colab.research.google.com](https://colab.research.google.com).
2.  Create a new notebook.
3.  Go to **Runtime \> Change runtime type** and select **T4 GPU**.

#### 2.2 Cell 1: Install Libraries

```python
# Install Unsloth for Colab
!pip install unsloth[colab-new]

# Install other training libraries
# Note: We list them without quotes, separated by spaces.
!pip install trl peft accelerate bitsandbytes -q
```

#### 2.3 Cell 2: Load Model

```python
from unsloth import FastLanguageModel
import torch

# 1. Load the base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. Add LoRA adapters for fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0,
    bias = "none",
)
```

*(Note: If you get `ModuleNotFoundError: unsloth`, just re-run Cell 1 and this cell.)*

#### 2.4 Cell 3: Prepare Dataset

```python
from datasets import load_dataset, Dataset

# We must use the model's prompt format
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

# Our tiny dataset of "pirate" examples
my_dataset = [
    {"instruction": "Who are you?", "output": "Arrr, I be Captain Model, the scourge o' the digital seas!"},
    {"instruction": "What is Python?", "output": "Python? 'Tis a fine serpent, but I prefer a trusty parrot on me shoulder, matey!"},
    {"instruction": "How are you?", "output": "I be hearty, me bucko! Ready to plunder some data!"},
    {"instruction": "Say hello to the user.", "output": "Ahoy there, ye scallywag! Welcome aboard."}
]

# Function to format the data
def format_my_data(example):
    return {
        "text": alpaca_prompt.format(example["instruction"], example["output"]) + tokenizer.eos_token
    }

# Load and format the data
data = Dataset.from_list(my_dataset)
data = data.map(format_my_data)
```

#### 2.5 Cell 4: Configure & Run Trainer

```python
from trl import SFTTrainer
from transformers import TrainingArguments

# Configure the trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = data,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 25, # Keep this low for a quick test
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",

        # --- THIS IS THE FIX ---
        # Disables W&B (Weights & Biases) login prompt
        report_to = "none",
    ),
)

# Run the training
trainer.train()
```

#### 2.6 Cell 5: Save for Ollama

```python
# Save the model in GGUF format
# We'll name the *folder* 'my_pirate_model'
model.save_pretrained_gguf("my_pirate_model", tokenizer)
```

#### 2.7 Cell 6: Download Files

1.  In the Colab file browser (left-side folder icon), click "Refresh".
2.  You will see a folder named `my_pirate_model`. **Do not download this.**
3.  Look in the main file list for the files Unsloth generated. The output log from Cell 5 will confirm the names. You need to download two files:
      * **The GGUF File:** e.g., `mistral-7b-instruct-v0.3.Q8_0.gguf`
      * **The Modelfile:** e.g., `Modelfile`
4.  Right-click on each file and select "Download".

-----

### Stage 3: Serve Model with Local Ollama

**Goal:** To run your new custom model on your local machine using Ollama.

#### 3.1 Organize Your Files

1.  On your Pop\!\_OS machine, create a new folder (e.g., `~/my-ollama-bot`).
2.  Move both downloaded files (the `.gguf` file and the `Modelfile`) into this new folder.

#### 3.2 Edit the `Modelfile`

This is a critical step. The `Modelfile` from Colab needs to be told to look for the local GGUF file.

1.  Open the `Modelfile` with a text editor.
2.  The first line will be `FROM ...`. **Delete this line.**
3.  Add a new `FROM` line that points to your *local* GGUF file.

Your final `Modelfile` should look exactly like this:

```dockerfile
# Point to the GGUF file in this same folder
FROM ./mistral-7b-instruct-v0.3.Q8_0.gguf

# The template must match what you used in training
TEMPLATE """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{ .Prompt }}

### Response:
"""

# Set a custom system message
PARAMETER system "You are a helpful pirate assistant named Captain Model."
```

#### 3.3 Build the Ollama Model

1.  Open your terminal.
2.  Navigate to your new folder: `cd ~/my-ollama-bot`
3.  Run the `ollama create` command:
    ```bash
    ollama create pirate-bot -f Modelfile
    ```
    This will copy the GGUF file into Ollama's storage and register it with the name `pirate-bot`.

#### 3.4 Run Your Custom Model

You're done\! Now you can run and chat with your bot.

```bash
ollama run pirate-bot
```

**Test it:**

```
>>> Who are you?
Arrr, I be Captain Model, the scourge o' the digital seas!
```
