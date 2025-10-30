Examples of how to use the Ollama command line, from managing models to interacting with them using various prompts.

### 📦 Model Management

First, you need to manage the models on your system.

**1. Pull (Download) a Model**
You need to pull a model before you can run it. The "tag" (e.g., `latest`, `7b`, `13b-chat`) is optional but recommended. If you don't provide one, `latest` is assumed.

```bash
# Pull the latest Llama 3 8B model
ollama pull llama3

# Pull a specific coding model
ollama pull codellama:7b

# Pull the Phi-3 mini model
ollama pull phi-3-mini

# Pull a multimodal (vision) model
ollama pull llava
```

**2. List Your Local Models**
This command shows all the models you have downloaded, their size, and when they were last used.

```bash
ollama list
```

**3. Remove a Model**
If you want to free up disk space, you can remove a model.

```bash
ollama rm codellama:7b
```

-----

### 🗣️ Interacting with Models

There are two primary ways to interact with a model: in an interactive chat session or by sending a single, non-interactive prompt.

**1. Interactive Chat (Most Common)**
Use `ollama run [model_name]`. This starts a chat session where the model remembers the context of your conversation.

```bash
ollama run llama3
```

Once you are in the session, you can type your prompts.

```
>>> What is the capital of France?
Paris is the capital of France.

>>> What is it famous for?
Paris is famous for many things, including the Eiffel Tower, the Louvre Museum, its art and culture, and its cuisine.

>>> /bye
```

(Type `/bye` or press `Ctrl+D` to exit the session).

**2. Single Prompt (Good for Scripting)**
If you just want a single answer without a back-and-forth chat, provide the prompt directly on the command line.

```bash
ollama run llama3 "What is the capital of France?"
```

The model will output its response directly to your shell and then exit.

-----

### ✨ Examples: Different Models & Prompts

Here’s how you can use different types of models for various tasks.

#### Example 1: General Q\&A & Creativity (using `llama3`)

`llama3` is an excellent all-purpose model for general knowledge, conversation, and creative tasks.

**Task: Simple Question**

```bash
ollama run llama3 "Explain the difference between HTTP and HTTPS in one paragraph."
```

**Task: Creative Writing**

```bash
ollama run llama3 "Write a short, 4-line poem about a computer dreaming."
```

**Task: Summarization**

```bash
ollama run llama3 "Summarize this article: [paste a long block of text here]"
```

#### Example 2: Code Generation & Explanation (using `codellama`)

`codellama` is fine-tuned specifically for programming tasks.

**Task: Write a Function**

```bash
ollama run codellama "Write a Python function to check if a string is a palindrome."
```

**Task: Explain Code**

```bash
ollama run codellama "Explain what this line of Bash code does: find . -name '*.txt' -delete"
```

**Task: Debugging (in interactive mode)**

```bash
ollama run codellama
>>> I have this Python code, but it's giving me an error:
def add(a, b)
  return a + b

>>>
The error is a syntax error. You are missing a colon ':' at the end of your function definition. It should be:
def add(a, b):
  return a + b
```

#### Example 3: Lightweight & Fast Tasks (using `phi-3-mini`)

`phi-3-mini` is a smaller model, making it very fast. It's great for simpler tasks like data extraction or quick summarization.

**Task: Data Extraction**

```bash
ollama run phi-3-mini "Extract the name and email address from this text: 'The user, John Doe, can be reached at john.doe@example.com for further questions.'"
```

**Task: Classification**

```bash
ollama run phi-3-mini "Classify this email as 'Spam' or 'Important': 'Subject: You've won! Click here to claim your prize!'"
```

#### Example 4: Vision (Multimodal) (using `llava`)

`llava` can understand both text and images. You provide a text prompt and one or more image paths.

**Task: Describe an Image**

```bash
# This command runs in non-interactive mode.
# Note that the image path is provided at the end.
ollama run llava "Describe what is happening in this image." /path/to/my-image.jpg
```

**Task: Interactive Q\&A about an Image**
You can also run it in interactive mode to ask follow-up questions.

```bash
ollama run llava
```

```
>>> What do you see in this picture? /Users/me/Downloads/my_pet.png
This is an image of a golden retriever dog sitting on a green lawn.

>>> What color is the dog's collar?
The dog is wearing a red collar.
```

-----

### 🚀 Bonus: Useful In-Chat Commands

When you are inside an interactive session (after `ollama run [model_name]`), you can use these commands:

  * **/set system [prompt]**: Change the system prompt to alter the model's behavior.
    ```
    >>> /set system You are a helpful assistant who always responds like a pirate.
    System prompt set.

    >>> What is the capital of France?
    Ahoy! The capital of France be Paris, matey!
    ```
  * **/show info**: See details about the current model.
  * **/bye**: Exit the chat session.
