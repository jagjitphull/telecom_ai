# 🚀 Quick Start Guide

## Prerequisites
- Python 3.8+
- Ollama (https://ollama.ai/)
- Modern web browser

## 5-Minute Setup

### 1. Download and Extract
- Extract the `telecom_analysis_app.zip` to a folder
- Open terminal/command prompt in that folder

### 2. Run Setup Script

**On macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows:**
```cmd
setup.bat
```

### 3. Start Ollama (Terminal 1)
```bash
ollama serve
```

Wait for: `Listening on 127.0.0.1:11434`

### 4. Pull a Model (Terminal 2)
```bash
ollama pull mistral
```

This downloads the AI model (~4GB). Only needed once.

### 5. Start Flask App (Terminal 3)
```bash
python app.py
```

Wait for: `Running on http://0.0.0.0:5000`

### 6. Open Browser
Navigate to: **http://localhost:5000**

## First Analysis

1. **Upload** - Drag & drop your telecom logs (or use `sample_telecom.log`)
2. **Select Model** - Choose `mistral` (fastest)
3. **Analyze** - Click "Analyze Logs"
4. **View Results** - See analysis in tabs
5. **Download** - Get HTML report

## What's in the Report

- 📊 Statistics dashboard
- 📋 Executive summary
- 🔍 Root cause analysis
- 🔮 Predictions
- 💡 Recommendations
- 🔗 Log correlations
- ❌ Error analysis

## Models

| Model | Speed | Quality | Memory |
|-------|-------|---------|--------|
| mistral | ⚡⚡⚡ | ⭐⭐⭐ | 4GB |
| llama2 | ⚡⚡ | ⭐⭐⭐⭐ | 4GB |

## Troubleshooting

### "Cannot connect to Ollama"
- Make sure `ollama serve` is running
- Check port 11434 is available

### "No models available"
- Run `ollama pull mistral` in terminal 2
- Restart the Flask app

### Slow Analysis
- Normal on first run (model loads)
- Try smaller files first
- Use `mistral` instead of larger models

### Out of Memory
- Close other applications
- Use smaller log files
- Try `mistral` model

## File Limits
- Max 5 files per upload
- Max 50MB per file
- Max 250MB total

## Need Help?
- Read README.md for full documentation
- Check Ollama: https://ollama.ai/docs
- Flask docs: https://flask.palletsprojects.com/

## Next Steps

✅ Try with `sample_telecom.log` first
✅ Upload your real logs
✅ Compare different AI models
✅ Generate reports for your team

---

**Happy analyzing! 🎯**
