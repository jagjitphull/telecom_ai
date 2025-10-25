# 📥 Complete Installation Guide

## System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| OS | Windows/macOS/Linux | Linux/macOS |
| Python | 3.8 | 3.10+ |
| RAM | 2GB | 4GB+ |
| Disk Space | 10GB | 20GB+ |
| Network | 100 Mbps | 1 Gbps |

## Pre-Installation Checklist

- [ ] Python 3.8+ installed
- [ ] pip package manager working
- [ ] Internet connection for model download
- [ ] 10GB+ free disk space
- [ ] Terminal/Command Prompt access
- [ ] Administrator/sudo access (for Ollama)

---

## Step-by-Step Installation

### Step 1: Install Python

**Windows:**
- Download from https://python.org/
- Run installer, check "Add Python to PATH"
- Verify: `python --version`

**macOS:**
```bash
brew install python@3.10
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3-pip
```

### Step 2: Install Ollama

**Windows/macOS:**
- Download from https://ollama.ai/
- Run installer, follow instructions

**Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

Verify installation:
```bash
ollama --version
```

### Step 3: Extract Application

1. Download `telecom_analysis_app.zip`
2. Extract to desired location
3. Open terminal in extracted folder

### Step 4: Install Python Dependencies

**macOS/Linux:**
```bash
python3 -m pip install -r requirements.txt
```

**Windows:**
```cmd
python -m pip install -r requirements.txt
```

### Step 5: Download AI Model

```bash
ollama pull mistral
```

Options:
- `mistral` - Fast (4GB) ⭐ Start here
- `llama2` - Balanced (4GB)
- `neural-chat` - Optimized (4GB)
- `dolphin-mixtral` - Advanced (26GB)

---

## Running the Application

### Terminal 1: Start Ollama
```bash
ollama serve
```

Wait for: `Listening on 127.0.0.1:11434`

### Terminal 2: Start Flask App
```bash
python app.py
```

Wait for: `Running on http://0.0.0.0:5000`

### Browser
Open: `http://localhost:5000`

---

## Using Setup Scripts

### macOS/Linux Automated Setup
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Check Python installation
- Install dependencies
- Create directories
- Optionally pull models

### Windows Automated Setup
```cmd
setup.bat
```

Same features as shell script.

---

## First Test Run

1. **Start Services** (follow "Running the Application" above)
2. **Open** http://localhost:5000
3. **Upload** the included `sample_telecom.log`
4. **Select** model (try `mistral`)
5. **Analyze** - Wait 1-2 minutes
6. **View** results in tabs
7. **Download** HTML report

---

## Installation Verification

### Check Python
```bash
python3 --version
# Output: Python 3.x.x
```

### Check Pip
```bash
pip3 list | grep Flask
# Output: Flask 3.0.0
```

### Check Ollama
```bash
curl http://localhost:11434/api/tags
# Output: JSON with model list
```

### Check Application
```bash
python3 app.py
# Output: Running on http://0.0.0.0:5000
```

---

## Troubleshooting Installation

### Python Not Found
```bash
# macOS/Linux - Use python3 explicitly
python3 app.py

# Windows - Add Python to PATH
# Settings > Environment Variables > PATH
```

### Permission Denied (Linux/macOS)
```bash
chmod +x setup.sh
chmod +x app.py
```

### Port 5000 Already in Use
```bash
# Change port in app.py line ~310
app.run(port=5001)  # Use different port
```

### Ollama Connection Failed
```bash
# Check if running
curl http://localhost:11434/api/tags

# If error, start Ollama
ollama serve

# If still failing, check firewall
sudo ufw allow 11434  # Linux
```

### Out of Memory During Download
- Use smaller model: `ollama pull mistral` instead of larger ones
- Close other applications
- Ensure 10GB+ free disk space

### Slow Download
- Check internet speed
- Pause other downloads
- Use faster model (mistral)

---

## Docker Installation (Optional)

For containerized deployment:

### Build Image
```bash
docker build -t telecom-analyzer .
```

### Run Container
```bash
docker run -p 5000:5000 -p 11434:11434 telecom-analyzer
```

---

## Production Deployment

### Before Going Live

1. **Security**
   - Implement authentication
   - Use HTTPS with SSL/TLS
   - Restrict to internal network only
   - Set CORS properly

2. **Performance**
   - Use reverse proxy (nginx)
   - Enable caching
   - Load balance if needed
   - Monitor resource usage

3. **Reliability**
   - Set up logging
   - Create backups
   - Monitor health
   - Set up alerts

### Production Configuration

Update `app.py`:
```python
app.run(
    debug=False,
    host='0.0.0.0',
    port=5000,
    threaded=True,
    ssl_context='adhoc'  # For HTTPS
)
```

---

## Post-Installation Steps

1. ✅ Verify all components working
2. ✅ Test with sample log file
3. ✅ Read README.md for full features
4. ✅ Customize as needed
5. ✅ Set up monitoring/logging
6. ✅ Train team on usage

---

## Quick Reference

### Start Application
```bash
# Terminal 1
ollama serve

# Terminal 2
python app.py

# Then open: http://localhost:5000
```

### Common Commands
```bash
# List models
ollama list

# Pull new model
ollama pull <model_name>

# Check Ollama status
curl http://localhost:11434/api/tags

# Stop Ollama
pkill ollama  # macOS/Linux
taskkill /IM ollama.exe  # Windows

# View Flask logs
python app.py 2>&1 | tee app.log
```

### File Locations
- **Uploads**: `./uploads/`
- **Reports**: `./reports/`
- **Logs**: `./app.log`
- **Models**: `~/.ollama/models/`

---

## Support Resources

| Resource | URL |
|----------|-----|
| Ollama Docs | https://ollama.ai/docs |
| Flask Docs | https://flask.palletsprojects.com/ |
| Python Docs | https://python.org/docs |
| GitHub Issues | Report bugs here |

---

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] Ollama installed and running
- [ ] Dependencies installed (pip install -r requirements.txt)
- [ ] Model downloaded (ollama pull mistral)
- [ ] Flask app running on :5000
- [ ] Web UI accessible at http://localhost:5000
- [ ] Upload/analysis working
- [ ] Sample report generated
- [ ] HTML report downloadable

---

**You're all set! Start analyzing telecom logs! 🚀**

If you encounter issues, check the troubleshooting section or review README.md.
