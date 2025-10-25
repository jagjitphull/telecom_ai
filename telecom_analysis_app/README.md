# 🚀 Telecom Logs Analysis - AI Agent

A powerful AI-powered telecom logs analysis system built with Flask and Ollama. This application analyzes telecom logs, identifies root causes of issues, predicts future problems, and generates professional HTML reports.

## 🎯 Features

- **Multi-file Upload**: Upload up to 5 log files (50MB each, 250MB total)
- **AI-Powered Analysis**: Uses Ollama LLMs for intelligent analysis
- **Root Cause Detection**: Identifies underlying issues in telecom systems
- **Predictive Analytics**: Forecasts potential future problems
- **Log Correlation**: Correlates logs by call IDs, users, IPs, and error patterns
- **Professional Reports**: Generates comprehensive HTML reports with statistics and insights
- **Model Selection**: Choose from multiple Ollama models for analysis
- **Session Management**: Track and manage multiple analysis sessions
- **Real-time Status**: Monitor Ollama connection status

## 📋 Requirements

- Python 3.8+
- Flask 3.0+
- Ollama (running locally on port 11434)
- 2GB+ available RAM for LLM inference
- Modern web browser

## 🔧 Installation

### Step 1: Clone or Download the Application

```bash
cd telecom_analysis_app
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install and Setup Ollama

1. **Download Ollama** from https://ollama.ai/
2. **Install** the application
3. **Pull a model** (required):

```bash
ollama pull mistral          # Fast, ~7B parameters
ollama pull llama2           # Balanced, ~7B parameters
ollama pull neural-chat      # Optimized for chat, ~7B parameters
ollama pull dolphin-mixtral  # Advanced, ~8x7B parameters
```

4. **Run Ollama** (it will start on localhost:11434):

```bash
ollama serve
```

Keep this running while using the application.

### Step 4: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## 💻 Usage

### 1. Access Web Interface

Open your browser and navigate to: `http://localhost:5000`

### 2. Upload Log Files

- Click the upload area or drag & drop files
- Supported formats: `.log`, `.txt`, `.csv`
- Max 5 files per analysis
- Max 50MB per file

### 3. Select AI Model

- Choose from available Ollama models
- Different models offer different analysis quality/speed trade-offs

### 4. Run Analysis

- Click "Analyze Logs" button
- Wait for AI analysis (takes 1-5 minutes depending on file size and model)

### 5. View Results

Results are displayed in tabs:
- **Summary**: Overview of the logs
- **Root Cause**: Identified issues and their causes
- **Predictions**: Forecasted problems
- **Reports**: Download full HTML report

### 6. Download Report

Click "Download HTML Report" to get a professional report with:
- Statistics dashboard
- Executive summary
- Root cause analysis
- Predictions and forecasts
- Recommendations
- Error analysis
- Correlation patterns

## 📊 Log Format Support

The analyzer supports various telecom log formats:

### Common Log Elements Recognized:
- **Timestamps**: Various formats (YYYY-MM-DD HH:MM:SS, HH:MM:SS, etc.)
- **Log Levels**: ERROR, WARNING, INFO, DEBUG
- **Call IDs**: Unique identifiers for calls
- **Users**: User identifiers or phone numbers
- **IP Addresses**: Source/destination IPs
- **Status Codes**: HTTP/SIP status codes
- **Latency**: Milliseconds or seconds
- **Exceptions**: Error messages

### Example Log Formats:

```
2024-01-15 10:23:45 [ERROR] Call-ID: SIP-123456-789 User: +1234567890 Latency: 250ms Status: 500
2024-01-15 10:23:46 [WARNING] IP: 192.168.1.100 Connection timeout after 5000ms
2024-01-15 10:23:47 [INFO] User: alice@example.com processed successfully in 150ms
```

## 🤖 Ollama Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| mistral | 4.1GB | ⚡⚡⚡ | ⭐⭐⭐ | Quick analysis |
| llama2 | 3.8GB | ⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| neural-chat | 4.1GB | ⚡⚡ | ⭐⭐⭐⭐ | Chat optimized |
| dolphin-mixtral | 26GB | ⚡ | ⭐⭐⭐⭐⭐ | Deep analysis |

### Pull Multiple Models:
```bash
ollama pull mistral
ollama pull llama2
ollama pull neural-chat
```

## 📁 Project Structure

```
telecom_analysis_app/
├── app.py                    # Main Flask application
├── agent.py                  # AI analysis agent
├── log_analyzer.py          # Log parsing and correlation
├── ollama_client.py         # Ollama API client
├── report_generator.py      # HTML report generation
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html          # Web UI
├── uploads/                 # Uploaded files (auto-created)
└── reports/                 # Generated reports (auto-created)
```

## 🔍 Analysis Process

1. **Log Parsing**: Extracts structured data from raw logs
2. **Correlation**: Links related logs by call ID, user, IP, etc.
3. **Statistics**: Calculates error rates, latency, patterns
4. **AI Analysis**: Uses LLM to identify root causes
5. **Prediction**: Forecasts potential future issues
6. **Recommendations**: Generates actionable advice
7. **Report Generation**: Creates professional HTML report

## 🚨 Troubleshooting

### "No models available" Error
```bash
# Ensure Ollama is running
ollama serve

# In another terminal, pull a model
ollama pull mistral
```

### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If error, restart Ollama
ollama serve
```

### Out of Memory Error
- Use smaller models (mistral instead of dolphin-mixtral)
- Reduce file size
- Close other applications

### Slow Analysis
- Normal for first run (model loads into memory)
- Consider using faster model (mistral)
- Reduce log file size

## 📈 Performance Tips

1. **Start with smaller logs**: Test with 1-2MB files first
2. **Use mistral model**: Good balance of speed and quality
3. **Batch similar logs**: Group related logs together
4. **Regular analysis**: Analyze daily/weekly for patterns

## 🔒 Security Notes

- Keep Ollama running on localhost only
- Do not expose port 5000 to the internet without authentication
- Use HTTPS in production environments
- Sanitize log files before uploading

## 📝 API Endpoints

### Health & Status
- `GET /api/health` - Check service health
- `GET /api/models` - List available models

### File Management
- `POST /api/upload` - Upload log files
- `GET /api/sessions` - List all sessions
- `GET /api/session/<id>` - Get session details

### Analysis
- `POST /api/analyze` - Analyze logs
- `GET /api/report/<id>` - Download HTML report

## 🎓 Example Workflow

```python
# Quick test with Python
import requests

# Check Ollama status
response = requests.get('http://localhost:5000/api/health')
print(response.json())

# Get available models
response = requests.get('http://localhost:5000/api/models')
models = response.json()['models']
print(f"Available models: {models}")

# Upload files
with open('telecom.log', 'rb') as f:
    files = {'files': f}
    response = requests.post('http://localhost:5000/api/upload', files=files)
    session_id = response.json()['session_id']

# Analyze
response = requests.post('http://localhost:5000/api/analyze', json={
    'session_id': session_id,
    'model': models[0]
})
analysis = response.json()
print(analysis)
```

## 📞 Support

For issues or questions:
1. Check the Ollama documentation: https://ollama.ai/docs
2. Review Flask documentation: https://flask.palletsprojects.com/
3. Check logs in the application terminal

## 📄 License

This application is provided as-is for telecom analysis purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional log format support
- Custom analysis templates
- Database integration
- Distributed analysis
- Advanced visualization

## ⭐ Features Roadmap

- [ ] Multi-user authentication
- [ ] Database persistence
- [ ] Advanced visualization dashboards
- [ ] Export to PDF/Excel
- [ ] Scheduled analysis jobs
- [ ] Machine learning model training
- [ ] Real-time log streaming
- [ ] Integration with monitoring systems

---

**Built with ❤️ for telecom excellence**
