# 🚀 Telecom Logs Analysis - AI Agent Application

## Overview

A production-ready, AI-powered telecom logs analysis system that leverages Ollama LLMs to analyze, correlate, and predict issues in telecom infrastructure. The application provides intelligent root cause analysis, predictive analytics, and generates professional HTML reports.

---

## 📦 What You Get

### Complete Package Contents:
```
telecom_analysis_app/
├── 🖥️  Web Application
│   ├── app.py                    # Flask backend (12.4KB)
│   ├── agent.py                  # AI analysis agent (6.7KB)
│   ├── log_analyzer.py          # Log parsing engine (6.7KB)
│   ├── ollama_client.py         # Ollama integration (2.5KB)
│   ├── report_generator.py      # Report creation (15.5KB)
│   └── templates/
│       └── index.html           # Web UI (modern, responsive)
│
├── 📚 Documentation
│   ├── README.md                # Full documentation
│   ├── QUICKSTART.md           # 5-minute setup
│   └── .env.example            # Configuration template
│
├── ⚙️  Setup & Configuration
│   ├── requirements.txt         # Python dependencies
│   ├── setup.sh               # macOS/Linux setup
│   └── setup.bat              # Windows setup
│
└── 📝 Sample Data
    └── sample_telecom.log     # Test log file
```

---

## 🎯 Key Features

### 1. **Multi-File Upload**
- Upload up to 5 log files simultaneously
- Support for `.log`, `.txt`, and `.csv` formats
- Max 50MB per file (250MB total)
- Real-time file size validation

### 2. **AI-Powered Analysis**
Uses Ollama LLMs for:
- **Intelligent Summarization**: AI understands log context
- **Root Cause Detection**: Identifies underlying issues
- **Predictive Analytics**: Forecasts future problems
- **Smart Recommendations**: Actionable improvement suggestions

### 3. **Log Correlation**
Automatically correlates logs by:
- Call IDs (track related events)
- User identifiers
- IP addresses
- Error types
- Status codes
- Latency patterns

### 4. **Comprehensive Statistics**
Calculates:
- Error rates and counts
- Warning frequencies
- Average latency
- High-latency incidents
- Unique users/calls/IPs
- Error distribution

### 5. **Professional Reports**
Generates HTML reports with:
- Beautiful dashboard UI
- Statistics visualizations
- Color-coded severity indicators
- Executive summary
- Detailed findings
- Actionable recommendations
- Print-friendly formatting

### 6. **Model Selection**
Choose from multiple Ollama models:
- **Mistral** (4GB) - Fast, good quality
- **Llama2** (4GB) - Balanced performance
- **Neural-Chat** (4GB) - Optimized for analysis
- **Dolphin-Mixtral** (26GB) - Advanced analysis

### 7. **Session Management**
- Track multiple analysis sessions
- Store analysis history
- Retrieve previous reports
- Session status tracking

---

## 🏗️ Architecture

### Frontend
- **Framework**: HTML5 + CSS3 + Vanilla JavaScript
- **Features**: Drag-and-drop upload, tab interface, real-time status
- **Responsive**: Works on desktop, tablet, mobile

### Backend
- **Framework**: Flask 3.0
- **Python Version**: 3.8+
- **Architecture**: RESTful API + Agent pattern

### Data Flow
```
Upload Files
    ↓
Parse & Extract
    ↓
Correlate Logs
    ↓
Generate Statistics
    ↓
AI Analysis (via Ollama)
    ├─→ Summary Generation
    ├─→ Root Cause Detection
    ├─→ Predictive Analytics
    └─→ Recommendations
    ↓
Generate HTML Report
    ↓
Display & Download
```

---

## ⚡ Quick Start (5 Minutes)

### Prerequisites
- Python 3.8+
- Ollama (https://ollama.ai/)
- 2GB+ RAM

### Step 1: Extract Files
```bash
unzip telecom_analysis_app.zip
cd telecom_analysis_app
```

### Step 2: Setup (Automatic)
**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

### Step 3: Pull AI Model
```bash
ollama pull mistral  # ~4GB download
```

### Step 4: Start Ollama (Terminal 1)
```bash
ollama serve
```

### Step 5: Start App (Terminal 2)
```bash
python app.py
```

### Step 6: Open Browser
```
http://localhost:5000
```

---

## 💼 Use Cases

### 1. **Network Troubleshooting**
- Identify connection failures
- Correlate network errors
- Predict degradation

### 2. **Performance Analysis**
- Analyze latency patterns
- Identify bottlenecks
- Capacity planning

### 3. **Incident Investigation**
- Root cause analysis
- Impact assessment
- Timeline reconstruction

### 4. **Predictive Maintenance**
- Forecast failures
- Resource exhaustion warnings
- Proactive alerts

### 5. **Compliance & Auditing**
- Generate audit reports
- Track system health
- Compliance documentation

---

## 📊 Sample Analysis Output

### Statistics Dashboard
```
Total Logs: 12,456
Error Rate: 8.5%
Warnings: 342
Average Latency: 245ms
High Latency Events: 89
Unique Users: 156
Unique Call IDs: 1,234
```

### Root Cause Analysis
```
Primary Issue: Server 192.168.1.50 Resource Exhaustion
- CPU Usage: 95%
- Memory: 92%
- Database Connections: 245/250

Secondary Issues:
- Network latency to gateway (2500ms)
- DNS resolution delays (800ms)
- SSL certificate expiring in 7 days
```

### Predictions
```
Risk: HIGH - 75% probability of system failure within 2 hours
Reason: Resource exhaustion trend, increasing error rate
Impact: 45 active users affected, ~2000 calls pending
Recommendation: Immediate failover to backup infrastructure
```

---

## 🔧 API Endpoints

### Health & Configuration
- `GET /api/health` - Service health check
- `GET /api/models` - Available AI models

### File Management
- `POST /api/upload` - Upload log files
- `GET /api/sessions` - List sessions
- `GET /api/session/<id>` - Session details

### Analysis
- `POST /api/analyze` - Run analysis
- `GET /api/report/<id>` - Download report

---

## 📋 Supported Log Formats

The analyzer recognizes:
- **Timestamps**: Multiple formats (YYYY-MM-DD HH:MM:SS, etc.)
- **Log Levels**: ERROR, WARNING, INFO, DEBUG
- **Call IDs**: SIP-*, Call-ID:, call_id=
- **Users**: Email, phone numbers, usernames
- **IP Addresses**: IPv4 format
- **Status Codes**: HTTP/SIP codes
- **Latency**: ms or s units
- **Exceptions**: Error message extraction

### Example Supported Log Lines:
```
2024-01-15 10:23:45 [ERROR] Call-ID: SIP-123456 User: +1234567890 Status: 500
2024-01-15 10:23:46 [WARNING] IP: 192.168.1.100 Latency: 2500ms
2024-01-15 10:23:47 [INFO] Connection established in 150ms
```

---

## 🔒 Security Features

- ✅ File type validation
- ✅ File size limits enforcement
- ✅ HTML content sanitization
- ✅ CORS protection
- ✅ Error handling
- ✅ Local Ollama only (no external APIs)
- ✅ Session isolation
- ✅ Secure file storage

### Security Notes:
- Keep Ollama on localhost only
- Use HTTPS in production
- Implement authentication for multi-user environments
- Regular backup of reports

---

## 🚀 Performance Metrics

### Typical Analysis Times (Mistral Model)
- Small logs (1MB): 30-60 seconds
- Medium logs (10MB): 2-4 minutes
- Large logs (50MB): 5-10 minutes

### Resource Requirements
- **CPU**: 2+ cores recommended
- **RAM**: 2GB (minimum), 4GB+ (recommended)
- **Disk**: 10GB available (for models + cache)
- **Network**: 100 Mbps (for model download)

### Optimization Tips
1. Start with smaller files to test
2. Use mistral model for speed
3. Close unnecessary applications
4. Batch similar logs together

---

## 🛠️ Advanced Configuration

### Environment Variables (.env)
```bash
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=mistral
MAX_FILES=5
MAX_FILE_SIZE_MB=50
ANALYSIS_TIMEOUT_SECONDS=600
LOG_LEVEL=INFO
```

### Custom Analysis Templates
Modify `agent.py` to:
- Add custom analysis prompts
- Create specialized detectors
- Implement domain-specific logic

### Database Integration
Extend `app.py` to:
- Store analysis history
- Track metrics over time
- Create dashboards
- Export to analytics tools

---

## 📚 Documentation

### Included Files:
1. **README.md** - Comprehensive guide (8KB)
2. **QUICKSTART.md** - 5-minute setup (2KB)
3. **This File** - Overview (this document)

### External Resources:
- [Ollama Documentation](https://ollama.ai/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Python Documentation](https://python.org/docs)

---

## 🐛 Troubleshooting

### Common Issues

**"Cannot connect to Ollama"**
```bash
# Ensure Ollama is running
ollama serve
```

**"No models available"**
```bash
ollama pull mistral
ollama pull llama2
```

**"Out of Memory"**
- Use smaller model (mistral)
- Reduce log file size
- Close other applications

**"Slow Analysis"**
- Normal for first run
- Try mistral model
- Process smaller batches

See README.md for more troubleshooting steps.

---

## ✨ Key Components Explained

### 1. **app.py** - Flask Server
- Handles file uploads
- Manages sessions
- Serves web interface
- Orchestrates analysis

### 2. **agent.py** - AI Analysis Agent
- Coordinates AI analysis
- Manages LLM calls
- Generates insights
- Tracks analysis history

### 3. **log_analyzer.py** - Log Processing
- Parses log lines
- Extracts structured data
- Correlates events
- Generates statistics

### 4. **ollama_client.py** - LLM Interface
- Communicates with Ollama
- Manages model selection
- Handles response streaming
- Error handling

### 5. **report_generator.py** - Report Creation
- Creates HTML reports
- Applies styling
- Includes visualizations
- Professional formatting

### 6. **index.html** - Web Interface
- File upload UI
- Model selection
- Results display
- Report download

---

## 🎓 Learning Path

1. **Start**: Use sample_telecom.log to understand features
2. **Experiment**: Try different AI models
3. **Customize**: Modify analysis prompts in agent.py
4. **Scale**: Integrate with your logging infrastructure
5. **Automate**: Create scheduled analysis jobs

---

## 📞 Support & Contributing

### Getting Help
1. Check README.md for detailed docs
2. Review QUICKSTART.md for setup issues
3. Check application logs for errors
4. Verify Ollama is running correctly

### Contributing Improvements
- Custom log format support
- New analysis templates
- Database integration
- API enhancements
- UI improvements

---

## 🎯 Next Steps

1. ✅ Extract the application
2. ✅ Run setup script
3. ✅ Start Ollama server
4. ✅ Launch Flask app
5. ✅ Open http://localhost:5000
6. ✅ Test with sample_telecom.log
7. ✅ Analyze your real logs

---

## 📝 Version Info

- **Application**: Telecom Log Analysis v1.0
- **Python**: 3.8+
- **Flask**: 3.0.0
- **Ollama**: Latest (supports all models)
- **Last Updated**: October 2024

---

**Ready to analyze telecom logs like never before? Let's go! 🚀**
