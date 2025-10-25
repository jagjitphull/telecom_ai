from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
import json
from io import BytesIO

from ollama_client import OllamaClient
from agent import TelecomAnalysisAgent
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
CORS(app)

# Upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
REPORTS_FOLDER = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250MB total (5 files x 50MB)
app.config['ALLOWED_EXTENSIONS'] = {'log', 'txt', 'csv'}

# Initialize components
ollama_client = OllamaClient()
agent = TelecomAnalysisAgent(ollama_client)
report_generator = ReportGenerator()

# Store analysis sessions
analysis_sessions = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    ollama_connected = ollama_client.check_connection()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ollama_connected': ollama_connected,
        'services': {
            'flask': 'running',
            'ollama': 'connected' if ollama_connected else 'disconnected'
        }
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available Ollama models"""
    try:
        models = ollama_client.get_available_models()
        
        if not models:
            return jsonify({
                'success': False,
                'error': 'No models found. Please ensure Ollama is running and models are pulled.',
                'models': []
            }), 503
        
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'models': []
        }), 500


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        # Check if files are in request
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        # Validate file count
        if len(files) > 5:
            return jsonify({'success': False, 'error': 'Maximum 5 files allowed'}), 400
        
        if len(files) == 0:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        uploaded_files = []
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Process each file
        total_size = 0
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': f'Invalid file type: {file.filename}. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
                }), 400
            
            # Check file size (50MB per file)
            file_size = len(file.read())
            file.seek(0)
            
            if file_size > 50 * 1024 * 1024:
                return jsonify({
                    'success': False,
                    'error': f'File {file.filename} exceeds 50MB limit'
                }), 400
            
            total_size += file_size
            
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(session_folder, filename)
            file.save(filepath)
            
            uploaded_files.append({
                'filename': filename,
                'size': file_size,
                'path': filepath
            })
        
        # Store session info
        analysis_sessions[session_id] = {
            'files': uploaded_files,
            'timestamp': datetime.now().isoformat(),
            'total_size': total_size,
            'status': 'uploaded'
        }
        
        logger.info(f"Files uploaded for session {session_id}: {len(uploaded_files)} files")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'files': uploaded_files,
            'total_size': total_size
        })
    
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_logs():
    """Analyze uploaded logs"""
    try:
        data = request.json
        session_id = data.get('session_id')
        model = data.get('model')
        
        if not session_id or not model:
            return jsonify({
                'success': False,
                'error': 'session_id and model are required'
            }), 400
        
        if session_id not in analysis_sessions:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404
        
        session = analysis_sessions[session_id]
        
        if session['status'] != 'uploaded':
            return jsonify({
                'success': False,
                'error': 'Session is not in uploaded state'
            }), 400
        
        # Read and combine all log files
        combined_logs = ""
        for file_info in session['files']:
            try:
                with open(file_info['path'], 'r', encoding='utf-8') as f:
                    combined_logs += f"=== File: {file_info['filename']} ===\n"
                    combined_logs += f.read() + "\n\n"
            except Exception as e:
                logger.error(f"Error reading file {file_info['filename']}: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Error reading file: {file_info["filename"]}'
                }), 500
        
        if not combined_logs.strip():
            return jsonify({
                'success': False,
                'error': 'No valid log content found'
            }), 400
        
        logger.info(f"Starting analysis for session {session_id} with model {model}")
        
        # Run analysis
        analysis_result = agent.analyze_logs(combined_logs, model)
        
        if not analysis_result['success']:
            return jsonify({
                'success': False,
                'error': analysis_result.get('error', 'Analysis failed')
            }), 500
        
        # Generate HTML report
        html_report = report_generator.generate_report(analysis_result)
        report_filename = f"report_{session_id}.html"
        report_path = os.path.join(REPORTS_FOLDER, report_filename)
        
        report_generator.save_report(html_report, report_path)
        
        # Update session
        session['status'] = 'analyzed'
        session['analysis'] = analysis_result
        session['report_path'] = report_path
        session['report_filename'] = report_filename
        session['analysis_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Analysis completed for session {session_id}")
        
        # Return summary (full report via download endpoint)
        return jsonify({
            'success': True,
            'session_id': session_id,
            'model': model,
            'statistics': analysis_result['statistics'],
            'summary': analysis_result['summary'][:500],  # Preview
            'root_cause': analysis_result['root_cause'][:500],  # Preview
            'predictions': analysis_result['predictions'],
            'recommendations': analysis_result['recommendations'],
            'report_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/report/<session_id>', methods=['GET'])
def get_report(session_id):
    """Download HTML report"""
    try:
        if session_id not in analysis_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = analysis_sessions[session_id]
        
        if session['status'] != 'analyzed':
            return jsonify({'error': 'Analysis not yet completed'}), 400
        
        report_path = session['report_path']
        
        if not os.path.exists(report_path):
            return jsonify({'error': 'Report not found'}), 404
        
        return send_file(
            report_path,
            mimetype='text/html',
            as_attachment=True,
            download_name=session['report_filename']
        )
    
    except Exception as e:
        logger.error(f"Error retrieving report: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session details"""
    try:
        if session_id not in analysis_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = analysis_sessions[session_id]
        
        # Don't expose full analysis data in session endpoint
        return jsonify({
            'success': True,
            'session_id': session_id,
            'status': session['status'],
            'files_count': len(session['files']),
            'total_size': session['total_size'],
            'timestamp': session['timestamp'],
            'has_report': session['status'] == 'analyzed'
        })
    
    except Exception as e:
        logger.error(f"Error retrieving session: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all sessions"""
    try:
        sessions_list = []
        for session_id, session in analysis_sessions.items():
            sessions_list.append({
                'session_id': session_id,
                'status': session['status'],
                'files_count': len(session['files']),
                'timestamp': session['timestamp'],
                'has_report': session['status'] == 'analyzed'
            })
        
        return jsonify({
            'success': True,
            'sessions': sorted(sessions_list, key=lambda x: x['timestamp'], reverse=True)
        })
    
    except Exception as e:
        logger.error(f"Error retrieving sessions: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File or total size exceeds limit. Max 250MB total (50MB per file, 5 files)'
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Telecom Log Analysis Application")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Reports folder: {REPORTS_FOLDER}")
    
    # Check Ollama connection
    if ollama_client.check_connection():
        logger.info("✓ Ollama connection successful")
        models = ollama_client.get_available_models()
        logger.info(f"✓ Available models: {', '.join(models) if models else 'None'}")
    else:
        logger.warning("⚠ Ollama not connected. Please ensure Ollama is running on localhost:11434")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
