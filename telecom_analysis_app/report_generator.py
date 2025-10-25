import json
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generate professional HTML reports for log analysis"""
    
    CSS_STYLES = """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        header {
            border-bottom: 3px solid #2c3e50;
            margin-bottom: 30px;
            padding-bottom: 20px;
        }
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.8em;
        }
        h3 {
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.3em;
        }
        .metadata {
            display: flex;
            gap: 30px;
            margin-bottom: 20px;
            font-size: 0.95em;
            color: #666;
        }
        .metadata-item {
            display: flex;
            flex-direction: column;
        }
        .metadata-label {
            font-weight: bold;
            color: #2c3e50;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-card.info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .stat-card.error {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .stat-card.warning {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }
        .stat-card.success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .analysis-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }
        .analysis-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .analysis-content {
            color: #555;
            line-height: 1.8;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        thead {
            background: #2c3e50;
            color: white;
        }
        th {
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .error-badge {
            display: inline-block;
            background: #e74c3c;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .warning-badge {
            display: inline-block;
            background: #f39c12;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .info-badge {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .success-badge {
            display: inline-block;
            background: #27ae60;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .alert {
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 15px;
            border-left: 4px solid;
        }
        .alert-danger {
            background: #fadbd8;
            border-color: #e74c3c;
            color: #a93226;
        }
        .alert-warning {
            background: #fdebd0;
            border-color: #f39c12;
            color: #9a6321;
        }
        .alert-info {
            background: #d6eaf8;
            border-color: #3498db;
            color: #1b4f72;
        }
        .alert-success {
            background: #d5f4e6;
            border-color: #27ae60;
            color: #0b5345;
        }
        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }
        .list-item {
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }
        .recommendation {
            background: #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid #27ae60;
        }
        @media print {
            body {
                background: white;
            }
            .container {
                box-shadow: none;
                padding: 0;
            }
        }
    </style>
    """
    
    def __init__(self):
        self.report_id = None
    
    def generate_report(self, analysis: Dict[str, Any], filename: str = None) -> str:
        """Generate complete HTML report"""
        timestamp = datetime.now()
        
        if not filename:
            filename = f"report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telecom Logs Analysis Report</title>
    {self.CSS_STYLES}
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Telecom Logs Analysis Report</h1>
            <div class="metadata">
                <div class="metadata-item">
                    <span class="metadata-label">Generated:</span>
                    <span>{timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">AI Model:</span>
                    <span>{analysis.get('model', 'Unknown')}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Logs Analyzed:</span>
                    <span>{analysis['statistics']['total_logs']:,}</span>
                </div>
            </div>
        </header>

        {self._generate_statistics_section(analysis)}
        {self._generate_executive_summary_section(analysis)}
        {self._generate_root_cause_section(analysis)}
        {self._generate_predictions_section(analysis)}
        {self._generate_recommendations_section(analysis)}
        {self._generate_correlations_section(analysis)}
        {self._generate_top_errors_section(analysis)}
        
        <footer>
            <p>This report was automatically generated by Telecom Log Analysis Agent</p>
            <p>For questions or support, contact your system administrator</p>
        </footer>
    </div>
</body>
</html>
"""
        return html_content
    
    def _generate_statistics_section(self, analysis: Dict) -> str:
        """Generate statistics dashboard section"""
        stats = analysis['statistics']
        error_rate = stats['error_rate']
        error_severity = "danger" if error_rate > 10 else "warning" if error_rate > 5 else "success"
        
        html = f"""
        <h2>📈 Statistics Dashboard</h2>
        <div class="stats-grid">
            <div class="stat-card info">
                <div class="stat-value">{stats['total_logs']:,}</div>
                <div class="stat-label">Total Logs</div>
            </div>
            <div class="stat-card error">
                <div class="stat-value">{stats['error_count']}</div>
                <div class="stat-label">Errors ({stats['error_rate']:.2f}%)</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-value">{stats['warning_count']}</div>
                <div class="stat-label">Warnings</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value">{stats['info_count']}</div>
                <div class="stat-label">Info Logs</div>
            </div>
            <div class="stat-card info">
                <div class="stat-value">{stats['avg_latency_ms']:.2f}</div>
                <div class="stat-label">Avg Latency (ms)</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-value">{stats['high_latency_count']}</div>
                <div class="stat-label">High Latency Events</div>
            </div>
            <div class="stat-card info">
                <div class="stat-value">{stats['unique_users']}</div>
                <div class="stat-label">Unique Users</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value">{stats['unique_call_ids']}</div>
                <div class="stat-label">Unique Call IDs</div>
            </div>
        </div>
        """
        return html
    
    def _generate_executive_summary_section(self, analysis: Dict) -> str:
        """Generate executive summary section"""
        summary = analysis['summary']
        return f"""
        <h2>📋 Executive Summary</h2>
        <div class="analysis-section">
            <div class="analysis-content">{self._escape_html(summary)}</div>
        </div>
        """
    
    def _generate_root_cause_section(self, analysis: Dict) -> str:
        """Generate root cause analysis section"""
        root_cause = analysis['root_cause']
        stats = analysis['statistics']
        
        alert_class = "danger" if stats['error_rate'] > 10 else "warning" if stats['error_rate'] > 5 else "info"
        
        return f"""
        <h2>🔍 Root Cause Analysis</h2>
        <div class="alert alert-{alert_class}">
            <strong>Error Rate: {stats['error_rate']:.2f}%</strong>
        </div>
        <div class="analysis-section">
            <div class="analysis-title">Identified Root Causes:</div>
            <div class="analysis-content">{self._escape_html(root_cause)}</div>
        </div>
        """
    
    def _generate_predictions_section(self, analysis: Dict) -> str:
        """Generate predictions section"""
        predictions = analysis['predictions']
        return f"""
        <h2>🔮 Predictions & Forecasts</h2>
        <div class="analysis-section">
            <div class="analysis-title">Predicted Issues and Trends:</div>
            <div class="analysis-content">{self._escape_html(predictions)}</div>
        </div>
        """
    
    def _generate_recommendations_section(self, analysis: Dict) -> str:
        """Generate recommendations section"""
        recommendations = analysis['recommendations']
        return f"""
        <h2>💡 Recommendations</h2>
        <div class="analysis-section">
            <div class="analysis-title">Recommended Actions:</div>
            <div class="analysis-content">{self._escape_html(recommendations)}</div>
        </div>
        """
    
    def _generate_correlations_section(self, analysis: Dict) -> str:
        """Generate correlations summary section"""
        correlations = analysis['correlations_summary']
        
        html = "<h2>🔗 Correlations & Patterns</h2>"
        
        # Problematic Call IDs
        if correlations.get('problematic_call_ids'):
            html += "<h3>Top Problematic Call IDs</h3><table><thead><tr><th>Call ID</th><th>Incidents</th></tr></thead><tbody>"
            for call_id, count in correlations['problematic_call_ids']:
                html += f"<tr><td>{self._escape_html(str(call_id))}</td><td><span class='error-badge'>{count}</span></td></tr>"
            html += "</tbody></table>"
        
        # Affected Users
        if correlations.get('affected_users'):
            html += "<h3>Most Affected Users</h3><table><thead><tr><th>User</th><th>Incidents</th></tr></thead><tbody>"
            for user, count in correlations['affected_users']:
                html += f"<tr><td>{self._escape_html(str(user))}</td><td><span class='warning-badge'>{count}</span></td></tr>"
            html += "</tbody></table>"
        
        # Problematic IPs
        if correlations.get('problematic_ips'):
            html += "<h3>Problematic IP Addresses</h3><table><thead><tr><th>IP Address</th><th>Incidents</th></tr></thead><tbody>"
            for ip, count in correlations['problematic_ips']:
                html += f"<tr><td>{self._escape_html(str(ip))}</td><td><span class='info-badge'>{count}</span></td></tr>"
            html += "</tbody></table>"
        
        return html
    
    def _generate_top_errors_section(self, analysis: Dict) -> str:
        """Generate top errors section"""
        errors = analysis['top_errors']
        
        if not errors:
            return "<h2>❌ Error Analysis</h2><p>No errors found in logs.</p>"
        
        html = "<h2>❌ Error Analysis</h2>"
        html += "<table><thead><tr><th>Error Message</th><th>Count</th></tr></thead><tbody>"
        
        for error, count in errors:
            html += f"<tr><td>{self._escape_html(str(error)[:100])}</td><td><span class='error-badge'>{count}</span></td></tr>"
        
        html += "</tbody></table>"
        return html
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        if not text:
            return ""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))
    
    def save_report(self, html_content: str, filepath: str) -> bool:
        """Save report to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Report saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return False
