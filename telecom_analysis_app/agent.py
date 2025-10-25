import logging
from typing import Dict, Any, List
from ollama_client import OllamaClient
from log_analyzer import LogParser
import json

logger = logging.getLogger(__name__)

class TelecomAnalysisAgent:
    """AI Agent for telecom logs analysis using Ollama"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client
        self.parser = LogParser()
        self.analysis_history = []
    
    def analyze_logs(self, log_content: str, model: str) -> Dict[str, Any]:
        """Main analysis orchestration"""
        logger.info(f"Starting log analysis with model: {model}")
        
        # Step 1: Parse logs
        parsed_logs, raw_analysis = self.parser.parse_logs(log_content)
        
        if not parsed_logs:
            return {
                'success': False,
                'error': 'No logs could be parsed',
                'model': model
            }
        
        # Step 2: Generate statistics
        stats = self.parser.generate_statistics(parsed_logs)
        
        # Step 3: Correlate logs
        correlations = self.parser.correlate_logs(parsed_logs)
        
        # Step 4: AI-powered analysis
        summary = self._generate_ai_summary(log_content, parsed_logs, stats, model)
        root_cause = self._identify_root_cause(log_content, stats, correlations, model)
        predictions = self._predict_issues(log_content, stats, parsed_logs, model)
        recommendations = self._generate_recommendations(root_cause, stats, model)
        
        analysis_result = {
            'success': True,
            'model': model,
            'statistics': stats,
            'summary': summary,
            'root_cause': root_cause,
            'predictions': predictions,
            'recommendations': recommendations,
            'correlations_summary': self._summarize_correlations(correlations),
            'parsed_logs_count': len(parsed_logs),
            'top_errors': self._get_top_errors(parsed_logs)
        }
        
        self.analysis_history.append(analysis_result)
        return analysis_result
    
    def _generate_ai_summary(self, log_content: str, parsed_logs: List[Dict], stats: Dict, model: str) -> str:
        """Generate AI-powered summary of logs"""
        prompt = f"""Analyze these telecom logs and provide a concise summary:

Statistics:
- Total Logs: {stats['total_logs']}
- Errors: {stats['error_count']} ({stats['error_rate']:.2f}%)
- Warnings: {stats['warning_count']}
- Average Latency: {stats['avg_latency_ms']:.2f}ms
- Unique Users: {stats['unique_users']}
- Unique Call IDs: {stats['unique_call_ids']}

Log Sample (first 2000 chars):
{log_content[:2000]}

Provide a 3-4 sentence technical summary of what's happening in these logs."""

        return self.ollama.generate_response(prompt, model)
    
    def _identify_root_cause(self, log_content: str, stats: Dict, correlations: Dict, model: str) -> str:
        """Use AI to identify root cause of issues"""
        error_logs = [log for correlation in correlations.get('by_level', {}).get('ERROR', []) 
                      for log in [correlation]]
        
        prompt = f"""Based on these telecom logs, identify the root cause of issues:

Key Issues:
- Error Rate: {stats['error_rate']:.2f}%
- Error Count: {stats['error_count']}
- High Latency Instances: {stats['high_latency_count']}
- Average Latency: {stats['avg_latency_ms']:.2f}ms

Log Patterns (first 1500 chars):
{log_content[:1500]}

Provide a detailed root cause analysis. Consider:
1. Error patterns and frequencies
2. Correlation with latency spikes
3. Impact on specific users/call IDs
4. Potential infrastructure or software issues

Be specific and technical."""

        return self.ollama.generate_response(prompt, model)
    
    def _predict_issues(self, log_content: str, stats: Dict, parsed_logs: List[Dict], model: str) -> str:
        """Predict potential future issues"""
        high_latency_percent = (stats['high_latency_count'] / max(1, stats['total_logs'])) * 100
        error_rate = stats['error_rate']
        
        # Build prediction prompt
        prompt = f"""Analyze these telecom system metrics and provide specific predictions about future issues.

SYSTEM METRICS:
- Error Rate: {error_rate:.2f}% (Critical if >10%, Warning if >5%)
- High Latency Incidents: {high_latency_percent:.2f}%
- Total Errors: {stats['error_count']}
- Average Latency: {stats['avg_latency_ms']:.2f}ms
- Unique Users: {stats['unique_users']}
- Unique IPs: {stats['unique_ips']}

RECENT LOG PATTERNS:
{log_content[:1000]}

PROVIDE PREDICTIONS IN THIS FORMAT:

1. FAILURE RISK ASSESSMENT
   - Risk Level: (CRITICAL/HIGH/MEDIUM/LOW)
   - Probability: X%
   - Timeframe: (Hours/Days/Weeks)

2. LIKELY FAILURE POINTS
   - Component 1: Description
   - Component 2: Description
   - Component 3: Description

3. USER IMPACT
   - Affected Users: X%
   - Affected Calls: X
   - Service Degradation: Severity

4. RECOMMENDED THRESHOLDS
   - Error Rate Alert: X%
   - Latency Alert: Xms
   - Connection Timeout: Xs

5. PREVENTIVE ACTIONS
   - Immediate: Action 1
   - Short-term: Action 2
   - Long-term: Action 3"""

        response = self.ollama.generate_response(prompt, model)
        
        # If response is too short or empty, generate intelligent fallback
        if not response or len(response.strip()) < 50:
            return self._generate_fallback_predictions(stats, error_rate, high_latency_percent)
        
        return response
    
    def _generate_fallback_predictions(self, stats: Dict, error_rate: float, latency_percent: float) -> str:
        """Generate intelligent predictions based on metrics when LLM fails"""
        
        predictions = []
        
        # Risk Assessment
        if error_rate > 15:
            risk_level = "CRITICAL"
            probability = 85
        elif error_rate > 10:
            risk_level = "HIGH"
            probability = 65
        elif error_rate > 5:
            risk_level = "MEDIUM"
            probability = 40
        else:
            risk_level = "LOW"
            probability = 15
        
        predictions.append(f"""1. FAILURE RISK ASSESSMENT
   - Risk Level: {risk_level}
   - Probability: {probability}%
   - Timeframe: {self._get_timeframe(error_rate)}""")
        
        # Likely failure points
        failure_points = self._identify_failure_points(stats, error_rate, latency_percent)
        predictions.append(f"""2. LIKELY FAILURE POINTS
{failure_points}""")
        
        # User impact
        affected_percent = min(error_rate * 10, 100)  # Estimate based on error rate
        predictions.append(f"""3. USER IMPACT
   - Affected Users: ~{affected_percent:.0f}%
   - Affected Calls: {stats['error_count'] * 2} (estimated)
   - Service Degradation: {'Severe' if error_rate > 10 else 'Moderate' if error_rate > 5 else 'Minor'}""")
        
        # Thresholds
        threshold_error = max(error_rate + 5, 10)
        threshold_latency = max(stats['avg_latency_ms'] + 200, 500)
        predictions.append(f"""4. RECOMMENDED THRESHOLDS
   - Error Rate Alert: {threshold_error:.1f}%
   - Latency Alert: {threshold_latency:.0f}ms
   - Connection Timeout: 30s""")
        
        # Actions
        predictions.append(f"""5. PREVENTIVE ACTIONS
   - Immediate: Monitor error rate closely, enable backup systems
   - Short-term: Increase capacity, optimize database queries
   - Long-term: Upgrade infrastructure, implement redundancy""")
        
        return "\n".join(predictions)
    
    def _generate_recommendations(self, root_cause: str, stats: Dict, model: str) -> str:
        """Generate actionable recommendations"""
        prompt = f"""Based on this root cause analysis, provide specific recommendations:

Root Cause Found:
{root_cause[:500]}

Statistics:
- Error Rate: {stats['error_rate']:.2f}%
- Average Latency: {stats['avg_latency_ms']:.2f}ms

Provide:
1. Immediate actions to resolve the issue
2. Short-term improvements (1-2 weeks)
3. Long-term architectural improvements
4. Monitoring and alerting recommendations
5. Prevention strategies

Be specific with technical details."""

        return self.ollama.generate_response(prompt, model)
    
    def _summarize_correlations(self, correlations: Dict) -> Dict[str, Any]:
        """Summarize key correlations"""
        summary = {
            'problematic_call_ids': self._get_top_items(correlations.get('by_call_id', {}), 5),
            'affected_users': self._get_top_items(correlations.get('by_user', {}), 5),
            'problematic_ips': self._get_top_items(correlations.get('by_ip', {}), 5),
            'error_distribution': dict(correlations.get('errors_by_type', {})),
            'total_by_level': {
                level: len(logs) 
                for level, logs in correlations.get('by_level', {}).items()
            }
        }
        return summary
    
    def _get_top_items(self, items_dict: Dict, limit: int = 5) -> List:
        """Get top items by frequency"""
        return sorted(
            [(k, len(v)) for k, v in items_dict.items()],
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def _get_top_errors(self, parsed_logs: List[Dict]) -> List[str]:
        """Extract top error messages"""
        errors = [log['exception'] for log in parsed_logs if log['exception']]
        error_counts = {}
        for error in errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _get_timeframe(self, error_rate: float) -> str:
        """Estimate failure timeframe based on error rate"""
        if error_rate > 20:
            return "Within 1 hour"
        elif error_rate > 15:
            return "Within 4 hours"
        elif error_rate > 10:
            return "Within 24 hours"
        elif error_rate > 5:
            return "Within 3 days"
        else:
            return "Within 1 week"
    
    def _identify_failure_points(self, stats: Dict, error_rate: float, latency_percent: float) -> str:
        """Identify likely failure points"""
        points = []
        
        if error_rate > 10:
            points.append("   - Application Server: High error rate indicates overload")
        if latency_percent > 10:
            points.append("   - Network Infrastructure: High latency suggests congestion")
        if stats['high_latency_count'] > 20:
            points.append("   - Database Tier: Slow queries causing response delays")
        if stats['unique_users'] > 100 and stats['error_count'] > 50:
            points.append("   - Load Balancer: Uneven distribution under high load")
        if stats['error_rate'] > 5:
            points.append("   - Authentication Service: Connection/credential issues")
        if not points:
            points.append("   - System appears stable with low error rates")
        
        return "\n".join(points)
