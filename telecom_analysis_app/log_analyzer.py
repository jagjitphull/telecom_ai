import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class LogParser:
    """Parse and extract information from telecom logs"""
    
    # Common telecom log patterns
    PATTERNS = {
        'timestamp': r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}|^\d{2}:\d{2}:\d{2}',
        'error': r'ERROR|CRITICAL|FATAL|Exception',
        'warning': r'WARN|WARNING',
        'info': r'INFO',
        'call_id': r'[Cc]all[_-]?[Ii]d[:\s]+([A-Za-z0-9\-_]+)',
        'user': r'[Uu]ser[:\s]+([A-Za-z0-9\-_@\.]+)',
        'ip': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'status_code': r'\b(200|201|400|401|403|404|500|502|503|504)\b',
        'latency': r'latency[:\s=]+(\d+\.?\d*)\s*(ms|s)',
        'exception': r'(?:Exception|Error):\s*([^\n]+)',
    }
    
    def __init__(self):
        self.compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE) 
            for key, pattern in self.PATTERNS.items()
        }
    
    def parse_logs(self, log_content: str) -> Tuple[List[Dict], str]:
        """Parse log content and extract structured data"""
        try:
            lines = log_content.split('\n')
            parsed_logs = []
            raw_analysis = []
            
            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                
                log_entry = self._parse_line(line)
                parsed_logs.append(log_entry)
                
                # Extract specific issues
                if self.compiled_patterns['error'].search(line):
                    raw_analysis.append(f"Line {idx + 1}: ERROR - {line[:100]}")
                elif self.compiled_patterns['warning'].search(line):
                    raw_analysis.append(f"Line {idx + 1}: WARNING - {line[:100]}")
            
            return parsed_logs, '\n'.join(raw_analysis)
        except Exception as e:
            logger.error(f"Error parsing logs: {e}")
            return [], str(e)
    
    def _parse_line(self, line: str) -> Dict[str, Any]:
        """Parse a single log line"""
        entry = {
            'raw': line,
            'timestamp': self._extract_timestamp(line),
            'level': self._extract_level(line),
            'call_id': self._extract_call_id(line),
            'user': self._extract_user(line),
            'ip': self._extract_ip(line),
            'status_code': self._extract_status_code(line),
            'latency_ms': self._extract_latency(line),
            'exception': self._extract_exception(line),
        }
        return entry
    
    def _extract_timestamp(self, line: str) -> Optional[str]:
        match = self.compiled_patterns['timestamp'].search(line)
        return match.group(0) if match else None
    
    def _extract_level(self, line: str) -> str:
        if self.compiled_patterns['error'].search(line):
            return 'ERROR'
        elif self.compiled_patterns['warning'].search(line):
            return 'WARNING'
        elif self.compiled_patterns['info'].search(line):
            return 'INFO'
        return 'DEBUG'
    
    def _extract_call_id(self, line: str) -> Optional[str]:
        match = self.compiled_patterns['call_id'].search(line)
        return match.group(1) if match else None
    
    def _extract_user(self, line: str) -> Optional[str]:
        match = self.compiled_patterns['user'].search(line)
        return match.group(1) if match else None
    
    def _extract_ip(self, line: str) -> Optional[str]:
        match = self.compiled_patterns['ip'].search(line)
        return match.group(0) if match else None
    
    def _extract_status_code(self, line: str) -> Optional[str]:
        match = self.compiled_patterns['status_code'].search(line)
        return match.group(0) if match else None
    
    def _extract_latency(self, line: str) -> Optional[float]:
        match = self.compiled_patterns['latency'].search(line)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            return value * 1000 if unit.lower() == 's' else value
        return None
    
    def _extract_exception(self, line: str) -> Optional[str]:
        match = self.compiled_patterns['exception'].search(line)
        return match.group(1) if match else None
    
    def correlate_logs(self, parsed_logs: List[Dict]) -> Dict[str, List]:
        """Correlate logs by various fields"""
        correlations = {
            'by_call_id': defaultdict(list),
            'by_user': defaultdict(list),
            'by_ip': defaultdict(list),
            'by_level': defaultdict(list),
            'by_status_code': defaultdict(list),
            'errors_by_type': defaultdict(int),
        }
        
        for log in parsed_logs:
            if log['call_id']:
                correlations['by_call_id'][log['call_id']].append(log)
            if log['user']:
                correlations['by_user'][log['user']].append(log)
            if log['ip']:
                correlations['by_ip'][log['ip']].append(log)
            
            correlations['by_level'][log['level']].append(log)
            
            if log['status_code']:
                correlations['by_status_code'][log['status_code']].append(log)
            
            if log['exception']:
                correlations['errors_by_type'][log['exception']] += 1
        
        return dict(correlations)
    
    def generate_statistics(self, parsed_logs: List[Dict]) -> Dict[str, Any]:
        """Generate statistics from parsed logs"""
        stats = {
            'total_logs': len(parsed_logs),
            'error_count': sum(1 for log in parsed_logs if log['level'] == 'ERROR'),
            'warning_count': sum(1 for log in parsed_logs if log['level'] == 'WARNING'),
            'info_count': sum(1 for log in parsed_logs if log['level'] == 'INFO'),
            'avg_latency_ms': 0,
            'high_latency_count': 0,
            'unique_call_ids': len(set(log['call_id'] for log in parsed_logs if log['call_id'])),
            'unique_users': len(set(log['user'] for log in parsed_logs if log['user'])),
            'unique_ips': len(set(log['ip'] for log in parsed_logs if log['ip'])),
            'error_rate': 0,
        }
        
        latencies = [log['latency_ms'] for log in parsed_logs if log['latency_ms']]
        if latencies:
            stats['avg_latency_ms'] = sum(latencies) / len(latencies)
            stats['high_latency_count'] = sum(1 for lat in latencies if lat > 1000)
        
        if stats['total_logs'] > 0:
            stats['error_rate'] = (stats['error_count'] / stats['total_logs']) * 100
        
        return stats
