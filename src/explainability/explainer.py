#!/usr/bin/env python3
"""
AI Explainability for Adaptive Intersection AI Copilot.
This module provides natural language explanations for AI decisions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficDecisionExplainer:
    """Explains traffic control decisions in natural language."""
    
    def __init__(self):
        """Initialize the explainer."""
        self.decision_history = []
        self.explanation_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load explanation templates."""
        return {
            'phase_change': [
                "Changed to {phase_name} because {reason}",
                "Switched to {phase_name} due to {reason}",
                "Extended {phase_name} for {reason}"
            ],
            'wait_time': [
                "{count} pedestrians waiting {time} seconds",
                "{count} vehicles waiting {time} seconds",
                "Average wait time: {time} seconds for {type}"
            ],
            'safety': [
                "Safety concern: {issue}",
                "Preventing risky behavior: {action}",
                "Ensuring safe crossing for {group}"
            ],
            'efficiency': [
                "Optimizing flow: {metric}",
                "Reducing congestion: {action}",
                "Improving throughput: {result}"
            ],
            'fairness': [
                "Balancing {stakeholder1} and {stakeholder2} needs",
                "Ensuring fair access for {group}",
                "Prioritizing based on {criteria}"
            ]
        }
    
    def explain_decision(self, decision_data: Dict) -> str:
        """
        Generate explanation for a traffic control decision.
        
        Args:
            decision_data: Dictionary containing decision information
            
        Returns:
            Natural language explanation
        """
        try:
            # Extract key information
            action = decision_data.get('action', 0)
            current_state = decision_data.get('state', {})
            previous_state = decision_data.get('previous_state', {})
            phase_info = decision_data.get('phase_info', {})
            
            # Generate explanation components
            explanation_parts = []
            
            # Phase change explanation
            if action != 0:
                phase_explanation = self._explain_phase_change(
                    action, current_state, previous_state, phase_info
                )
                explanation_parts.append(phase_explanation)
            
            # Wait time explanation
            wait_explanation = self._explain_wait_times(current_state)
            if wait_explanation:
                explanation_parts.append(wait_explanation)
            
            # Safety explanation
            safety_explanation = self._explain_safety_considerations(current_state)
            if safety_explanation:
                explanation_parts.append(safety_explanation)
            
            # Efficiency explanation
            efficiency_explanation = self._explain_efficiency(current_state, previous_state)
            if efficiency_explanation:
                explanation_parts.append(efficiency_explanation)
            
            # Fairness explanation
            fairness_explanation = self._explain_fairness(current_state)
            if fairness_explanation:
                explanation_parts.append(fairness_explanation)
            
            # Combine explanations
            full_explanation = self._combine_explanations(explanation_parts)
            
            # Store in history
            self._store_explanation(decision_data, full_explanation)
            
            return full_explanation
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return "Unable to generate explanation at this time."
    
    def _explain_phase_change(self, action: int, current_state: Dict, 
                            previous_state: Dict, phase_info: Dict) -> str:
        """Explain phase change decision."""
        phase_names = {
            0: "North-South Green",
            1: "North-South Yellow", 
            2: "East-West Green",
            3: "East-West Yellow"
        }
        
        phase_name = phase_names.get(action, "Unknown Phase")
        
        # Determine reason for phase change
        reasons = []
        
        # Check pedestrian demand
        pedestrians_ns = current_state.get('pedestrians_ns', 0)
        pedestrians_ew = current_state.get('pedestrians_ew', 0)
        
        if pedestrians_ns > pedestrians_ew + 2:
            reasons.append(f"high pedestrian demand from north-south ({pedestrians_ns} people)")
        elif pedestrians_ew > pedestrians_ns + 2:
            reasons.append(f"high pedestrian demand from east-west ({pedestrians_ew} people)")
        
        # Check vehicle demand
        vehicles_ns = current_state.get('vehicles_ns', 0)
        vehicles_ew = current_state.get('vehicles_ew', 0)
        
        if vehicles_ns > vehicles_ew + 3:
            reasons.append(f"high vehicle demand from north-south ({vehicles_ns} vehicles)")
        elif vehicles_ew > vehicles_ns + 3:
            reasons.append(f"high vehicle demand from east-west ({vehicles_ew} vehicles)")
        
        # Check wait times
        avg_pedestrian_wait = current_state.get('avg_pedestrian_wait', 0)
        avg_vehicle_wait = current_state.get('avg_vehicle_wait', 0)
        
        if avg_pedestrian_wait > 30:
            reasons.append(f"pedestrians waiting too long ({avg_pedestrian_wait:.1f}s)")
        if avg_vehicle_wait > 45:
            reasons.append(f"vehicles waiting too long ({avg_vehicle_wait:.1f}s)")
        
        # Default reason
        if not reasons:
            reasons.append("balanced traffic flow")
        
        reason = " and ".join(reasons)
        
        # Select template
        template = np.random.choice(self.explanation_templates['phase_change'])
        return template.format(phase_name=phase_name, reason=reason)
    
    def _explain_wait_times(self, current_state: Dict) -> str:
        """Explain current wait times."""
        explanations = []
        
        pedestrians_ns = current_state.get('pedestrians_ns', 0)
        pedestrians_ew = current_state.get('pedestrians_ew', 0)
        avg_pedestrian_wait = current_state.get('avg_pedestrian_wait', 0)
        
        vehicles_ns = current_state.get('vehicles_ns', 0)
        vehicles_ew = current_state.get('vehicles_ew', 0)
        avg_vehicle_wait = current_state.get('avg_vehicle_wait', 0)
        
        # Pedestrian wait times
        if pedestrians_ns > 0 or pedestrians_ew > 0:
            total_pedestrians = pedestrians_ns + pedestrians_ew
            if avg_pedestrian_wait > 20:
                explanations.append(
                    f"{total_pedestrians} pedestrians waiting an average of {avg_pedestrian_wait:.1f} seconds"
                )
        
        # Vehicle wait times
        if vehicles_ns > 0 or vehicles_ew > 0:
            total_vehicles = vehicles_ns + vehicles_ew
            if avg_vehicle_wait > 30:
                explanations.append(
                    f"{total_vehicles} vehicles waiting an average of {avg_vehicle_wait:.1f} seconds"
                )
        
        return "; ".join(explanations) if explanations else ""
    
    def _explain_safety_considerations(self, current_state: Dict) -> str:
        """Explain safety considerations."""
        explanations = []
        
        avg_pedestrian_wait = current_state.get('avg_pedestrian_wait', 0)
        pedestrians_ns = current_state.get('pedestrians_ns', 0)
        pedestrians_ew = current_state.get('pedestrians_ew', 0)
        
        # High wait time safety concern
        if avg_pedestrian_wait > 45:
            explanations.append("preventing risky jaywalking due to long wait times")
        
        # High pedestrian count safety concern
        if pedestrians_ns + pedestrians_ew > 15:
            explanations.append("ensuring safe crossing for large pedestrian group")
        
        # Vulnerable groups
        hour = current_state.get('hour', 12)
        if 7 <= hour <= 9 or 14 <= hour <= 16:  # School hours
            explanations.append("prioritizing safety for students during school hours")
        
        return "; ".join(explanations) if explanations else ""
    
    def _explain_efficiency(self, current_state: Dict, previous_state: Dict) -> str:
        """Explain efficiency considerations."""
        explanations = []
        
        # Throughput improvement
        current_throughput = (current_state.get('vehicles_ns', 0) + 
                            current_state.get('vehicles_ew', 0) +
                            current_state.get('pedestrians_ns', 0) + 
                            current_state.get('pedestrians_ew', 0))
        
        if previous_state:
            previous_throughput = (previous_state.get('vehicles_ns', 0) + 
                                 previous_state.get('vehicles_ew', 0) +
                                 previous_state.get('pedestrians_ns', 0) + 
                                 previous_state.get('pedestrians_ew', 0))
            
            if current_throughput > previous_throughput:
                explanations.append(f"improving throughput ({current_throughput} vs {previous_throughput})")
        
        # Congestion reduction
        avg_vehicle_wait = current_state.get('avg_vehicle_wait', 0)
        if avg_vehicle_wait < 20:
            explanations.append("maintaining smooth vehicle flow")
        
        return "; ".join(explanations) if explanations else ""
    
    def _explain_fairness(self, current_state: Dict) -> str:
        """Explain fairness considerations."""
        explanations = []
        
        pedestrians_ns = current_state.get('pedestrians_ns', 0)
        pedestrians_ew = current_state.get('pedestrians_ew', 0)
        vehicles_ns = current_state.get('vehicles_ns', 0)
        vehicles_ew = current_state.get('vehicles_ew', 0)
        
        # Balance between directions
        if abs(pedestrians_ns - pedestrians_ew) <= 2:
            explanations.append("balancing pedestrian access across all directions")
        
        if abs(vehicles_ns - vehicles_ew) <= 3:
            explanations.append("balancing vehicle flow across all directions")
        
        # Balance between pedestrians and vehicles
        total_pedestrians = pedestrians_ns + pedestrians_ew
        total_vehicles = vehicles_ns + vehicles_ew
        
        if total_pedestrians > 0 and total_vehicles > 0:
            explanations.append("balancing needs of pedestrians and vehicles")
        
        # Accessibility considerations
        hour = current_state.get('hour', 12)
        if 9 <= hour <= 11 or 14 <= hour <= 16:  # Peak hours
            explanations.append("ensuring fair access during peak hours")
        
        return "; ".join(explanations) if explanations else ""
    
    def _combine_explanations(self, explanation_parts: List[str]) -> str:
        """Combine explanation parts into a coherent explanation."""
        if not explanation_parts:
            return "No specific explanation available."
        
        # Filter out empty explanations
        valid_parts = [part for part in explanation_parts if part.strip()]
        
        if len(valid_parts) == 1:
            return valid_parts[0]
        elif len(valid_parts) == 2:
            return f"{valid_parts[0]}. {valid_parts[1]}"
        else:
            return f"{valid_parts[0]}. {valid_parts[1]}. {valid_parts[2]}"
    
    def _store_explanation(self, decision_data: Dict, explanation: str):
        """Store explanation in history."""
        explanation_record = {
            'timestamp': datetime.now().isoformat(),
            'decision_data': decision_data,
            'explanation': explanation
        }
        
        self.decision_history.append(explanation_record)
        
        # Keep only last 100 explanations
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
    
    def get_explanation_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent explanation history.
        
        Args:
            limit: Maximum number of explanations to return
            
        Returns:
            List of recent explanations
        """
        return self.decision_history[-limit:]
    
    def generate_summary_report(self, time_period: str = "last_hour") -> str:
        """
        Generate a summary report of decisions and explanations.
        
        Args:
            time_period: Time period for summary
            
        Returns:
            Summary report
        """
        if not self.decision_history:
            return "No decision history available."
        
        # Filter by time period (simplified)
        recent_explanations = self.decision_history[-20:]  # Last 20 decisions
        
        # Count decision types
        phase_changes = 0
        safety_decisions = 0
        efficiency_decisions = 0
        
        for record in recent_explanations:
            explanation = record['explanation'].lower()
            if 'changed to' in explanation or 'switched to' in explanation:
                phase_changes += 1
            if 'safety' in explanation or 'risky' in explanation:
                safety_decisions += 1
            if 'efficiency' in explanation or 'throughput' in explanation:
                efficiency_decisions += 1
        
        # Generate summary
        summary = f"""
Traffic Control Summary ({time_period}):
- Total decisions made: {len(recent_explanations)}
- Phase changes: {phase_changes}
- Safety-focused decisions: {safety_decisions}
- Efficiency-focused decisions: {efficiency_decisions}

Recent key decisions:
"""
        
        # Add recent explanations
        for record in recent_explanations[-3:]:
            summary += f"- {record['explanation']}\n"
        
        return summary
    
    def export_explanations(self, filepath: str):
        """Export explanation history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.decision_history, f, indent=2)
        logger.info(f"Explanations exported to {filepath}")

class PerformanceAnalyzer:
    """Analyzes performance metrics and provides insights."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.metrics_history = []
        
    def analyze_performance(self, metrics: Dict) -> Dict[str, str]:
        """
        Analyze performance metrics and provide insights.
        
        Args:
            metrics: Dictionary containing performance metrics
            
        Returns:
            Dictionary of insights
        """
        insights = {}
        
        # Wait time analysis
        avg_vehicle_wait = metrics.get('avg_vehicle_wait', 0)
        avg_pedestrian_wait = metrics.get('avg_pedestrian_wait', 0)
        
        if avg_pedestrian_wait > 30:
            insights['pedestrian_wait'] = "Pedestrian wait times are high. Consider prioritizing pedestrian phases."
        elif avg_pedestrian_wait < 15:
            insights['pedestrian_wait'] = "Pedestrian wait times are optimal."
        
        if avg_vehicle_wait > 45:
            insights['vehicle_wait'] = "Vehicle wait times are high. Consider optimizing vehicle flow."
        elif avg_vehicle_wait < 25:
            insights['vehicle_wait'] = "Vehicle wait times are optimal."
        
        # Throughput analysis
        total_throughput = (metrics.get('vehicles', 0) + metrics.get('pedestrians', 0))
        if total_throughput > 20:
            insights['throughput'] = "High traffic volume detected. System is handling peak demand well."
        elif total_throughput < 5:
            insights['throughput'] = "Low traffic volume. System is operating efficiently."
        
        # Safety analysis
        if avg_pedestrian_wait > 45 and metrics.get('pedestrians', 0) > 10:
            insights['safety'] = "Safety concern: High pedestrian wait times with many pedestrians present."
        
        return insights
    
    def compare_controllers(self, fixed_metrics: Dict, rl_metrics: Dict) -> str:
        """
        Compare performance between fixed-time and RL controllers.
        
        Args:
            fixed_metrics: Fixed-time controller metrics
            rl_metrics: RL controller metrics
            
        Returns:
            Comparison summary
        """
        # Calculate improvements
        vehicle_wait_improvement = ((fixed_metrics.get('avg_vehicle_wait', 0) - 
                                   rl_metrics.get('avg_vehicle_wait', 0)) / 
                                  max(fixed_metrics.get('avg_vehicle_wait', 1), 1) * 100)
        
        pedestrian_wait_improvement = ((fixed_metrics.get('avg_pedestrian_wait', 0) - 
                                      rl_metrics.get('avg_pedestrian_wait', 0)) / 
                                     max(fixed_metrics.get('avg_pedestrian_wait', 1), 1) * 100)
        
        comparison = f"""
Controller Performance Comparison:

Fixed-Time Controller:
- Average vehicle wait: {fixed_metrics.get('avg_vehicle_wait', 0):.1f}s
- Average pedestrian wait: {fixed_metrics.get('avg_pedestrian_wait', 0):.1f}s

RL Controller:
- Average vehicle wait: {rl_metrics.get('avg_vehicle_wait', 0):.1f}s
- Average pedestrian wait: {rl_metrics.get('avg_pedestrian_wait', 0):.1f}s

Improvements:
- Vehicle wait time: {vehicle_wait_improvement:+.1f}%
- Pedestrian wait time: {pedestrian_wait_improvement:+.1f}%
"""
        
        if vehicle_wait_improvement > 0 or pedestrian_wait_improvement > 0:
            comparison += "\nThe RL controller shows improved performance!"
        else:
            comparison += "\nThe RL controller needs further training."
        
        return comparison

# Example usage
if __name__ == "__main__":
    # Initialize explainer
    explainer = TrafficDecisionExplainer()
    
    # Sample decision data
    decision_data = {
        'action': 2,  # East-West Green
        'state': {
            'pedestrians_ns': 3,
            'pedestrians_ew': 8,
            'vehicles_ns': 2,
            'vehicles_ew': 5,
            'avg_vehicle_wait': 25.0,
            'avg_pedestrian_wait': 35.0,
            'hour': 14
        },
        'previous_state': {
            'pedestrians_ns': 2,
            'pedestrians_ew': 6,
            'vehicles_ns': 1,
            'vehicles_ew': 4,
            'avg_vehicle_wait': 30.0,
            'avg_pedestrian_wait': 40.0
        },
        'phase_info': {
            'current_phase': 0,
            'phase_duration': 45
        }
    }
    
    # Generate explanation
    explanation = explainer.explain_decision(decision_data)
    print("TrafficDecisionExplainer initialized successfully!")
    print(f"Explanation: {explanation}")