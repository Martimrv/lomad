"""
Explanation Generator Module

Generates natural language explanations for AIS anomalies.
Currently uses template-based generation, designed to be easily
swapped with LLM-based generation in the future.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from .prompt_encoder import PromptEncoder, VesselState, AnomalyContext


@dataclass
class Explanation:
    """Generated explanation for a vessel's anomalies"""
    mmsi: int
    risk_level: str
    summary: str
    detailed_explanation: str
    anomaly_breakdown: List[Dict[str, Any]]
    recommendations: List[str]
    
    # Metadata
    anomaly_count: int
    anomaly_rate: float
    avg_error_km: float
    
    def to_dict(self) -> Dict:
        return {
            "mmsi": self.mmsi,
            "risk_level": self.risk_level,
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "anomaly_breakdown": self.anomaly_breakdown,
            "recommendations": self.recommendations,
            "metadata": {
                "anomaly_count": self.anomaly_count,
                "anomaly_rate": self.anomaly_rate,
                "avg_error_km": round(self.avg_error_km, 2)
            }
        }


class ExplanationGenerator(ABC):
    """
    Abstract base class for explanation generators.
    
    Subclasses can implement:
    - TemplateGenerator: Rule-based template filling
    - LLMGenerator: OpenAI, Ollama, HuggingFace, etc.
    """
    
    @abstractmethod
    def generate(
        self,
        mmsi: int,
        points: List[Dict],
        metrics: Optional[Dict] = None
    ) -> Explanation:
        """Generate an explanation for a vessel's anomalies"""
        pass


class TemplateGenerator(ExplanationGenerator):
    """
    Template-based explanation generator.
    
    Uses predefined templates and rules to generate human-readable
    explanations based on anomaly characteristics.
    """
    
    # Summary templates by risk level
    SUMMARY_TEMPLATES = {
        "critical": "CRITICAL: Vessel {mmsi} shows highly irregular behavior with {anomaly_count} anomalies detected ({anomaly_rate:.1f}% of observations).",
        "high": "WARNING: Vessel {mmsi} exhibits significant deviations from expected trajectory with {anomaly_count} anomalies ({anomaly_rate:.1f}%).",
        "medium": "ATTENTION: Vessel {mmsi} has moderate irregularities with {anomaly_count} anomalies detected ({anomaly_rate:.1f}%).",
        "low": "INFO: Vessel {mmsi} shows minor deviations with {anomaly_count} anomalies ({anomaly_rate:.1f}%). Behavior is mostly within expected parameters.",
        "none": "OK: Vessel {mmsi} trajectory is within normal parameters. No significant anomalies detected."
    }
    
    # Detailed explanation templates by anomaly type
    ANOMALY_TYPE_TEMPLATES = {
        "position": {
            "high": "The vessel deviated significantly from its predicted position, with an average error of {avg_error:.2f} km. This could indicate GPS spoofing, AIS manipulation, or an unexpected course change.",
            "medium": "Moderate position deviations were detected, averaging {avg_error:.2f} km from predicted locations. The vessel may be maneuvering in an unusual pattern.",
            "low": "Minor position variations of approximately {avg_error:.2f} km were observed, likely due to normal navigation adjustments."
        },
        "speed": {
            "high": "Significant speed anomalies detected, with deviations up to {max_speed_error:.1f} knots from expected values. This could indicate engine issues, evasive maneuvers, or data manipulation.",
            "medium": "Moderate speed variations observed, with some readings differing by {max_speed_error:.1f} knots from predictions.",
            "low": "Minor speed fluctuations detected, within acceptable operational variance."
        },
        "course": {
            "high": "Sharp course changes of up to {max_course_error:.0f}° were detected, significantly different from predicted heading. This suggests sudden maneuvers or potential navigation system issues.",
            "medium": "Moderate heading deviations of up to {max_course_error:.0f}° observed, indicating non-standard navigation patterns.",
            "low": "Minor course adjustments detected, consistent with normal navigation."
        },
        "combined": {
            "high": "Multiple anomaly indicators triggered simultaneously, including position ({avg_error:.2f} km deviation), speed, and course changes. This combination suggests coordinated irregular behavior requiring immediate attention.",
            "medium": "Several metrics show concurrent deviations from expected values. The vessel's overall behavior pattern is moderately irregular.",
            "low": "Minor concurrent variations in position, speed, and course detected, but within acceptable ranges."
        }
    }
    
    # Recommendations by severity
    RECOMMENDATIONS = {
        "critical": [
            "Immediate verification of vessel identity and position recommended",
            "Cross-reference with other tracking systems (radar, satellite)",
            "Consider alerting maritime authorities",
            "Monitor for potential AIS spoofing indicators"
        ],
        "high": [
            "Verify vessel position through secondary sources",
            "Review recent trajectory history for patterns",
            "Continue enhanced monitoring"
        ],
        "medium": [
            "Monitor vessel for continued irregularities",
            "Review if behavior corresponds to known maritime activities"
        ],
        "low": [
            "Standard monitoring procedures sufficient",
            "No immediate action required"
        ],
        "none": [
            "Continue normal monitoring"
        ]
    }
    
    def __init__(self):
        self.encoder = PromptEncoder()
    
    def generate(
        self,
        mmsi: int,
        points: List[Dict],
        metrics: Optional[Dict] = None
    ) -> Explanation:
        """
        Generate a template-based explanation for vessel anomalies.
        
        Args:
            mmsi: Vessel identifier
            points: List of prediction points with anomaly flags
            metrics: Optional pre-computed metrics for the vessel
        
        Returns:
            Explanation object with natural language descriptions
        """
        if not points:
            return self._generate_no_data_explanation(mmsi)
        
        # Encode trajectory summary
        traj_summary = self.encoder.encode_trajectory_summary(points)
        
        # Analyze anomaly points
        anomaly_points = [p for p in points if p.get("is_anomaly", 0) == 1]
        anomaly_contexts = self._analyze_anomaly_points(anomaly_points)
        
        # Determine overall risk level
        risk_level = self._determine_overall_risk(traj_summary, anomaly_contexts)
        
        # Generate summary
        summary = self._generate_summary(mmsi, traj_summary, risk_level)
        
        # Generate detailed explanation
        detailed = self._generate_detailed_explanation(
            anomaly_contexts, traj_summary, metrics
        )
        
        # Generate anomaly breakdown
        breakdown = self._generate_breakdown(anomaly_contexts, points)
        
        # Get recommendations
        recommendations = self.RECOMMENDATIONS.get(risk_level, self.RECOMMENDATIONS["low"])
        
        return Explanation(
            mmsi=mmsi,
            risk_level=risk_level,
            summary=summary,
            detailed_explanation=detailed,
            anomaly_breakdown=breakdown,
            recommendations=recommendations,
            anomaly_count=traj_summary["anomaly_count"],
            anomaly_rate=traj_summary["anomaly_rate"] * 100,
            avg_error_km=traj_summary["avg_position_error_km"]
        )
    
    def _generate_no_data_explanation(self, mmsi: int) -> Explanation:
        """Generate explanation when no data is available"""
        return Explanation(
            mmsi=mmsi,
            risk_level="none",
            summary=f"No prediction data available for vessel {mmsi}.",
            detailed_explanation="Unable to analyze vessel behavior due to insufficient data points.",
            anomaly_breakdown=[],
            recommendations=["Ensure vessel has sufficient tracking data"],
            anomaly_count=0,
            anomaly_rate=0.0,
            avg_error_km=0.0
        )
    
    def _analyze_anomaly_points(self, anomaly_points: List[Dict]) -> List[AnomalyContext]:
        """Analyze each anomaly point and encode context"""
        contexts = []
        for point in anomaly_points:
            context = self.encoder.encode_anomaly_context(
                true_sog=point.get("true_sog", 0),
                pred_sog=point.get("pred_sog", 0),
                true_cog=point.get("true_cog", 0),
                pred_cog=point.get("pred_cog", 0),
                true_lat=point.get("true_latitude", 0),
                pred_lat=point.get("pred_latitude", 0),
                true_lon=point.get("true_longitude", 0),
                pred_lon=point.get("pred_longitude", 0),
                combined_error=point.get("combined_error", 0),
                is_anomaly=True
            )
            contexts.append(context)
        return contexts
    
    def _determine_overall_risk(
        self,
        traj_summary: Dict,
        anomaly_contexts: List[AnomalyContext]
    ) -> str:
        """Determine overall risk level from trajectory and anomalies"""
        if not anomaly_contexts:
            return "none"
        
        # Get highest severity from contexts
        severities = [c.severity for c in anomaly_contexts]
        severity_order = ["critical", "high", "medium", "low"]
        
        for severity in severity_order:
            if severity in severities:
                highest_severity = severity
                break
        else:
            highest_severity = "low"
        
        # Adjust based on anomaly rate
        anomaly_rate = traj_summary["anomaly_rate"]
        if anomaly_rate > 0.15 and highest_severity not in ["critical", "high"]:
            # Elevate risk if too many anomalies
            highest_severity = "high" if highest_severity == "medium" else "medium"
        
        return highest_severity
    
    def _generate_summary(
        self,
        mmsi: int,
        traj_summary: Dict,
        risk_level: str
    ) -> str:
        """Generate summary text"""
        template = self.SUMMARY_TEMPLATES.get(risk_level, self.SUMMARY_TEMPLATES["low"])
        return template.format(
            mmsi=mmsi,
            anomaly_count=traj_summary["anomaly_count"],
            anomaly_rate=traj_summary["anomaly_rate"] * 100
        )
    
    def _generate_detailed_explanation(
        self,
        anomaly_contexts: List[AnomalyContext],
        traj_summary: Dict,
        metrics: Optional[Dict]
    ) -> str:
        """Generate detailed explanation based on anomaly patterns"""
        if not anomaly_contexts:
            return "The vessel's trajectory closely matches predictions. All metrics are within normal parameters."
        
        # Determine dominant anomaly type
        type_counts = {}
        for ctx in anomaly_contexts:
            type_counts[ctx.anomaly_type] = type_counts.get(ctx.anomaly_type, 0) + 1
        
        dominant_type = max(type_counts, key=type_counts.get)
        
        # Get severity
        severities = [ctx.severity for ctx in anomaly_contexts]
        if "critical" in severities or "high" in severities:
            severity = "high"
        elif "medium" in severities:
            severity = "medium"
        else:
            severity = "low"
        
        # Calculate max errors for templates
        max_speed_error = max((ctx.speed_error_knots for ctx in anomaly_contexts), default=0)
        max_course_error = max((ctx.course_error_deg for ctx in anomaly_contexts), default=0)
        avg_error = traj_summary["avg_position_error_km"]
        
        # Get appropriate template
        type_templates = self.ANOMALY_TYPE_TEMPLATES.get(
            dominant_type, 
            self.ANOMALY_TYPE_TEMPLATES["combined"]
        )
        template = type_templates.get(severity, type_templates["low"])
        
        # Fill template
        explanation = template.format(
            avg_error=avg_error,
            max_speed_error=max_speed_error,
            max_course_error=max_course_error
        )
        
        # Add trend information
        trend = traj_summary.get("trend", "stable")
        if trend == "increasing":
            explanation += " The anomaly frequency appears to be increasing over time, suggesting evolving irregular behavior."
        elif trend == "decreasing":
            explanation += " The anomaly frequency is decreasing, suggesting the vessel may be returning to normal operations."
        
        return explanation
    
    def _generate_breakdown(
        self,
        anomaly_contexts: List[AnomalyContext],
        all_points: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate breakdown of anomaly factors"""
        if not anomaly_contexts:
            return []
        
        # Aggregate by type
        type_stats = {}
        for ctx in anomaly_contexts:
            atype = ctx.anomaly_type
            if atype not in type_stats:
                type_stats[atype] = {
                    "count": 0,
                    "position_errors": [],
                    "speed_errors": [],
                    "course_errors": [],
                    "severities": []
                }
            type_stats[atype]["count"] += 1
            type_stats[atype]["position_errors"].append(ctx.position_error_km)
            type_stats[atype]["speed_errors"].append(ctx.speed_error_knots)
            type_stats[atype]["course_errors"].append(ctx.course_error_deg)
            type_stats[atype]["severities"].append(ctx.severity)
        
        breakdown = []
        type_labels = {
            "position": "Position Deviation",
            "speed": "Speed Anomaly",
            "course": "Course Change",
            "combined": "Multiple Factors",
            "minor": "Minor Irregularity",
            "none": "Normal"
        }
        
        for atype, stats in type_stats.items():
            import numpy as np
            
            # Determine dominant severity for this type
            sev_counts = {}
            for s in stats["severities"]:
                sev_counts[s] = sev_counts.get(s, 0) + 1
            dominant_sev = max(sev_counts, key=sev_counts.get)
            
            breakdown.append({
                "type": atype,
                "label": type_labels.get(atype, atype.title()),
                "count": stats["count"],
                "severity": dominant_sev,
                "avg_position_error_km": round(float(np.mean(stats["position_errors"])), 2),
                "avg_speed_error_knots": round(float(np.mean(stats["speed_errors"])), 1),
                "avg_course_error_deg": round(float(np.mean(stats["course_errors"])), 0),
                "description": self._get_type_description(atype, dominant_sev)
            })
        
        # Sort by count (most frequent first)
        breakdown.sort(key=lambda x: x["count"], reverse=True)
        
        return breakdown
    
    def _get_type_description(self, anomaly_type: str, severity: str) -> str:
        """Get a short description for an anomaly type"""
        descriptions = {
            "position": {
                "high": "Significant position deviations from predicted trajectory",
                "medium": "Moderate position drift from expected route",
                "low": "Minor position variations"
            },
            "speed": {
                "high": "Major speed irregularities detected",
                "medium": "Notable speed fluctuations",
                "low": "Minor speed variations"
            },
            "course": {
                "high": "Sharp unexpected course changes",
                "medium": "Moderate heading deviations",
                "low": "Minor course adjustments"
            },
            "combined": {
                "high": "Multiple concurrent anomaly indicators",
                "medium": "Several metrics showing deviations",
                "low": "Minor combined variations"
            }
        }
        
        type_descs = descriptions.get(anomaly_type, descriptions["combined"])
        return type_descs.get(severity, type_descs["low"])


class TrajectoryAnomalyGenerator(ExplanationGenerator):
    """
    Template-based explanation generator for trajectory-based anomalies.
    
    Handles anomalies detected by direct trajectory analysis:
    - Speed changes (acceleration/deceleration)
    - Course changes (sharp turns)
    - Loitering
    - Drift
    - Zigzag patterns
    """
    
    # Labels for anomaly types
    ANOMALY_LABELS = {
        "speed_change_critical": "Critical Speed Change",
        "speed_change_warning": "Speed Change Warning",
        "course_change_critical": "Critical Course Change",
        "course_change_warning": "Course Change Warning",
        "speed_too_high": "Excessive Speed",
        "speed_too_low": "Very Low Speed",
        "loitering": "Loitering Behavior",
        "drift_detected": "Drift Detected",
        "zigzag_pattern": "Zigzag Pattern",
        "none": "Normal"
    }
    
    # Descriptions for each anomaly type
    ANOMALY_DESCRIPTIONS = {
        "speed_change_critical": "Vessel experienced a severe sudden change in speed (>6 knots). This could indicate emergency maneuvers, equipment failure, or intentional evasive action.",
        "speed_change_warning": "Moderate sudden speed change detected (3-6 knots). May indicate maneuvering or response to external conditions.",
        "course_change_critical": "Sharp sudden turn detected (>60°). This represents a significant deviation from the vessel's previous heading.",
        "course_change_warning": "Moderate sudden course change (30-60°). The vessel altered its heading noticeably.",
        "speed_too_high": "Vessel traveling at unusually high speed (>25 knots). May warrant verification of vessel type and operating conditions.",
        "speed_too_low": "Vessel nearly stationary (<0.5 knots). Could indicate anchoring, drifting, or equipment issues.",
        "loitering": "Vessel remaining in a confined area at low speed. This pattern may indicate waiting, searching, or suspicious activity.",
        "drift_detected": "Heading differs significantly from course over ground. Vessel appears to be pushed by currents or wind.",
        "zigzag_pattern": "Frequent alternating course changes detected. May indicate searching, fishing operations, or evasive maneuvering."
    }
    
    # Severity mapping
    SEVERITY_MAP = {
        "none": 0,
        "speed_too_low": 1,
        "loitering": 2,
        "drift_detected": 2,
        "zigzag_pattern": 3,
        "speed_change_warning": 3,
        "course_change_warning": 3,
        "speed_too_high": 4,
        "speed_change_critical": 5,
        "course_change_critical": 5
    }
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        "critical": 5,
        "high": 4,
        "medium": 3,
        "low": 1
    }
    
    # Summary templates
    SUMMARY_TEMPLATES = {
        "critical": "CRITICAL: Vessel {mmsi} shows highly irregular movement patterns with {anomaly_count} trajectory anomalies ({anomaly_rate:.1f}% of observations).",
        "high": "WARNING: Vessel {mmsi} exhibits significant trajectory irregularities with {anomaly_count} anomalies ({anomaly_rate:.1f}%).",
        "medium": "ATTENTION: Vessel {mmsi} has notable trajectory changes with {anomaly_count} anomalies ({anomaly_rate:.1f}%).",
        "low": "INFO: Vessel {mmsi} shows minor trajectory irregularities with {anomaly_count} anomalies ({anomaly_rate:.1f}%).",
        "none": "OK: Vessel {mmsi} trajectory is normal. No significant anomalies detected."
    }
    
    # Recommendations by risk level
    RECOMMENDATIONS = {
        "critical": [
            "Immediate investigation of vessel behavior recommended",
            "Review full trajectory history for context",
            "Consider alerting maritime authorities if behavior persists",
            "Cross-reference with AIS data from other sources"
        ],
        "high": [
            "Enhanced monitoring recommended",
            "Review recent trajectory for patterns",
            "Verify vessel identity and reported destination"
        ],
        "medium": [
            "Monitor vessel for continued irregularities",
            "Note trajectory pattern for future reference"
        ],
        "low": [
            "Standard monitoring procedures sufficient",
            "Minor variations within expected parameters"
        ],
        "none": [
            "Continue normal monitoring"
        ]
    }
    
    def generate(
        self,
        mmsi: int,
        points: List[Dict],
        metrics: Optional[Dict] = None
    ) -> Explanation:
        """
        Generate explanation for trajectory-based anomalies.
        
        Args:
            mmsi: Vessel identifier
            points: List of trajectory points with anomaly_type field
            metrics: Optional vessel metrics
        """
        if not points:
            return self._generate_no_data_explanation(mmsi)
        
        # Filter anomaly points
        anomaly_points = [p for p in points if p.get("is_anomaly", 0) == 1]
        anomaly_count = len(anomaly_points)
        anomaly_rate = anomaly_count / len(points) if points else 0
        
        # Count anomaly types
        type_counts = {}
        max_severity = 0
        for p in anomaly_points:
            atype = p.get("anomaly_type", "none")
            type_counts[atype] = type_counts.get(atype, 0) + 1
            severity = self.SEVERITY_MAP.get(atype, 0)
            max_severity = max(max_severity, severity)
        
        # Determine risk level
        risk_level = self._severity_to_risk(max_severity, anomaly_rate)
        
        # Generate summary
        summary = self.SUMMARY_TEMPLATES.get(risk_level, self.SUMMARY_TEMPLATES["low"]).format(
            mmsi=mmsi,
            anomaly_count=anomaly_count,
            anomaly_rate=anomaly_rate * 100
        )
        
        # Generate detailed explanation
        detailed = self._generate_detailed_explanation(type_counts, anomaly_points)
        
        # Generate breakdown
        breakdown = self._generate_breakdown(type_counts, anomaly_points)
        
        # Get recommendations
        recommendations = self.RECOMMENDATIONS.get(risk_level, self.RECOMMENDATIONS["low"])
        
        # Calculate avg error (use max speed/course change as proxy)
        avg_speed_change = sum(p.get("speed_change", 0) or 0 for p in anomaly_points) / max(1, len(anomaly_points))
        
        return Explanation(
            mmsi=mmsi,
            risk_level=risk_level,
            summary=summary,
            detailed_explanation=detailed,
            anomaly_breakdown=breakdown,
            recommendations=recommendations,
            anomaly_count=anomaly_count,
            anomaly_rate=anomaly_rate * 100,
            avg_error_km=round(avg_speed_change * 0.1, 2)  # Rough km estimate
        )
    
    def _severity_to_risk(self, max_severity: int, anomaly_rate: float) -> str:
        """Convert max severity to risk level, adjusted by anomaly rate"""
        if max_severity >= 5:
            return "critical"
        elif max_severity >= 4:
            return "high"
        elif max_severity >= 3 or anomaly_rate > 0.15:
            return "medium" if max_severity >= 2 else "low"
        elif max_severity >= 1:
            return "low"
        return "none"
    
    def _generate_no_data_explanation(self, mmsi: int) -> Explanation:
        """Generate explanation when no data is available"""
        return Explanation(
            mmsi=mmsi,
            risk_level="none",
            summary=f"No trajectory data available for vessel {mmsi}.",
            detailed_explanation="Unable to analyze vessel behavior due to insufficient data points.",
            anomaly_breakdown=[],
            recommendations=["Ensure vessel has sufficient tracking data"],
            anomaly_count=0,
            anomaly_rate=0.0,
            avg_error_km=0.0
        )
    
    def _generate_detailed_explanation(
        self,
        type_counts: Dict[str, int],
        anomaly_points: List[Dict]
    ) -> str:
        """Generate detailed explanation based on detected anomaly types"""
        if not type_counts:
            return "The vessel's trajectory shows normal movement patterns. No significant anomalies detected."
        
        # Find dominant anomaly type
        dominant_type = max(type_counts, key=type_counts.get)
        
        # Start with dominant type description
        explanation_parts = []
        
        if dominant_type in self.ANOMALY_DESCRIPTIONS:
            explanation_parts.append(self.ANOMALY_DESCRIPTIONS[dominant_type])
        
        # Add context about other types if present
        other_types = [t for t in type_counts if t != dominant_type and t != "none"]
        if other_types:
            other_labels = [self.ANOMALY_LABELS.get(t, t) for t in other_types[:2]]
            explanation_parts.append(
                f"Additionally, {' and '.join(other_labels).lower()} behaviors were also detected."
            )
        
        # Add statistics
        if anomaly_points:
            max_speed_change = max((p.get("speed_change") or 0 for p in anomaly_points), default=0)
            max_course_change = max((p.get("course_change") or 0 for p in anomaly_points), default=0)
            
            if max_speed_change > 0 or max_course_change > 0:
                stats = []
                if max_speed_change > 0:
                    stats.append(f"maximum speed change of {max_speed_change:.1f} knots")
                if max_course_change > 0:
                    stats.append(f"maximum course change of {max_course_change:.0f}°")
                explanation_parts.append(f"The vessel showed {' and '.join(stats)}.")
        
        return " ".join(explanation_parts)
    
    def _generate_breakdown(
        self,
        type_counts: Dict[str, int],
        anomaly_points: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate breakdown of anomaly types"""
        breakdown = []
        
        for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            if atype == "none":
                continue
            
            # Get points of this type
            type_points = [p for p in anomaly_points if p.get("anomaly_type") == atype]
            
            # Calculate averages
            avg_speed_change = sum(p.get("speed_change") or 0 for p in type_points) / max(1, len(type_points))
            avg_course_change = sum(p.get("course_change") or 0 for p in type_points) / max(1, len(type_points))
            
            severity = self.SEVERITY_MAP.get(atype, 0)
            severity_label = "critical" if severity >= 5 else "high" if severity >= 4 else "medium" if severity >= 3 else "low"
            
            breakdown.append({
                "type": atype,
                "label": self.ANOMALY_LABELS.get(atype, atype),
                "count": count,
                "severity": severity_label,
                "avg_position_error_km": round(avg_speed_change * 0.1, 2),
                "avg_speed_error_knots": round(avg_speed_change, 1),
                "avg_course_error_deg": round(avg_course_change, 0),
                "description": self.ANOMALY_DESCRIPTIONS.get(atype, "Trajectory irregularity detected")
            })
        
        return breakdown


# Future LLM-based generator placeholder
class LLMGenerator(ExplanationGenerator):
    """
    LLM-based explanation generator (placeholder for future implementation).
    
    Can be implemented with:
    - OpenAI API (GPT-3.5/GPT-4)
    - Local Ollama (Llama, Mistral)
    - HuggingFace Transformers
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.encoder = PromptEncoder()
        # TODO: Initialize LLM client
    
    def generate(
        self,
        mmsi: int,
        points: List[Dict],
        metrics: Optional[Dict] = None
    ) -> Explanation:
        """Generate explanation using LLM"""
        # TODO: Implement LLM-based generation
        # 1. Encode data using prompt encoder
        # 2. Format as structured prompt
        # 3. Call LLM API
        # 4. Parse response into Explanation object
        
        # For now, fall back to template generator
        template_gen = TemplateGenerator()
        return template_gen.generate(mmsi, points, metrics)

