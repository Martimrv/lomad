# AIS-LLM Explanation Module

## Overview

The explanation module converts numerical AIS predictions and anomaly scores into **human-readable natural language explanations**. It's designed to help maritime operators understand *why* a vessel's behavior was flagged as anomalous.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Explanation Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   AIS Predictions ──→ Prompt Encoder ──→ Generator ──→ Text     │
│   (numbers)           (structured      (template/    (natural   │
│                        tokens)          LLM)          language) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Prompt Encoder (`prompt_encoder.py`)

Converts numerical AIS data into structured text tokens.

#### Token Format

```xml
<SPEED>12.5 kn</SPEED>
<COURSE>185°</COURSE>
<POSITION>(28.4160, -80.6220)</POSITION>
<ANOMALY_SCORE>0.82</ANOMALY_SCORE>
<RISK>HIGH</RISK>
<POSITION_ERROR>2.45 km</POSITION_ERROR>
```

#### Classes

**`VesselState`** - Encoded current vessel state
```python
@dataclass
class VesselState:
    mmsi: int
    speed_token: str      # "<SPEED>12.5 kn</SPEED>"
    course_token: str     # "<COURSE>185°</COURSE>"
    position_token: str   # "<POSITION>(lat, lon)</POSITION>"
    anomaly_score_token: str
    risk_level: str       # "low", "medium", "high", "critical"
    risk_token: str
    raw_values: Dict
```

**`AnomalyContext`** - Anomaly analysis context
```python
@dataclass
class AnomalyContext:
    has_anomaly: bool
    anomaly_type: str     # "position", "speed", "course", "combined"
    severity: str         # "low", "medium", "high", "critical"
    position_error_km: float
    speed_error_knots: float
    course_error_deg: float
    combined_error: float
    deviation_tokens: Dict[str, str]
```

#### Methods

| Method | Description |
|--------|-------------|
| `encode_vessel_state()` | Encode current position, speed, course, risk |
| `encode_anomaly_context()` | Analyze and encode anomaly details |
| `encode_trajectory_summary()` | Summarize trajectory patterns |

### 2. Explanation Generator (`generator.py`)

Generates natural language explanations from encoded data.

#### Abstract Interface

```python
class ExplanationGenerator(ABC):
    @abstractmethod
    def generate(self, mmsi: int, points: List[Dict], 
                 metrics: Optional[Dict]) -> Explanation:
        pass
```

#### Implementations

**`TemplateGenerator`** (Current)
- Rule-based template filling
- No external dependencies
- Fast and predictable

**`LLMGenerator`** (Placeholder for future)
- OpenAI API integration
- Local Ollama support
- HuggingFace Transformers

#### Output Structure

```python
@dataclass
class Explanation:
    mmsi: int
    risk_level: str           # "critical", "high", "medium", "low", "none"
    summary: str              # One-line summary
    detailed_explanation: str # Full analysis paragraph
    anomaly_breakdown: List   # Breakdown by anomaly type
    recommendations: List[str] # Actionable suggestions
    anomaly_count: int
    anomaly_rate: float
    avg_error_km: float
```

## API Usage

### REST Endpoint

```
GET /vessel/{mmsi}/explanation
```

**Response:**
```json
{
  "mmsi": 211002010,
  "risk_level": "critical",
  "summary": "CRITICAL: Vessel 211002010 shows highly irregular behavior with 5 anomalies detected (5.9% of observations).",
  "detailed_explanation": "Multiple anomaly indicators triggered simultaneously, including position (2.45 km deviation), speed, and course changes...",
  "anomaly_breakdown": [
    {
      "type": "combined",
      "label": "Multiple Factors",
      "count": 3,
      "severity": "high",
      "avg_position_error_km": 2.45,
      "avg_speed_error_knots": 3.2,
      "avg_course_error_deg": 25,
      "description": "Multiple concurrent anomaly indicators"
    }
  ],
  "recommendations": [
    "Immediate verification of vessel identity and position recommended",
    "Cross-reference with other tracking systems (radar, satellite)",
    "Consider alerting maritime authorities"
  ],
  "metadata": {
    "anomaly_count": 5,
    "anomaly_rate": 5.9,
    "avg_error_km": 2.45
  }
}
```

### Python Usage

```python
from backend.explanation import TemplateGenerator

generator = TemplateGenerator()

# Points from prediction data
points = [
    {
        "true_sog": 12.5, "pred_sog": 10.2,
        "true_cog": 185, "pred_cog": 160,
        "true_latitude": 28.416, "pred_latitude": 28.410,
        "true_longitude": -80.622, "pred_longitude": -80.615,
        "combined_error": 0.75,
        "is_anomaly": 1
    },
    # ... more points
]

explanation = generator.generate(mmsi=211002010, points=points)
print(explanation.summary)
print(explanation.detailed_explanation)
```

## Risk Level Classification

| Level | Combined Error | Description |
|-------|----------------|-------------|
| **Critical** | ≥ 0.8 | Highly irregular, immediate attention |
| **High** | ≥ 0.6 | Significant deviations |
| **Medium** | ≥ 0.4 | Moderate irregularities |
| **Low** | < 0.4 | Minor deviations |
| **None** | N/A | No anomalies detected |

## Anomaly Type Detection

The encoder analyzes errors to determine the primary anomaly type:

| Type | Trigger | Description |
|------|---------|-------------|
| `position` | Position error dominant | GPS spoofing, route deviation |
| `speed` | Speed error dominant | Engine issues, evasive maneuvers |
| `course` | Course error dominant | Sudden heading changes |
| `combined` | Multiple factors | Coordinated irregular behavior |
| `minor` | Low errors | Normal operational variance |

## Templates

### Summary Templates

```python
SUMMARY_TEMPLATES = {
    "critical": "CRITICAL: Vessel {mmsi} shows highly irregular behavior...",
    "high": "WARNING: Vessel {mmsi} exhibits significant deviations...",
    "medium": "ATTENTION: Vessel {mmsi} has moderate irregularities...",
    "low": "INFO: Vessel {mmsi} shows minor deviations...",
    "none": "OK: Vessel {mmsi} trajectory is within normal parameters."
}
```

### Recommendations by Severity

**Critical:**
- Immediate verification of vessel identity
- Cross-reference with radar/satellite
- Alert maritime authorities
- Monitor for AIS spoofing

**High:**
- Verify position through secondary sources
- Review trajectory history
- Enhanced monitoring

**Medium:**
- Monitor for continued irregularities
- Review against known activities

**Low:**
- Standard monitoring sufficient

## Extending to LLM

The module is designed for easy LLM integration:

### OpenAI Example

```python
class OpenAIGenerator(ExplanationGenerator):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.encoder = PromptEncoder()
    
    def generate(self, mmsi, points, metrics):
        # Encode data
        traj_summary = self.encoder.encode_trajectory_summary(points)
        
        # Build prompt
        prompt = f"""
        Analyze this vessel's behavior:
        {traj_summary['summary_token']}
        
        Anomaly points detected: {traj_summary['anomaly_count']}
        Average position error: {traj_summary['avg_position_error_km']:.2f} km
        
        Provide a brief maritime-specific explanation.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response into Explanation object
        ...
```

### Ollama (Local) Example

```python
class OllamaGenerator(ExplanationGenerator):
    def __init__(self, model: str = "mistral"):
        self.model = model
        self.encoder = PromptEncoder()
    
    def generate(self, mmsi, points, metrics):
        import ollama
        
        traj_summary = self.encoder.encode_trajectory_summary(points)
        prompt = self._build_prompt(traj_summary)
        
        response = ollama.generate(model=self.model, prompt=prompt)
        # Parse and return...
```

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `prompt_encoder.py` | Numerical → Token conversion |
| `generator.py` | Explanation generation |
| `README.md` | This documentation |

## Dependencies

Core (no external LLM):
```
numpy
```

For LLM integration (optional):
```
openai      # For OpenAI API
ollama      # For local Ollama
transformers  # For HuggingFace
```

