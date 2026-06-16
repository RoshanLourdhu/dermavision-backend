import requests
import subprocess
import time

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def bool_to_text(value):
    return "Yes" if value else "No"

OLLAMA_URL = "http://localhost:11434"
TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"
GENERATE_ENDPOINT = f"{OLLAMA_URL}/api/generate"

def is_ollama_running() -> bool:
    """Check if Ollama server is responding at the tags endpoint."""
    try:
        resp = requests.get(TAGS_ENDPOINT, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

def start_ollama_server() -> None:
    """Attempt to start Ollama server in background.
    Assumes `ollama` executable is in PATH.
    """
    try:
        # Start without blocking the current process.
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        # If starting fails, we will handle later.
        print(f"Failed to start Ollama server: {e}")

def ensure_ollama_running(max_wait_seconds: int = 30) -> bool:
    """Make sure Ollama is running, start it if needed, and wait until reachable.
    Returns True if reachable, False otherwise.
    """
    if is_ollama_running():
        return True
    # Try to start the server.
    start_ollama_server()
    # Poll until reachable or timeout.
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        if is_ollama_running():
            return True
        time.sleep(2)
    return False

def fallback_report(data: dict) -> str:
    """Generate a simple text fallback report when Ollama is unavailable.
    This mirrors the structure of the desired report but without AI generation.
    """
    lines = []
    lines.append("--- FALLBACK CLINICAL REPORT ---")
    lines.append(f"Classification: {data.get('class', 'N/A')}")
    lines.append(f"Confidence Level: {data.get('confidence', 'N/A')}")
    lines.append(f"Risk Level: {data.get('risk', 'N/A')}")
    lines.append("")
    lines.append("Morphology:")
    lines.append(f"- Area: {data.get('area', 'N/A')}")
    lines.append(f"- Perimeter: {data.get('perimeter', 'N/A')}")
    lines.append(f"- Roughness: {data.get('roughness', 'N/A')}")
    lines.append("")
    lines.append("Depth:")
    lines.append(f"- Max Depth: {data.get('max_depth', 'N/A')}")
    lines.append(f"- Mean Depth: {data.get('mean_depth', 'N/A')}")
    lines.append("")
    lines.append("Patient Symptoms:")
    symptoms = data.get('symptoms', {})
    for name, val in symptoms.items():
        lines.append(f"- {name.capitalize()}: {bool_to_text(val)}")
    lines.append("")
    lines.append("Lesion History:")
    history = data.get('history', {})
    for name, val in history.items():
        lines.append(f"- {name.replace('_', ' ').capitalize()}: {val}")
    lines.append("")
    lines.append(f"Alerts: {data.get('alerts', '')}")
    lines.append(f"Recommended Action: {data.get('action', '')}")
    lines.append("--- END OF FALLBACK REPORT ---")
    return "\n".join(lines)

# -------------------------
# MAIN FUNCTION
# -------------------------

def generate_medical_report(data):
    """Generate a medical report using Ollama LLM, with fallback if Ollama is unavailable."""
    if not ensure_ollama_running():
        # Ollama not reachable after attempts; return fallback.
        return fallback_report(data)

    prompt = f"""
You are an AI dermatology assistant generating a structured clinical report.

----------------------------------------
INPUT DATA
----------------------------------------

Classification: {data['class']}
Confidence Level: {data['confidence']}
Risk Level: {data['risk']}

Morphology:
- Area: {data['area']}
- Perimeter: {data['perimeter']}
- Roughness: {data['roughness']}

Depth:
- Max Depth: {data['max_depth']}
- Mean Depth: {data['mean_depth']}

Patient Symptoms:
- Itching: {bool_to_text(data['symptoms']['itching'])}
- Pain: {bool_to_text(data['symptoms']['pain'])}
- Bleeding: {bool_to_text(data['symptoms']['bleeding'])}
- Oozing: {bool_to_text(data['symptoms']['oozing'])}

Lesion History:
- Duration: {data['history']['duration']}
- Growth: {bool_to_text(data['history']['growth'])}
- Color Change: {bool_to_text(data['history']['color_change'])}
- Border Irregularity: {bool_to_text(data['history']['border_change'])}

Alerts: {data['alerts']}
Recommended Action: {data['action']}

----------------------------------------
OUTPUT FORMAT (STRICT)
----------------------------------------

----------------------------------------
AI DERMATOLOGY REPORT
----------------------------------------

Patient Analysis Summary:
Include:
- primary classification
- alternative diagnosis (ONLY if clearly present)
- confidence level

Morphological Assessment:
- Area
- Perimeter
- Roughness
- Depth (Max and Mean)

Patient Symptoms:
List key symptoms briefly.

Lesion History:
Summarize duration and changes.

Clinical Interpretation:
- Combine morphology + depth + symptoms + history
- Highlight:
  • color change
  • irregular borders
  • growth
  • depth variation
- Explain what increases concern

Risk Assessment:
- State risk level
- Justify using:
  • confidence
  • depth
  • morphology
  • history

Recommendation:
Provide clear action:
- Monitor / Clinical evaluation / Urgent consultation

Disclaimer:
AI-assisted analysis; clinical evaluation advised.

----------------------------------------

IMPORTANT RULES:
- DO NOT include introductory lines like "Here is the report"
- Round all numerical values to 4 decimal places
- If alternative class exists, explicitly name it
- Avoid vague alternatives like "another lesion"
- Summarize symptoms including both present and absent findings
- Clearly explain which features increase clinical concern
- Do NOT add extra text outside format
- Do NOT be conversational
- Do NOT make definitive diagnosis
- Use phrases like:
  "appears consistent with"
  "suggests"
  "may indicate"

- If confidence is LOW:
  → clearly express uncertainty

- If multiple classes exist:
  → include both

- If no strong second class:
  → do NOT invent alternatives

- If risk is MODERATE or HIGH:
  → do NOT bias toward benign interpretation

- High depth (>0.85):
  → treat as significant structural variation

- Use symptoms + history to influence interpretation:
  - color change
  - border irregularity
  - growth
  - duration

- Keep report concise and clinically meaningful
"""

    try:
        response = requests.post(
            GENERATE_ENDPOINT,
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 300
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"⚠️ Ollama API error: {response.status_code}"

    except Exception as e:
        return f"⚠️ Ollama connection error: {str(e)}"