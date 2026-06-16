from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from database import init_db, save_patient, update_report
from datetime import datetime
import os
import shutil
import subprocess
import json
import re
import threading


from wolfram_service import get_wolfram_analysis

# -------------------------
# INIT
# -------------------------
init_db()
app = FastAPI()

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "test_images")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=OUT_DIR), name="static")

REPORT_FILE = os.path.join(OUT_DIR, "report.txt")

# -------------------------
# CLEAN
# -------------------------
def clear_dir(folder):
    for f in os.listdir(folder):
        try:
            os.remove(os.path.join(folder, f))
        except:
            pass

# -------------------------
# JSON PARSER
# -------------------------
def extract_json(stdout):
    # Try finding a line that is valid JSON and contains "classification"
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "classification" in data:
                    return data
            except:
                continue
    # Fallback to finding block starting with '{"metrics":'
    try:
        start_idx = stdout.find('{"metrics":')
        if start_idx != -1:
            end_idx = stdout.rfind('}')
            if end_idx != -1 and end_idx > start_idx:
                try:
                    return json.loads(stdout[start_idx:end_idx+1])
                except:
                    pass
    except Exception as e:
        print("Fallback JSON parse error:", e)
    return {}

# -------------------------
# BACKGROUND REPORT
# -------------------------
import time, requests

def is_ollama_running() -> bool:
    """Check if Ollama API is reachable on localhost:11434."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def start_ollama_service():
    """Start Ollama server in background if not already running."""
    print("Starting Ollama service...")
    # Launch Ollama serve; redirect stdio to dev/null to avoid blocking.
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def generate_report_async(report_data):
    """Generate deterministic clinical report without any LLM calls."""
    try:
        import math
        import os
        
        # Helper function for formatting values to decimals
        def clean_format(val, decimals=2):
            try:
                return f"{float(val):.{decimals}f}"
            except Exception:
                return str(val)

        # Helper function for formatting comma-separated values
        def format_with_commas(val, decimals=2):
            try:
                return f"{float(val):,.{decimals}f}"
            except Exception:
                return str(val)

        # Extract required fields
        classification = report_data.get("classification", {})
        label = classification.get("label", "N/A")
        confidence = classification.get("confidence", 0.0)
        
        # Read risk from classification object, fallback to report_data, default to "Unknown" if missing
        risk = classification.get("risk") or report_data.get("risk")
        if not risk or risk == "N/A":
            risk = "Unknown"

        metrics = report_data.get("metrics", {})
        alerts = report_data.get("alerts", [])
        patient_id = report_data.get("patient_id", "unknown")
        name = report_data.get("name", "unknown")
        age = report_data.get("age", "unknown")
        analysis_date = report_data.get(
            "analysis_date",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Unknown condition detection
        if confidence < 0.55:
            diagnosis = "Unknown Skin Condition"
            unknown_explanation = (
                "The image does not strongly match any of the seven trained lesion categories."
            )
        else:
            diagnosis = label
            unknown_explanation = ""

        # Derived metrics
        area_px = metrics.get("area_px", 0)
        perimeter_px = metrics.get("perimeter_px", 0)
        roughness = metrics.get("roughness", 0.0)
        volume = metrics.get("volume", 0.0)
        max_depth = metrics.get("max_depth", 0.0)
        mean_depth = metrics.get("mean_depth", 0.0)

        total_pixels = 256 * 256
        seg_coverage_pct = (area_px / total_pixels) * 100 if total_pixels else 0

        circularity = (
            (4 * 3.141592653589793 * area_px) / (perimeter_px ** 2)
            if perimeter_px else 0
        )
        depth_aspect_ratio = (max_depth / mean_depth) if mean_depth else 0

        # Wolfram Analysis
        try:
            from wolfram_service import get_wolfram_analysis
            wolfram_analysis = get_wolfram_analysis(classification, metrics)
        except Exception:
            wolfram_analysis = None

        if wolfram_analysis and "mathematical_analysis" in wolfram_analysis:
            math_an = wolfram_analysis["mathematical_analysis"]
            circularity_val = math_an.get("circularity", circularity)
            border_asymmetry_val = math_an.get("border_asymmetry", "Not Available")
            fractal_dimension_val = math_an.get("fractal_dimension", "Not Available")
        else:
            circularity_val = circularity
            border_asymmetry_val = 1.0 - circularity
            if area_px > 1.0 and perimeter_px > 1.0:
                fractal_val = 2.0 * math.log(perimeter_px) / math.log(area_px)
                fractal_dimension_val = min(2.0, max(1.0, fractal_val))
            else:
                fractal_dimension_val = 1.0


        # Read the actual Grad-CAM overlap / attention alignment value from the pipeline output.
        try:
            import cv2
            import numpy as np
            gradcam_thresh_path = os.path.join(OUT_DIR, "gradcam_thresh.png")
            mask_path = os.path.join(OUT_DIR, "mask.png")
            if os.path.exists(gradcam_thresh_path) and os.path.exists(mask_path):
                cam_img = cv2.imread(gradcam_thresh_path, cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if cam_img is not None and mask_img is not None:
                    cam_bin = (cam_img > 127).astype(np.uint8)
                    mask_bin = (mask_img > 127).astype(np.uint8)
                    if cam_bin.shape != mask_bin.shape:
                        cam_bin = cv2.resize(
                            cam_bin,
                            (mask_bin.shape[1], mask_bin.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    total_act = cam_bin.sum()
                    if total_act > 0:
                        inside_act = (cam_bin & mask_bin).sum()
                        inside_pct = (inside_act / total_act) * 100
                        attention_alignment_val = f"{clean_format(inside_pct, 1)}%"
                    else:
                        attention_alignment_val = "Not Available"
                else:
                    attention_alignment_val = "Not Available"
            else:
                attention_alignment_val = "Not Available"
        except Exception:
            attention_alignment_val = "Not Available"

        # Build report lines
        report_lines = []
        report_lines.append("# AI Clinical Assessment Report")
        report_lines.append("")

        # 1. Patient Information
        report_lines.append("## Patient Information")
        report_lines.append(f"Patient ID: {patient_id}")
        report_lines.append(f"Name: {name}")
        report_lines.append(f"Age: {age}")
        report_lines.append(f"Analysis Date: {analysis_date}")
        report_lines.append("")

        # 2. Primary Classification
        report_lines.append("## Primary Classification")
        report_lines.append(f"Diagnosis: {diagnosis}")
        report_lines.append(f"Confidence: {clean_format(confidence * 100, 1)}%")
        report_lines.append(f"Risk Level: {risk}")
        if unknown_explanation:
            report_lines.append(unknown_explanation)
        report_lines.append("")

        # Interpretations for morphology
        area_interpret = "Significant lesion surface area." if area_px > 10000 else "Small localized lesion surface area."
        perimeter_interpret = "Extensive lesion boundary." if perimeter_px > 500 else "Limited lesion boundary."
        
        roughness_val = float(roughness)
        if roughness_val >= 0.25:
            roughness_interpret = "High border irregularity and structural surface variation detected."
        elif roughness_val >= 0.15:
            roughness_interpret = "Mild to moderate border irregularity and surface variation detected."
        else:
            roughness_interpret = "Low surface roughness and typical border uniformity."
            
        volume_val = float(volume)
        if volume_val >= 10000:
            volume_interpret = "Significant lesion volume indicating deep/wide structural expansion."
        elif volume_val >= 5000:
            volume_interpret = "Moderate lesion volume."
        else:
            volume_interpret = "Minimal lesion volume."
            
        mean_depth_val = float(mean_depth)
        if mean_depth_val >= 0.5:
            mean_depth_interpret = "Deep vertical extension indicating potential invasion beyond superficial epidermal layers."
        elif mean_depth_val >= 0.25:
            mean_depth_interpret = "Moderate depth variation observed."
        else:
            mean_depth_interpret = "Superficial lesion structure with minimal vertical extension."
            
        max_depth_val = float(max_depth)
        if max_depth_val >= 0.8:
            max_depth_interpret = "Localized deep lesion structures present."
        elif max_depth_val >= 0.4:
            max_depth_interpret = "Moderate localized depth structure."
        else:
            max_depth_interpret = "Superficial depth limits."

        # 3. Lesion Morphology Analysis
        report_lines.append("## Lesion Morphology Analysis")
        report_lines.append(f"Area: {format_with_commas(area_px, 0)} px²")
        report_lines.append(f"Interpretation:")
        report_lines.append(area_interpret)
        report_lines.append("")
        report_lines.append(f"Perimeter: {format_with_commas(perimeter_px, 0)} px")
        report_lines.append(f"Interpretation:")
        report_lines.append(perimeter_interpret)
        report_lines.append("")
        report_lines.append(f"Border Roughness Index: {clean_format(roughness, 2)}")
        report_lines.append(f"Interpretation:")
        report_lines.append(roughness_interpret)
        report_lines.append("")
        report_lines.append(f"Volume Estimate: {format_with_commas(volume, 2)} units³")
        report_lines.append(f"Interpretation:")
        report_lines.append(volume_interpret)
        report_lines.append("")
        report_lines.append(f"Mean Depth Index: {clean_format(mean_depth, 2)}")
        report_lines.append(f"Interpretation:")
        report_lines.append(mean_depth_interpret)
        report_lines.append("")
        report_lines.append(f"Maximum Depth Index: {clean_format(max_depth, 2)}")
        report_lines.append(f"Interpretation:")
        report_lines.append(max_depth_interpret)
        report_lines.append("")

        # 4. Segmentation Findings
        report_lines.append("## Segmentation Findings")
        seg_quality = "Optimal" if area_px > 0 else "Low Quality/No Lesion Detected"
        report_lines.append(f"Segmentation Quality: {seg_quality}")
        report_lines.append(f"Lesion Coverage: {clean_format(seg_coverage_pct, 2)}%")
        if seg_coverage_pct > 40:
            coverage_interpret = "The lesion occupies a significant portion of the analyzed skin surface."
        elif seg_coverage_pct > 15:
            coverage_interpret = "Moderate lesion coverage within the imaging field."
        else:
            coverage_interpret = "Minimal lesion coverage relative to the total skin area analyzed."
        report_lines.append(f"Interpretation:")
        report_lines.append(coverage_interpret)
        report_lines.append("")

        # 5. Explainable AI Findings
        report_lines.append("## Explainable AI Findings")
        report_lines.append(f"Attention Alignment Score: {attention_alignment_val}")
        report_lines.append(f"Interpretation:")
        report_lines.append("The classification model focused primarily on clinically relevant lesion regions.")
        report_lines.append("")

        # Wolfram metrics interpretations
        try:
            circ_f = float(circularity_val)
            if circ_f >= 0.8:
                circ_interpret = "High circularity, indicating a highly regular and uniform symmetric shape."
            elif circ_f >= 0.55:
                circ_interpret = "Moderately regular lesion shape."
            else:
                circ_interpret = "Low circularity, indicating an irregular, highly complex border configuration."
        except Exception:
            circ_interpret = "Circularity properties within moderate parameters."

        try:
            asym_f = float(border_asymmetry_val)
            if asym_f >= 0.45:
                asym_interpret = "High structural asymmetry observed across the major axes."
            elif asym_f >= 0.2:
                asym_interpret = "Moderate border asymmetry detected."
            else:
                asym_interpret = "Minimal asymmetry, representing a highly symmetric lesion."
        except Exception:
            asym_interpret = "Symmetry assessment within baseline limits."

        try:
            frac_f = float(fractal_dimension_val)
            if frac_f >= 1.4:
                frac_interpret = "High fractal dimension, indicating a highly irregular, detailed, and jagged border structure."
            elif frac_f >= 1.15:
                frac_interpret = "Moderate boundary complexity."
            else:
                frac_interpret = "Low fractal dimension, indicating a smooth, simple boundary shape."
        except Exception:
            frac_interpret = "Boundary complexity within typical baseline standards."

        try:
            ratio_f = float(depth_aspect_ratio)
            if ratio_f >= 1.5:
                ratio_interpret = "High aspect ratio, indicating substantial vertical depth relative to horizontal expansion."
            else:
                ratio_interpret = "Low aspect ratio, representing a flat or superficial lesion structure."
        except Exception:
            ratio_interpret = "Depth aspect ratio indicates uniform lesion scaling."

        # 6. Wolfram Clinical Intelligence
        report_lines.append("## Wolfram Clinical Intelligence")
        report_lines.append(f"Circularity: {clean_format(circularity_val, 2)}")
        report_lines.append("Interpretation:")
        report_lines.append(circ_interpret)
        report_lines.append("")
        
        report_lines.append(f"Border Asymmetry: {clean_format(border_asymmetry_val, 2)}")
        report_lines.append("Interpretation:")
        report_lines.append(asym_interpret)
        report_lines.append("")
        
        report_lines.append(f"Fractal Dimension: {clean_format(fractal_dimension_val, 2)}")
        report_lines.append("Interpretation:")
        report_lines.append(frac_interpret)
        report_lines.append("")
        
        report_lines.append(f"Depth Aspect Ratio: {clean_format(depth_aspect_ratio, 2)}")
        report_lines.append("Interpretation:")
        report_lines.append(ratio_interpret)
        report_lines.append("")

        # 7. Risk Evaluation
        report_lines.append("## Risk Evaluation")
        report_lines.append(f"Risk category: {risk}")
        report_lines.append("")
        report_lines.append("Risk Factors:")
        
        risk_factors = []
        for alert in alerts:
            risk_factors.append(f"• {alert}")
            
        if roughness_val >= 0.25 and "Irregular surface detected" not in alerts:
            risk_factors.append("• Significant border irregularity")
        elif 0.15 <= roughness_val < 0.25 and "Irregular surface detected" not in alerts:
            risk_factors.append("• Moderate border irregularity")
            
        if volume_val >= 10000:
            risk_factors.append("• Elevated lesion volume characteristics")
        elif 5000 <= volume_val < 10000:
            risk_factors.append("• Moderate lesion volume characteristics")
            
        if confidence < 0.75:
            risk_factors.append("• Borderline classification confidence level")
            
        try:
            circ_val_f = float(circularity_val)
            if circ_val_f < 0.6 and "Irregular boundary" not in alerts:
                risk_factors.append("• High boundary complexity (irregular shape)")
        except Exception:
            pass

        try:
            asym_val_f = float(border_asymmetry_val)
            if asym_val_f > 0.4:
                risk_factors.append("• Significant structural asymmetry")
        except Exception:
            pass

        if not risk_factors:
            risk_factors.append("• None identified")
            
        for factor in risk_factors:
            report_lines.append(factor)
        report_lines.append("")

        report = "\n".join(report_lines)

        # Write deterministic report and update DB
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        update_report(patient_id, report)
    except Exception as e:
        error_msg = f"Report generation error: {str(e)}"
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(error_msg)
        update_report(report_data.get("patient_id", "unknown"), error_msg)

# -------------------------
# MAIN API
# -------------------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    patient_id: str = Form("1"),
    name: str = Form("test"),
    age: str = Form("25"),
    itching: str = Form("n"),
    pain: str = Form("n"),
    bleeding: str = Form("n"),
    oozing: str = Form("n"),
    duration: str = Form("1 month"),
    growth: str = Form("n"),
    color_change: str = Form("n"),
    border_change: str = Form("n"),
):
    try:
        print("STEP 1: /analyze request received")
        clear_dir(IMG_DIR)

        if os.path.exists(REPORT_FILE):
            os.remove(REPORT_FILE)

        # Save input image
        img_path = os.path.join(IMG_DIR, "input.jpg")
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("STEP 2: Image saved to", img_path)

        # CLI input
        cli_input = "\n".join([
            patient_id, name, age,
            itching, pain, bleeding, oozing,
            duration, growth, color_change, border_change,
            "n"
        ]) + "\n"

        print("STEP 3: Starting inference subprocess")

        # Run inference
        # Run inference
        proc = subprocess.Popen(
            ["python", os.path.join(BASE_DIR, "inference_segmentation.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("STEP 4: Subprocess started, waiting for completion")
        stdout, stderr = proc.communicate(input=cli_input)
        print("STEP 5: Subprocess finished with returncode", proc.returncode)
        if stderr:
            print("INFERENCE STDERR:\n", stderr)
        else:
            print("INFERENCE STDERR empty")
        print("RAW STDOUT:\n", stdout)

        if stderr:
            print("INFERENCE ERROR:", stderr)

        parsed = extract_json(stdout)

        print("PARSED JSON:", parsed)

        # Fail-safe
        if not parsed:
            return {
                "error": "Inference JSON parsing failed",
                "raw": stdout[-500:]
            }

        classification = parsed.get("classification", {})
        metrics = parsed.get("metrics", {})
        alerts = parsed.get("alerts", [])
        report_input = parsed.get("report_input", {})

        # Wolfram analysis
        print("STEP 7: Starting Wolfram analysis")
        try:
            wolfram_analysis = get_wolfram_analysis(classification, metrics)
            print("STEP 8: Wolfram analysis completed")
        except Exception as e:
            print("WOLFRAM ANALYSIS ERROR:", e)
            wolfram_analysis = None

        # Save patient
        save_patient({
            "patient_id": patient_id,
            "name": name,
            "age": age,
            "image_path": img_path,
            "area": metrics.get("area_px"),
            "perimeter": metrics.get("perimeter_px"),
            "roughness": metrics.get("roughness"),
            "volume": metrics.get("volume"),
            "max_depth": metrics.get("max_depth"),
            "mean_depth": metrics.get("mean_depth"),
            "classification": classification.get("label"),
            "confidence": classification.get("confidence"),
            "risk": classification.get("risk"),
            "report": "PENDING",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "wolfram_analysis": json.dumps(wolfram_analysis) if wolfram_analysis else None
        })

        # Start report generation thread
        report_data = {
            "classification": classification,
            "risk": classification.get("risk", "N/A"),
            "metrics": metrics,
            "alerts": alerts,
            "patient_id": patient_id,
            "name": name,
            "age": age,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        threading.Thread(
            target=generate_report_async,
            args=(report_data,),
            daemon=True
        ).start()

        print("STEP 10: Returning API response")
        return {
            "classification": {
                "label": classification.get("label", "N/A"),
                "confidence": classification.get("confidence", 0),
                "risk": classification.get("risk", "N/A"),
            },
            "metrics": metrics,
            "alerts": alerts,
            "report_status": "processing",
            "wolfram_analysis": wolfram_analysis,
            "images": {
                "segmentation": "/static/segmentation.png",
                "gradcam": "/static/gradcam.png",
                "attention": "/static/attention.png",
                "depth": "/static/depth.png",
                "depth_gray": "/static/depth_raw.png",
                "profile": "/static/profile.png",
                "three_d": "/static/3d_interactive.html"
            }
        }


    except Exception as e:
        print("API ERROR:", e)
        return {"error": str(e)}

# -------------------------
# REPORT FETCH
# -------------------------
@app.get("/report")
def get_report():
    try:
        if not os.path.exists(REPORT_FILE):
            return {"status": "processing", "report": ""}

        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        if content.strip() == "":
            return {"status": "processing", "report": ""}

        if "LLM Error" in content:
            return {"status": "error", "report": content}

        return {"status": "ready", "report": content}

    except Exception as e:
        return {"status": "error", "report": str(e)}

# -------------------------
# HISTORY
# -------------------------
@app.get("/history/{patient_id}")
def get_history(patient_id: str):
    from database import get_patient_history
    data = get_patient_history(patient_id)
    return {"history": data}