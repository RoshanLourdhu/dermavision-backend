import os
import math
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WolframService")

# Retrieve API configurations from environment
WOLFRAM_API_URL = os.environ.get("WOLFRAM_API_URL")
WOLFRAM_API_KEY = os.environ.get("WOLFRAM_API_KEY")

def lognormal_cdf(x, mu, sigma):
    """Computes exact CDF of LogNormalDistribution[mu, sigma] using math.erf"""
    if x <= 0:
        return 0.0
    try:
        return 0.5 * (1.0 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2.0))))
    except Exception:
        return 0.5

def beta_pdf(t, a, b):
    """Computes PDF of BetaDistribution[a, b]"""
    if t <= 0 or t >= 1:
        return 0.0
    return (t ** (a - 1.0)) * ((1.0 - t) ** (b - 1.0))

def beta_cdf(x, a=2.5, b=5.0):
    """Computes CDF of BetaDistribution[a, b] using Simpson's rule integration"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    try:
        n = 100
        h = x / n
        # B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)
        beta_val = (math.gamma(a) * math.gamma(b)) / math.gamma(a + b)
        
        s = beta_pdf(0, a, b) + beta_pdf(x, a, b)
        for i in range(1, n):
            t = i * h
            weight = 4 if i % 2 == 1 else 2
            s += weight * beta_pdf(t, a, b)
        
        cdf_val = (s * h / 3.0) / beta_val
        return min(1.0, max(0.0, cdf_val))
    except Exception:
        return 0.5

def gamma_cdf(x, k=3.0, theta=50.0):
    """Computes exact CDF of GammaDistribution[3.0, 50.0] analytically for integer k=3"""
    if x <= 0:
        return 0.0
    try:
        z = x / theta
        # CDF = 1 - e^-z * (1 + z + z^2/2)
        cdf_val = 1.0 - math.exp(-z) * (1.0 + z + 0.5 * (z ** 2))
        return min(1.0, max(0.0, cdf_val))
    except Exception:
        return 0.5

def evaluate_locally(params):
    """
    Computes exact math/statistical logic locally as a precise offline fallback
    to match the Wolfram Language API output.
    """
    label = params.get("class", "Unknown")
    confidence = float(params.get("confidence", 0.8))
    risk = params.get("risk", "LOW").upper()
    area = float(params.get("area", 1000.0))
    perimeter = float(params.get("perimeter", 100.0))
    roughness = float(params.get("roughness", 0.1))
    volume = float(params.get("volume", 50.0))
    max_depth = float(params.get("max_depth", 0.5))
    mean_depth = float(params.get("mean_depth", 0.2))

    # 1. Mathematical Lesion Analysis
    # Circularity = 4 * pi * area / perimeter^2
    circularity = (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0
    circularity = min(1.0, max(0.0, circularity))
    border_asymmetry = 1.0 - circularity
    
    # Fractal dimension estimate: 2 * log(perimeter) / log(area)
    if area > 1.0 and perimeter > 1.0:
        fractal_dimension = 2.0 * math.log(perimeter) / math.log(area)
        fractal_dimension = min(2.0, max(1.0, fractal_dimension))
    else:
        fractal_dimension = 1.0
        
    lesion_density = volume / area if area > 0 else 0.0
    aspect_ratio = max_depth / math.sqrt(area / math.pi) if area > 0 else 0.0

    # 2. Lesion Severity Score (0-100)
    base_risk = 60 if risk == "HIGH" else (35 if risk == "MODERATE" else 10)
    circularity_penalty = min(10.0, max(0.0, (1.0 / max(circularity, 0.01) - 1.0) * 5.0))
    morph_score = min(20.0, max(0.0, roughness * 40.0 + circularity_penalty))
    depth_score = min(20.0, max(0.0, max_depth * 12.0 + mean_depth * 8.0))
    
    severity_score = int(min(100, max(0, round((base_risk * confidence) + morph_score + depth_score))))

    # 3. Advanced Risk Analytics (Risk Index 0.0 - 1.0)
    base_risk_weight = 1.0 if risk == "HIGH" else (0.5 if risk == "MODERATE" else 0.1)
    structural_risk = min(1.0, max(0.0, (1.0 - circularity) * 0.7 + roughness * 0.6))
    invasion_risk = min(1.0, max(0.0, max_depth * 0.6 + mean_depth * 0.4))
    
    risk_index = round(0.4 * invasion_risk + 0.3 * structural_risk + 0.3 * base_risk_weight, 2)
    risk_index = min(1.0, max(0.0, risk_index))

    # 4. Statistical Interpretation
    # Benchmarks: LogNormalDistribution[8.5, 1.2], BetaDistribution[2.5, 5.0], GammaDistribution[3.0, 50.0]
    area_pct = int(round(100.0 * lognormal_cdf(area, 8.5, 1.2)))
    depth_pct = int(round(100.0 * beta_cdf(mean_depth, 2.5, 5.0)))
    volume_pct = int(round(100.0 * gamma_cdf(volume, 3.0, 50.0)))

    # 5. Clinical Insights
    insights = []
    if circularity < 0.6:
        insights.append(f"Lesion boundary exhibits high fractal irregularity (Circularity: {circularity:.2f}), representing potential atypical growth patterns.")
    if mean_depth > 0.45:
        insights.append(f"Depth analytics reveal significant vertical extension (Mean Depth: {mean_depth:.2f}), indicating potential invasion beyond epidermal layers.")
    if severity_score >= 65:
        insights.append(f"Elevated Lesion Severity Score ({severity_score}/100) indicates an urgent clinical evaluation or histopathological verification is recommended.")
    if roughness > 0.25:
        insights.append("Elevated structural roughness deviation indicates high topographical heterogeneity across the lesion surface.")
    if risk == "HIGH":
        insights.append(f"AI diagnosis classification ({label}) has high risk classification. Dermatological review is highly recommended.")
    else:
        insights.append("Lesion classifications indicate low-to-moderate immediate risk. Continued periodic monitoring is advised.")
        
    if not insights:
        insights.append("Lesion characteristics are within typical baseline margins. Regular skin self-examination is advised.")

    return {
        "severity_score": severity_score,
        "risk_index": risk_index,
        "mathematical_analysis": {
            "circularity": round(circularity, 3),
            "border_asymmetry": round(border_asymmetry, 3),
            "fractal_dimension": round(fractal_dimension, 3),
            "lesion_density": round(lesion_density, 3),
            "aspect_ratio": round(aspect_ratio, 3)
        },
        "statistical_interpretation": {
            "area_percentile": area_pct,
            "depth_percentile": depth_pct,
            "volume_percentile": volume_pct
        },
        "clinical_insights": insights
    }

def get_wolfram_analysis(classification, metrics):
    """
    Computes analysis using Wolfram Cloud APIs, or falls back to local evaluation
    if the API endpoint is not set.
    """
    params = {
        "class": classification.get("label", "Unknown"),
        "confidence": float(classification.get("confidence", 0.0)),
        "risk": classification.get("risk", "LOW"),
        "area": float(metrics.get("area_px", 0.0)),
        "perimeter": float(metrics.get("perimeter_px", 0.0)),
        "roughness": float(metrics.get("roughness", 0.0)),
        "volume": float(metrics.get("volume", 0.0)),
        "max_depth": float(metrics.get("max_depth", 0.0)),
        "mean_depth": float(metrics.get("mean_depth", 0.0))
    }

    if WOLFRAM_API_URL:
        try:
            logger.info("Connecting to Wolfram Cloud API...")
            headers = {}
            if WOLFRAM_API_KEY:
                headers["Authorization"] = f"Bearer {WOLFRAM_API_KEY}"
            
            # Send parameters
            response = requests.post(WOLFRAM_API_URL, json=params, headers=headers, timeout=12)
            if response.status_code == 200:
                result = response.json()
                logger.info("Successfully fetched results from Wolfram Cloud.")
                # Make sure fields are parsed and correct
                if "severity_score" in result:
                    return result
            logger.warning(f"Wolfram Cloud returned invalid response or code {response.status_code}. Running local fallback.")
        except Exception as e:
            logger.warning(f"Error querying Wolfram Cloud API: {str(e)}. Running local fallback.")
            
    # Local fallback
    logger.info("Running local mathematical/statistical Wolfram simulation fallback.")
    return evaluate_locally(params)
