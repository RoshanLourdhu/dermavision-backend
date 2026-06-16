(* Wolfram Language API for Skin Lesion Clinical Intelligence *)
(* Deploy this script using: CloudDeploy[APIFunction[...], "SkinLesionAnalysisAPI", Permissions -> "Public"] *)

APIFunction[
  {
    "class" -> "String",
    "confidence" -> "Real",
    "risk" -> "String",
    "area" -> "Real",
    "perimeter" -> "Real",
    "roughness" -> "Real",
    "volume" -> "Real",
    "max_depth" -> "Real",
    "mean_depth" -> "Real"
  },
  Function[params,
    Module[
      {
        class = params["class"],
        confidence = params["confidence"],
        risk = params["risk"],
        area = params["area"],
        perimeter = params["perimeter"],
        roughness = params["roughness"],
        volume = params["volume"],
        maxDepth = params["max_depth"],
        meanDepth = params["mean_depth"],
        
        (* Outputs *)
        severity, riskIdx,
        circularity, borderAsymmetry, fractalDim, density, aspectRatio,
        areaPct, depthPct, volumePct,
        insights = {}
      },
      
      (* 1. Mathematical Lesion Analysis *)
      circularity = If[perimeter > 0, Clip[(4 * Pi * area) / (perimeter^2), {0.0, 1.0}], 1.0];
      borderAsymmetry = 1.0 - circularity;
      fractalDim = If[area > 1 && perimeter > 1, Clip[2 * Log[perimeter] / Log[area], {1.0, 2.0}], 1.0];
      density = If[area > 0, volume / area, 0.0];
      aspectRatio = If[area > 0, maxDepth / Sqrt[area / Pi], 0.0];
      
      (* 2. Lesion Severity Score (0-100) *)
      Module[
        {baseRisk, morphScore, depthScore, circularityPenalty},
        baseRisk = Which[
          ToUpperCase[risk] == "HIGH", 60,
          ToUpperCase[risk] == "MODERATE", 35,
          True, 10
        ];
        circularityPenalty = Clip[(1.0/Max[circularity, 0.01] - 1.0) * 5.0, {0.0, 10.0}];
        morphScore = Clip[roughness * 40.0 + circularityPenalty, {0.0, 20.0}];
        depthScore = Clip[maxDepth * 12.0 + meanDepth * 8.0, {0.0, 20.0}];
        
        severity = Clip[Round[(baseRisk * confidence) + morphScore + depthScore], {0, 100}]
      ];
      
      (* 3. Advanced Risk Analytics (Risk Index 0.0 - 1.0) *)
      Module[
        {baseRiskWeight, structuralRisk, invasionRisk},
        baseRiskWeight = Which[
          ToUpperCase[risk] == "HIGH", 1.0,
          ToUpperCase[risk] == "MODERATE", 0.5,
          True, 0.1
        ];
        structuralRisk = Clip[(1.0 - circularity) * 0.7 + roughness * 0.6, {0.0, 1.0}];
        invasionRisk = Clip[maxDepth * 0.6 + meanDepth * 0.4, {0.0, 1.0}];
        
        riskIdx = Clip[Round[0.4 * invasionRisk + 0.3 * structuralRisk + 0.3 * baseRiskWeight, 0.01], {0.0, 1.0}]
      ];
      
      (* 4. Statistical Interpretation *)
      (* Area benchmark distribution (LogNormalDistribution) *)
      areaPct = Round[100.0 * CDF[LogNormalDistribution[8.5, 1.2], area]];
      (* Depth benchmark distribution (BetaDistribution) *)
      depthPct = Round[100.0 * CDF[BetaDistribution[2.5, 5.0], meanDepth]];
      (* Volume benchmark distribution (GammaDistribution) *)
      volumePct = Round[100.0 * CDF[GammaDistribution[3.0, 50.0], volume]];
      
      (* 5. Clinical Insights *)
      If[circularity < 0.6,
        AppendTo[insights, "Lesion boundary exhibits high fractal irregularity (Circularity: " <> ToString[Round[circularity, 0.01]] <> "), representing potential atypical growth patterns."]
      ];
      If[meanDepth > 0.45,
        AppendTo[insights, "Depth analytics reveal significant vertical extension (Mean Depth: " <> ToString[Round[meanDepth, 0.02]] <> "), indicating potential invasion beyond epidermal layers."]
      ];
      If[severity >= 65,
        AppendTo[insights, "Elevated Lesion Severity Score (" <> ToString[severity] <> "/100) indicates an urgent clinical evaluation or histopathological verification is recommended."]
      ];
      If[roughness > 0.25,
        AppendTo[insights, "Elevated structural roughness deviation indicates high topographical heterogeneity across the lesion surface."]
      ];
      If[ToUpperCase[risk] == "HIGH",
        AppendTo[insights, "AI diagnosis classification (" <> class <> ") has high risk classification. Dermatological review is highly recommended."],
        AppendTo[insights, " Lesion classifications indicate low-to-moderate immediate risk. Continued periodic monitoring is advised."]
      ];
      If[Length[insights] == 0,
        AppendTo[insights, "Lesion characteristics are within typical baseline margins. Regular skin self-examination is advised."]
      ];
      
      (* Return JSON payload *)
      ExportString[
        <|
          "severity_score" -> severity,
          "risk_index" -> riskIdx,
          "mathematical_analysis" -> <|
            "circularity" -> Round[circularity, 0.001],
            "border_asymmetry" -> Round[borderAsymmetry, 0.001],
            "fractal_dimension" -> Round[fractalDim, 0.001],
            "lesion_density" -> Round[density, 0.001],
            "aspect_ratio" -> Round[aspectRatio, 0.001]
          |>,
          "statistical_interpretation" -> <|
            "area_percentile" -> areaPct,
            "depth_percentile" -> depthPct,
            "volume_percentile" -> volumePct
          |>,
          "clinical_insights" -> insights
        |>,
        "JSON"
      ]
    ]
  ],
  "JSON"
]
