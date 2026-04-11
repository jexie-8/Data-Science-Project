# mappings.py

# Default risk for any ID not explicitly found in these maps
GLOBAL_MEAN_RISK = 0.35

# Feature list exactly as expected by your XGBoost model
MODEL_FEATURES = [
    "Marital status", "Application order", "Course", "Daytime/evening attendance",
    "Previous qualification", "Displaced", "Gender", "Scholarship holder",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (approved)",
    "Academic_Efficiency_Ratio", "Financial_Risk_Score", "is_mature_entry",
    "is_zero_sem1", "is_single", "is_standard_secondary", "has_credits",
    "Total_Academic_Momentum", "late_switcher", "Grade_Velocity_Trend",
    "Recovery_Signal", "Mother's_occupation_risk", "Father's_occupation_risk",
    "Application_mode_risk", "Age_at_Enrollment_Log"
]

# Risk based on dropout rates per occupation (mapped from your HTML values)
# Values represent probability weights (0.0 to 1.0)
MOTHER_OCC_MAP = {
    1: 0.15,  # Student
    2: 0.05,  # Legislative/Senior Official (Low Risk)
    3: 0.04,  # Scientific/Intellectual
    4: 0.08,  # Intermediate Tech
    5: 0.12,  # Administrative
    6: 0.22,  # Service/Sales
    7: 0.28,  # Agriculture/Fishery
    8: 0.25,  # Skilled Industrial
    9: 0.38,  # Unskilled Worker (Higher Risk)
    10: 0.10, # Armed Forces
    99: 0.35  # Other/Unknown
}

# Usually similar to Mother's but with slight variations in the dataset
FATHER_OCC_MAP = {
    1: 0.18, 2: 0.06, 3: 0.05, 4: 0.09, 5: 0.14, 
    6: 0.24, 7: 0.30, 8: 0.27, 9: 0.42, 10: 0.11, 99: 0.35
}

# Crucial: Different entry modes have very different dropout risks
APP_MODE_MAP = {
    1: 0.18,   # 1st phase - general (Low Risk)
    2: 0.25,   # Ordinance
    5: 0.20,   # Special quota
    7: 0.15,   # Holders of other higher courses
    8: 0.45,   # Over 23 years old (High Risk)
    15: 0.35,  # Change of course
    17: 0.48,  # Special admissions (over 23)
    18: 0.30,  # Change of institution
    39: 0.52   # Over 23 (Special phase)
}

# Course risk (Some STEM courses have higher attrition than Social Services)
COURSE_MAP = {
    33: 0.40,   # Biofuel
    171: 0.25,  # Animation
    8014: 0.15, # Social Service
    9003: 0.30, # Agronomy
    9070: 0.22, # Communication Design
    9085: 0.18, # Veterinary
    9119: 0.45, # Informatics Engineering (High Risk)
    9130: 0.35, # Equiniculture
    9147: 0.28, # Management
    9238: 0.20, # Social Service (Evening)
    9254: 0.32, # Tourism
    9500: 0.10, # Nursing (Very Low Risk)
    9556: 0.12, # Oral Hygiene
    9670: 0.24, # Advertising
    9773: 0.20, # Journalism
    9853: 0.15, # Basic Education
    9991: 0.25  # Management (Evening)
}