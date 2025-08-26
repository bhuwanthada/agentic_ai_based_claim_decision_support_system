BIOMARKER_WEIGHTS = {
    "pd-l1": 0.004,  # multiplier → % * 0.004 (100% → 0.4)
    "tmb": 0.01,  # multiplier → mut/Mb * 0.01 (10 → 0.1)
    "msi-h": 0.1,
    "dmmr": 0.1,
    "egfr+": 0.05,
    "alk+": 0.05,
    "braf+": 0.05,
    "her2+": 0.05,
    "progressive disease": 0.15,
    "metastasis": 0.15,
    "failed chemotherapy": 0.1,
    "prior therapy failure": 0.1,
}
TOP_K = 5
EXCLUDED_KEYS = ["risk_score", "decision", "case_text", "patient_id"]
PUBMED_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
GUIDELINE_API_URL = "https://clinicaltrials.gov/api/v2/studies"
