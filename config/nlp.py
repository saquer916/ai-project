"""
Company name → ticker lookup (prototype mapping).
Production: use OpenFIGI + NER for robust extraction.
"""

DEFAULT_MAPPING = {
    "steel components": {
        "suppliers": [("Nucor", "NUE"), ("US Steel", "X"), ("Arcelor Mittal", "AMTX")],
        "customers": [("Ford Motor", "F"), ("General Motors", "GM"), ("Tesla", "TSLA")]
    },
    "car parts": {
        "suppliers": [("Aisin", "AISFF"), ("Magna International", "MGA"), ("Aptiv", "APTV")],
        "customers":  [("Ford Motor", "F"), ("Toyota", "TM"), ("Volkswagen", "VWAGY")]
    },
    "electronics": {
        "suppliers": [("Intel", "INTC"), ("TSMC", "TSM"), ("Samsung", "SSNLF")],
        "customers": [("Apple", "AAPL"), ("Microsoft", "MSFT"), ("Meta", "META")]
    },
    "chemicals": {
        "suppliers": [("DuPont", "DD"), ("Eastman Chemical", "EMN"), ("Dow", "DOW")],
        "customers": [("Procter & Gamble", "PG"), ("Coca-Cola", "KO"), ("Nestle", "NSRGY")]
    },
    "textiles": {
        "suppliers": [("Invista", "N/A"), ("Lenzing", "LENZ"), ("Toray", "TYEYY")],
        "customers": [("Nike", "NKE"), ("Adidas", "ADDYY"), ("H&M", "HNNMY")]
    }
}

def extract_companies_from_product(product_text):
    """
    Map product description to supplier/customer tickers.
    Production: use spaCy NER + OpenFIGI API for fuzzy matching.
    """
    key = product_text.strip().lower()
    return DEFAULT_MAPPING.get(key, {"suppliers": [], "customers": []})