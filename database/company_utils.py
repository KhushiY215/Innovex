"""
Utility functions related to company identifiers.

Generates a deterministic INT8 company_id from company_name.
This ensures the same company always maps to the same ID.
"""

import hashlib


def generate_company_id(company_name: str) -> int:
    """
    Generate a stable BIGINT (INT8) company_id from a company name.

    Steps:
    1. Normalize the name
    2. Hash using SHA256
    3. Convert first 12 hex chars → integer
    """

    if not company_name:
        raise ValueError("company_name cannot be empty")

    normalized = company_name.strip().lower()

    # SHA256 hash
    hashed = hashlib.sha256(normalized.encode()).hexdigest()

    # Convert part of hash to integer
    company_id = int(hashed[:12], 16)

    return company_id