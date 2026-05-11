# report_generator.py
# DockExplain — shim that delegates to explanation_pdf_generator.py
#
# This replaces the old report_generator.py. Any code that does:
#   from report_generator import generate_report
# will now get the new, working PDF generator.

import os

def generate_report(json_path: str, out_pdf_path: str | None = None) -> str:
    """
    Generate the DockExplain PDF report.
    Delegates to explanation_pdf_generator.generate_explanation_pdf().

    Accepts either a results JSON or an explanation JSON — automatically
    finds the explanation JSON if a results JSON is passed.
    """
    from explanation_pdf_generator import generate_explanation_pdf

    # If given a results JSON, look for the companion explanation JSON
    if json_path.endswith("_results.json"):
        explanation_json = json_path.replace("_results.json", "_explanation.json")
        if os.path.exists(explanation_json):
            json_path = explanation_json
        # else fall through and use results JSON as-is

    return generate_explanation_pdf(json_path, out_pdf_path)