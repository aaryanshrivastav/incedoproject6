"""Generate deterministic sample corpus for Northwind Labs with planted facts."""

import argparse
import csv
import json
import random
from pathlib import Path

import openpyxl
from docx import Document
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table


def generate_corpus(out_dir: Path, seed: int = 7) -> None:
    """Generate deterministic corpus of >= 36 files plus manifest.json."""
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Directory '{out_dir}' already exists and is not empty. Skipping generation.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    manifest: dict[str, dict[str, str]] = {}

    styles = getSampleStyleSheet()

    # --- 1. Text PDFs (10 files) ---
    pdf_specs = [
        (
            "hr_benefits_guide.pdf",
            "Northwind Labs 2026 Benefits Guide",
            "Standard employee health coverage covers 85% of dental expenses.",
        ),
        (
            "code_of_conduct.pdf",
            "Corporate Code of Conduct",
            "The maximum allowable gift value from suppliers is strictly $50.",
        ),
        (
            "remote_work_policy.pdf",
            "Global Remote Work Policy",
            "Remote employees receive a $500 annual home office equipment stipend.",
        ),
        (
            "cloud_security_standard.pdf",
            "Cloud Infrastructure Security Standard",
            "All production S3 buckets must enforce AES-256 server-side encryption.",
        ),
        (
            "incident_response_runbook.pdf",
            "Sev-1 Incident Response Protocol",
            "Severity 1 outages require incident command activation within 15 minutes.",
        ),
        (
            "data_retention_policy.pdf",
            "Corporate Data Retention Policy",
            "Customer financial audit records are archived for precisely 7 years.",
        ),
        (
            "travel_reimbursement.pdf",
            "Business Travel and Expense Policy",
            "Daily meal allowance for domestic travel is capped at $75 per day.",
        ),
        (
            "serverless_architecture.pdf",
            "Serverless Microservices Guidelines",
            "Lambda function timeouts must never exceed 30 seconds for synchronous APIs.",
        ),
        (
            "procurement_guidelines.pdf",
            "Vendor Procurement Guidelines",
            "Purchases over $25,000 require approval from the Chief Financial Officer.",
        ),
        (
            "open_source_compliance.pdf",
            "Open Source Software Usage Policy",
            "AGPL licensed libraries are strictly prohibited in all client-facing binaries.",
        ),
    ]

    for filename, title, fact in pdf_specs:
        filepath = out_dir / filename
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        elements = [
            Paragraph(title, styles["Title"]),
            Paragraph("Northwind Labs Enterprise Documentation", styles["Heading2"]),
            Paragraph(fact, styles["Normal"]),
            Paragraph(
                "This policy is strictly enforced across all business divisions.",
                styles["Normal"],
            ),
            Table(
                [
                    ["Parameter", "Specification", "Compliance"],
                    ["Primary Owner", "Operations & Legal", "Mandatory"],
                    ["Review Cycle", "Annual (Q1)", "Active"],
                ],
                style=[("GRID", (0, 0), (-1, -1), 1, colors.black)],
            ),
        ]
        doc.build(elements)
        manifest[filename] = {"title": title, "planted_fact": fact}

    # --- 2. Scanned PDFs via Pillow (2 files) ---
    scanned_specs = [
        (
            "scanned_patent_filing.pdf",
            "PATENT APPLICATION #US-2026-0912",
            "Quantum Resistant Key Distribution Protocol #QR-991.",
        ),
        (
            "scanned_lease_agreement.pdf",
            "OFFICE LEASE AGREEMENT",
            "Northwind Labs Tokyo branch lease expires on December 31 2028.",
        ),
    ]
    for filename, title, fact in scanned_specs:
        filepath = out_dir / filename
        img = Image.new("RGB", (850, 450), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((60, 60), f"NORTHWIND LABS CONFIDENTIAL: {title}", fill="black")
        draw.text((60, 140), fact, fill="black")
        draw.text((60, 220), "Executed by Authorized Signatory: Dr. Marcus Vance", fill="black")
        img.save(str(filepath), "PDF", resolution=300.0)
        manifest[filename] = {"title": title, "planted_fact": fact}

    # --- 3. DOCX Documents (8 files) ---
    docx_specs = [
        (
            "q3_business_review.docx",
            "Q3 2026 Executive Review",
            "Net operating profit increased by 18.4% in the APAC sector.",
        ),
        (
            "database_migration_plan.docx",
            "PostgreSQL 16 Migration Plan",
            "The production database cutover is scheduled for Saturday 02:00 UTC.",
        ),
        (
            "ai_governance_framework.docx",
            "AI Model Governance Framework",
            "All generative AI model prompts must pass automated toxicity and PII scans.",
        ),
        (
            "api_design_standards.docx",
            "RESTful API Standards v3",
            "All public endpoints must support idempotency via the Idempotency-Key HTTP header.",
        ),
        (
            "mobile_device_policy.docx",
            "Corporate Device Management Policy",
            "Lost mobile devices must be reported to the IT Helpdesk within 2 hours.",
        ),
        (
            "disaster_recovery_plan.docx",
            "Disaster Recovery Runbook",
            "The Recovery Time Objective (RTO) for core database clusters is 45 minutes.",
        ),
        (
            "performance_review_process.docx",
            "Annual Performance Evaluation Guide",
            "Peer feedback submissions close on November 15 each calendar year.",
        ),
        (
            "workplace_ergonomics.docx",
            "Workplace Ergonomics Standards",
            "Employees may request sit-stand desks without requiring a medical certificate.",
        ),
    ]
    for filename, title, fact in docx_specs:
        filepath = out_dir / filename
        doc = Document()
        doc.add_heading(title, level=1)
        doc.add_paragraph("Northwind Labs Confidential Document")
        doc.add_heading("Key Policies and Objectives", level=2)
        doc.add_paragraph(fact)
        doc.add_paragraph(
            "Additional operational requirements apply as documented in the engineering handbook."
        )
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Category"
        table.cell(0, 1).text = "Target"
        table.cell(1, 0).text = "Compliance"
        table.cell(1, 1).text = "100% Required"
        doc.save(str(filepath))
        manifest[filename] = {"title": title, "planted_fact": fact}

    # --- 4. HTML Documents (6 files) ---
    html_specs = [
        (
            "deployment_guide.html",
            "Production Deployment Guide",
            "Kubernetes cluster rolling updates require a minimum of 3 healthy replicas.",
        ),
        (
            "frontend_guidelines.html",
            "Frontend Component Architecture",
            "Design tokens require WCAG AAA contrast ratio for all button states.",
        ),
        (
            "onboarding_checklist.html",
            "New Hire Engineering Onboarding",
            "All new engineers must complete SOC2 compliance training within their first 7 days.",
        ),
        (
            "observability_guide.html",
            "Observability and Tracing Standard",
            "OpenTelemetry spans must include org_id and user_id attributes on all external calls.",
        ),
        (
            "release_notes_v2.html",
            "Platform Release Notes v2.4",
            "Version 2.4 deprecates the legacy XML export endpoint in favor of Parquet streaming.",
        ),
        (
            "security_advisory_2026.html",
            "Security Advisory 2026-04",
            "Security patch KB-9941 resolved a race condition in the session revocation cache.",
        ),
    ]
    for filename, title, fact in html_specs:
        filepath = out_dir / filename
        content = f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
  <h1>{title}</h1>
  <p>Northwind Labs Engineering Portal</p>
  <h2>Core Requirements</h2>
  <p>{fact}</p>
  <table>
    <tr><th>Protocol</th><th>Status</th></tr>
    <tr><td>Standardization</td><td>Enforced</td></tr>
  </table>
</body>
</html>"""
        filepath.write_text(content, encoding="utf-8")
        manifest[filename] = {"title": title, "planted_fact": fact}

    # --- 5. CSV Files (4 files) ---
    csv_specs = [
        (
            "hardware_inventory.csv",
            "hardware_inventory",
            [
                ["Asset_ID", "Model", "Assignment", "Location"],
                ["HW-101", "MacBook Pro M3 Max", "Lead Architect", "Tokyo Office"],
                ["HW-102", "Dell PowerEdge R750", "Build Server 01", "Datacenter Ashburn"],
            ],
            "HW-101 is assigned to Lead Architect at Tokyo Office.",
        ),
        (
            "sales_targets_2026.csv",
            "sales_targets_2026",
            [
                ["Region", "Q1_Target", "Q2_Target", "Annual_Quota"],
                ["North America", "$12.5M", "$14.0M", "$55.0M"],
                ["EMEA", "$8.0M", "$9.5M", "$38.0M"],
            ],
            "Annual quota for North America region is $55.0M.",
        ),
        (
            "sla_benchmarks.csv",
            "sla_benchmarks",
            [
                ["Tier", "Availability_Commitment", "Response_Time"],
                ["Enterprise Gold", "99.99%", "15 minutes"],
                ["Standard", "99.9%", "4 hours"],
            ],
            "Enterprise Gold SLA tier guarantees 99.99% availability.",
        ),
        (
            "customer_satisfaction.csv",
            "customer_satisfaction",
            [
                ["Cohort", "CSAT_Score", "NPS"],
                ["Enterprise SaaS", "94.2%", "+68"],
                ["Mid-Market", "88.7%", "+52"],
            ],
            "Enterprise SaaS CSAT score reached 94.2%.",
        ),
    ]
    for filename, title, rows, fact in csv_specs:
        filepath = out_dir / filename
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        manifest[filename] = {"title": title, "planted_fact": fact}

    # --- 6. XLSX Files (3 files) ---
    xlsx_specs = [
        (
            "financial_forecast.xlsx",
            "financial_forecast",
            [
                (
                    "Projections",
                    [
                        ["Quarter", "Revenue", "EBITDA"],
                        ["Q1 2026", "$28.4M", "$6.2M"],
                        ["Q2 2026", "$31.0M", "$7.1M"],
                    ],
                ),
            ],
            "Q1 2026 projected revenue is $28.4M.",
        ),
        (
            "headcount_budget.xlsx",
            "headcount_budget",
            [
                (
                    "Engineering",
                    [["Role", "Openings", "Budget"], ["Principal ML Engineer", "3", "$750K"]],
                ),
                ("Product", [["Role", "Openings", "Budget"], ["Staff Technical PM", "1", "$210K"]]),
            ],
            "Engineering department has 3 open Principal ML Engineer positions.",
        ),
        (
            "vendor_cost_allocations.xlsx",
            "vendor_cost_allocations",
            [
                (
                    "Cloud",
                    [["Vendor", "Annual_Cost", "Contract_End"], ["AWS", "$1.8M", "2027-06-30"]],
                ),
            ],
            "Annual AWS cloud expenditure is contracted at $1.8M ending June 2027.",
        ),
    ]
    for filename, title, sheets, fact in xlsx_specs:
        filepath = out_dir / filename
        wb = openpyxl.Workbook()
        # remove default sheet
        wb.remove(wb.active)
        for s_name, s_rows in sheets:
            ws = wb.create_sheet(title=s_name)
            for r in s_rows:
                ws.append(r)
        wb.save(str(filepath))
        manifest[filename] = {"title": title, "planted_fact": fact}

    # --- 7. Duplicate Content File (1 file) ---
    # Same content as hr_benefits_guide.pdf under a different name
    orig_bytes = (out_dir / "hr_benefits_guide.pdf").read_bytes()
    (out_dir / "hr_benefits_guide_duplicate.pdf").write_bytes(orig_bytes)
    manifest["hr_benefits_guide_duplicate.pdf"] = {
        "title": "Northwind Labs 2026 Benefits Guide",
        "planted_fact": "Duplicate of hr_benefits_guide.pdf for deduplication testing.",
    }

    # --- 8. Bad / Corrupt Files (3 files) ---
    (out_dir / "corrupted_archive.pdf").write_bytes(b"%PDF-1.4\nTRUNCATED_GARBAGE_PAYLOAD")
    (out_dir / "empty_document.txt").write_bytes(b"")
    (out_dir / "fake_extension.pdf").write_text(
        "This is an unadulterated text file masquerading as a PDF.", encoding="utf-8"
    )

    manifest["corrupted_archive.pdf"] = {
        "title": "corrupted_archive",
        "planted_fact": "Malformed PDF bytes",
    }
    manifest["empty_document.txt"] = {"title": "empty_document", "planted_fact": "Zero byte file"}
    manifest["fake_extension.pdf"] = {
        "title": "fake_extension",
        "planted_fact": "Plain text with .pdf suffix",
    }

    # Write manifest.json
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_files = len(list(out_dir.glob("*.*")))
    print(f"Generated {total_files} files in '{out_dir}' with manifest at '{manifest_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample corpus for Northwind Labs")
    parser.add_argument(
        "--out", type=Path, default=Path("data/raw/sample"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    generate_corpus(args.out, args.seed)


if __name__ == "__main__":
    main()
