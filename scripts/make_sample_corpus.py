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


def _build_manifest_enrichment() -> tuple[
    dict[str, dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Return per-file QA metadata, unanswerable questions, and multi-hop entries."""
    per_file: dict[str, dict[str, object]] = {
        "hr_benefits_guide.pdf": {
            "question": "What percentage of dental expenses does standard employee health coverage pay?",
            "answer": "Standard employee health coverage covers 85% of dental expenses.",
            "evidence_span": "Standard employee health coverage covers 85% of dental expenses.",
            "kind": "factual",
        },
        "code_of_conduct.pdf": {
            "question": "What is the maximum allowable gift value from suppliers under the Corporate Code of Conduct?",
            "answer": "The maximum allowable gift value from suppliers is strictly $50.",
            "evidence_span": "The maximum allowable gift value from suppliers is strictly $50.",
            "kind": "factual",
        },
        "remote_work_policy.pdf": {
            "question": "How much is the annual home office equipment stipend for remote employees?",
            "answer": "Remote employees receive a $500 annual home office equipment stipend.",
            "evidence_span": "Remote employees receive a $500 annual home office equipment stipend.",
            "kind": "factual",
        },
        "cloud_security_standard.pdf": {
            "question": "What server-side encryption standard is required for all production S3 buckets?",
            "answer": "All production S3 buckets must enforce AES-256 server-side encryption.",
            "evidence_span": "All production S3 buckets must enforce AES-256 server-side encryption.",
            "kind": "factual",
        },
        "incident_response_runbook.pdf": {
            "question": "Within how many minutes must incident command be activated for Severity 1 outages?",
            "answer": "Severity 1 outages require incident command activation within 15 minutes.",
            "evidence_span": "Severity 1 outages require incident command activation within 15 minutes.",
            "kind": "factual",
        },
        "data_retention_policy.pdf": {
            "question": "How long are customer financial audit records archived under the Corporate Data Retention Policy?",
            "answer": "Customer financial audit records are archived for precisely 7 years.",
            "evidence_span": "Customer financial audit records are archived for precisely 7 years.",
            "kind": "factual",
        },
        "travel_reimbursement.pdf": {
            "question": "What is the daily meal allowance cap for domestic business travel?",
            "answer": "Daily meal allowance for domestic travel is capped at $75 per day.",
            "evidence_span": "Daily meal allowance for domestic travel is capped at $75 per day.",
            "kind": "factual",
            "acl_hint": "finance",
        },
        "serverless_architecture.pdf": {
            "question": "What is the maximum allowed Lambda function timeout for synchronous APIs?",
            "answer": "Lambda function timeouts must never exceed 30 seconds for synchronous APIs.",
            "evidence_span": "Lambda function timeouts must never exceed 30 seconds for synchronous APIs.",
            "kind": "factual",
        },
        "procurement_guidelines.pdf": {
            "question": "Who must approve vendor purchases over $25,000?",
            "answer": "Purchases over $25,000 require approval from the Chief Financial Officer.",
            "evidence_span": "Purchases over $25,000 require approval from the Chief Financial Officer.",
            "kind": "factual",
            "acl_hint": "finance",
        },
        "open_source_compliance.pdf": {
            "question": "Which open source license is strictly prohibited in all client-facing binaries?",
            "answer": "AGPL licensed libraries are strictly prohibited in all client-facing binaries.",
            "evidence_span": "AGPL licensed libraries are strictly prohibited in all client-facing binaries.",
            "kind": "factual",
        },
        "scanned_patent_filing.pdf": {
            "question": "What protocol is described in patent application #US-2026-0912?",
            "answer": "Patent application #US-2026-0912 describes Quantum Resistant Key Distribution Protocol #QR-991.",
            "evidence_span": "Key Distribution Protocol",
            "kind": "factual",
        },
        "scanned_lease_agreement.pdf": {
            "question": "When does the Northwind Labs Tokyo branch office lease expire?",
            "answer": "Northwind Labs Tokyo branch lease expires on December 31 2028.",
            "evidence_span": "December 31 2028",
            "kind": "factual",
        },
        "q3_business_review.docx": {
            "question": "By what percentage did net operating profit increase in the APAC sector in Q3 2026?",
            "answer": "Net operating profit increased by 18.4% in the APAC sector.",
            "evidence_span": "Net operating profit increased by 18.4% in the APAC sector.",
            "kind": "factual",
            "acl_hint": "finance",
        },
        "database_migration_plan.docx": {
            "question": "When is the PostgreSQL 16 production database cutover scheduled?",
            "answer": "The production database cutover is scheduled for Saturday 02:00 UTC.",
            "evidence_span": "The production database cutover is scheduled for Saturday 02:00 UTC.",
            "kind": "factual",
        },
        "ai_governance_framework.docx": {
            "question": "What automated checks must all generative AI model prompts pass?",
            "answer": "All generative AI model prompts must pass automated toxicity and PII scans.",
            "evidence_span": "All generative AI model prompts must pass automated toxicity and PII scans.",
            "kind": "factual",
        },
        "api_design_standards.docx": {
            "question": "Which HTTP header must all public endpoints support for idempotency?",
            "answer": "All public endpoints must support idempotency via the Idempotency-Key HTTP header.",
            "evidence_span": "All public endpoints must support idempotency via the Idempotency-Key HTTP header.",
            "kind": "factual",
        },
        "mobile_device_policy.docx": {
            "question": "Within how many hours must lost mobile devices be reported to the IT Helpdesk?",
            "answer": "Lost mobile devices must be reported to the IT Helpdesk within 2 hours.",
            "evidence_span": "Lost mobile devices must be reported to the IT Helpdesk within 2 hours.",
            "kind": "factual",
        },
        "disaster_recovery_plan.docx": {
            "question": "What is the Recovery Time Objective (RTO) for core database clusters?",
            "answer": "The Recovery Time Objective (RTO) for core database clusters is 45 minutes.",
            "evidence_span": "The Recovery Time Objective (RTO) for core database clusters is 45 minutes.",
            "kind": "factual",
        },
        "performance_review_process.docx": {
            "question": "On what date do peer feedback submissions close each calendar year?",
            "answer": "Peer feedback submissions close on November 15 each calendar year.",
            "evidence_span": "Peer feedback submissions close on November 15 each calendar year.",
            "kind": "factual",
        },
        "workplace_ergonomics.docx": {
            "question": "Do employees need a medical certificate to request a sit-stand desk?",
            "answer": "Employees may request sit-stand desks without requiring a medical certificate.",
            "evidence_span": "Employees may request sit-stand desks without requiring a medical certificate.",
            "kind": "factual",
        },
        "deployment_guide.html": {
            "question": "How many healthy replicas are required as a minimum during Kubernetes rolling updates?",
            "answer": "Kubernetes cluster rolling updates require a minimum of 3 healthy replicas.",
            "evidence_span": "Kubernetes cluster rolling updates require a minimum of 3 healthy replicas.",
            "kind": "factual",
        },
        "frontend_guidelines.html": {
            "question": "What contrast ratio do frontend design tokens require for all button states?",
            "answer": "Design tokens require WCAG AAA contrast ratio for all button states.",
            "evidence_span": "Design tokens require WCAG AAA contrast ratio for all button states.",
            "kind": "factual",
        },
        "onboarding_checklist.html": {
            "question": "Within how many days must new engineers complete SOC2 compliance training?",
            "answer": "All new engineers must complete SOC2 compliance training within their first 7 days.",
            "evidence_span": "All new engineers must complete SOC2 compliance training within their first 7 days.",
            "kind": "factual",
        },
        "observability_guide.html": {
            "question": "Which attributes must OpenTelemetry spans include on all external calls?",
            "answer": "OpenTelemetry spans must include org_id and user_id attributes on all external calls.",
            "evidence_span": "OpenTelemetry spans must include org_id and user_id attributes on all external calls.",
            "kind": "factual",
        },
        "release_notes_v2.html": {
            "question": "What does Platform Release v2.4 deprecate and what replaces it?",
            "answer": "Version 2.4 deprecates the legacy XML export endpoint in favor of Parquet streaming.",
            "evidence_span": "Version 2.4 deprecates the legacy XML export endpoint in favor of Parquet streaming.",
            "kind": "factual",
        },
        "security_advisory_2026.html": {
            "question": "Which security patch resolved the race condition in the session revocation cache?",
            "answer": "Security patch KB-9941 resolved the race condition in the session revocation cache.",
            "evidence_span": "Security patch KB-9941 resolved a race condition in the session revocation cache.",
            "kind": "factual",
        },
        "hardware_inventory.csv": {
            "question": "Who is asset HW-101 assigned to and at which office location?",
            "answer": "HW-101 is a MacBook Pro M3 Max assigned to the Lead Architect at the Tokyo Office.",
            "evidence_span": "Asset_ID: HW-101 | Model: MacBook Pro M3 Max | Assignment: Lead Architect | Location: Tokyo Office",
            "kind": "table",
            "extra_rows": [
                {
                    "question": "Where is Build Server 01 (asset HW-102) located?",
                    "answer": "HW-102 (Dell PowerEdge R750, Build Server 01) is located at Datacenter Ashburn.",
                    "evidence_span": "Asset_ID: HW-102 | Model: Dell PowerEdge R750 | Assignment: Build Server 01 | Location: Datacenter Ashburn",
                }
            ],
        },
        "sales_targets_2026.csv": {
            "question": "What is the 2026 annual sales quota for the North America region?",
            "answer": "The annual quota for North America is $55.0M.",
            "evidence_span": "Region: North America | Q1_Target: $12.5M | Q2_Target: $14.0M | Annual_Quota: $55.0M",
            "kind": "table",
            "extra_rows": [
                {
                    "question": "What is the 2026 annual sales quota for the EMEA region?",
                    "answer": "The annual quota for EMEA is $38.0M.",
                    "evidence_span": "Region: EMEA | Q1_Target: $8.0M | Q2_Target: $9.5M | Annual_Quota: $38.0M",
                }
            ],
        },
        "sla_benchmarks.csv": {
            "question": "What availability percentage and response time are guaranteed by the Enterprise Gold SLA tier?",
            "answer": "Enterprise Gold SLA guarantees 99.99% availability and a 15 minutes response time.",
            "evidence_span": "Tier: Enterprise Gold | Availability_Commitment: 99.99% | Response_Time: 15 minutes",
            "kind": "table",
            "extra_rows": [
                {
                    "question": "What is the response time commitment for the Standard SLA tier?",
                    "answer": "The Standard SLA tier guarantees 99.9% availability with a 4 hours response time.",
                    "evidence_span": "Tier: Standard | Availability_Commitment: 99.9% | Response_Time: 4 hours",
                }
            ],
        },
        "customer_satisfaction.csv": {
            "question": "What CSAT score and NPS did Enterprise SaaS achieve?",
            "answer": "Enterprise SaaS achieved a 94.2% CSAT score and an NPS of +68.",
            "evidence_span": "Cohort: Enterprise SaaS | CSAT_Score: 94.2% | NPS: +68",
            "kind": "table",
            "extra_rows": [
                {
                    "question": "What CSAT score and NPS did the Mid-Market cohort achieve?",
                    "answer": "Mid-Market achieved an 88.7% CSAT score and an NPS of +52.",
                    "evidence_span": "Cohort: Mid-Market | CSAT_Score: 88.7% | NPS: +52",
                }
            ],
        },
        "financial_forecast.xlsx": {
            "question": "What are the Q1 2026 projected revenue and EBITDA in the financial forecast?",
            "answer": "Q1 2026 projected revenue is $28.4M and EBITDA is $6.2M.",
            "evidence_span": "Quarter: Q1 2026 | Revenue: $28.4M | EBITDA: $6.2M",
            "kind": "table",
            "acl_hint": "finance",
            "extra_rows": [
                {
                    "question": "What are the Q2 2026 projected revenue and EBITDA in the financial forecast?",
                    "answer": "Q2 2026 projected revenue is $31.0M and EBITDA is $7.1M.",
                    "evidence_span": "Quarter: Q2 2026 | Revenue: $31.0M | EBITDA: $7.1M",
                }
            ],
        },
        "headcount_budget.xlsx": {
            "question": "How many open Principal ML Engineer positions are budgeted in Engineering?",
            "answer": "Engineering has 3 open Principal ML Engineer positions with a $750K budget.",
            "evidence_span": "Role: Principal ML Engineer | Openings: 3 | Budget: $750K",
            "kind": "table",
            "acl_hint": "finance",
            "extra_rows": [
                {
                    "question": "What is the headcount budget for Staff Technical PM in Product?",
                    "answer": "Product has 1 open Staff Technical PM position with a $210K budget.",
                    "evidence_span": "Role: Staff Technical PM | Openings: 1 | Budget: $210K",
                }
            ],
        },
        "vendor_cost_allocations.xlsx": {
            "question": "What is the contracted annual AWS cloud expenditure and contract end date?",
            "answer": "Annual AWS cloud expenditure is contracted at $1.8M ending 2027-06-30.",
            "evidence_span": "Vendor: AWS | Annual_Cost: $1.8M | Contract_End: 2027-06-30",
            "kind": "table",
            "acl_hint": "finance",
        },
    }

    unanswerable: list[dict[str, object]] = [
        {
            "question": "What is Northwind Labs' current NASDAQ ticker symbol and IPO share price?",
            "distinctive_tokens": ["nasdaq", "ticker", "ipo"],
            "notes": "Stock listing not in corpus",
        },
        {
            "question": "What is the guest Wi-Fi password for the London branch office?",
            "distinctive_tokens": ["wi-fi", "wifi", "london"],
            "notes": "No London office or Wi-Fi credentials",
        },
        {
            "question": "How many paid bereavement leave days are granted to part-time contractors in Brazil?",
            "distinctive_tokens": ["bereavement", "brazil"],
            "notes": "Contractor bereavement policy in Brazil absent",
        },
        {
            "question": "Which espresso roast is stocked in the Tokyo office breakroom?",
            "distinctive_tokens": ["espresso", "breakroom"],
            "notes": "Breakroom inventory absent",
        },
        {
            "question": "What was the final score of the 2026 Northwind Labs company softball tournament?",
            "distinctive_tokens": ["softball", "tournament"],
            "notes": "Social sports events absent",
        },
        {
            "question": "Who is the external catering vendor for the Zurich executive retreat?",
            "distinctive_tokens": ["catering", "zurich"],
            "notes": "Zurich retreat catering absent",
        },
        {
            "question": "What discount percentage do employees receive on Northwind Labs branded bicycles?",
            "distinctive_tokens": ["bicycles", "branded"],
            "notes": "Bicycle merchandise discount absent",
        },
        {
            "question": "When is the annual fire drill scheduled for the Berlin warehouse?",
            "distinctive_tokens": ["berlin", "warehouse"],
            "notes": "Berlin warehouse absent",
        },
        {
            "question": "What is the maximum pet insurance reimbursement for remote employees?",
            "distinctive_tokens": ["pet", "veterinary"],
            "notes": "Pet insurance policy absent",
        },
        {
            "question": "Which cryptocurrency wallet is approved for customer invoice payments?",
            "distinctive_tokens": ["cryptocurrency", "bitcoin"],
            "notes": "Crypto payment policy absent",
        },
    ]

    multi_hop: list[dict[str, object]] = [
        {
            "question": "What dental coverage percentage and annual remote equipment stipend do eligible employees receive?",
            "answer": "Standard employee health coverage covers 85% of dental expenses, and remote employees receive a $500 annual home office equipment stipend.",
            "sources": ["hr_benefits_guide.pdf", "remote_work_policy.pdf"],
            "evidence_spans": [
                "Standard employee health coverage covers 85% of dental expenses.",
                "Remote employees receive a $500 annual home office equipment stipend.",
            ],
            "notes": "HR benefits + remote work",
        },
        {
            "question": "What encryption standard is required for production S3 buckets and what is the max synchronous Lambda timeout?",
            "answer": "All production S3 buckets must enforce AES-256 server-side encryption, and Lambda function timeouts must never exceed 30 seconds for synchronous APIs.",
            "sources": ["cloud_security_standard.pdf", "serverless_architecture.pdf"],
            "evidence_spans": [
                "All production S3 buckets must enforce AES-256 server-side encryption.",
                "Lambda function timeouts must never exceed 30 seconds for synchronous APIs.",
            ],
            "notes": "Cloud security + serverless",
        },
        {
            "question": "When is the PostgreSQL 16 cutover scheduled and what is the RTO for core database clusters?",
            "answer": "The production database cutover is scheduled for Saturday 02:00 UTC, and the Recovery Time Objective (RTO) for core database clusters is 45 minutes.",
            "sources": ["database_migration_plan.docx", "disaster_recovery_plan.docx"],
            "evidence_spans": [
                "The production database cutover is scheduled for Saturday 02:00 UTC.",
                "The Recovery Time Objective (RTO) for core database clusters is 45 minutes.",
            ],
            "notes": "DB migration + DR",
        },
        {
            "question": "What is the supplier gift limit in the Code of Conduct and which open source license is prohibited in client-facing binaries?",
            "answer": "The maximum allowable gift value from suppliers is strictly $50, and AGPL licensed libraries are strictly prohibited in all client-facing binaries.",
            "sources": ["code_of_conduct.pdf", "open_source_compliance.pdf"],
            "evidence_spans": [
                "The maximum allowable gift value from suppliers is strictly $50.",
                "AGPL licensed libraries are strictly prohibited in all client-facing binaries.",
            ],
            "notes": "Code of conduct + OSS",
        },
        {
            "question": "Which HTTP header enforces API idempotency and what export endpoint was deprecated in Release v2.4?",
            "answer": "All public endpoints must support idempotency via the Idempotency-Key HTTP header, and Version 2.4 deprecates the legacy XML export endpoint in favor of Parquet streaming.",
            "sources": ["api_design_standards.docx", "release_notes_v2.html"],
            "evidence_spans": [
                "All public endpoints must support idempotency via the Idempotency-Key HTTP header.",
                "Version 2.4 deprecates the legacy XML export endpoint in favor of Parquet streaming.",
            ],
            "notes": "API standards + release notes",
        },
        {
            "question": "What automated scans are required for AI prompts and which patch fixed the session revocation cache race condition?",
            "answer": "All generative AI model prompts must pass automated toxicity and PII scans, and Security patch KB-9941 resolved the race condition in the session revocation cache.",
            "sources": ["ai_governance_framework.docx", "security_advisory_2026.html"],
            "evidence_spans": [
                "All generative AI model prompts must pass automated toxicity and PII scans.",
                "Security patch KB-9941 resolved a race condition in the session revocation cache.",
            ],
            "notes": "AI governance + security advisory",
        },
        {
            "question": "Within how many minutes must Sev-1 incident command be activated and within how many hours must lost mobile devices be reported?",
            "answer": "Severity 1 outages require incident command activation within 15 minutes, and lost mobile devices must be reported to the IT Helpdesk within 2 hours.",
            "sources": ["incident_response_runbook.pdf", "mobile_device_policy.docx"],
            "evidence_spans": [
                "Severity 1 outages require incident command activation within 15 minutes.",
                "Lost mobile devices must be reported to the IT Helpdesk within 2 hours.",
            ],
            "notes": "Incident response + mobile policy",
        },
        {
            "question": "How long are customer financial audit records archived and when does the Tokyo branch office lease expire?",
            "answer": "Customer financial audit records are archived for precisely 7 years, and the Northwind Labs Tokyo branch lease expires on December 31 2028.",
            "sources": ["data_retention_policy.pdf", "scanned_lease_agreement.pdf"],
            "evidence_spans": [
                "Customer financial audit records are archived for precisely 7 years.",
                "December 31 2028",
            ],
            "notes": "Retention + scanned lease",
        },
        {
            "question": "When do peer feedback submissions close and can employees request sit-stand desks without a medical certificate?",
            "answer": "Peer feedback submissions close on November 15 each calendar year, and employees may request sit-stand desks without requiring a medical certificate.",
            "sources": ["performance_review_process.docx", "workplace_ergonomics.docx"],
            "evidence_spans": [
                "Peer feedback submissions close on November 15 each calendar year.",
                "Employees may request sit-stand desks without requiring a medical certificate.",
            ],
            "notes": "Performance review + ergonomics",
        },
        {
            "question": "What is the Enterprise Gold SLA availability commitment and what CSAT score did Enterprise SaaS achieve?",
            "answer": "Enterprise Gold SLA guarantees 99.99% availability, and Enterprise SaaS achieved a 94.2% CSAT score.",
            "sources": ["sla_benchmarks.csv", "customer_satisfaction.csv"],
            "evidence_spans": [
                "Tier: Enterprise Gold | Availability_Commitment: 99.99% | Response_Time: 15 minutes",
                "Cohort: Enterprise SaaS | CSAT_Score: 94.2% | NPS: +68",
            ],
            "notes": "SLA + CSAT tables",
        },
    ]

    return per_file, unanswerable, multi_hop


def enrich_manifest(manifest_path: Path) -> None:
    """Enrich manifest.json in-place without overwriting existing enriched fields."""
    if not manifest_path.exists():
        return
    manifest: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_file, unanswerable, multi_hop = _build_manifest_enrichment()

    for filename, enrichment in per_file.items():
        existing = manifest.get(filename)
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.setdefault("source", filename)
            for k, v in enrichment.items():
                merged.setdefault(k, v)
            manifest[filename] = merged

    if "unanswerable_questions" not in manifest:
        manifest["unanswerable_questions"] = unanswerable
    if "multi_hop" not in manifest:
        manifest["multi_hop"] = multi_hop

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_corpus(out_dir: Path, seed: int = 7) -> None:
    """Generate deterministic corpus of >= 36 files plus manifest.json."""
    if out_dir.exists() and any(out_dir.iterdir()):
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            enrich_manifest(manifest_path)
            print(
                f"Directory '{out_dir}' already exists. Preserved files and enriched '{manifest_path}'."
            )
        else:
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

    # Write manifest.json and enrich with QA/unanswerable/multi_hop metadata
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    enrich_manifest(manifest_path)

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
