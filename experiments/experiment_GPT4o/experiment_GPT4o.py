"""
Automated GPT-4o experiment runner for LLM-assisted sketch-based
requirements elicitation and prototype generation.

Folder structure expected:

experiment_GPT4o/
│
├── experiment_GPT4o.py
│
├── service_portal/
│   └── sketches/
│       ├── Screen1.png
│       ├── Screen2.png
│       └── Screen3.png
|       └── Screen4.png
|       └── Screen5.png
|       
├── Ecommerce/
│   └── sketches/
│       ├── Screen1.png
│       ├── Screen2.png
│       └── Screen3.png
|       └── Screen4.png
|       └── Screen5.png
|       └── Screen6.png
│
└── Healthcare/
    └── sketches/
        ├── Screen1.png
        ├── Screen2.png
        └── Screen3.png
        └── Screen4.png
        └── Screen5.png
"""
from __future__ import annotations

import base64
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI


# =========================
# CONFIGURATION
# =========================

MODEL_LABEL = "OpenAI GPT-4o"
MODEL_ID = "gpt-4o"

TEMPERATURE = 0.2
TOP_P = 1.0
TOP_K_REPORTED = "not supported by model/API"
MAX_OUTPUT_TOKENS = 16384
RUNS_PER_SCENARIO = 3

IMAGE_DETAIL = "high"
TIMEOUT_SECONDS = 180.0
MAX_RETRIES = 2

BASE_DIR = Path(__file__).resolve().parent

SCENARIOS = {
    "service_portal": BASE_DIR / "service_portal",
    "Ecommerce": BASE_DIR / "Ecommerce",
    "Healthcare": BASE_DIR / "Healthcare",
}

SCREEN_FILES = [
    "Screen1.png",
    "Screen2.png",
    "Screen3.png",
    "Screen4.png",
    "Screen5.png",
]

client = OpenAI(timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)


# =========================
# PHASE 1 PROMPT
# =========================

REQUIREMENTS_PROMPT = """
You are given multiple low-fidelity, hand-drawn sketches representing different screens of the same software system.

Treat the sketches as a connected multi-screen workflow created during early-stage requirements elicitation.

Your task is to generate ONLY requirements-related artifacts. Do NOT generate HTML, CSS, or JavaScript in this phase.

Tasks:
1. Identify each screen and briefly describe:
   - Screen name
   - Purpose
   - Key UI components
   - Main user interactions

2. Generate approximately 12–20 functional requirements using formal “shall” statements.

3. Ensure all requirements are grounded in the sketches and workflow context.
   - Do not introduce unsupported advanced functionality.
   - Reasonable assumptions are allowed only when needed to complete the workflow.

4. Maintain traceability by linking requirements to relevant screen(s) and interface elements.

5. Identify ambiguities, missing information, unclear elements, and assumptions.

Output Format:

A. Screen Descriptions

B. Functional Requirements
- Use identifiers such as FR-1, FR-2, etc.
- Use formal “shall” statements.
- Group logically.

C. Traceability Mapping
Screen → Requirement IDs → Interface Elements

D. Ambiguities and Assumptions

Keep the response concise and avoid unnecessary repetition.
"""

# =========================
# PHASE 2 PROMPT: PROTOTYPE ONLY
# =========================

PROTOTYPE_PROMPT = """
You are given:
1. Multiple low-fidelity sketches representing screens of the same software system.
2. Structured requirements, assumptions, and traceability mappings generated in Phase 1.

Your task is to generate ONLY the interactive prototype.

The prototype must:
- Be implemented as ONE self-contained HTML file.
- Embed ALL CSS and JavaScript inside the same HTML file.
- Include ALL screens and workflows represented in the sketches.
- Preserve workflow continuity, navigation consistency, terminology consistency, and data-flow consistency across screens.
- Maintain strong alignment with the sketches and Phase 1 requirements.
- Avoid introducing unsupported or hallucinated functionality.

Prototype Design Rules:
- Reuse shared CSS classes and reusable UI components.
- Keep the implementation concise, modular, and lightweight.
- Avoid unnecessary nesting, excessive whitespace, verbose comments, and redundant markup.
- Do NOT use external frameworks or libraries unless absolutely necessary.
- Do NOT embed base64 images, large assets, SVG blobs, or unnecessary media.
- Use lightweight placeholders for missing images or icons when needed.
- Keep JavaScript minimal and focused only on interactions directly implied by the sketches or requirements.

Allowed lightweight interactions include:
- Screen navigation
- Form validation feedback
- Status messages
- Show/hide sections
- Dashboard updates
- Appointment confirmations
- Ticket confirmations
- Cart updates
- Modal toggles
- Basic filtering/search

Do NOT implement:
- Backend/database logic
- Authentication security
- APIs
- Persistent storage
- Complex animations
- Advanced business logic
- Features not grounded in the sketches or Phase 1 outputs

Where appropriate, enrich the prototype with realistic but lightweight UI elements such as:
- Buttons
- Labels
- Tables
- Cards
- Notifications
- Status indicators
- Sample records
- Dashboard statistics
- Informational panels
- Placeholder charts/icons

These additions must:
- Remain semantically consistent with the sketches
- Preserve intended workflows
- Support usability without expanding system scope

Code Quality Requirements:
- Ensure valid and properly closed HTML.
- Ensure CSS and JavaScript are functional and non-redundant.
- Ensure screen sections are clearly organized.
- Ensure responsive layout behavior for desktop and mobile widths.
- Use semantic HTML where appropriate.

Output Constraints:
- Output ONLY ONE complete ```html code block.
- Do NOT include explanations, markdown text, summaries, or analysis outside the code block.
- Ensure the generated HTML can run directly in a browser without modification.
- Prioritize completeness and correctness over visual perfection.
"""
# =========================
# LOGGING
# =========================

def append_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def write_experiment_log_header(log_path: Path) -> None:
    today = datetime.now().strftime("%Y-%m-%d")

    content = f"""Experiment Log

Model label: {MODEL_LABEL}
Requested model ID: {MODEL_ID}
Pipeline: Two-phase generation

Phase 1: Requirements, traceability, ambiguities
Phase 2: Prototype generation using sketches + Phase 1 output

Temperature: {TEMPERATURE}
Top-p: {TOP_P}
Top-k: {TOP_K_REPORTED}
Max output tokens per phase: {MAX_OUTPUT_TOKENS}
Runs per scenario: {RUNS_PER_SCENARIO}
Image detail / quality: {IMAGE_DETAIL}
Timeout: {TIMEOUT_SECONDS} seconds
Retries: {MAX_RETRIES}
Date: {today}

Base folder: {BASE_DIR}

------------------------------------------------------------
"""
    log_path.write_text(content, encoding="utf-8")


# =========================
# CONNECTION CHECK
# =========================

def check_openai_connection(log_path: Path) -> bool:
    try:
        response = client.responses.create(
            model=MODEL_ID,
            input="Say exactly: connection successful",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=20,
        )

        print("OpenAI connection successful.")
        print("Response:", response.output_text)
        print("Returned model:", getattr(response, "model", "unknown"))

        append_log(log_path, "OpenAI connection successful.")
        append_log(log_path, f"Connection test returned model: {getattr(response, 'model', 'unknown')}")
        return True

    except Exception as e:
        print("OpenAI connection failed.")
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR DETAILS:", e)

        append_log(log_path, "OpenAI connection failed.")
        append_log(log_path, f"ERROR TYPE: {type(e).__name__}")
        append_log(log_path, f"ERROR DETAILS: {e}")
        return False


# =========================
# HELPERS
# =========================

def encode_image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()

    if suffix == ".png":
        mime_type = "image/png"
    elif suffix in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif suffix == ".webp":
        mime_type = "image/webp"
    else:
        raise ValueError(f"Unsupported image file type: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def validate_scenario_folder(scenario_name: str, scenario_path: Path) -> List[Path]:
    sketch_dir = scenario_path / "sketches"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario folder: {scenario_path}")

    if not sketch_dir.exists():
        raise FileNotFoundError(f"Missing sketches folder: {sketch_dir}")

    image_paths = []

    for screen_file in SCREEN_FILES:
        expected_path = sketch_dir / screen_file

        if expected_path.exists():
            image_paths.append(expected_path)
            continue

        matches = [
            p for p in sketch_dir.iterdir()
            if p.name.lower() == screen_file.lower()
        ]

        if matches:
            image_paths.append(matches[0])
            continue

        raise FileNotFoundError(
            f"Missing required sketch for {scenario_name}: {expected_path}"
        )

    return image_paths


def extract_html_from_output(output_text: str) -> Optional[str]:
    html_match = re.search(
        r"```html\s*(.*?)```",
        output_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if html_match:
        return html_match.group(1).strip()

    raw_match = re.search(
        r"(<!DOCTYPE html.*?</html>|<html.*?</html>)",
        output_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if raw_match:
        return raw_match.group(1).strip()

    return None


def get_actual_model_id(response) -> str:
    return getattr(response, "model", "unknown")


def get_response_status(response) -> str:
    parts = []

    for attr in ["status", "finish_reason", "incomplete_details"]:
        value = getattr(response, attr, None)
        if value:
            parts.append(f"{attr}={value}")

    try:
        if hasattr(response, "output") and response.output:
            for item in response.output:
                item_status = getattr(item, "status", None)
                if item_status:
                    parts.append(f"output_item_status={item_status}")
    except Exception:
        pass

    return "; ".join(parts) if parts else "unknown"


def save_manifest_row(manifest_path: Path, row: dict, fieldnames: List[str]) -> None:
    retries = 3

    for attempt in range(retries):
        try:
            file_exists = manifest_path.exists()

            with manifest_path.open("a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)

            return

        except PermissionError:
            if attempt < retries - 1:
                print("Manifest file is open or locked. Retrying in 2 seconds...")
                time.sleep(2)
            else:
                raise


# =========================
# COMPLETENESS CHECKS
# =========================

def check_requirements_completeness(output_text: str) -> Tuple[bool, str]:
    required = [
        "A. Screen Descriptions",
        "B. Functional Requirements",
        "C. Traceability Mapping",
        "D. Ambiguities and Assumptions"
    ]

    issues = []

    for section in required:
        if section.lower() not in output_text.lower():
            issues.append(f"Missing section: {section}")

    if "FR-" not in output_text:
        issues.append("No functional requirement IDs detected")

    if output_text.count("```") % 2 != 0:
        issues.append("Unclosed markdown code block")

    flagged = len(issues) > 0
    return flagged, " | ".join(issues) if flagged else "None"


def check_prototype_completeness(
    output_text: str,
    html_code: Optional[str],
    response_status: str,
) -> Tuple[bool, str]:
    issues = []

    if not html_code:
        issues.append("No HTML extracted")
    else:
        lower_html = html_code.lower()

        if "<html" not in lower_html:
            issues.append("Missing opening <html> tag")
        if "</html>" not in lower_html:
            issues.append("Missing closing </html> tag")
        if "<body" not in lower_html:
            issues.append("Missing opening <body> tag")
        if "</body>" not in lower_html:
            issues.append("Missing closing </body> tag")
        if "<style" not in lower_html:
            issues.append("Missing embedded CSS/style block")
        if "<script" not in lower_html:
            issues.append("Missing embedded JavaScript/script block")

    if output_text.count("```") % 2 != 0:
        issues.append("Unclosed markdown code block")

    lower_status = response_status.lower()
    if "incomplete" in lower_status or "max" in lower_status or "length" in lower_status:
        issues.append(f"Response metadata suggests truncation: {response_status}")

    flagged = len(issues) > 0
    return flagged, " | ".join(issues) if flagged else "None"


# =========================
# API CALLS
# =========================

def build_image_content(image_paths: List[Path]) -> list:
    content = []

    for image_path in image_paths:
        content.append(
            {
                "type": "input_image",
                "image_url": encode_image_to_data_url(image_path),
                "detail": IMAGE_DETAIL,
            }
        )

    return content


def run_phase1_requirements(scenario_name: str, image_paths: List[Path]):
    content = [
        {
            "type": "input_text",
            "text": f"Scenario name: {scenario_name}\n\n{REQUIREMENTS_PROMPT}",
        }
    ]

    content.extend(build_image_content(image_paths))

    response = client.responses.create(
        model=MODEL_ID,
        input=[{"role": "user", "content": content}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    return response


def run_phase2_prototype(
    scenario_name: str,
    image_paths: List[Path],
    requirements_text: str,
):
    content = [
        {
            "type": "input_text",
            "text": (
                f"Scenario name: {scenario_name}\n\n"
                f"PHASE 1 REQUIREMENTS OUTPUT:\n{requirements_text}\n\n"
                f"{PROTOTYPE_PROMPT}"
            ),
        }
    ]

    content.extend(build_image_content(image_paths))

    response = client.responses.create(
        model=MODEL_ID,
        input=[{"role": "user", "content": content}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    return response


# =========================
# MAIN
# =========================

def main() -> None:
    log_path = BASE_DIR / "experiment_log.txt"
    manifest_path = BASE_DIR / "experiment_manifest.csv"

    write_experiment_log_header(log_path)

    if not check_openai_connection(log_path):
        print("Stopping experiment because OpenAI API/model connection failed.")
        return

    manifest_fields = [
        "timestamp",
        "scenario",
        "run",
        "phase",
        "model_label",
        "requested_model_id",
        "actual_model_id",
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "runs_per_scenario",
        "image_detail_quality",
        "timeout_seconds",
        "max_retries",
        "response_status_finish_reason",
        "output_txt",
        "output_html",
        "html_extracted",
        "flagged_incomplete_or_truncated",
        "truncation_or_completeness_issues",
        "status",
        "error",
    ]

    append_log(log_path, "Two-phase GPT-4o experiment started.")

    for scenario_name, scenario_path in SCENARIOS.items():
        try:
            image_paths = validate_scenario_folder(scenario_name, scenario_path)
        except Exception as e:
            append_log(log_path, f"ERROR validating {scenario_name}: {e}")
            print(f"ERROR validating {scenario_name}: {e}")
            continue

        append_log(
            log_path,
            f"Scenario {scenario_name}: found sketches: "
            + ", ".join(p.name for p in image_paths),
        )

        for run_number in range(1, RUNS_PER_SCENARIO + 1):
            print(f"\nRunning {scenario_name}, run {run_number}...")

            requirements_output_path = scenario_path / f"gpt4o_requirements_run{run_number}.txt"
            prototype_raw_output_path = scenario_path / f"gpt4o_prototype_raw_run{run_number}.txt"
            prototype_html_path = scenario_path / f"gpt4o_prototype_run{run_number}.html"

            # -------------------------
            # PHASE 1 — REQUIREMENTS
            # -------------------------

            phase = "Phase 1 - Requirements"
            status = "success"
            error_message = ""
            actual_model_id = "unknown"
            response_status = "unknown"
            flagged = "Unknown"
            issues = "Not evaluated"

            try:
                start = time.time()

                response1 = run_phase1_requirements(
                    scenario_name=scenario_name,
                    image_paths=image_paths,
                )

                actual_model_id = get_actual_model_id(response1)
                response_status = get_response_status(response1)
                requirements_text = response1.output_text or ""

                requirements_output_path.write_text(
                    requirements_text,
                    encoding="utf-8",
                )

                is_flagged, issue_summary = check_requirements_completeness(
                    requirements_text,
                )

                flagged = "Yes" if is_flagged else "No"
                issues = issue_summary

                elapsed = round(time.time() - start, 2)

                append_log(
                    log_path,
                    f"{scenario_name} run {run_number} Phase 1 completed in {elapsed}s. Flagged: {flagged}",
                )

            except Exception as e:
                status = "failed"
                error_message = str(e)
                requirements_text = ""

                append_log(
                    log_path,
                    f"ERROR {scenario_name} run {run_number} Phase 1: {e}",
                )

                print(f"ERROR {scenario_name} run {run_number} Phase 1: {e}")

            save_manifest_row(
                manifest_path=manifest_path,
                fieldnames=manifest_fields,
                row={
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "scenario": scenario_name,
                    "run": run_number,
                    "phase": phase,
                    "model_label": MODEL_LABEL,
                    "requested_model_id": MODEL_ID,
                    "actual_model_id": actual_model_id,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K_REPORTED,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "runs_per_scenario": RUNS_PER_SCENARIO,
                    "image_detail_quality": IMAGE_DETAIL,
                    "timeout_seconds": TIMEOUT_SECONDS,
                    "max_retries": MAX_RETRIES,
                    "response_status_finish_reason": response_status,
                    "output_txt": str(requirements_output_path),
                    "output_html": "",
                    "html_extracted": "N/A",
                    "flagged_incomplete_or_truncated": flagged,
                    "truncation_or_completeness_issues": issues,
                    "status": status,
                    "error": error_message,
                },
            )

            if not requirements_text:
                print("Skipping Phase 2 because Phase 1 failed or returned empty output.")
                continue

            # -------------------------
            # PHASE 2 — PROTOTYPE
            # -------------------------

            phase = "Phase 2 - Prototype"
            status = "success"
            error_message = ""
            actual_model_id = "unknown"
            response_status = "unknown"
            html_extracted = "No"
            flagged = "Unknown"
            issues = "Not evaluated"

            try:
                start = time.time()

                response2 = run_phase2_prototype(
                    scenario_name=scenario_name,
                    image_paths=image_paths,
                    requirements_text=requirements_text,
                )

                actual_model_id = get_actual_model_id(response2)
                response_status = get_response_status(response2)
                prototype_output_text = response2.output_text or ""

                prototype_raw_output_path.write_text(
                    prototype_output_text,
                    encoding="utf-8",
                )

                html_code = extract_html_from_output(prototype_output_text)

                if html_code:
                    html_extracted = "Yes"
                    prototype_html_path.write_text(
                        html_code,
                        encoding="utf-8",
                    )
                else:
                    status = "success_no_html_extracted"

                is_flagged, issue_summary = check_prototype_completeness(
                    prototype_output_text,
                    html_code,
                    response_status,
                )

                flagged = "Yes" if is_flagged else "No"
                issues = issue_summary

                elapsed = round(time.time() - start, 2)

                append_log(
                    log_path,
                    f"{scenario_name} run {run_number} Phase 2 completed in {elapsed}s. Flagged: {flagged}",
                )

                print(f"Completed {scenario_name}, run {run_number}. Prototype flagged: {flagged}")

            except Exception as e:
                status = "failed"
                error_message = str(e)

                append_log(
                    log_path,
                    f"ERROR {scenario_name} run {run_number} Phase 2: {e}",
                )

                print(f"ERROR {scenario_name} run {run_number} Phase 2: {e}")

            save_manifest_row(
                manifest_path=manifest_path,
                fieldnames=manifest_fields,
                row={
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "scenario": scenario_name,
                    "run": run_number,
                    "phase": phase,
                    "model_label": MODEL_LABEL,
                    "requested_model_id": MODEL_ID,
                    "actual_model_id": actual_model_id,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K_REPORTED,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "runs_per_scenario": RUNS_PER_SCENARIO,
                    "image_detail_quality": IMAGE_DETAIL,
                    "timeout_seconds": TIMEOUT_SECONDS,
                    "max_retries": MAX_RETRIES,
                    "response_status_finish_reason": response_status,
                    "output_txt": str(prototype_raw_output_path),
                    "output_html": str(prototype_html_path) if prototype_html_path.exists() else "",
                    "html_extracted": html_extracted,
                    "flagged_incomplete_or_truncated": flagged,
                    "truncation_or_completeness_issues": issues,
                    "status": status,
                    "error": error_message,
                },
            )

    append_log(log_path, "Two-phase GPT-4o experiment completed.")
    print("\nExperiment completed. Check experiment_log.txt and experiment_manifest.csv.")


if __name__ == "__main__":
    main()