"""
Automated Gemini 2.5 Pro experiment runner for LLM-assisted sketch-based
requirements elicitation and prototype generation.

Folder structure expected:

Gemini_2.5_Pro/
│
├── Gemini_2.5_Pro.py
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

import csv
import mimetypes
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from google import genai
from google.genai import types

# =========================
# CONFIGURATION
# =========================

MODEL_LABEL = "Google Gemini 2.5 Pro"
MODEL_ID = "gemini-2.5-pro"

TEMPERATURE = 0.2
TOP_P = 1.0
TOP_K = "not set / provider default"
TOP_K_REPORTED = TOP_K

MAX_OUTPUT_TOKENS = 65535
RUNS_PER_SCENARIO = 3

IMAGE_DETAIL = "native/original"
TIMEOUT_SECONDS = 180
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

# =========================
# API CONFIGURATION
# =========================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable.")

client = genai.Client(api_key="")

gen_config = types.GenerateContentConfig(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_output_tokens=MAX_OUTPUT_TOKENS,
)

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

Phase 1: Requirements, traceability
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
# GEMINI RETRY HELPER
# =========================

def call_gemini_with_retries(model: str, contents: list, config):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_error = e
            print(f"Gemini attempt {attempt} failed: {e}")

            if attempt <= MAX_RETRIES:
                time.sleep(10)

    raise last_error


# =========================
# CONNECTION CHECK
# =========================
def check_gemini_connection(log_path: Path) -> bool:
    try:
        response = call_gemini_with_retries(
            model=MODEL_ID,
            contents=["Respond with plain text only. Say exactly: connection successful"],
            config=types.GenerateContentConfig(
                temperature=0,
                top_p=TOP_P,
                max_output_tokens=50,
            ),
        )

        text = safe_get_response_text(response).strip()

        if not text:
            text = "[No text returned, but API call completed]"

        print("Gemini connection successful.")
        print("Response:", text)

        append_log(log_path, "Gemini connection successful.")
        append_log(log_path, f"Connection response: {text}")

        return True

    except Exception as e:
        print(f"Gemini connection failed: {e}")
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR DETAILS:", e)

        append_log(log_path, "Gemini connection failed.")
        append_log(log_path, f"ERROR TYPE: {type(e).__name__}")
        append_log(log_path, f"ERROR DETAILS: {e}")

        return False

# =========================
# HELPERS
# =========================

def get_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)

    if mime_type in ["image/png", "image/jpeg", "image/webp"]:
        return mime_type

    return "image/png"


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


def build_gemini_contents(prompt_text: str, image_paths: List[Path]) -> list:
    contents = [prompt_text]

    for image_path in image_paths:
        with Image.open(image_path) as img:
            contents.append(img.copy())

    return contents


def safe_get_response_text(response) -> str:
    try:
        return response.text or ""
    except Exception:
        pass

    parts = []

    try:
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    parts.append(part.text)
    except Exception:
        pass

    return "\n".join(parts).strip()


def get_actual_model_id() -> str:
    return MODEL_ID


def get_response_status(response) -> str:
    parts = []

    try:
        if response.candidates:
            candidate = response.candidates[0]

            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                parts.append(f"finish_reason={finish_reason}")

            safety_ratings = getattr(candidate, "safety_ratings", None)
            if safety_ratings:
                parts.append("safety_ratings_present=True")
    except Exception:
        pass

    return "; ".join(parts) if parts else "unknown"


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

    if not html_code or not html_code.strip():
        issues.append("No HTML extracted")
    else:
        lower_html = html_code.lower()

        # Structural tag checks
        if "<html" not in lower_html:
            issues.append("Missing opening <html> tag")
        if "</html>" not in lower_html:
            issues.append("Missing closing </html> tag")
        if "<body" not in lower_html:
            issues.append("Missing opening <body> tag")
        if "</body>" not in lower_html:
            issues.append("Missing closing </body> tag")

        # Component checks for the self-contained requirement
        if "<style" not in lower_html:
            issues.append("Missing embedded CSS/style block")
        if "<script" not in lower_html:
            issues.append("Missing embedded JavaScript/script block")

    # Check for markdown formatting errors
    if output_text.count("```") % 2 != 0:
        issues.append("Unclosed markdown code block")

    # Metadata Check:
    # We remove "stop" because finish_reason="STOP" means success.
    # We only flag "max" or "length" which indicate the model was cut off.
    lower_status = response_status.lower()
    if "max" in lower_status or "length" in lower_status:
        issues.append(f"Response metadata confirms truncation: {response_status}")

    flagged = len(issues) > 0
    return flagged, " | ".join(issues) if flagged else "None"


# =========================
# API CALLS
# =========================

def run_phase1_requirements(scenario_name: str, image_paths: List[Path]):
    prompt_text = f"Scenario name: {scenario_name}\n\n{REQUIREMENTS_PROMPT}"

    response = call_gemini_with_retries(
        model=MODEL_ID,
        contents=build_gemini_contents(prompt_text, image_paths),
        config=gen_config,
    )

    return response


def run_phase2_prototype(
    scenario_name: str,
    image_paths: List[Path],
    requirements_text: str,
):
    prompt_text = (
        f"Scenario name: {scenario_name}\n\n"
        f"PHASE 1 REQUIREMENTS OUTPUT:\n{requirements_text}\n\n"
        f"{PROTOTYPE_PROMPT}"
    )

    response = call_gemini_with_retries(
        model=MODEL_ID,
        contents=build_gemini_contents(prompt_text, image_paths),
        config=gen_config,
    )

    return response


# =========================
# MAIN
# =========================

def main() -> None:
    log_path = BASE_DIR / "experiment_log.txt"
    manifest_path = BASE_DIR / "experiment_manifest.csv"

    write_experiment_log_header(log_path)

    if not check_gemini_connection(log_path):
        print("Stopping experiment because Gemini API/model connection failed.")
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

    append_log(log_path, "Two-phase Gemini experiment started.")

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

            requirements_output_path = scenario_path / f"gemini_requirements_run{run_number}.txt"
            prototype_raw_output_path = scenario_path / f"gemini_prototype_raw_run{run_number}.txt"
            prototype_html_path = scenario_path / f"gemini_prototype_run{run_number}.html"

            phase = "Phase 1 - Requirements"
            status = "success"
            error_message = ""
            actual_model_id = get_actual_model_id()
            response_status = "unknown"
            flagged = "Unknown"
            issues = "Not evaluated"
            requirements_text = ""

            try:
                start = time.time()

                response1 = run_phase1_requirements(
                    scenario_name=scenario_name,
                    image_paths=image_paths,
                )

                response_status = get_response_status(response1)
                requirements_text = safe_get_response_text(response1)

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

            phase = "Phase 2 - Prototype"
            status = "success"
            error_message = ""
            actual_model_id = get_actual_model_id()
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

                response_status = get_response_status(response2)
                prototype_output_text = safe_get_response_text(response2)

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
                print(f"Issues: {issues}")

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

    append_log(log_path, "Two-phase Gemini experiment completed.")
    print("\nExperiment completed. Check experiment_log.txt and experiment_manifest.csv.")


if __name__ == "__main__":
    main()