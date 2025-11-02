import json
import os
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz
import gradio as gr
import re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def pdf_to_lines(pdf_bytes: bytes) -> List[str]:
    lines: List[str] = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        lines.append(line)
    return lines

ANCHORS: Dict[str, List[str]] = {
    "nome": ["nome", "nome do profissional", "profissional", "titular"],
    "inscricao": ["inscrição", "inscricao", "nº inscrição", "registro", "número"],
    "seccional": ["seccional", "uf", "seção", "secção", "estado"],
    "cpf": ["cpf", "documento", "cpf do titular"],
    "oab": ["oab", "carteira oab", "nº oab"],
}

REGEX_PATTERNS: Dict[str, re.Pattern[str]] = {
    "cpf": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "cep": re.compile(r"\b\d{5}-?\d{3}\b"),
    "inscricao": re.compile(r"\b\d{3,6}\b"),
    "oab": re.compile(r"\b\d{3,6}\b"),
}

UF_SET = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
}


def normalize_cpf(value: str) -> Optional[str]:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) != 11 or len(set(digits)) == 1:
        return None

    def calc_digit(v: str) -> int:
        factor = len(v) + 1
        total = sum(int(num) * (factor - idx) for idx, num in enumerate(v))
        remainder = (total * 10) % 11
        return remainder if remainder < 10 else 0

    first = calc_digit(digits[:9])
    second = calc_digit(digits[:9] + str(first))
    if digits[-2:] != f"{first}{second}":
        return None
    return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"


def normalize_cep(value: str) -> Optional[str]:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) != 8:
        return None
    return f"{digits[:5]}-{digits[5:]}"


def normalize_uf(value: str) -> Optional[str]:
    if not value:
        return None
    code = value.strip().upper()
    if code in UF_SET:
        return code
    return None


def normalize_value(field: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    value = str(value).strip()
    if not value:
        return None
    field_lower = field.lower()
    if field_lower == "cpf":
        return normalize_cpf(value)
    if field_lower == "cep":
        return normalize_cep(value)
    if field_lower in {"uf", "seccional"}:
        return normalize_uf(value)
    return value

def build_anchor_set(field: str, description: str) -> List[str]:
    base = set(ANCHORS.get(field.lower(), []))
    base.add(field.lower())
    if description:
        desc_tokens = [description.lower()]
        desc_tokens.extend(description.lower().split())
        base.update(token for token in desc_tokens if token)
    return [token for token in base if token]


def guess_field(lines: List[str], field: str, description: str) -> Tuple[Optional[str], float, Optional[int]]:
    field_lower = field.lower()
    regex = REGEX_PATTERNS.get(field_lower)
    if regex:
        for idx, line in enumerate(lines):
            match = regex.search(line)
            if match:
                value = match.group(0).strip()
                return value, 0.9, idx
    anchors = build_anchor_set(field, description)
    best_score = 0.0
    best_idx: Optional[int] = None
    for idx, raw_line in enumerate(lines):
        line = raw_line.lower()
        for anchor in anchors:
            score = fuzz.partial_ratio(anchor, line)
            if score > best_score:
                best_score = score
                best_idx = idx
    if best_idx is None or best_score <= 0:
        return None, 0.0, None
    candidate_value: Optional[str] = None
    target_line = lines[best_idx]
    if ":" in target_line:
        after = target_line.split(":", 1)[1].strip()
        if after:
            candidate_value = after
    if not candidate_value:
        next_idx = best_idx + 1
        if next_idx < len(lines):
            candidate_value = lines[next_idx].strip()
    confidence = best_score / 100
    if regex and candidate_value:
        regex_match = regex.search(candidate_value)
        if regex_match:
            candidate_value = regex_match.group(0).strip()
            confidence = max(confidence, 0.9)
    return candidate_value, confidence, best_idx


def heuristic_extract(
    lines: List[str],
    schema_map: Dict[str, str],
    conf_threshold: float,
) -> Tuple[Dict[str, Optional[str]], Dict[str, float], Dict[str, Optional[int]], List[str]]:
    partial: Dict[str, Optional[str]] = {}
    confidences: Dict[str, float] = {}
    positions: Dict[str, Optional[int]] = {}
    missing: List[str] = []
    for field, description in schema_map.items():
        value, confidence, idx = guess_field(lines, field, description or "")
        partial[field] = value
        confidences[field] = confidence
        positions[field] = idx
        if value is None or confidence < conf_threshold:
            missing.append(field)
    return partial, confidences, positions, missing


def build_excerpts(
    lines: List[str],
    positions: Dict[str, Optional[int]],
    targets: List[str],
    radius: int = 2,
) -> Dict[str, str]:
    excerpts: Dict[str, str] = {}
    default_excerpt = "\n".join(lines[: min(len(lines), 10)])
    for field in targets:
        idx = positions.get(field)
        if idx is None:
            excerpts[field] = default_excerpt
        else:
            start = max(0, idx - radius)
            end = min(len(lines), idx + radius + 1)
            excerpts[field] = "\n".join(lines[start:end])
    return excerpts

def llm_fill_missing_gpt5mini(
    schema_subset: Dict[str, str],
    excerpts: Dict[str, str],
) -> Dict[str, Optional[str]]:
    if not schema_subset or client is None:
        return {}
    system_message = (
        "Você é um assistente que extrai campos estruturados de documentos. "
        "Responda apenas em JSON válido, com exatamente as chaves solicitadas. "
        "Use apenas as informações fornecidas nos trechos. Se um dado estiver ausente, "
        "ambíguo ou inválido, retorne null."
    )
    schema_lines = [
        f"- {key}: {schema_subset[key]}" for key in schema_subset
    ]
    context_parts = []
    for key, excerpt in excerpts.items():
        context_parts.append(f"Campo: {key}\n{excerpt}")
    user_message = (
        "Campos solicitados:\n"
        + "\n".join(schema_lines)
        + "\n\nTrechos relevantes:\n"
        + "\n---\n".join(context_parts)
        + "\n\nForneça JSON estrito com exatamente essas chaves."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception:
        return {}
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        return {}
    cleaned: Dict[str, Optional[str]] = {}
    for key in schema_subset:
        cleaned[key] = data.get(key)
    return cleaned

def run_pipeline(
    pdf_file: Optional[gr.File],
    label: str,
    schema_text: str,
    conf_threshold: float,
    allow_llm: bool,
) -> Tuple[str, str]:
    start_time = time.perf_counter()
    if pdf_file is None:
        return json.dumps({}, ensure_ascii=False, indent=2), "Nenhum PDF fornecido."
    try:
        schema_map = json.loads(schema_text or "{}")
        if not isinstance(schema_map, dict):
            raise ValueError("Schema deve ser um objeto JSON.")
    except Exception as exc:
        return json.dumps({}, ensure_ascii=False, indent=2), f"Schema inválido: {exc}"

    try:
        pdf_bytes = pdf_file.read()
    except Exception as exc:
        return json.dumps({}, ensure_ascii=False, indent=2), f"Erro ao ler PDF: {exc}"

    lines = pdf_to_lines(pdf_bytes)
    if not lines:
        return json.dumps({}, ensure_ascii=False, indent=2), "PDF sem texto extraído."

    heuristics, confidences, positions, initial_missing = heuristic_extract(
        lines, schema_map, conf_threshold
    )

    normalized: Dict[str, Optional[str]] = {}
    llm_targets: List[str] = []
    for field in schema_map:
        value = heuristics.get(field)
        normalized_value = normalize_value(field, value)
        normalized[field] = normalized_value if normalized_value is not None else None
        if normalized[field] is None or confidences.get(field, 0.0) < conf_threshold:
            llm_targets.append(field)

    llm_results: Dict[str, Optional[str]] = {}
    llm_used = False
    if allow_llm and llm_targets:
        excerpts = build_excerpts(lines, positions, llm_targets)
        schema_subset = {field: schema_map[field] for field in llm_targets}
        llm_results = llm_fill_missing_gpt5mini(schema_subset, excerpts)
        llm_used = bool(llm_results)

    final_data: Dict[str, Optional[str]] = {}
    for field in schema_map:
        value = normalized.get(field)
        if field in llm_results:
            llm_value = normalize_value(field, llm_results[field])
            if llm_value is not None or value is None:
                value = llm_value
        final_data[field] = value

    final_missing = [key for key, value in final_data.items() if value is None]

    elapsed = time.perf_counter() - start_time
    log_lines = [
        f"Label: {label}",
        f"Campos no schema: {', '.join(schema_map.keys())}",
        f"Campos faltantes após heurística: {', '.join(initial_missing) if initial_missing else 'nenhum'}",
        f"Confidências heurísticas: "
        + ", ".join(
            f"{field}={confidences.get(field, 0.0):.2f}" for field in schema_map
        ),
        f"LLM habilitado: {'sim' if allow_llm else 'não'}",
        f"LLM acionado: {'sim' if llm_targets else 'não'}",
        f"LLM retornou dados: {'sim' if llm_used else 'não'}",
        f"Campos finais ausentes/invalidos: {', '.join(final_missing) if final_missing else 'nenhum'}",
        f"Tempo total: {elapsed:.2f}s",
    ]

    json_output = json.dumps(final_data, ensure_ascii=False, indent=2)
    log_text = "\n".join(log_lines)
    return json_output, log_text

def build_interface() -> gr.Blocks:
    default_schema = json.dumps(
        {
            "nome": "Nome do profissional.",
            "inscricao": "Número de inscrição/registro.",
            "seccional": "Seccional/UF (ex: PR, SP).",
            "cpf": "CPF do titular (se constar).",
        },
        ensure_ascii=False,
        indent=2,
    )

    with gr.Blocks(title="Extrator PDF → JSON orientado por schema") as demo:
        gr.Markdown("## Extrator PDF → JSON orientado por schema")
        with gr.Row():
            pdf_input = gr.File(label="PDF (1 página, texto embutido)", file_types=[".pdf"])
            label_input = gr.Textbox(label="Label", placeholder="Identificador livre")
        schema_input = gr.Code(
            label="Schema de extração (JSON)",
            value=default_schema,
            language="json",
            lines=12,
        )
        with gr.Row():
            threshold_input = gr.Slider(
                minimum=0.5,
                maximum=0.95,
                value=0.70,
                step=0.01,
                label="Limiar de confiança heurística",
            )
            llm_checkbox = gr.Checkbox(label="Usar fallback LLM", value=True)
        extract_button = gr.Button("Extrair")
        json_output = gr.JSON(label="Resultado JSON")
        log_output = gr.Textbox(label="Log", lines=12)

        def on_extract(pdf_file, label_value, schema_text, threshold, allow_llm):
            return run_pipeline(pdf_file, label_value or "", schema_text, float(threshold), bool(allow_llm))

        extract_button.click(
            fn=on_extract,
            inputs=[pdf_input, label_input, schema_input, threshold_input, llm_checkbox],
            outputs=[json_output, log_output],
        )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
