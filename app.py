import argparse
import copy
import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


CACHE: Dict[str, Dict[str, Any]] = {}


DEFAULT_SCHEMA: Dict[str, str] = {
	"nome": "Nome do profissional, normalmente no canto superior esquerdo da imagem",
	"inscricao": "Número de inscrição do profissional",
	"seccional": "Seccional do profissional",
	"subsecao": "Subseção à qual o profissional faz parte",
	"categoria": "Categoria, pode ser ADVOGADO, ADVOGADA, SUPLEMENTAR, ESTAGIARIO, ESTAGIARIA",
	"endereco_profissional": "Endereço profissional completo",
	"situacao": "Situação do profissional, normalmente no canto inferior direito.",
}

PROMPT_TEMPLATE = (
	"Extrai os dados desse arquivo no seguinte formato: Caso não encontrar o campo, "
	"adicione null. Retorne somente o arquivo json {schema_json}\n\n"
	"Texto do documento:\n{document_text}"
)


def pdf_to_text(pdf_bytes: bytes) -> Tuple[str, Optional[str]]:
	lines: List[str] = []
	try:
		with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
			for page in pdf.pages:
				text = page.extract_text() or ""
				for raw_line in text.splitlines():
					line = raw_line.strip()
					if line:
						lines.append(line)
	except Exception as exc:  # noqa: BLE001
		return "", f"Erro ao processar PDF: {exc}"
	if not lines:
		return "", "Nenhum texto extraído do PDF."
	return "\n".join(lines), None


def read_pdf_bytes(pdf_input: Any) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
	potential_paths: List[str] = []
	display_name: Optional[str] = None

	if isinstance(pdf_input, dict):
		for key in ("path", "name"):
			value = pdf_input.get(key)
			if isinstance(value, str):
				potential_paths.append(value)
		display_name = pdf_input.get("name") if isinstance(pdf_input.get("name"), str) else None

	if hasattr(pdf_input, "name") and isinstance(pdf_input.name, str):
		potential_paths.append(pdf_input.name)
		display_name = pdf_input.name

	if isinstance(pdf_input, str):
		potential_paths.append(pdf_input)
		display_name = pdf_input

	for path in potential_paths:
		if os.path.exists(path):
			try:
				with open(path, "rb") as handle:
					return handle.read(), display_name or os.path.basename(path), None
			except Exception as exc:  # noqa: BLE001
				return None, display_name or path, f"Erro ao ler PDF: {exc}"

	if hasattr(pdf_input, "read"):
		try:
			data = pdf_input.read()
			if isinstance(data, bytes):
				return data, display_name or "arquivo.pdf", None
		except Exception as exc:  # noqa: BLE001
			return None, display_name or "arquivo.pdf", f"Erro ao ler PDF: {exc}"

	return None, display_name or "arquivo.pdf", "Não foi possível acessar os bytes do PDF enviado."


def compute_cache_key(schema_signature: str, pdf_bytes: bytes) -> str:
	schema_hash = hashlib.sha256(schema_signature.encode("utf-8")).hexdigest()
	pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
	return f"{schema_hash}|{pdf_hash}"


def parse_schema(schema_text: str) -> Tuple[Optional[Dict[str, str]], Optional[str], Optional[str]]:
	trimmed = schema_text.strip() if schema_text else ""
	if not trimmed:
		schema_map = DEFAULT_SCHEMA.copy()
		schema_json = json.dumps(schema_map, ensure_ascii=False, indent=2, sort_keys=True)
		return schema_map, schema_json, None

	try:
		raw = json.loads(trimmed)
	except Exception as exc:  # noqa: BLE001
		return None, None, f"Schema inválido: {exc}"
	if not isinstance(raw, dict):
		return None, None, "Schema deve ser um objeto JSON."
	schema_map: Dict[str, str] = {}
	for key, value in raw.items():
		if not isinstance(key, str):
			return None, None, "Todas as chaves do schema devem ser strings."
		schema_map[key] = value if isinstance(value, str) else str(value)
	schema_json = json.dumps(schema_map, ensure_ascii=False, indent=2, sort_keys=True)
	return schema_map, schema_json, None


def call_llm(
	document_text: str,
	schema_map: Dict[str, str],
	schema_json: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
	if client is None:
		return None, "OPENAI_API_KEY não configurada no ambiente."
	prompt = PROMPT_TEMPLATE.format(schema_json=schema_json, document_text=document_text)
	try:
		response = client.chat.completions.create(
			model="gpt-5-mini",
			messages=[
				{
					"role": "system",
					"content": "Você é um assistente que extrai dados estruturados e responde apenas em JSON válido.",
				},
				{"role": "user", "content": prompt},
			],
			response_format={"type": "json_object"},
		)
	except Exception as exc:  # noqa: BLE001
		return None, f"Falha na chamada ao modelo: {exc}"

	content = response.choices[0].message.content
	try:
		data = json.loads(content)
	except Exception as exc:  # noqa: BLE001
		return None, f"Resposta do modelo não pôde ser convertida em JSON: {exc}"
	cleaned: Dict[str, Any] = {}
	for key in schema_map:
		cleaned[key] = data.get(key)
	return cleaned, None


def build_relevant_context(
	document_text: str,
	schema_map: Dict[str, str],
	max_chars: int = 4000,
	max_lines: int = 200,
) -> Tuple[str, bool]:
	if len(document_text) <= max_chars:
		return document_text, False

	lines = [line.strip() for line in document_text.splitlines() if line.strip()]
	if not lines:
		return document_text[:max_chars], True

	keywords: List[str] = []
	for key, description in schema_map.items():
		keywords.append(key.lower())
		if isinstance(description, str):
			keywords.extend(token for token in description.lower().split() if len(token) > 3)

	relevant: List[str] = []
	seen: set[str] = set()
	for line in lines:
		lower = line.lower()
		if any(keyword in lower for keyword in keywords):
			if line not in seen:
				relevant.append(line)
				seen.add(line)
		if len(relevant) >= max_lines:
			break

	if not relevant:
		relevant = lines[:max_lines]

	snippet = "\n".join(relevant)
	if len(snippet) > max_chars:
		snippet = snippet[:max_chars]
	return snippet, True


def process_pdf(
	pdf_input: Any,
	schema_map: Dict[str, str],
	schema_json: str,
) -> Tuple[str, Optional[Dict[str, Any]], str]:
	pdf_bytes, display_name, error = read_pdf_bytes(pdf_input)
	if pdf_bytes is None:
		return display_name or "arquivo.pdf", None, error or "Erro desconhecido ao ler PDF."

	cache_key = compute_cache_key(schema_json, pdf_bytes)
	if cache_key in CACHE:
		cached_data = CACHE[cache_key].get("data")
		return display_name or "arquivo.pdf", copy.deepcopy(cached_data), "cache"

	document_text, text_error = pdf_to_text(pdf_bytes)
	if text_error is not None:
		return display_name or "arquivo.pdf", None, text_error

	reduced_text, truncated = build_relevant_context(document_text, schema_map)
	llm_result, llm_error = call_llm(reduced_text, schema_map, schema_json)
	if llm_error is not None:
		return display_name or "arquivo.pdf", None, llm_error

	CACHE[cache_key] = {"data": copy.deepcopy(llm_result)}
	status = "ok" if not truncated else "ok (contexto reduzido)"
	return display_name or "arquivo.pdf", llm_result, status


def process_batch(
	pdf_inputs: Optional[List[Any]],
	schema_map: Dict[str, str],
	schema_json: str,
	label: str,
) -> Tuple[List[Dict[str, Any]], str]:
	if not pdf_inputs:
		return [], "Nenhum PDF fornecido."

	results: List[Dict[str, Any]] = []
	logs: List[str] = []

	for pdf_input in pdf_inputs:
		name, data, status = process_pdf(pdf_input, schema_map, schema_json)
		record = {
			"label": label or "",
			"arquivo": str(name),
			"dados": data,
			"status": status,
		}
		results.append(record)
		if label:
			logs.append(f"{label}:{name}: {status}")
		else:
			logs.append(f"{name}: {status}")

	filtered_results = [entry.get("dados") for entry in results]

	return filtered_results, "\n".join(logs)


def run_cli(pdf_dir: str, label: str, schema_text: Optional[str]) -> int:
	schema_map, schema_json, schema_error = parse_schema(schema_text or "")
	if schema_error is not None:
		print(f"Erro no schema: {schema_error}", file=sys.stderr)
		return 1

	directory = Path(pdf_dir)
	if not directory.exists() or not directory.is_dir():
		print(f"Diretório inválido: {pdf_dir}", file=sys.stderr)
		return 1

	pdf_files = sorted(path for path in directory.glob("*.pdf") if path.is_file())
	if not pdf_files:
		print(f"Nenhum arquivo .pdf encontrado em {pdf_dir}", file=sys.stderr)
		return 1

	outputs: List[Dict[str, Any]] = []
	logs: List[str] = []

	for pdf_path in pdf_files:
		name, data, status = process_pdf(str(pdf_path), schema_map, schema_json)
		outputs.append(
			{
				"label": label,
				"arquivo": str(pdf_path),
				"dados": data,
				"status": status,
			}
		)
		if label:
			logs.append(f"{label}:{pdf_path.name}: {status}")
		else:
			logs.append(f"{pdf_path.name}: {status}")

	print(json.dumps(outputs, ensure_ascii=False, indent=2))
	if logs:
		print("\n".join(logs), file=sys.stderr)

	return 0


def build_interface() -> gr.Blocks:
	schema_default = json.dumps(DEFAULT_SCHEMA, ensure_ascii=False, indent=2, sort_keys=True)
	with gr.Blocks(title="Extrator PDF → JSON") as demo:
		gr.Markdown("## Extrator PDF → JSON")
		gr.Markdown(
			"Envie um ou mais PDFs, informe o label identificador e defina o schema que deve ser utilizado na extração."
		)

		pdf_input = gr.File(
			label="PDFs",
			file_types=[".pdf"],
			file_count="multiple",
		)

		with gr.Row():
			label_input = gr.Textbox(label="Label", placeholder="carteira_oab")
			schema_input = gr.Code(
				label="Schema (JSON)",
				value=schema_default,
				language="json",
				lines=12,
			)

		extract_button = gr.Button("Extrair")
		json_output = gr.JSON(label="Resultados (por arquivo)")
		log_output = gr.Textbox(label="Log", lines=10)

		def on_extract(files: List[Any], label_value: str, schema_text: str):
			if not label_value or not label_value.strip():
				return [], "Label é obrigatório."
			schema_map, schema_json, schema_error = parse_schema(schema_text)
			if schema_error is not None:
				return [], schema_error
			return process_batch(files, schema_map, schema_json, label_value.strip())

		extract_button.click(
			fn=on_extract,
			inputs=[pdf_input, label_input, schema_input],
			outputs=[json_output, log_output],
		)

	return demo


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Extrator PDF → JSON orientado por schema")
	parser.add_argument(
		"--cli",
		action="store_true",
		help="Executa em modo linha de comando processando todos os PDFs de uma pasta.",
	)
	parser.add_argument(
		"--pdf-dir",
		type=str,
		help="Caminho da pasta que contém arquivos PDF (obrigatório no modo CLI).",
	)
	parser.add_argument(
		"--label",
		type=str,
		help="Label identificador aplicado aos resultados (obrigatório no modo CLI).",
	)
	parser.add_argument(
		"--schema-file",
		type=str,
		help="Arquivo JSON com o schema a ser utilizado (opcional).",
	)
	parser.add_argument(
		"--schema",
		type=str,
		help="Schema em JSON fornecido diretamente na linha de comando (sobrepõe --schema-file).",
	)

	args = parser.parse_args(argv)

	if args.cli:
		if not args.pdf_dir:
			parser.error("--pdf-dir é obrigatório quando --cli é utilizado.")
		if not args.label:
			parser.error("--label é obrigatório quando --cli é utilizado.")

		schema_text: Optional[str] = None
		if args.schema_file:
			try:
				schema_text = Path(args.schema_file).read_text(encoding="utf-8")
			except Exception as exc:  # noqa: BLE001
				print(f"Erro ao ler schema: {exc}", file=sys.stderr)
				return 1
		if args.schema is not None:
			schema_text = args.schema

		return run_cli(args.pdf_dir, args.label, schema_text)

	interface = build_interface()
	interface.launch()
	return 0


if __name__ == "__main__":
	sys.exit(main())
