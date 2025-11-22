#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_subject_to_csv.py

Lee mmlu.json y uno o más CSVs de respuestas (por ejemplo: *answers*.csv)
y agrega una columna "subject" al CSV, mapeando cada user_prompt a su materia.
El mapeo se hace limpiando el prompt (removiendo opciones A., B., ... y el bloque
de instrucciones de CoT) y comparándolo con las "question" de mmlu.json.

- No requiere argumentos.
- Escribe un nuevo archivo por cada CSV de entrada con sufijo "_with_subject.csv".
"""

import json
import re
import sys
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


# ---------------------- Utilidades de normalización/limpieza ----------------------

COT_MARKER_PATTERN = re.compile(r"This is a Chain-of-Thought\s*\(CoT\)\s*task\.", re.IGNORECASE)

# Patrón de línea de opción estilo:
#   A. texto
#   B) texto
#   C : texto     (por si aparece alguna variación rara)
OPTION_LINE_RE = re.compile(r"^[A-Z]\s*[\.\):]\s", re.UNICODE)

def normalize_text(s: str) -> str:
    """Normaliza texto para comparación robusta (lower, NFKC, espacios)."""
    if s is None:
        return ""
    # Normaliza unicode (ej. comillas “ ” => ")
    s = unicodedata.normalize("NFKC", s)
    # Minúsculas
    s = s.lower()
    # Quita espacios en extremos de cada línea
    lines = [ln.strip() for ln in s.splitlines()]
    s = " ".join([ln for ln in lines if ln])  # colapsa saltos en un solo espacio
    # Colapsa espacios múltiples
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_cot_and_options(user_prompt: str) -> str:
    """
    Remueve del prompt:
      - todo desde el marcador del CoT en adelante (si existe)
      - líneas de opciones múltiples (A., B., C., ...), y todo lo que sigue si se desea
    Devuelve el 'stem' de la pregunta (incluyendo contexto, si aplica).
    """
    if not user_prompt:
        return ""

    # Corta en el bloque CoT (si existe)
    parts = COT_MARKER_PATTERN.split(user_prompt, maxsplit=1)
    pre_cot = parts[0]

    # Recorre líneas y toma solo las anteriores a la PRIMERA opción
    kept_lines = []
    saw_option = False
    for ln in pre_cot.splitlines():
        if OPTION_LINE_RE.match(ln.strip()):
            saw_option = True
            break  # dejamos de recolectar al encontrar la primera opción
        kept_lines.append(ln)

    stem = "\n".join(kept_lines).strip()
    return stem


# ---------------------- Carga de mmlu.json y armado del índice ----------------------

def load_mmlu_index(mmlu_path: Path):
    """
    Carga mmlu.json que contiene:
      {
        "question": [ ... ],
        "subject":  [ ... ]
      }
    Devuelve:
      - map_q2subj: dict { normalized_question: subject }
      - questions_norm: lista de preguntas normalizadas (en el mismo orden)
      - subjects: lista de subjects (en el mismo orden)
    """
    with mmlu_path.open("r", encoding="utf-8") as f:
        mmlu = json.load(f)

    questions = mmlu.get("question", [])
    subjects = mmlu.get("subject", [])
    if not isinstance(questions, list) or not isinstance(subjects, list):
        raise ValueError("mmlu.json no tiene listas 'question' y 'subject'.")
    if len(questions) != len(subjects):
        raise ValueError("mmlu.json: 'question' y 'subject' no tienen el mismo largo.")

    questions_norm = [normalize_text(q) for q in questions]
    map_q2subj = {q_norm: subjects[i] for i, q_norm in enumerate(questions_norm)}
    return map_q2subj, questions_norm, subjects


# ---------------------- Matching: exacto y con tolerancia ----------------------

def best_match_subject(clean_stem_norm: str, map_q2subj: dict, questions_norm: list, subjects: list):
    """
    Intenta encontrar el subject para 'clean_stem_norm' (pregunta ya normalizada).
    Estrategia:
      1) Match exacto
      2) Si no hay exacto:
         - match por contención (stem in qnorm o qnorm in stem)
         - si sigue sin éxito, fuzzy matching DiffLib con umbral alto
    Devuelve: subject (str) o None si no encuentra con suficiente confianza.
    """
    # 1) exacto
    subj = map_q2subj.get(clean_stem_norm)
    if subj is not None:
        return subj

    # 2) contención
    candidates = []
    for i, qn in enumerate(questions_norm):
        if clean_stem_norm and (clean_stem_norm in qn or qn in clean_stem_norm):
            candidates.append((i, 1.0))  # máxima similitud por contención

    if len(candidates) == 1:
        return subjects[candidates[0][0]]
    elif len(candidates) > 1:
        # Si hay varias por contención, elige la de mayor similitud difflib
        best_i, best_score = -1, -1.0
        for i, _ in candidates:
            score = SequenceMatcher(None, clean_stem_norm, questions_norm[i]).ratio()
            if score > best_score:
                best_i, best_score = i, score
        # exígimos umbral relativamente alto para evitar colisiones
        if best_score >= 0.92:
            return subjects[best_i]

    # 3) fuzzy general
    best_i, best_score = -1, -1.0
    for i, qn in enumerate(questions_norm):
        score = SequenceMatcher(None, clean_stem_norm, qn).ratio()
        if score > best_score:
            best_i, best_score = i, score

    if best_score >= 0.94:
        return subjects[best_i]

    return None


# ---------------------- Proceso principal ----------------------

def process_csv(csv_path: Path, map_q2subj, questions_norm, subjects):
    df = pd.read_csv(csv_path)

    if "user_prompt" not in df.columns:
        print(f"[AVISO] {csv_path.name} no contiene la columna 'user_prompt'. Se omite.")
        return None

    subjects_col = []
    not_found = 0

    for idx, prompt in enumerate(df["user_prompt"].astype(str).tolist()):
        stem = strip_cot_and_options(prompt)
        stem_norm = normalize_text(stem)
        subj = best_match_subject(stem_norm, map_q2subj, questions_norm, subjects)
        if subj is None:
            not_found += 1
        subjects_col.append(subj)

    df["subject"] = subjects_col

    out_path = csv_path.with_name(csv_path.stem + "_with_subject.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Escribí: {out_path.name}  (no encontrados: {not_found} de {len(df)})")
    return out_path


def main():
    root = Path(".").resolve()

    # 1) mmlu.json obligatorio en datasets/
    mmlu_path = root / "datasets" / "mmlu.json"
    if not mmlu_path.exists():
        print("ERROR: No encuentro 'datasets/mmlu.json'.")
        sys.exit(1)

    map_q2subj, questions_norm, subjects = load_mmlu_index(mmlu_path)

    # 2) Buscar CSVs candidatos en results/ (por defecto: *answers*.csv)
    csvs = sorted((root / "results").glob("*answers*.csv"))
    if not csvs:
        # fallback a cualquier CSV
        csvs = sorted((root / "results").glob("*.csv"))

    if not csvs:
        print("ERROR: No encontré CSVs para procesar en el directorio actual.")
        sys.exit(1)

    for csv in csvs:
        try:
            process_csv(csv, map_q2subj, questions_norm, subjects)
        except Exception as e:
            print(f"[ERROR] Procesando {csv.name}: {e}")

if __name__ == "__main__":
    main()
