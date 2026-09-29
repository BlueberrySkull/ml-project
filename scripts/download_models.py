"""Скачивает предобученные модели проекта в папку models/.

Модели в git не хранятся (см. .gitignore) - их нужно скачивать локально
и на этапе сборки Docker-образа этим скриптом.

Использование:
    python scripts/download_models.py              # скачать всё
    python scripts/download_models.py --llm-only
    python scripts/download_models.py --embedder-only
    python scripts/download_models.py --force       # перекачать, даже если уже есть
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

# На Windows консоль иногда использует cp1251/cp866 и не может напечатать
# кириллицу из наших сообщений - переключаем stdout/stderr на UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# LLM для генерации ответов: Meno-Tiny-0.1 (Qwen2.5-1.5B, дообучена под русский RAG/QA).
# В этом репозитории на HF есть только fp16-версия (~2.9 ГБ), квантованных файлов
# автор не выкладывал - при необходимости меньшего размера квантуем сами через
# llama.cpp (llama-quantize) на следующем этапе.
LLM_REPO = "bond005/Meno-Tiny-0.1-GGUF"
LLM_FILENAME = "meno-tiny-0.1-fp16.gguf"
LLM_MIN_SIZE_BYTES = 2_900_000_000  # ожидаемый размер файла, для проверки целостности

# Эмбеддер для векторного поиска по базе инструкций.
# rubert-entity-embedder (был скачан вручную ранее) для этого не подходит:
# это Siamese-модель для эмбеддингов сущностей (NER), а не эмбеддингов предложений/абзацев.
EMBEDDER_REPO = "cointegrated/rubert-tiny2"


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} ТБ"


def download_llm(force: bool) -> Path:
    print(f"[LLM] Скачиваю {LLM_FILENAME} из {LLM_REPO} ...")
    dest_dir = MODELS_DIR / "llm"
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=LLM_REPO,
            filename=LLM_FILENAME,
            local_dir=dest_dir,
            force_download=force,
        )
    )
    size = path.stat().st_size
    print(f"[LLM] Готово: {path} ({human_size(size)})")
    if size < LLM_MIN_SIZE_BYTES:
        raise RuntimeError(
            f"[LLM] Файл меньше ожидаемого ({human_size(size)} < "
            f"{human_size(LLM_MIN_SIZE_BYTES)}) - похоже, скачивание оборвалось. "
            "Запустите скрипт ещё раз с --force."
        )
    return path


def download_embedder(force: bool) -> Path:
    print(f"[Embedder] Скачиваю {EMBEDDER_REPO} ...")
    dest_dir = MODELS_DIR / "embedder"
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        snapshot_download(
            repo_id=EMBEDDER_REPO,
            local_dir=dest_dir,
            force_download=force,
        )
    )
    files = sorted(p for p in path.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"[Embedder] Готово: {path} ({len(files)} файлов, {human_size(total)})")
    if not any(p.name in ("pytorch_model.bin", "model.safetensors") for p in files):
        raise RuntimeError(
            "[Embedder] Не нашёл файл весов (model.safetensors / pytorch_model.bin) "
            "среди скачанного - похоже, скачивание оборвалось. Запустите с --force."
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--llm-only", action="store_true", help="скачать только LLM")
    group.add_argument(
        "--embedder-only", action="store_true", help="скачать только эмбеддер"
    )
    parser.add_argument(
        "--force", action="store_true", help="перекачать, даже если файлы уже есть"
    )
    args = parser.parse_args()

    try:
        if not args.embedder_only:
            download_llm(args.force)
        if not args.llm_only:
            download_embedder(args.force)
    except Exception as exc:  # noqa: BLE001 - выводим понятную ошибку и код возврата
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1

    print("Все модели скачаны в", MODELS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
