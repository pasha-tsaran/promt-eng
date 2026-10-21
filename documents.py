"""Локальное чтение текста и таблиц DOCX, без загрузки документа в сервисы."""

import hashlib
import io
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 120_000
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"


def _visible_children(node: ET.Element) -> list[ET.Element]:
    if node.tag in {W + "del", W + "moveFrom"}:
        return []
    if node.tag == MC + "AlternateContent":
        branch = node.find(MC + "Choice")
        if branch is None:
            branch = node.find(MC + "Fallback")
        return list(branch) if branch is not None else []
    return list(node)


def _walk(node: ET.Element):
    yield node
    for child in _visible_children(node):
        yield from _walk(child)


def _paragraph_text(paragraph: ET.Element) -> str:
    def tokens(node: ET.Element):
        if node is not paragraph and node.tag == W + "p":
            return
        if node.tag == W + "t":
            yield node.text or ""
        elif node.tag == W + "tab":
            yield "\t"
        elif node.tag in {W + "br", W + "cr"}:
            yield "\n"
        for child in _visible_children(node):
            yield from tokens(child)
    return "".join(tokens(paragraph)).strip()


def extract_docx(data: bytes) -> list[str]:
    try:
        with ZipFile(io.BytesIO(data)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > MAX_XML_BYTES:
                raise ValueError("XML документа больше 8 МБ. Разделите договор на части.")
            xml = archive.read(info)
        if b"<!DOCTYPE" in xml.upper():
            raise ValueError("DOCX содержит неподдерживаемое объявление DTD.")
        body = ET.fromstring(xml).find(W + "body")
    except (BadZipFile, KeyError, ET.ParseError, RuntimeError) as exc:
        raise ValueError("Не удалось прочитать DOCX. Пересохраните файл в Word как .docx.") from exc
    if body is None:
        raise ValueError("В DOCX нет основного текста.")
    return [text for node in _walk(body) if node.tag == W + "p" and (text := _paragraph_text(node))]


def read_contract(path: Path) -> tuple[str, dict[str, Any]]:
    if path.suffix.lower() not in {".docx", ".txt", ".md"}:
        raise ValueError("Нужен .docx, .txt или .md. Старый .doc сохраните в Word как .docx.")
    if not path.is_file():
        raise ValueError("Файл не найден. Проверьте полный путь к договору.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Файл больше 20 МБ. Сохраните текст договора отдельно.")
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Файл больше 20 МБ.")
    if path.suffix.lower() == ".docx":
        paragraphs = extract_docx(data)
    else:
        try:
            paragraphs = [line.strip() for line in data.decode("utf-8-sig").splitlines() if line.strip()]
        except UnicodeDecodeError:
            raise ValueError("Сохраните текстовый файл в UTF-8.") from None
    if not paragraphs:
        raise ValueError("Текст не найден. Сканы и изображения внутри DOCX не распознаются.")
    text = "\n".join(f"[Абзац {number}] {value}" for number, value in enumerate(paragraphs, 1))
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("Текст больше 120 000 символов. Разделите документ; текст не обрезан.")
    return text, {"file_name": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                  "paragraphs": len(paragraphs), "characters": len(text)}
