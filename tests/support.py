"""Временная рабочая папка для тестов в пределах проекта."""

from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from uuid import uuid4


@contextmanager
def workspace():
    parent = Path(__file__).resolve().parent
    folder = parent / f".tmp-{uuid4().hex}"
    folder.mkdir()
    try:
        yield folder
    finally:
        if not folder.resolve().is_relative_to(parent) or not folder.name.startswith(".tmp-"):
            raise RuntimeError("Небезопасный путь временной папки.")
        rmtree(folder)
