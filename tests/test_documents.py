import io
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from support import workspace

import documents


def docx(body):
    buffer = io.BytesIO()
    xml = (f'<w:document xmlns:w="{documents.W[1:-1]}" xmlns:mc="{documents.MC[1:-1]}">'
           f'<w:body>{body}</w:body></w:document>')
    with ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


class DocumentTests(unittest.TestCase):
    def test_runs_tables_and_breaks(self):
        data = docx('<w:p><w:r><w:t>Договор </w:t><w:t>услуг</w:t></w:r></w:p>'
                    '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Цена</w:t><w:tab/><w:t>100</w:t>'
                    '<w:br/><w:t>рублей</w:t></w:r></w:p></w:tc></w:tr></w:tbl>')
        self.assertEqual(documents.extract_docx(data), ["Договор услуг", "Цена\t100\nрублей"])

    def test_nested_text_and_alternate_content_are_not_duplicated(self):
        data = docx('<w:p><w:r><w:t>Основной текст</w:t><w:txbxContent><w:p><w:r>'
                    '<w:t>Текст в блоке</w:t></w:r></w:p></w:txbxContent></w:r></w:p>'
                    '<mc:AlternateContent><mc:Choice><w:p><w:r><w:t>Условие</w:t></w:r></w:p></mc:Choice>'
                    '<mc:Fallback><w:p><w:r><w:t>Копия условия</w:t></w:r></w:p></mc:Fallback></mc:AlternateContent>')
        self.assertEqual(documents.extract_docx(data), ["Основной текст", "Текст в блоке", "Условие"])

    def test_deleted_text_is_not_analyzed_as_current_condition(self):
        data = docx('<w:p><w:del><w:r><w:t>Удаленное условие</w:t></w:r></w:del>'
                    '<w:ins><w:r><w:t>Новое условие</w:t></w:r></w:ins></w:p>')
        self.assertEqual(documents.extract_docx(data), ["Новое условие"])

    def test_corrupt_docx_and_dtd_are_rejected(self):
        with self.assertRaises(ValueError):
            documents.extract_docx(b"not DOCX")
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", '<!DOCTYPE x [<!ENTITY y "text">]><x/>')
        with self.assertRaises(ValueError):
            documents.extract_docx(buffer.getvalue())

    def test_real_file_labels_metadata_and_limits(self):
        with workspace() as folder:
            path = folder / "contract.docx"
            path.write_bytes(docx('<w:p><w:r><w:t>Условие</w:t></w:r></w:p>'))
            text, source = documents.read_contract(path)
            self.assertEqual(text, "[Абзац 1] Условие")
            self.assertEqual(source["paragraphs"], 1)
            self.assertEqual(len(source["sha256"]), 64)
            self.assertNotIn("text", source)
            with patch("documents.MAX_TEXT_CHARS", 3), self.assertRaises(ValueError):
                documents.read_contract(path)
            with patch("documents.MAX_FILE_BYTES", 1), self.assertRaises(ValueError):
                documents.read_contract(path)
            with self.assertRaises(ValueError):
                documents.read_contract(folder / "contract.doc")
            with self.assertRaises(ValueError):
                documents.read_contract(folder / "missing.docx")
            path.write_bytes(docx(""))
            with self.assertRaises(ValueError):
                documents.read_contract(path)

    def test_utf8_text_and_encoding_error(self):
        with workspace() as folder:
            path = folder / "contract.txt"
            path.write_text("Первый абзац\n\nВторой", encoding="utf-8-sig")
            self.assertEqual(documents.read_contract(path)[1]["paragraphs"], 2)
            path.write_bytes(b"\xff\xfe\x80")
            with self.assertRaises(ValueError):
                documents.read_contract(path)


if __name__ == "__main__":
    unittest.main()
