"""Tests for .docx generation from legal documents."""

import os
import tempfile

import pytest

from legalbot.agent.tools.document import _text_to_docx


class TestTextToDocx:
    def test_basic_docx_generation(self):
        text = "民事起诉状\n\n一、当事人信息\n原告：张三\n被告：李四\n\n二、诉讼请求\n1. 判令被告返还借款10万元\n\n三、事实与理由\n2024年1月，被告向原告借款10万元"
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            path = f.name
        try:
            result = _text_to_docx(text, path)
            assert result == path
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_docx_with_empty_lines(self):
        text = "标题\n\n\n段落1\n\n\n段落2"
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            path = f.name
        try:
            result = _text_to_docx(text, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_docx_nested_directory(self):
        text = "测试文书"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "test.docx")
            result = _text_to_docx(text, path)
            assert result == path
            assert os.path.exists(path)
