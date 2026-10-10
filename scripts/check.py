"""Check source syntax and local documentation links without network access."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
}
_FENCED_BLOCK_RE = re.compile(r"(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\n]*$")
_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(([^\n]+?)\)")
_REQUIRED_README_SECTIONS = [
    "## 프로젝트 소개",
    "## 핵심 특징",
    "## 아키텍처",
]


def _check_markdown_links(path: Path) -> list[str]:
    errors = []
    text = _FENCED_BLOCK_RE.sub("", path.read_text(encoding="utf-8"))
    for target in _MARKDOWN_LINK_RE.findall(text):
        target = target.strip()
        if target.startswith("<"):
            target = target[1 : target.index(">")]
        else:
            target = target.split(' "', 1)[0]
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        destination = (
            ROOT / unquote(parsed.path.lstrip("/"))
            if parsed.path.startswith("/")
            else path.parent / unquote(parsed.path)
        )
        if destination.is_file() and destination.suffix == ".md" and parsed.fragment:
            anchors = _markdown_anchors(destination.read_text(encoding="utf-8"))
            if unquote(parsed.fragment) not in anchors:
                errors.append(f"{path.relative_to(ROOT)}: 없는 문서 앵커 {target}")
        if not destination.exists():
            errors.append(f"{path.relative_to(ROOT)}: 없는 링크 대상 {target}")
    return errors


def _markdown_anchors(text: str) -> set[str]:
    content = _FENCED_BLOCK_RE.sub("", text)
    anchors = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", content, re.MULTILINE):
        cleaned = re.sub(r"[^\w\-\s]", "", heading.lower().replace("`", ""))
        anchors.add(re.sub(r"\s", "-", cleaned))
    return anchors


def _check_readme_structure() -> list[str]:
    errors = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    heading_text = _FENCED_BLOCK_RE.sub("", readme)
    headings = re.findall(r"^#{1,2} .+$", heading_text, re.M)
    if len(headings) < 4 or headings[1:4] != _REQUIRED_README_SECTIONS:
        errors.append(
            "README 첫 섹션 순서: 프로젝트명 → 프로젝트 소개 → 핵심 특징 → 아키텍처"
        )
    section = readme.split("## 아키텍처", 1)[-1].split("\n## ", 1)[0]
    if not re.search(r"(?m)^\|.+\|$", section):
        errors.append(
            "README 아키텍처에는 구성 요소와 역할을 설명하는 표가 필요합니다."
        )
    if not re.search(r"(?m)^\|.+\|\n\s*\n```mermaid\n", section):
        errors.append("README 아키텍처 표 바로 아래에 Mermaid 다이어그램이 필요합니다.")
    return errors


def main() -> int:
    """Validate repository files and return a nonzero exit code on errors."""
    errors = []
    count = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or EXCLUDED.intersection(path.relative_to(ROOT).parts):
            continue
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".sh":
                result = subprocess.run(
                    ["bash", "-n", str(path)], capture_output=True, text=True
                )
                if result.returncode:
                    errors.append(result.stderr.strip())
            elif path.suffix == ".md":
                # 템플릿의 상대 링크는 렌더링된 출력 경로를 기준으로 검사해야 한다.
                if "src" in path.relative_to(ROOT).parts and "templates" in path.parts:
                    continue
                errors.extend(_check_markdown_links(path))
            else:
                continue
            count += 1
        except (SyntaxError, ValueError, OSError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    errors.extend(_check_readme_structure())
    for error in errors:
        print(error, file=sys.stderr)
    print(f"문법·문서 검사: {count}개 파일, 오류 {len(errors)}개")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
