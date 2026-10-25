"""Build the flat archive required by the platform file copier."""
import hashlib
import re
from pathlib import Path
import zipfile


def build(root=None):
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    output = root / "dist"
    output.mkdir(exist_ok=True)
    archive = output / "submission.zip"
    files = {"solution.py": (root / "solution.py").read_bytes(),
             "requirements.txt": (root / "reference" / "requirements.txt").read_bytes()}
    for path in sorted((root / "research").glob("*.py")):
        if path.stem != "__init__":
            files[f"automl_{path.stem}.py"] = path.read_bytes()
    # The platform copies only root Python files. Also rewrite the generated
    # worker import string; the source package stays convenient for development.
    for name, data in list(files.items()):
        if name.endswith(".py"):
            data = data.replace(b"research.", b"automl_")
            compile(data, name, "exec")
            if re.search(rb"\b(?:from|import) research\b", data):
                raise ValueError(f"Unflattened import in {name}")
        if re.search(rb"sk-[A-Za-z0-9_-]{16,}", data):
            raise ValueError(f"Possible API key in {name}")
        files[name] = data
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert "solution.py" in bundle.namelist()
        assert all(Path(name).name == name for name in bundle.namelist())
        assert all(name.endswith(".py") or name == "requirements.txt" for name in bundle.namelist())
    if archive.stat().st_size > 100 * 1024 ** 2:
        raise ValueError("Submission exceeds the 100 MB platform limit")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "submission.sha256").write_text(f"{digest}  submission.zip\n", encoding="ascii")
    print(f"{archive} ({archive.stat().st_size} bytes)")
    print(f"SHA256 {digest}")
    return archive


if __name__ == "__main__":
    build()
