import hashlib
import json
from pathlib import Path
import zipfile


def build(root=None):
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    output = root / "dist"
    output.mkdir(exist_ok=True)
    archive = output / "submission.zip"
    files = [root / name for name in ["solution.py", "requirements.txt", "requirements-dev.txt", "README.md", "Dockerfile", ".env.example"]]
    files += sorted((root / "research").glob("*.py"))
    files += sorted((root / "tools").glob("*.py"))
    files += sorted((root / "tests").glob("*.py"))
    manifest = {}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            name = path.relative_to(root).as_posix()
            data = path.read_bytes()
            compile(data, name, "exec") if path.suffix == ".py" else None
            manifest[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
        bundle.writestr("source_manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert "solution.py" in bundle.namelist()
        assert not any(Path(name).name == ".env" or name.startswith("data/") or "model.joblib" in name for name in bundle.namelist())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "submission.sha256").write_text(f"{digest}  submission.zip\n", encoding="ascii")
    print(f"{archive} ({archive.stat().st_size} bytes)")
    print(f"SHA256 {digest}")
    return archive


if __name__ == "__main__":
    build()
