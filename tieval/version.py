from pathlib import Path

PWD = Path(__file__).parent.resolve()

def string():
    try:
        version = (PWD / "VERSION").read_text(encoding="utf-8").strip()
        if version:
            return version
    except:
        pass
    return "unknown (git checkout)"
