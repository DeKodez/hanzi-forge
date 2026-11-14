"""Filesystem locations shared by the package and scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
FONTS = ASSETS / "fonts"
INDEX_DIR = DATA / "index"
MODELS = DATA / "models"
OUTPUTS = ROOT / "outputs"

BRUSH_FONT = FONTS / "MaShanZheng-Regular.ttf"
PEN_FONT = FONTS / "LXGWWenKai-Regular.ttf"
DICTIONARY = DATA / "makemeahanzi_dictionary.txt"
CEDICT = DATA / "cedict.txt.gz"
HANZIDB = DATA / "hanzidb.csv"
IDS_DB = DATA / "ids.txt"


def ensure_dirs() -> None:
    for path in (DATA, ASSETS, FONTS, INDEX_DIR, MODELS, OUTPUTS):
        path.mkdir(parents=True, exist_ok=True)
