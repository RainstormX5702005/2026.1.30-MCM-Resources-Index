from pathlib import Path

import matplotlib as mpl


def set_encoding_method(font_name: str = "Microsoft Yahei") -> None:
    mpl.rcParams["font.family"] = font_name
    mpl.rcParams["axes.unicode_minus"] = False


DATA_DIR = Path(__file__).parent.parent.parent / "pyplot_examples"
FIG_DIR = DATA_DIR / "assets"
