"""Display names of the sample videos (and the two uploads), for the paper's tables and the README."""

from __future__ import annotations

import re

NAMES = [  # (keyword in the file or run name, lower case; display name)
    ("above clouds", "PNB Merdeka 118"), ("above_clouds", "PNB Merdeka 118"), ("rural", "Rural farmland"),
    ("angkor", "Angkor Wat"), ("hanoi", "Hanoi Opera House"), ("colosseum", "Colosseum"), ("eiffel", "Eiffel Tower"),
    ("iceland", "Iceland canyon (FPV)"), ("gopro", "GoPro waterfall (FPV)"), ("hagia", "Hagia Sophia"),
    ("jal mahal", "Jal Mahal"), ("jal_mahal", "Jal Mahal"), ("notre dame", "Notre-Dame"), ("notre_dame", "Notre-Dame"),
    ("petronas", "Petronas Towers (FPV)"), ("qutub", "Qutub Minar"), ("reichstag", "Reichstag"),
    ("messiah", "Cristo Redentor"), ("cristo", "Cristo Redentor"), ("kinbane", "Kinbane Castle"), ("dunluce", "Dunluce Castle"),
]


def video_name(s: str) -> str:
    """``'HAGIA SOPHIA 4K ｜ Vibes of Istanbul [mTlqTdui6Zo].webm'`` or ``'map_hagia_sophia_...'`` -> ``'Hagia Sophia'``."""
    low = s.lower()
    for key, name in NAMES:
        if key in low:
            return name
    return re.split(r"[｜|：:,]| - |\[|\(", s)[0].strip()[:26] or s
