#!/usr/bin/env python3
"""Build an editable PowerPoint framework figure with frozen/trainable states.

The slide is generated through LibreOffice UNO so that blocks, labels, arrows,
badges, and lock icons remain editable PowerPoint objects.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import time

import uno
from com.sun.star.awt import Point, Size
from com.sun.star.beans import PropertyValue


DASH = uno.Enum("com.sun.star.drawing.LineStyle", "DASH")
SOLID = uno.Enum("com.sun.star.drawing.LineStyle", "SOLID")
MIDDLE = uno.Enum("com.sun.star.drawing.TextVerticalAdjust", "CENTER")
CENTER = uno.Enum("com.sun.star.style.ParagraphAdjust", "CENTER")
LEFT = uno.Enum("com.sun.star.style.ParagraphAdjust", "LEFT")


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "outputs" / "figures" / "framework_asset_pack"
OUTPUT_DIR = ROOT / "outputs" / "figures" / "framework_pptx"
PPTX_PATH = OUTPUT_DIR / "figure1_framework_trainable_frozen.pptx"
PDF_PATH = OUTPUT_DIR / "figure1_framework_trainable_frozen.pdf"

PAGE_W = 33867
PAGE_H = 19050


COLORS = {
    "navy": 0x17233D,
    "blue": 0x275DAD,
    "blue_mid": 0x6E98DF,
    "blue_light": 0xEAF2FF,
    "blue_panel": 0xF3F7FF,
    "source_panel": 0xF7F9FC,
    "gray": 0x5E687A,
    "gray_mid": 0xAEB7C5,
    "gray_light": 0xEEF1F5,
    "gray_lock": 0x6E7683,
    "white": 0xFFFFFF,
    "black": 0x111111,
    "teal": 0x078B6C,
    "teal_light": 0xE0F5EE,
    "orange": 0xDA8618,
    "orange_light": 0xFFF3DC,
    "magenta": 0xA344A4,
    "magenta_light": 0xF6E7F7,
    "red": 0xBC3D44,
}


def prop(name: str, value: object) -> PropertyValue:
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def inch(value: float) -> int:
    return int(round(value * 2540.0))


def set_if(obj: object, name: str, value: object) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        pass


class SlideBuilder:
    def __init__(self, doc: object, slide: object) -> None:
        self.doc = doc
        self.slide = slide

    def add(self, service: str) -> object:
        shape = self.doc.createInstance(service)
        self.slide.add(shape)
        return shape

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: int,
        line: int,
        line_width: float = 0.8,
        radius: float = 0.08,
        transparency: int = 0,
    ) -> object:
        shape = self.add("com.sun.star.drawing.RectangleShape")
        shape.Position = Point(inch(x), inch(y))
        shape.Size = Size(inch(w), inch(h))
        shape.FillColor = fill
        shape.FillTransparence = transparency
        shape.LineColor = line
        shape.LineWidth = max(1, inch(line_width / 72.0))
        set_if(shape, "CornerRadius", inch(radius))
        return shape

    def ellipse(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: int,
        line: int,
        line_width: float = 0.8,
        transparency: int = 0,
    ) -> object:
        shape = self.add("com.sun.star.drawing.EllipseShape")
        shape.Position = Point(inch(x), inch(y))
        shape.Size = Size(inch(w), inch(h))
        shape.FillColor = fill
        shape.FillTransparence = transparency
        shape.LineColor = line
        shape.LineWidth = max(1, inch(line_width / 72.0))
        return shape

    def text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        *,
        size: float = 10.0,
        color: int = COLORS["navy"],
        bold: bool = False,
        align: str = "left",
        valign: str = "middle",
        font: str = "Liberation Sans",
        margin: float = 0.02,
    ) -> object:
        shape = self.add("com.sun.star.drawing.TextShape")
        shape.Position = Point(inch(x), inch(y))
        shape.Size = Size(inch(w), inch(h))
        shape.String = value
        shape.TextAutoGrowHeight = False
        shape.TextAutoGrowWidth = False
        shape.TextLeftDistance = inch(margin)
        shape.TextRightDistance = inch(margin)
        shape.TextUpperDistance = inch(margin)
        shape.TextLowerDistance = inch(margin)
        set_if(shape, "TextVerticalAdjust", MIDDLE if valign == "middle" else 0)
        cursor = shape.createTextCursor()
        cursor.gotoEnd(True)
        cursor.CharFontName = font
        cursor.CharHeight = float(size)
        cursor.CharColor = color
        cursor.CharWeight = 150.0 if bold else 100.0
        cursor.ParaAdjust = CENTER if align == "center" else LEFT
        # Setting String can make LibreOffice expand a text shape to its default
        # minimum height. Reassert the requested geometry after formatting so
        # compact labels and badges stay inside their intended bounds in PPTX.
        shape.Position = Point(inch(x), inch(y))
        shape.Size = Size(inch(w), inch(h))
        return shape

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: int = COLORS["gray"],
        width: float = 1.4,
        arrow: bool = True,
        dashed: bool = False,
    ) -> object:
        shape = self.add("com.sun.star.drawing.LineShape")
        shape.Position = Point(inch(x1), inch(y1))
        shape.Size = Size(inch(x2 - x1), inch(y2 - y1))
        shape.LineColor = color
        shape.LineWidth = max(1, inch(width / 72.0))
        shape.LineStyle = DASH if dashed else SOLID
        if arrow:
            set_if(shape, "LineEndName", "Arrow")
            set_if(shape, "LineEndWidth", inch(0.10))
            set_if(shape, "LineEndCenter", True)
        return shape

    def image(self, path: Path, x: float, y: float, w: float, h: float) -> object:
        self.rect(x - 0.025, y - 0.025, w + 0.05, h + 0.05, fill=COLORS["black"], line=COLORS["black"], radius=0.0)
        shape = self.add("com.sun.star.drawing.GraphicObjectShape")
        shape.Position = Point(inch(x), inch(y))
        shape.Size = Size(inch(w), inch(h))
        shape.GraphicURL = uno.systemPathToFileUrl(str(path.resolve()))
        return shape

    def badge(
        self,
        x: float,
        y: float,
        w: float,
        label: str,
        *,
        fill: int,
        line: int,
        color: int,
        size: float = 5.8,
    ) -> None:
        self.rect(x, y, w, 0.22, fill=fill, line=line, line_width=0.7, radius=0.10)
        self.text(x, y + 0.005, w, 0.20, label, size=size, color=color, bold=True, align="center")

    def lock(self, x: float, y: float, scale: float = 1.0, *, color: int = COLORS["gray_lock"]) -> None:
        # Ellipse behind a body rectangle produces a reliable editable padlock.
        self.ellipse(
            x + 0.04 * scale,
            y,
            0.18 * scale,
            0.20 * scale,
            fill=COLORS["white"],
            line=color,
            line_width=1.1,
            transparency=100,
        )
        self.rect(
            x,
            y + 0.10 * scale,
            0.26 * scale,
            0.20 * scale,
            fill=color,
            line=color,
            line_width=0.6,
            radius=0.025,
        )
        self.ellipse(
            x + 0.105 * scale,
            y + 0.155 * scale,
            0.05 * scale,
            0.05 * scale,
            fill=COLORS["white"],
            line=COLORS["white"],
            line_width=0.4,
        )

    def card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str = "",
        *,
        fill: int = COLORS["white"],
        line: int = COLORS["gray_mid"],
        title_color: int = COLORS["navy"],
        title_size: float = 9.0,
        subtitle_size: float = 6.3,
    ) -> None:
        self.rect(x, y, w, h, fill=fill, line=line, line_width=1.0, radius=0.13)
        title_h = 0.31 if subtitle else h
        self.text(x + 0.08, y + 0.06, w - 0.16, title_h, title, size=title_size, color=title_color, bold=True, align="center")
        if subtitle:
            self.text(x + 0.08, y + 0.37, w - 0.16, h - 0.41, subtitle, size=subtitle_size, color=COLORS["gray"], align="center")


def connect_office(port: int = 2002) -> tuple[object, subprocess.Popen[bytes]]:
    profile = OUTPUT_DIR / "lo_profile"
    profile.mkdir(parents=True, exist_ok=True)
    command = [
        "libreoffice",
        "--headless",
        "--nologo",
        "--nodefault",
        "--norestore",
        "--nolockcheck",
        f"-env:UserInstallation={uno.systemPathToFileUrl(str(profile.resolve()))}",
        f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext",
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    deadline = time.time() + 20.0
    while time.time() < deadline:
        try:
            context = resolver.resolve(
                f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
            )
            return context, process
        except Exception:
            if process.poll() is not None:
                raise RuntimeError("LibreOffice exited before the UNO connection was ready")
            time.sleep(0.25)
    process.terminate()
    raise TimeoutError("Timed out connecting to LibreOffice")


def build_slide(builder: SlideBuilder) -> None:
    b = builder

    # Background and title.
    b.rect(0, 0, 13.333, 7.5, fill=COLORS["white"], line=COLORS["white"], radius=0.0)
    b.text(0.22, 0.08, 8.2, 0.32, "Figure 1. Overall proposed framework", size=18.5, bold=True)
    b.text(
        0.22,
        0.40,
        11.8,
        0.22,
        "Unsupervised source training and parameter-efficient test-time adaptation for cross-contrast MRI reconstruction",
        size=7.7,
        color=COLORS["gray"],
    )

    # Main panels.
    # Keep the two large domains rectangular in PPTX. LibreOffice otherwise
    # exports large rounded rectangles as near-capsules when reopened.
    b.rect(0.06, 0.69, 3.05, 6.38, fill=COLORS["source_panel"], line=0xC9D1DC, line_width=1.0, radius=0.0)
    b.rect(3.32, 0.69, 9.95, 6.38, fill=COLORS["blue_panel"], line=0xC9D1DC, line_width=1.0, radius=0.0)
    b.text(0.29, 0.82, 1.7, 0.24, "SOURCE DOMAIN", size=10.2, bold=True)
    b.text(0.29, 1.06, 1.8, 0.20, "T1-weighted MRI", size=7.0, color=COLORS["gray"])
    b.badge(2.10, 0.84, 0.76, "SOURCE TRAINING", fill=0xDCE9FF, line=0xDCE9FF, color=COLORS["blue"], size=5.5)

    b.text(3.58, 0.82, 1.7, 0.24, "TARGET DOMAIN", size=10.2, bold=True)
    b.text(3.58, 1.06, 2.8, 0.20, "Unseen T2 / FLAIR / POST contrasts", size=7.0, color=COLORS["gray"])
    b.badge(11.93, 0.84, 0.88, "TEST TIME", fill=0xDCE9FF, line=0xDCE9FF, color=COLORS["blue"], size=6.0)

    # Source-domain images.
    b.image(ASSET_DIR / "01_source_T1_reference.png", 0.33, 1.45, 1.04, 1.04)
    b.image(ASSET_DIR / "02_source_T1_zero_filled_R4.png", 1.72, 1.45, 1.04, 1.04)
    b.text(0.32, 2.52, 1.06, 0.20, "T1 reference", size=6.2, color=COLORS["gray"], align="center")
    b.text(1.69, 2.52, 1.10, 0.20, "Zero-filled  Rₐcc=4", size=6.2, color=COLORS["gray"], align="center")
    b.badge(0.45, 2.33, 0.80, "not used as GT", fill=COLORS["white"], line=0xD7DCE5, color=COLORS["gray"], size=5.2)

    b.line(1.53, 2.72, 1.53, 2.98, color=COLORS["black"], width=1.1)
    b.card(
        0.42,
        3.00,
        2.34,
        0.66,
        "TRUE-ENSURE source training",
        "self-supervised learning from acquired measurements",
        fill=COLORS["white"],
        line=COLORS["blue"],
        title_size=9.0,
        subtitle_size=5.7,
    )
    b.line(1.59, 3.68, 1.59, 3.94, color=COLORS["black"], width=1.1)
    b.card(
        0.43,
        3.98,
        2.32,
        0.64,
        "Source checkpoint",
        "12 cascades: {W₀, λ₀}",
        fill=COLORS["white"],
        line=COLORS["gray_mid"],
        title_size=9.2,
        subtitle_size=6.3,
    )
    b.lock(2.40, 4.06, 0.82)
    b.badge(0.72, 4.78, 1.74, "BASE CNN FROZEN AT TEST TIME", fill=COLORS["gray_light"], line=COLORS["gray_mid"], color=COLORS["gray"], size=5.6)
    b.card(
        0.35,
        5.40,
        2.47,
        0.72,
        "No fully sampled supervision",
        "source: TRUE-ENSURE  •  target: measured k-space only",
        fill=0xE2EEFF,
        line=0xE2EEFF,
        title_color=COLORS["blue"],
        title_size=8.0,
        subtitle_size=5.6,
    )
    b.text(
        0.35,
        6.31,
        2.44,
        0.38,
        "Source weights initialize every target case;\nLoRA branches are inserted with zero output.",
        size=6.0,
        color=COLORS["gray"],
        align="center",
    )

    # Target examples and the actual mask.
    target_items = [
        ("03_target_T2_reference.png", "T2", 3.63),
        ("04_target_FLAIR_reference.png", "FLAIR", 4.68),
        ("05_target_POST_reference.png", "POST", 5.73),
    ]
    for filename, label, x in target_items:
        b.image(ASSET_DIR / filename, x, 1.44, 0.86, 0.86)
        b.text(x, 2.32, 0.86, 0.18, label, size=6.2, color=COLORS["gray"], align="center")
    b.image(ASSET_DIR / "06_actual_sampling_mask_R4.png", 6.85, 1.44, 0.48, 0.86)
    b.text(6.64, 2.32, 0.90, 0.18, "actual R=3.81 mask", size=5.6, color=COLORS["gray"], align="center")
    b.card(
        7.68,
        1.45,
        3.12,
        0.84,
        "Target images are illustrative only",
        "No target reference enters TTA. The displayed mask is one exact\nsample-specific Bernoulli–Gaussian realization (nominal Rₐcc=4).",
        fill=COLORS["white"],
        line=0xC9D1DC,
        title_size=7.2,
        subtitle_size=5.6,
    )
    b.card(
        11.05,
        1.45,
        1.86,
        0.84,
        "Trainable at test time",
        "36 Conv-LoRA adapters\n+ 12 DC scalars λₖ",
        fill=COLORS["teal_light"],
        line=COLORS["teal"],
        title_color=COLORS["teal"],
        title_size=6.7,
        subtitle_size=5.6,
    )

    # Connection from the source checkpoint into the target model.
    b.line(2.76, 4.30, 3.56, 4.30, color=COLORS["blue"], width=1.7)

    # Unrolled test-time ribbon.
    b.rect(3.57, 2.67, 9.36, 0.84, fill=COLORS["white"], line=COLORS["blue_mid"], line_width=1.2, radius=0.15)
    b.text(3.73, 2.70, 2.20, 0.20, "12-STAGE UNROLLED RECONSTRUCTION", size=7.4, color=COLORS["blue"], bold=True)
    b.text(3.75, 3.02, 0.60, 0.22, "x⁽⁰⁾", size=8.5, bold=True, align="center")
    b.line(4.30, 3.13, 4.57, 3.13, color=COLORS["blue"], width=1.2)

    cascade_xs = [4.58, 6.28, 9.12]
    cascade_labels = ["CASCADE 1", "CASCADE 2", "CASCADE 12"]
    for x, label in zip(cascade_xs, cascade_labels):
        b.rect(x, 2.89, 1.28, 0.48, fill=COLORS["gray_light"], line=COLORS["gray_mid"], line_width=0.8, radius=0.08)
        b.text(x + 0.05, 2.92, 1.18, 0.18, label, size=5.8, color=COLORS["gray"], bold=True, align="center")
        b.text(x + 0.11, 3.13, 0.72, 0.14, "CNN + DC", size=5.1, color=COLORS["gray"], align="center")
        b.lock(x + 0.93, 3.05, 0.60)
        b.badge(x + 0.05, 2.71, 0.73, "LoRA + λₖ", fill=COLORS["teal_light"], line=COLORS["teal"], color=COLORS["teal"], size=4.7)
    b.line(5.86, 3.13, 6.27, 3.13, color=COLORS["blue"], width=1.2)
    b.text(7.63, 2.91, 1.10, 0.37, "•••", size=16.0, color=COLORS["blue"], bold=True, align="center")
    b.line(7.56, 3.13, 7.75, 3.13, color=COLORS["blue"], width=1.2, arrow=False)
    b.line(8.71, 3.13, 9.10, 3.13, color=COLORS["blue"], width=1.2)
    b.line(10.40, 3.13, 10.72, 3.13, color=COLORS["blue"], width=1.2)
    b.text(10.73, 3.01, 0.43, 0.25, "x̂", size=11.0, color=COLORS["blue"], bold=True, align="center")
    b.text(11.14, 2.72, 1.62, 0.20, "FROZEN BASE • TRAINABLE INSERTS", size=5.1, color=COLORS["gray"], bold=True, align="center")
    b.line(10.99, 3.30, 11.55, 3.71, color=COLORS["blue"], width=1.1)

    # Zoomed adaptive cascade.
    b.rect(3.73, 3.74, 6.90, 2.33, fill=COLORS["white"], line=COLORS["blue"], line_width=1.2, radius=0.0)
    b.text(3.91, 3.79, 3.25, 0.22, "Zoom: one adaptive cascade k", size=8.6, color=COLORS["blue"], bold=True)
    b.text(7.20, 3.79, 3.16, 0.22, "same structure repeated for k = 1, …, 12", size=5.8, color=COLORS["gray"], align="right")
    b.text(3.91, 4.51, 0.58, 0.25, "x⁽ᵏ⁻¹⁾", size=8.2, bold=True, align="center")
    b.line(4.46, 4.64, 4.73, 4.64, color=COLORS["gray"], width=1.2)

    # Frozen base branch.
    b.rect(4.76, 4.18, 2.12, 0.56, fill=COLORS["gray_light"], line=COLORS["gray_mid"], line_width=1.0, radius=0.09)
    b.text(4.86, 4.21, 1.52, 0.22, "Base Conv 3×3, W₀", size=7.0, color=COLORS["gray"], bold=True, align="center")
    b.text(4.86, 4.46, 1.55, 0.16, "64 → 64  •  frozen", size=5.5, color=COLORS["gray"], align="center")
    b.lock(6.52, 4.29, 0.75)

    # Trainable LoRA bypass.
    b.line(4.61, 4.65, 4.61, 5.33, color=COLORS["teal"], width=1.2, arrow=False)
    b.line(4.61, 5.33, 4.76, 5.33, color=COLORS["teal"], width=1.2)
    b.rect(4.76, 5.03, 0.88, 0.58, fill=COLORS["teal_light"], line=COLORS["teal"], line_width=1.0, radius=0.09)
    b.text(4.79, 5.06, 0.82, 0.22, "LoRA-down A", size=6.2, color=COLORS["teal"], bold=True, align="center")
    b.text(4.79, 5.31, 0.82, 0.16, "3×3: 64→2", size=5.0, color=COLORS["teal"], align="center")
    b.line(5.65, 5.33, 5.80, 5.33, color=COLORS["teal"], width=1.2)
    b.rect(5.81, 5.03, 0.88, 0.58, fill=COLORS["teal_light"], line=COLORS["teal"], line_width=1.0, radius=0.09)
    b.text(5.84, 5.06, 0.82, 0.22, "LoRA-up B", size=6.2, color=COLORS["teal"], bold=True, align="center")
    b.text(5.84, 5.31, 0.82, 0.16, "1×1: 2→64", size=5.0, color=COLORS["teal"], align="center")
    b.badge(5.17, 4.78, 1.12, "TRAINABLE  rLoRA=2", fill=COLORS["teal"], line=COLORS["teal"], color=COLORS["white"], size=5.0)

    # Add branches and pass through soft DC.
    b.line(6.89, 4.46, 7.23, 4.65, color=COLORS["gray"], width=1.2)
    b.line(6.69, 5.33, 7.23, 4.74, color=COLORS["teal"], width=1.2)
    b.ellipse(7.22, 4.52, 0.38, 0.38, fill=COLORS["white"], line=COLORS["blue"], line_width=1.2)
    b.text(7.22, 4.54, 0.38, 0.28, "+", size=11.0, color=COLORS["blue"], bold=True, align="center")
    b.line(7.60, 4.71, 7.82, 4.71, color=COLORS["blue"], width=1.2)

    b.rect(7.84, 4.22, 1.65, 0.98, fill=COLORS["orange_light"], line=COLORS["orange"], line_width=1.0, radius=0.10)
    b.text(7.93, 4.27, 1.47, 0.22, "Soft data consistency", size=6.8, color=0x8B560F, bold=True, align="center")
    b.text(7.96, 4.54, 1.40, 0.20, "AᴴA and measured yΩ", size=5.2, color=COLORS["gray"], align="center")
    b.lock(8.02, 4.78, 0.63)
    b.badge(8.55, 4.78, 0.72, "λₖ TRAINABLE", fill=COLORS["orange"], line=COLORS["orange"], color=COLORS["white"], size=4.6)
    b.line(9.50, 4.71, 9.78, 4.71, color=COLORS["blue"], width=1.2)
    b.text(9.79, 4.57, 0.48, 0.28, "x⁽ᵏ⁾", size=8.2, bold=True, align="center")
    b.text(
        7.78,
        5.35,
        2.55,
        0.40,
        "xₖ = xₖ₋½ − λₖ(AᴴAxₖ₋½ − x₀)\nOnly λₖ changes; the physics operator stays fixed.",
        size=5.5,
        color=COLORS["gray"],
        align="center",
    )

    # Self-supervised objective and final result.
    b.card(
        10.90,
        3.76,
        2.02,
        1.08,
        "Self-supervised TTA",
        "95% measured k-space → normalized complex L1\n5% held out → select best step (including step 0)",
        fill=COLORS["magenta_light"],
        line=COLORS["magenta"],
        title_color=COLORS["magenta"],
        title_size=7.8,
        subtitle_size=5.3,
    )
    b.badge(11.42, 3.59, 0.96, "NO TARGET GT", fill=COLORS["white"], line=COLORS["magenta"], color=COLORS["magenta"], size=5.0)

    # Gradient paths touch only trainable components.
    b.line(11.05, 4.91, 9.04, 5.00, color=COLORS["magenta"], width=1.1, arrow=True, dashed=True)
    b.line(10.96, 5.02, 6.28, 5.44, color=COLORS["magenta"], width=1.1, arrow=True, dashed=True)
    b.text(9.32, 5.09, 1.18, 0.18, "gradient", size=5.1, color=COLORS["magenta"], bold=True, align="center")
    b.text(6.82, 5.54, 1.06, 0.18, "gradient", size=5.1, color=COLORS["magenta"], bold=True, align="center")

    b.line(11.91, 4.86, 11.91, 5.18, color=COLORS["blue"], width=1.3)
    b.image(ASSET_DIR / "07_proposed_LoRA_TTA_POST_reconstruction.png", 11.39, 5.24, 1.04, 1.04)
    b.text(10.99, 6.31, 1.84, 0.20, "Adapted POST reconstruction", size=6.4, color=COLORS["blue"], bold=True, align="center")
    b.text(10.99, 6.52, 1.84, 0.20, "without target ground truth", size=5.4, color=COLORS["gray"], align="center")

    # Footer legend and exact code-specific parameter accounting.
    b.line(3.60, 6.84, 12.96, 6.84, color=0xD7DDE7, width=0.7, arrow=False)
    b.lock(3.77, 6.91, 0.70)
    b.text(4.01, 6.88, 1.20, 0.22, "Frozen pretrained weight", size=5.7, color=COLORS["gray"])
    b.badge(5.30, 6.91, 0.68, "TRAINABLE", fill=COLORS["teal"], line=COLORS["teal"], color=COLORS["white"], size=4.5)
    b.text(6.04, 6.88, 1.52, 0.22, "LoRA A/B or DC scalar λₖ", size=5.7, color=COLORS["gray"])
    b.text(
        7.67,
        6.87,
        4.95,
        0.25,
        "Code-matched: 12 independent denoisers × 3 adapters + 12 λₖ = 46,092 trainable / 1,400,844 total (3.29%)",
        size=5.6,
        color=COLORS["navy"],
        bold=True,
        align="right",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for required in (
        "01_source_T1_reference.png",
        "02_source_T1_zero_filled_R4.png",
        "03_target_T2_reference.png",
        "04_target_FLAIR_reference.png",
        "05_target_POST_reference.png",
        "06_actual_sampling_mask_R4.png",
        "07_proposed_LoRA_TTA_POST_reconstruction.png",
    ):
        if not (ASSET_DIR / required).exists():
            raise FileNotFoundError(f"Missing framework asset: {ASSET_DIR / required}")

    context, process = connect_office()
    try:
        service_manager = context.ServiceManager
        desktop = service_manager.createInstanceWithContext("com.sun.star.frame.Desktop", context)
        document = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, ())
        slides = document.getDrawPages()
        slide = slides.getByIndex(0)
        slide.Width = PAGE_W
        slide.Height = PAGE_H
        set_if(slide, "Layout", 0)
        while slide.getCount() > 0:
            slide.remove(slide.getByIndex(0))

        build_slide(SlideBuilder(document, slide))

        document.storeAsURL(
            uno.systemPathToFileUrl(str(PPTX_PATH.resolve())),
            (prop("FilterName", "Impress MS PowerPoint 2007 XML"), prop("Overwrite", True)),
        )
        document.close(True)

        # Reopen the saved PPTX before rendering. This validates the actual
        # PowerPoint package and gives linked graphics time to load fully.
        time.sleep(1.0)
        rendered = desktop.loadComponentFromURL(
            uno.systemPathToFileUrl(str(PPTX_PATH.resolve())),
            "_blank",
            0,
            (prop("Hidden", True),),
        )
        time.sleep(1.0)
        rendered.storeToURL(
            uno.systemPathToFileUrl(str(PDF_PATH.resolve())),
            (prop("FilterName", "impress_pdf_Export"), prop("Overwrite", True)),
        )
        rendered.close(True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    subprocess.run(
        ["pdftoppm", "-f", "1", "-singlefile", "-png", "-r", "180", str(PDF_PATH), str(OUTPUT_DIR / "figure1_framework_trainable_frozen_preview")],
        check=True,
    )
    print(PPTX_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
