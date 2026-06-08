from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("social_assets")
OUT.mkdir(exist_ok=True)

W, H = 1600, 900
BG = "#07090d"
PANEL = "#11151b"
PANEL2 = "#0d1117"
BORDER = "#293241"
GRID = "#151a22"
TEXT = "#eef2f7"
MUTED = "#a5adba"
DIM = "#778294"
GREEN = "#58e6a6"
BLUE = "#7ab7ff"
YELLOW = "#ffd166"
RED = "#ff8c8c"
PURPLE = "#b794ff"

FONT_DIR = Path("C:/Windows/Fonts")


def load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in candidates:
        path = FONT_DIR / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def bold(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return load_font(["segoeuib.ttf", "arialbd.ttf"], size)


def regular(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return load_font(["segoeui.ttf", "arial.ttf"], size)


def mono(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return load_font(["CascadiaMono.ttf", "consola.ttf"], size)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    for x in range(90, W, 90):
        draw.line((x, 70, x, H - 70), fill=GRID, width=1)
    for y in range(85, H, 90):
        draw.line((50, y, W - 50, y), fill=GRID, width=1)
    return image, draw


def round_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str = PANEL,
    outline: str = BORDER,
    radius: int = 14,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def write(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = TEXT,
) -> None:
    draw.text(xy, value, font=font, fill=fill)


def header(draw: ImageDraw.ImageDraw, title: str, subtitle: str, title_size: int = 58) -> None:
    write(draw, (70, 54), "AlgoRadar", mono(30), GREEN)
    write(draw, (70, 112), title, bold(title_size), TEXT)
    write(draw, (72, 178), subtitle, regular(25), MUTED)


def footer(draw: ImageDraw.ImageDraw) -> None:
    write(draw, (70, 822), "github.com/drawmebaaz/AlgoRadar", mono(24), DIM)


def metric_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    label: str,
    value: str,
    note: str,
    color: str = GREEN,
) -> None:
    round_rect(draw, (x, y, x + width, y + height), fill=PANEL, radius=12)
    write(draw, (x + 24, y + 22), label.upper(), mono(15), DIM)
    write(draw, (x + 24, y + 62), value, bold(34), TEXT)
    write(draw, (x + 24, y + 110), note, regular(19), MUTED)
    draw.rounded_rectangle((x + 24, y + height - 22, x + width - 48, y + height - 14), radius=5, fill="#222a35")
    draw.rounded_rectangle(
        (x + 24, y + height - 22, x + 24 + int((width - 72) * 0.62), y + height - 14),
        radius=5,
        fill=color,
    )


def bar(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, label: str, pct: float, color: str = GREEN) -> None:
    write(draw, (x, y), label, regular(21), MUTED)
    write(draw, (x + width - 72, y), f"{pct:.0f}%", bold(19), TEXT)
    draw.rounded_rectangle((x, y + 32, x + width, y + 44), radius=6, fill="#222a35")
    draw.rounded_rectangle((x, y + 32, x + int(width * pct / 100), y + 44), radius=6, fill=color)


def save(image: Image.Image, name: str) -> None:
    image.save(OUT / name, optimize=True)


def remove_old_assets() -> None:
    for path in OUT.glob("algoradar-*.png"):
        path.unlink()


def build_overview() -> None:
    image, draw = canvas()
    header(draw, "Multi-Platform CP Intelligence", "Analyze Codeforces, CodeChef, and LeetCode handles from one focused dashboard.")
    card_width = 344
    metric_card(draw, 70, 252, card_width, 155, "Platforms", "3", "combined only when handles exist", GREEN)
    metric_card(draw, 439, 252, card_width, 155, "CF rating", "1200 / 1418", "current and max rating", BLUE)
    metric_card(draw, 808, 252, card_width, 155, "Solved signal", "842", "public solves across profiles", YELLOW)
    metric_card(draw, 1177, 252, card_width, 155, "Focus tags", "dp, graphs", "highest repair priority", PURPLE)

    round_rect(draw, (70, 468, 930, 758), fill=PANEL2, radius=14)
    write(draw, (100, 496), "Contest and practice trend", bold(28), TEXT)
    points = [(125, 690), (245, 672), (365, 652), (485, 618), (605, 642), (725, 575), (845, 606)]
    draw.line(points, fill=GREEN, width=4)
    for point in points:
        draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=GREEN)
    write(draw, (105, 710), "rating, contest deltas, and solved volume stay separate per platform", regular(20), MUTED)

    round_rect(draw, (970, 468, 1530, 758), fill=PANEL2, radius=14)
    write(draw, (1000, 496), "What changed", bold(28), TEXT)
    rows = [
        ("No cross-platform rating mixing", 92, GREEN),
        ("Cached public data pulls", 78, BLUE),
        ("Tag-aware solve model", 84, YELLOW),
        ("Platform-wise sections", 88, PURPLE),
    ]
    for index, (label, pct, color) in enumerate(rows):
        bar(draw, 1000, 550 + index * 52, 460, label, pct, color)
    footer(draw)
    save(image, "algoradar-01-multi-platform-profile.png")


def build_weakness_map() -> None:
    image, draw = canvas()
    header(draw, "Weakness Map", "Find tags that deserve practice using attempts, solved depth, recent failures, and tag coverage.")
    round_rect(draw, (70, 240, 860, 745), fill=PANEL2, radius=14)
    write(draw, (100, 270), "Tag priority matrix", bold(30), TEXT)
    rows = [
        ("graphs", "Weak", 36, 1180, 1500, RED),
        ("dp", "Over-attempted", 41, 1240, 1600, YELLOW),
        ("binary search", "Stable", 68, 1320, 1700, BLUE),
        ("math", "Strong", 76, 1420, 1900, GREEN),
        ("trees", "Untouched", 0, 0, 0, PURPLE),
    ]
    for index, (tag, level, solved, _avg, top, color) in enumerate(rows):
        y = 330 + index * 72
        draw.rounded_rectangle((100, y, 830, y + 54), radius=10, fill="#131922", outline="#242d3a")
        write(draw, (124, y + 13), tag, bold(22), TEXT)
        write(draw, (320, y + 14), level, regular(20), color)
        write(draw, (525, y + 14), f"{solved} solved", mono(18), MUTED)
        write(draw, (665, y + 14), f"top {top or '--'}", mono(18), MUTED)

    round_rect(draw, (920, 240, 1490, 745), fill=PANEL2, radius=14)
    write(draw, (950, 270), "Repair priority inputs", bold(30), TEXT)
    rows = [
        ("Low solved depth", 86, RED),
        ("Recent failed attempts", 73, YELLOW),
        ("Enough evidence", 69, BLUE),
        ("Accuracy signal", 38, GREEN),
        ("Untouched penalty", 57, PURPLE),
    ]
    for index, (label, pct, color) in enumerate(rows):
        bar(draw, 955, 350 + index * 70, 460, label, pct, color)
    write(draw, (950, 690), "Accuracy is a signal, not the whole verdict.", regular(22), MUTED)
    footer(draw)
    save(image, "algoradar-02-weakness-map.png")


def build_recommendations() -> None:
    image, draw = canvas()
    header(
        draw,
        "Platform-Separated Recommendations",
        "Confidence, growth, and stretch queues are ranked separately for each platform.",
        title_size=54,
    )
    panel_width = 450
    platforms = [("Codeforces", 70), ("CodeChef", 575), ("LeetCode", 1080)]
    items = {
        "Codeforces": [
            ("Confidence", "1100-1200", "implementation"),
            ("Growth", "1300-1500", "dp + graphs"),
            ("Stretch", "1500-1700", "trees"),
        ],
        "CodeChef": [
            ("Confidence", "Starters B", "math"),
            ("Growth", "1200-1400", "greedy"),
            ("Stretch", "1500+", "dp"),
        ],
        "LeetCode": [
            ("Confidence", "Medium Q1-Q2", "arrays"),
            ("Growth", "Medium Q2-Q3", "graphs"),
            ("Stretch", "Hard Q3-Q4", "dp"),
        ],
    }
    for name, x in platforms:
        round_rect(draw, (x, 255, x + panel_width, 735), fill=PANEL2, radius=14)
        write(draw, (x + 28, 285), name, bold(30), TEXT)
        for index, (bucket, target, tags) in enumerate(items[name]):
            y = 350 + index * 115
            color = GREEN if bucket == "Confidence" else YELLOW if bucket == "Growth" else RED
            draw.rounded_rectangle((x + 28, y, x + panel_width - 28, y + 84), radius=10, fill=PANEL, outline="#252f3d")
            write(draw, (x + 50, y + 15), bucket, bold(22), color)
            write(draw, (x + 220, y + 16), target, mono(18), TEXT)
            write(draw, (x + 50, y + 49), tags, regular(19), MUTED)
        write(draw, (x + 28, 682), "Ranked by probability, weak tags, popularity, diversity", regular(16), DIM)
    footer(draw)
    save(image, "algoradar-03-recommendations.png")


def build_solve_probability() -> None:
    image, draw = canvas()
    header(draw, "Calibrated Solve Probability", "Estimate a realistic solve chance from solved volume, tag strength, rating gap, and platform scale.", title_size=55)
    round_rect(draw, (70, 250, 820, 735), fill=PANEL2, radius=14)
    write(draw, (100, 280), "Probability falls as target rating rises", bold(30), TEXT)
    points = [(130, 365), (245, 410), (360, 475), (475, 545), (590, 610), (705, 660)]
    draw.line(points, fill=GREEN, width=4)
    for point in points:
        draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=GREEN)
    for x, label in [(130, "1000"), (245, "1200"), (360, "1400"), (475, "1600"), (590, "1800"), (705, "2000")]:
        write(draw, (x - 20, 682), label, mono(17), DIM)
    for y, label in [(360, "80%"), (430, "60%"), (500, "40%"), (570, "20%")]:
        draw.line((105, y, 760, y), fill="#1a2029", width=1)
        write(draw, (715, y - 13), label, mono(16), DIM)
    write(draw, (105, 705), "A harder problem should not randomly become easier.", regular(21), MUTED)

    round_rect(draw, (880, 250, 1490, 735), fill=PANEL2, radius=14)
    write(draw, (910, 280), "Model factors", bold(30), TEXT)
    rows = [
        ("Solved volume on selected tags", 82, GREEN),
        ("Hardest solved minus target", 64, BLUE),
        ("Average solved difficulty", 58, YELLOW),
        ("Recent failures", 31, RED),
        ("Rating-source confidence", 70, PURPLE),
    ]
    for index, (label, pct, color) in enumerate(rows):
        bar(draw, 915, 350 + index * 62, 500, label, pct, color)
    write(draw, (915, 675), "Accuracy is downweighted because public solves can include help/editorials.", regular(19), MUTED)
    footer(draw)
    save(image, "algoradar-04-solve-probability.png")


def build_social_card() -> None:
    image, draw = canvas()
    write(draw, (90, 85), "AlgoRadar", mono(40), GREEN)
    write(draw, (90, 160), "AI Competitive Programming", bold(66), TEXT)
    write(draw, (90, 240), "Weakness Analyzer", bold(66), TEXT)
    write(draw, (95, 342), "Multi-platform analytics, calibrated solve probability,", regular(29), MUTED)
    write(draw, (95, 382), "and personalized practice queues for CP learners.", regular(29), MUTED)

    chips = [
        ("Weakness map", GREEN),
        ("Problem recommender", YELLOW),
        ("Solve probability", BLUE),
        ("Real public data", PURPLE),
    ]
    for index, (label, color) in enumerate(chips):
        x = 95 + (index % 2) * 430
        y = 495 + (index // 2) * 105
        round_rect(draw, (x, y, x + 370, y + 72), fill=PANEL, radius=12)
        draw.ellipse((x + 24, y + 24, x + 48, y + 48), fill=color)
        write(draw, (x + 70, y + 20), label, bold(24), TEXT)

    round_rect(draw, (1070, 150, 1480, 690), fill=PANEL2, radius=16)
    rows = [("Confidence", 78, GREEN), ("Growth", 56, YELLOW), ("Stretch", 34, RED), ("Avoid", 14, DIM)]
    for index, (bucket, pct, color) in enumerate(rows):
        y = 210 + index * 105
        write(draw, (1110, y), bucket, bold(25), TEXT)
        write(draw, (1375, y), f"{pct}%", mono(25), color)
        draw.rounded_rectangle((1110, y + 46, 1440, y + 60), radius=7, fill="#222a35")
        draw.rounded_rectangle((1110, y + 46, 1110 + int(330 * pct / 100), y + 60), radius=7, fill=color)
    footer(draw)
    save(image, "algoradar-00-social-card.png")


def main() -> None:
    remove_old_assets()
    build_social_card()
    build_overview()
    build_weakness_map()
    build_recommendations()
    build_solve_probability()
    for path in sorted(OUT.glob("*.png")):
        image = Image.open(path)
        print(f"{path.name}: {image.size[0]}x{image.size[1]}")


if __name__ == "__main__":
    main()
