#!/usr/bin/env python3
"""Draw the README flowcharts as editable, self-contained SVGs.

Generate: python tools/generate_diagrams.py
Check:    python tools/generate_diagrams.py --check

Update this source when a pipeline changes, then regenerate the SVG files.
Only Python 3.10 or newer is needed, no extra packages.
"""

import argparse
from html import escape
from math import hypot
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "thesis_data" / "diagrams"
WIDTH = 1280
BG = "#07131f"
TEXT = "#edf7ff"
MUTED = "#a3b9ca"
LINE = "#78a6bd"
CUDA = "#52e69e"
OPENCL = "#59d8f3"
CPU = "#ffad85"
VIOLET = "#c3a0ff"
SANS = "'Segoe UI', Inter, Helvetica, Arial, sans-serif"
MONO = "Consolas, 'Liberation Mono', monospace"


def tint(color, amount=0.08):
    return "#" + "".join(
        f"{round(int(BG[i:i + 2], 16) * (1 - amount) + int(color[i:i + 2], 16) * amount):02x}"
        for i in (1, 3, 5)
    )


class Flowchart:
    def __init__(self, height, title, description, label, subtitle):
        self.height = height
        self.group_headers = []
        self.parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
            'role="img" aria-labelledby="title description">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="description">{escape(description)}</desc>',
            '<defs>',
            '<radialGradient id="ambient" gradientUnits="userSpaceOnUse" '
            f'cx="640" cy="{height * 0.35}" r="880">',
            '<stop offset="0" stop-color="#173747" stop-opacity="0.35"/>',
            '<stop offset="1" stop-color="#173747" stop-opacity="0"/>',
            '</radialGradient>',
            '<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">',
            '<path d="M 32 0 H 0 V 32" fill="none" stroke="#91c4df" '
            'stroke-opacity="0.035" stroke-width="1"/>',
            '</pattern>',
            '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%" '
            'color-interpolation-filters="sRGB">',
            '<feGaussianBlur stdDeviation="2.4"/>',
            '</filter>',
        ]
        for color in (LINE, CUDA, OPENCL, CPU, VIOLET):
            self.parts.append(
                f'<marker id="arrow-{color[1:]}" viewBox="0 0 10 10" '
                'refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                f'<path d="M 1 1 L 8 5 L 1 9" fill="none" stroke="{color}" '
                'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
                '</marker>'
            )
        self.parts.append('</defs>')
        self.rect(1, 1, WIDTH - 2, height - 2, fill=BG, stroke="#203a4c", radius=24)
        self.rect(2, 2, WIDTH - 4, height - 4, fill="url(#ambient)", radius=23)
        self.rect(2, 2, WIDTH - 4, height - 4, fill="url(#grid)", radius=23)
        self.text(44, 48, label.upper(), size=14, color=OPENCL, mono=True, tracking=1.8)
        self.text(44, 94, title, size=34, weight=700)
        self.text(44, 126, subtitle, size=19, color=MUTED)

    def rect(self, x, y, width, height, *, fill="none", stroke="none", radius=0, extra=""):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}"{extra}/>'
        )

    def text(self, x, y, value, *, size=18, color=TEXT, weight=400,
             anchor="start", mono=False, tracking=0):
        family = MONO if mono else SANS
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{escape(family, quote=True)}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'letter-spacing="{tracking}" fill="{color}">{escape(value)}</text>'
        )

    def lines(self, x, y, values, *, gap=26, **style):
        for index, value in enumerate(values):
            self.text(x, y + index * gap, value, **style)

    def group(self, x, y, width, height, label, color, *, note=""):
        self.rect(x, y, width, height, fill=tint(color, 0.035), stroke=color,
                  radius=18, extra=' stroke-opacity="0.5" stroke-dasharray="3 7"')
        self.group_headers.append((x, y, width, label, color, note))

    def node(self, x, y, width, height, title, subtitle="", *, color=LINE,
             terminal=False, size=22, body_size=17):
        self.parts.append(
            f'<g data-node="{escape(title, quote=True)}" '
            f'data-bounds="{x} {y} {width} {height}">'
        )
        radius = height / 2 if terminal else 11
        self.rect(x, y, width, height, stroke=color, radius=radius,
                  extra=' stroke-width="3" opacity="0.28" filter="url(#glow)"')
        self.rect(x, y, width, height, fill=tint(color), stroke=color, radius=radius,
                  extra=' stroke-width="1.6"')
        if subtitle:
            title_offset, body_offset = (-15, 30) if body_size >= 24 else (-4, 23)
            self.text(x + width / 2, y + height / 2 + title_offset, title, size=size,
                      weight=650, anchor="middle")
            self.text(x + width / 2, y + height / 2 + body_offset, subtitle,
                      size=body_size, color=MUTED, anchor="middle")
        else:
            self.text(x + width / 2, y + height / 2 + size * 0.34, title,
                      size=size, weight=650, anchor="middle")
        self.parts.append('</g>')

    def decision(self, cx, cy, width, height, rows, *, color=OPENCL, size=20):
        self.parts.append(
            f'<g data-node="decision" data-bounds="{cx - width / 2} '
            f'{cy - height / 2} {width} {height}" data-shape="diamond">'
        )
        points = f'{cx},{cy - height / 2} {cx + width / 2},{cy} {cx},{cy + height / 2} {cx - width / 2},{cy}'
        self.parts.append(
            f'<polygon points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3" opacity="0.3" filter="url(#glow)"/>'
        )
        self.parts.append(
            f'<polygon points="{points}" fill="{tint(color)}" '
            f'stroke="{color}" stroke-width="1.7" stroke-linejoin="round"/>'
        )
        gap = max(27, round(size * 1.45))
        start_y = cy + size * 0.34 - (len(rows) - 1) * gap / 2
        self.lines(cx, start_y, rows, gap=gap, size=size, weight=600, anchor="middle")
        self.parts.append('</g>')

    def link(self, points, *, color=LINE, arrow=True, dashed=False):
        # Round bends without changing the requested route or its endpoints.
        commands = [f'M {points[0][0]} {points[0][1]}']
        for index, (x, y) in enumerate(points[1:-1], 1):
            px, py = points[index - 1]
            nx, ny = points[index + 1]
            before, after = hypot(x - px, y - py), hypot(nx - x, ny - y)
            if not before or not after:
                continue
            radius = min(12, before / 2, after / 2)
            ax, ay = x + (px - x) * radius / before, y + (py - y) * radius / before
            bx, by = x + (nx - x) * radius / after, y + (ny - y) * radius / after
            commands.extend((f'L {ax:g} {ay:g}', f'Q {x} {y} {bx:g} {by:g}'))
        commands.append(f'L {points[-1][0]} {points[-1][1]}')
        route = " ".join(commands)
        dash = ' stroke-dasharray="6 6"' if dashed else ""
        marker = f' marker-end="url(#arrow-{color[1:]})"' if arrow else ""
        self.parts.append(
            f'<path d="{route}" fill="none" stroke="{color}" stroke-width="4" '
            f'opacity="0.15" filter="url(#glow)"{dash}/>'
        )
        self.parts.append(
            f'<path d="{route}" fill="none" stroke="{color}" stroke-width="1.9" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash}{marker}/>'
        )

    def tag(self, cx, y, value, *, color=MUTED, size=16):
        width = len(value) * size * 0.53 + 20
        self.rect(cx - width / 2, y - size, width, size + 8,
                  fill=BG, radius=6)
        self.text(cx, y, value, size=size, color=color, anchor="middle", weight=500)

    def note(self, y, value, *, color=MUTED, size=17):
        self.text(44, y, value, color=color, size=size)

    def svg(self):
        # Group names sit above crossing wires and use the group's own background.
        for x, y, width, label, color, note in self.group_headers:
            self.rect(x + 13, y + 14, len(label) * 10 + 36, 29, fill=tint(color, 0.035), radius=4)
            self.rect(x + 20, y + 22, 5, 14, fill=color, radius=2)
            self.text(x + 38, y + 34, label.upper(), size=15, color=color,
                      weight=600, mono=True, tracking=1)
            if note:
                self.text(x + width - 20, y + 34, note, size=16, color=MUTED, anchor="end")
        return "\n".join(self.parts + ['</svg>', ''])


def image_pipeline():
    chart = Flowchart(
        1120, "Image watermarking",
        "Single images and folders use the same watermark code. CUDA and OpenCL "
        "run on the GPU, Eigen runs on the CPU. CUDA tries nvJPEG for supported "
        "JPEG files, CPU image libraries handle other inputs and fallback. "
        "Embedding saves an image, detection returns a correlation value. "
        "Qt supports single-image embed and detect, CLI single-image mode embeds.",
        "Image mode", "One image or a folder, the same steps on your chosen backend",
    )
    chart.group(40, 464, 1200, 198, "Watermark processing", OPENCL, note="One backend per build")
    chart.link([(320, 264), (320, 298), (640, 298), (640, 326)])
    chart.link([(960, 264), (960, 298), (640, 298)], arrow=False)
    chart.link([(640, 410), (640, 442), (240, 442), (240, 530)], color=CUDA)
    chart.link([(640, 442), (640, 530)], color=OPENCL)
    chart.link([(640, 442), (1040, 442), (1040, 530)], color=CPU)
    chart.link([(240, 616), (240, 702), (640, 702), (640, 730)], arrow=False)
    chart.link([(1040, 616), (1040, 702), (640, 702)], arrow=False)
    chart.link([(640, 616), (640, 730)])
    chart.link([(510, 788), (320, 788), (320, 890)], color=CUDA)
    chart.link([(770, 788), (960, 788), (960, 890)], color=VIOLET)
    chart.node(130, 176, 380, 88, "Single image", "Qt embed or detect, CLI embed", terminal=True, color=OPENCL)
    chart.node(770, 176, 380, 88, "Image folder", "Qt or CLI, embed or detect", terminal=True, color=OPENCL)
    chart.node(420, 326, 440, 84, "Load and prepare pixels", "Read the file, apply its rotation")
    for x, title, subtitle, color in (
        (76, "CUDA", "GPU watermark, embed or detect", CUDA),
        (476, "OpenCL", "GPU watermark, embed or detect", OPENCL),
        (876, "Eigen + OpenMP", "CPU watermark, embed or detect", CPU),
    ):
        chart.node(x, 530, 328, 86, title, subtitle, color=color, size=24)
    chart.decision(640, 788, 260, 116, ["Embed", "or detect"], color=OPENCL, size=21)
    chart.tag(404, 778, "Embed", color=CUDA)
    chart.tag(876, 778, "Detect", color=VIOLET)
    chart.node(110, 890, 420, 88, "Save the image", "Single file or watermark_output/ folder", terminal=True, color=CUDA)
    chart.node(750, 890, 420, 88, "Read the correlation", "A detection result, no image to save", terminal=True, color=VIOLET)
    chart.note(1024, "CUDA can read and write JPEGs with nvJPEG, other image formats use CPU libraries.")
    chart.note(1054, "Batches load and save in the background, the same password and image size give the same watermark.")
    chart.note(1084, "p sets the prediction window, PSNR sets the watermark strength.")
    return chart


def video_pipeline():
    chart = Flowchart(
        1630, "Video watermarking",
        "FFmpeg reads the video. CUDA tries NVDEC, OpenCL tries D3D11VA and "
        "QSV on Intel. Both hardware paths test the first frame before use, "
        "unsupported decoding uses CPU decode and GPU upload. Eigen stays on "
        "the CPU. Frames become 8-bit SDR before luma watermarking. Supported "
        "HDR uses GPU Mobius, other HDR uses CPU filters. Detection accepts "
        "only 8-bit SDR input, returns correlation results and stops. Embedding uses CPU encoding by default, "
        "optional hardware encoding uses NVENC, QSV or AMF on the same GPU. "
        "FFmpeg joins video with compatible audio and subtitle streams.",
        "Video mode", "Decode, prepare frames, watermark, then encode or read the detection result",
    )
    for x, label, color in ((40, "CUDA / NVIDIA", CUDA), (456, "OPENCL / GPU", OPENCL), (872, "EIGEN / CPU", CPU)):
        chart.group(x, 384, 368, 596, label, color)
    chart.link([(640, 214), (640, 258)])
    chart.link([(640, 334), (640, 358), (224, 358), (224, 448)], color=CUDA)
    chart.link([(640, 358), (640, 448)], color=OPENCL)
    chart.link([(640, 358), (1056, 358), (1056, 448)], color=CPU)
    for center, decision_x, fallback_x, color in (
        (224, 142, 254, CUDA), (640, 558, 670, OPENCL),
    ):
        chart.link([(center, 520), (center, 542), (decision_x, 542), (decision_x, 566)], color=color)
        chart.link([(decision_x + 78, 612), (fallback_x, 612)], color=CPU, dashed=True)
        chart.link([(decision_x, 658), (decision_x, 694), (center, 694), (center, 724)], color=color)
        chart.link([(fallback_x + 69, 650), (fallback_x + 69, 694), (center, 694)], color=CPU, arrow=False, dashed=True)
    chart.link([(1056, 520), (1056, 724)], color=CPU)
    for center, color in ((224, CUDA), (640, OPENCL), (1056, CPU)):
        chart.link([(center, 804), (center, 866)], color=color)
        chart.link([(center, 946), (center, 1000), (640, 1000)], arrow=False)
    chart.link([(640, 1000), (640, 1020)])
    chart.link([(760, 1078), (916, 1078)], color=VIOLET)
    chart.link([(640, 1136), (640, 1194)])
    chart.link([(640, 1194), (224, 1194), (224, 1230)], color=CUDA)
    chart.link([(640, 1194), (640, 1230)], color=OPENCL)
    chart.link([(640, 1194), (1056, 1194), (1056, 1230)], color=CPU)
    chart.link([(224, 1318), (224, 1350), (640, 1350)], arrow=False)
    chart.link([(1056, 1318), (1056, 1350), (640, 1350)], arrow=False)
    chart.link([(640, 1318), (640, 1384)])
    chart.node(470, 158, 340, 56, "Input video", terminal=True, color=OPENCL)
    chart.node(450, 258, 380, 76, "Read the video stream", "Keep audio and subtitles for the output")
    chart.node(62, 448, 324, 72, "NVDEC decode", "NVIDIA hardware decoder", color=CUDA)
    chart.node(478, 448, 324, 72, "D3D11VA decode", "Intel can also try QSV", color=OPENCL)
    chart.node(894, 448, 324, 72, "CPU decode", "FFmpeg software decoder", color=CPU)
    for center, fallback_x, color in ((142, 254, CUDA), (558, 670, OPENCL)):
        chart.decision(center, 612, 156, 92, ["Frame", "ready"], color=color, size=18)
        chart.node(fallback_x, 574, 138, 76, "CPU decode", "Upload frames", color=CPU, size=17, body_size=15)
        chart.tag(center + 18, 681, "Yes", color=color, size=15)
        chart.tag(center + 95, 600, "No", color=CPU, size=15)
    chart.tag(1117, 624, "CPU frames", color=CPU)
    chart.node(62, 724, 324, 80, "Make 8-bit SDR", "CUDA conversion and HDR tone map*", color=CUDA, size=22, body_size=16)
    chart.node(478, 724, 324, 80, "Make 8-bit SDR", "GPU conversion and HDR tone map*", color=OPENCL, size=22, body_size=16)
    chart.node(894, 724, 324, 80, "Make 8-bit SDR", "CPU filters and HDR tone map", color=CPU)
    chart.tag(640, 840, "Watermark the selected frames", color=MUTED, size=17)
    for x, title, subtitle, color in (
        (62, "Watermark luma", "CUDA kernels", CUDA),
        (478, "Watermark luma", "OpenCL kernels", OPENCL),
        (894, "Watermark luma", "Eigen and OpenMP", CPU),
    ):
        chart.node(x, 866, 324, 80, title, subtitle, color=color)
    chart.decision(640, 1078, 240, 116, ["Embed", "or detect"], color=OPENCL, size=21)
    chart.tag(835, 1064, "Detect", color=VIOLET)
    chart.node(916, 1040, 284, 76, "Correlation results", "8-bit SDR input, detection ends here", terminal=True, color=VIOLET, size=21, body_size=15)
    chart.tag(672, 1168, "Embed", color=CUDA)
    chart.tag(640, 1210, "Choose an encoder, keep the same backend and GPU", size=17)
    chart.node(62, 1230, 324, 88, "CPU or NVENC", "Software or NVIDIA hardware", color=CUDA)
    chart.node(478, 1230, 324, 88, "CPU, QSV, AMF or NVENC", "Hardware follows the GPU vendor", color=OPENCL, size=20, body_size=16)
    chart.node(894, 1230, 324, 88, "CPU encode", "Eigen uses software encoding", color=CPU)
    chart.node(420, 1384, 440, 80, "Write the output video", "FFmpeg joins video, audio and subtitles", terminal=True, color=OPENCL)
    chart.note(1502, "CPU encoding is the default, --hw_encoder enables GPU encoding. SDR keeps 4:2:0, 4:2:2 or 4:4:4 when supported.")
    chart.note(1534, "*GPU HDR supports limited-range BT.2020 PQ, 4:2:0, 10-bit input. Other HDR uses CPU filters.")
    chart.note(1566, "CPU SDR converts before upload, skipped frames still get needed conversion, HDR tone mapping outputs 8-bit 4:2:0.")
    chart.note(1598, "D3D11 frames use GPU copies and sharing for OpenCL. Watermarking changes luma (Y), the brightness plane.")
    return chart


def video_architecture():
    chart = Flowchart(
        1430, "Video threads and queues",
        "Video embedding uses a main thread and a separate encode and mux thread. "
        "The decode pump runs on the main thread, reads packets, sends them to "
        "FFmpeg and drains decoded frames. That thread also prepares frames and "
        "submits CUDA or OpenCL watermark work, or runs Eigen on the CPU. OpenCL "
        "hardware decoding holds two frames ahead and stages GPU copies. Owned "
        "video frames and audio or subtitle packets pass through one queue with "
        "a maximum of eight items. A full queue makes the main thread wait. The "
        "encode thread checks frame copies when needed, encodes video frames "
        "and writes all output packets. After the worker is joined, the main "
        "thread flushes the encoder and finishes the file. Detection accepts "
        "8-bit SDR input and returns results without starting the encode thread. "
        "FFmpeg and Eigen may also use CPU worker threads.",
        "Video architecture", "Two application threads during embedding, GPU engines can work on different frames",
    )
    chart.group(40, 188, 600, 780, "Main thread", LINE, note="Decode and watermark loop")
    chart.group(880, 188, 360, 780, "Encode thread", LINE, note="Embed only")
    chart.group(40, 1160, 1200, 190, "GPU work can overlap", LINE,
                note="Depends on the device, codec and copies")

    chart.link([(340, 306), (340, 362)])
    chart.link([(340, 448), (340, 510)])
    chart.link([(340, 596), (340, 654)])
    chart.link([(340, 740), (340, 798)])
    chart.link([(562, 841), (614, 841), (614, 724), (664, 724)], color=VIOLET)
    # Auxiliary packets bypass the watermark work and share the same queue.
    chart.link([(562, 405), (760, 405), (760, 660)], color=VIOLET)
    chart.link([(856, 724), (900, 724), (900, 414), (970, 414)], color=VIOLET)
    chart.link([(1060, 462), (1060, 534)])
    chart.link([(1150, 414), (1222, 414), (1222, 894), (1204, 894)], color=VIOLET)
    chart.link([(1060, 614), (1060, 682)])
    chart.link([(1060, 780), (1060, 854)])
    chart.link([(1060, 934), (1060, 1026)])
    chart.link([(118, 697), (72, 697), (72, 1000), (320, 1000), (320, 1026)], color=VIOLET)

    chart.node(158, 250, 364, 56, "Input video", terminal=True, color=OPENCL)
    chart.node(118, 362, 444, 86, "Decode pump", "Read packets, feed FFmpeg, drain frames")
    chart.node(118, 510, 444, 86, "Prepare the frame", "Convert to 8-bit SDR, start needed copies")
    chart.tag(340, 626, "OpenCL HW decode holds two frames ahead", color=OPENCL, size=16)
    chart.node(118, 654, 444, 86, "Watermark selected frames", "Submit CUDA or OpenCL work, or run Eigen")
    chart.tag(84, 682, "Detect", color=VIOLET, size=15)
    chart.tag(374, 776, "Embed", color=LINE, size=15)
    chart.node(118, 798, 444, 86, "Hand off an owned frame", "GPU surface or CPU buffer")
    chart.text(78, 940, "The main loop continues while the encoder works.", size=18, color=MUTED)

    # Stacked outlines distinguish this shared queue from a processing step.
    chart.rect(656, 652, 192, 128, fill=tint(VIOLET), stroke=VIOLET, radius=11,
               extra=' stroke-opacity="0.25"')
    chart.rect(660, 656, 192, 128, fill=tint(VIOLET), stroke=VIOLET, radius=11,
               extra=' stroke-opacity="0.45"')
    chart.node(664, 660, 192, 128, "Encode queue", "8 items max", color=VIOLET, size=19, body_size=16)
    chart.tag(760, 642, "Frames + packets", color=VIOLET, size=16)
    chart.tag(760, 814, "If full, main waits", color=VIOLET, size=16)
    chart.tag(760, 390, "Audio / subtitles", color=VIOLET, size=16)

    chart.decision(1060, 414, 180, 96, ["Frame or", "packet"], color=VIOLET, size=18)
    chart.tag(1094, 502, "Frame", size=16)
    chart.tag(1187, 398, "Packet", color=VIOLET, size=15)
    chart.node(916, 534, 288, 80, "Wait for frame copy", "Only if a copy is still pending", size=20, body_size=16)
    chart.node(916, 682, 288, 98, "Encode the frame", "CPU or GPU encoder", size=21)
    chart.node(916, 854, 288, 80, "Write output packets", "Video, audio and subtitles", size=20, body_size=16)
    chart.tag(1094, 994, "At end", size=16)
    chart.node(880, 1026, 360, 76, "Finish the file", "Drain queue, join, flush, write trailer", terminal=True, color=OPENCL, body_size=16)
    chart.node(40, 1026, 560, 76, "Detection results", "8-bit SDR input, no encode thread", terminal=True, color=VIOLET)

    for x, title, subtitle, color in (
        (76, "Decode a later frame", "NVDEC or D3D11VA / QSV", LINE),
        (476, "Watermark a frame", "CUDA or OpenCL kernels", LINE),
        (876, "Encode an earlier frame", "NVENC, QSV or AMF", LINE),
    ):
        chart.node(x, 1228, 328, 80, title, subtitle, color=color, size=21, body_size=16)
    chart.note(1384, "The two thread groups show application threads, FFmpeg codecs and Eigen may also use CPU workers.")
    chart.note(1414, "During the loop, only the encode thread writes output. Final flushing runs on the main thread after joining it.")
    return chart


def ui_benchmark():
    chart = Flowchart(
        1390, "The Qt benchmark",
        "The Qt Benchmark tab runs six images, four prediction windows and "
        "eleven PSNR levels, making 264 cases. Setup, loading, pixel preparation "
        "and previews are outside timing. Embedding and detection each warm up "
        "once and repeat until timing is stable or a time or run limit is reached. "
        "The UI shows each result, then moves to the next case. On completion, "
        "the score is round of ten times the square root of the product of the "
        "geometric-mean embed and detect FPS. Stopping the run produces no score.",
        "Qt benchmark tab", "Measure watermark speed on the selected CPU or GPU, then build one score",
    )
    chart.group(40, 592, 376, 176, "Timed embed", CUDA)
    chart.group(864, 592, 376, 176, "Timed detect", OPENCL)
    chart.link([(640, 218), (640, 266)])
    chart.link([(640, 346), (640, 432)])
    chart.link([(640, 520), (640, 558), (228, 558), (228, 650)], color=CUDA)
    chart.link([(392, 690), (478, 690)])
    chart.link([(802, 690), (888, 690)], color=OPENCL)
    chart.link([(1052, 730), (1052, 884), (860, 884)])
    chart.link([(640, 926), (640, 980)])
    chart.link([(520, 1038), (20, 1038), (20, 476), (430, 476)], color=CUDA)
    chart.link([(640, 1096), (640, 1162)], color=OPENCL)
    chart.node(470, 162, 340, 56, "Start benchmark", terminal=True, color=OPENCL)
    chart.node(430, 266, 420, 80, "Set up the test", "Prepare the selected CPU or GPU")
    chart.tag(640, 390, "6 images × 4 p values × 11 PSNR levels = 264 cases", size=19, color=OPENCL)
    chart.node(430, 432, 420, 88, "Prepare the next case", "Load if needed, set p and PSNR")
    chart.node(64, 650, 328, 80, "Embed the watermark", "Repeat, wait for GPU work, record FPS", color=CUDA, body_size=16)
    chart.node(478, 650, 324, 80, "Prepare detection pixels", "Convert once, outside the timers", size=21, body_size=16)
    chart.node(888, 650, 328, 80, "Detect the watermark", "Repeat, read correlation, record FPS", color=OPENCL, body_size=16)
    chart.tag(640, 632, "Not timed", size=16)
    chart.text(44, 802, "Warm up once outside the timer, then repeat each step until its timing is stable.", size=19, color=TEXT)
    chart.text(44, 830, "Target: 50 ms and timing variation below 10%, budget: 250 ms, up to 300 runs.", size=18, color=MUTED)
    chart.node(420, 846, 440, 80, "Show the preview and results", "Update progress, timing and correlation")
    chart.decision(640, 1038, 240, 116, ["More", "cases"], color=OPENCL, size=21)
    chart.tag(382, 1023, "Yes, next case", color=CUDA)
    chart.tag(676, 1134, "No", color=OPENCL)
    chart.node(390, 1162, 500, 106, "Final score", "round(10 × √(E × D))", terminal=True, color=CUDA, size=24, body_size=28)
    chart.note(1310, "E and D are the geometric mean embed and detect FPS across all cases.")
    chart.note(1340, "Loading and previews are outside the timers, slow cases can use one or two runs, stopping gives no score.")
    chart.note(1370, "Images: 18×18, 512×512, 480p, 720p, 1080p, 4K. p: 3, 5, 7, 9. PSNR: 10–60 dB, step 5.")
    return chart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check SVGs without writing")
    options = parser.parse_args()
    diagrams = {
        "image-pipeline.svg": image_pipeline,
        "video-pipeline.svg": video_pipeline,
        "video-architecture.svg": video_architecture,
        "ui-benchmark.svg": ui_benchmark,
    }
    stale = []
    if not options.check:
        OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, factory in diagrams.items():
        path = OUTPUT / name
        data = factory().svg()
        if options.check:
            if not path.exists() or path.read_text(encoding="utf-8") != data:
                stale.append(name)
        else:
            path.write_text(data, encoding="utf-8", newline="\n")
            print(f"Wrote {path.relative_to(ROOT).as_posix()}")
    if stale:
        parser.exit(1, "Outdated diagrams: " + ", ".join(stale) + "\nRun python tools/generate_diagrams.py\n")
    if options.check:
        print(f"All {len(diagrams)} architecture diagrams are current.")


if __name__ == "__main__":
    main()
