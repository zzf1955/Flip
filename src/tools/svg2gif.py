"""Convert animated SVG files to high-quality GIFs using Playwright.

Usage:
    python scripts/svg2gif.py                          # convert all SVGs in clawd dir
    python scripts/svg2gif.py path/to/file.svg         # convert single file
    python scripts/svg2gif.py --size 512 --fps 30      # custom size and fps
    python scripts/svg2gif.py --workers 8              # parallel conversion
"""

import asyncio
import argparse
import glob
import io
import math
import os
import re
from pathlib import Path
from PIL import Image
from playwright.async_api import async_playwright


def _parse_animation_shorthand(decl):
    """Parse a single CSS animation shorthand and return effective cycle (s).

    Handles: animation: name 2s ease-in-out infinite alternate;
    'alternate' doubles the effective period.
    """
    tokens = decl.strip().rstrip(';').split()
    duration = None
    is_alternate = False
    for tok in tokens:
        # duration: first token matching Ns or Nms
        if duration is None:
            m = re.match(r'^([\d.]+)(s|ms)$', tok)
            if m:
                val = float(m.group(1))
                duration = val / 1000 if m.group(2) == 'ms' else val
                continue
        if tok in ('alternate', 'alternate-reverse'):
            is_alternate = True
    if duration is None or duration < 0.01:
        return None
    return duration * 2 if is_alternate else duration


def detect_cycle_duration(svg_path, max_cap=10.0):
    """Parse SVG, compute each animation's effective cycle, return LCM.

    Handles CSS animation shorthand including 'alternate' direction.
    Returns duration in seconds for one complete loop of all animations.
    """
    with open(svg_path) as f:
        content = f.read()

    cycles = []

    # Match 'animation:' property values (possibly multi-line)
    for m in re.finditer(r'animation\s*:\s*([^;}]+)', content):
        val = m.group(1).strip()
        # Multiple animations separated by commas
        for part in val.split(','):
            c = _parse_animation_shorthand(part)
            if c is not None:
                cycles.append(c)

    # Also match longhand animation-duration + check for alternate
    for m in re.finditer(r'animation-duration\s*:\s*([\d.]+)(s|ms)', content):
        val = float(m.group(1))
        dur = val / 1000 if m.group(2) == 'ms' else val
        if dur > 0.01:
            cycles.append(dur)

    if not cycles:
        return 2.0

    # Deduplicate and filter very fast cycles (< 0.1s, e.g. typing jitter)
    # These don't affect visual loop point
    visual_cycles = sorted(set(round(c, 3) for c in cycles if c >= 0.1))
    if not visual_cycles:
        return 2.0

    # Compute LCM of effective cycles (in centiseconds for integer math)
    def lcm(a, b):
        return a * b // math.gcd(a, b)

    ints = [round(v * 100) for v in visual_cycles]
    ints = [x for x in ints if x > 0]

    result = ints[0]
    for x in ints[1:]:
        result = lcm(result, x)
        if result > max_cap * 100:
            # LCM too large, use max individual cycle
            result = max(ints)
            break

    dur = result / 100.0
    return min(dur, max_cap)


async def svg_to_gif(svg_path, output_path, size=500, fps=20,
                     duration_s=None, max_cap=10.0, browser=None):
    """Render animated SVG to GIF by capturing frames with headless browser."""
    svg_path = os.path.abspath(svg_path)

    if duration_s is None:
        duration_s = detect_cycle_duration(svg_path, max_cap)

    n_frames = max(int(duration_s * fps), 2)
    frame_delay_ms = 1000 / fps

    own_browser = browser is None
    p_ctx = None
    if own_browser:
        p_ctx = await async_playwright().start()
        browser = await p_ctx.chromium.launch(headless=True)

    page = await browser.new_page(viewport={"width": size, "height": size})

    with open(svg_path, "r") as f:
        svg_content = f.read()

    html = f"""<!DOCTYPE html>
<html><head><style>
  html, body {{ margin:0; padding:0; width:{size}px; height:{size}px;
               background: transparent; overflow: hidden;
               display: flex; align-items: center; justify-content: center; }}
  svg {{ width: {size}px; height: {size}px; }}
</style></head>
<body>{svg_content}</body></html>"""

    await page.set_content(html)
    await page.wait_for_timeout(300)

    frames = []
    for _ in range(n_frames):
        screenshot = await page.screenshot(
            type="png", omit_background=True,
            clip={"x": 0, "y": 0, "width": size, "height": size}
        )
        img = Image.open(io.BytesIO(screenshot)).convert("RGBA")
        frames.append(img)
        await page.wait_for_timeout(frame_delay_ms)

    await page.close()
    if own_browser:
        await browser.close()
        await p_ctx.stop()

    # High-quality GIF with transparency
    gif_frames = []
    for frame in frames:
        alpha = frame.split()[3]
        # Use median cut for better color quantization
        p_img = frame.convert("RGB").convert(
            "P", palette=Image.ADAPTIVE, colors=255)
        # Transparent pixels -> index 255
        mask = Image.eval(alpha, lambda a: 255 if a < 128 else 0)
        p_img.paste(255, mask)
        gif_frames.append(p_img)

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(frame_delay_ms),
        loop=0,
        optimize=False,
        transparency=255,
        disposal=2,
    )
    return len(gif_frames), duration_s


async def worker(queue, results, browser, size, fps, max_cap, outdir):
    """Worker that pulls SVG paths from queue and converts them."""
    while True:
        svg_path = await queue.get()
        if svg_path is None:
            queue.task_done()
            break
        name = Path(svg_path).stem
        out = os.path.join(outdir, f"{name}.gif")
        try:
            n, dur = await svg_to_gif(svg_path, out, size, fps,
                                      duration_s=None, max_cap=max_cap,
                                      browser=browser)
            sz = os.path.getsize(out) / 1024
            results.append((name, n, sz, dur, None))
            print(f"  \u2713 {name}.gif ({n} frames, {dur:.1f}s cycle, {sz:.0f}KB)")
        except Exception as e:
            results.append((name, 0, 0, 0, str(e)))
            print(f"  \u2717 {name}: {e}")
        queue.task_done()


async def main():
    parser = argparse.ArgumentParser(description="Convert animated SVGs to GIFs")
    parser.add_argument("input", nargs="?", help="SVG file or directory")
    parser.add_argument("--size", type=int, default=512,
                        help="Output size in px (default 512)")
    parser.add_argument("--fps", type=int, default=25,
                        help="Frames per second (default 25)")
    parser.add_argument("--max-duration", type=float, default=10.0,
                        help="Max capture duration in seconds (default 10)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Force fixed duration (overrides auto-detect)")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--workers", "-j", type=int, default=4,
                        help="Parallel workers (default 4)")
    args = parser.parse_args()

    if args.input is None:
        svg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "test_results", "clawd", "clawd-on-desk-svg")
        svgs = sorted(glob.glob(os.path.join(svg_dir, "*.svg")))
        outdir = args.outdir or os.path.join(os.path.dirname(svg_dir),
                                             "converted-gif")
    elif os.path.isdir(args.input):
        svgs = sorted(glob.glob(os.path.join(args.input, "*.svg")))
        outdir = args.outdir or os.path.join(args.input, "gif")
    else:
        svgs = [args.input]
        outdir = args.outdir or os.path.dirname(args.input)

    os.makedirs(outdir, exist_ok=True)
    n_workers = min(args.workers, len(svgs))

    if args.duration is not None:
        print(f"Converting {len(svgs)} SVGs \u2192 GIF "
              f"({args.size}px, {args.fps}fps, fixed {args.duration}s, "
              f"{n_workers} workers)")
    else:
        print(f"Converting {len(svgs)} SVGs \u2192 GIF "
              f"({args.size}px, {args.fps}fps, auto-detect cycle \u2264{args.max_duration}s, "
              f"{n_workers} workers)")

    async with async_playwright() as p:
        browsers = [await p.chromium.launch(headless=True)
                    for _ in range(n_workers)]

        queue = asyncio.Queue()
        results = []

        tasks = [asyncio.create_task(
            worker(queue, results, browsers[i],
                   args.size, args.fps, args.max_duration, outdir))
            for i in range(n_workers)]

        for svg in svgs:
            await queue.put(svg)
        for _ in range(n_workers):
            await queue.put(None)

        await asyncio.gather(*tasks)
        for b in browsers:
            await b.close()

    ok = sum(1 for r in results if r[4] is None)
    print(f"\nDone! {ok}/{len(svgs)} converted \u2192 {outdir}")


if __name__ == "__main__":
    asyncio.run(main())
