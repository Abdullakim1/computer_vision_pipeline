"""One-command portfolio generation for CineForge.

Renders a curated set of clips spanning every procedural theme, exports
MP4 / GIF / poster assets, builds an analysis report, and produces a
brand-new self-contained ``index.html`` gallery with embedded base64 previews
so the portfolio can be opened locally or shared as a single file.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2

from . import metrics


# Curated order + a short cinematic prompt per theme (kept tasteful and short
# so at-a-glance the generator reads like a production shot list).
_THEME_SHOTS = [
    ("aurora",   "Aurora borealis ripples over a snowy valley, teal and violet curtains of light, orbit camera"),
    ("sunset",   "Warm sunset over a quiet horizon, amber and rose clouds, slow pan right"),
    ("golden",   "Golden hour light washing over a meadow, soft morning glow, gentle dolly in"),
    ("neon",     "Neon-lit cyberpunk street at night, magenta and cyan signs, tracking shot"),
    ("ocean",    "Calm ocean waves rolling under a low sun, deep blue water, slow zoom in"),
    ("mountains", "Bare granite peaks under a crisp sky, faint mist in the valleys, majestic crane up"),
    ("mono",     "Black-and-white winter landscape, drifting snow, monochrome film grain, pull-back dolly"),
    ("campfire", "A lone campfire flickering in the night, warm embers drifting up, subtle orbit"),
    ("cyber",    "Holographic grid dissolving into the dark, streams of glowing data, dutch tilt"),
    ("moody",    "Heavy fog rolling through a dark forest, muted tones, slow handheld drift"),
    ("meadow",   "Endless green meadow in spring, pollen drifting in golden light, smooth pan"),
    ("storm",    "Storm clouds gathering with distant lightning, rain beginning to fall, dramatic dutch tilt"),
]

# A representative camera move to showcase variety across the set.
_MOTIONS = ["orbit", "pan_right", "dolly_in", "tracking", "crane_up", "pan_left",
            "dolly_out", "orbit", "dutch_tilt", "handheld", "pan_left", "crane_down"]


def _banner():
    print("=" * 70)
    print("  CINEFORGE — PORTFOLIO GENERATOR")
    print("=" * 70)


def _export_clip(clip, theme, outdir, gif_scale=0.4, gif_fps=8):
    """Write mp4 + preview gif + poster for one theme; return asset paths."""
    outdir = Path(outdir)
    mp4 = clip.write_video(outdir / f"{theme}.mp4")
    gif = clip.to_gif(str(outdir / f"{theme}.gif"), fps=gif_fps, scale=gif_scale)
    import numpy as np
    poster = np.ascontiguousarray(clip.frames[clip.T // 2][:, :, ::-1])
    poster_path = outdir / f"{theme}.jpg"
    import cv2
    cv2.imwrite(str(poster_path), poster)
    return {"theme": theme, "mp4": mp4, "gif": gif, "poster": str(poster_path)}


def _data_uri(path: str, mime: str = "image/gif") -> str:
    """Return a base64 data URI of a file so an HTML page is fully self-contained."""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _gallery_html(rows, outdir, generated_stats):
    """Render a branded, self-contained gallery from portfolio entries."""
    cards = []
    for row in rows:
        gif_uri = _data_uri(row["gif"], "image/gif")
        poster_uri = _data_uri(row["poster"], "image/jpeg")
        cards.append("""
      <div class="card">
        <div class="media"><img src="{gif}" alt="{theme}" loading="lazy"
                                onerror="this.src='{poster}'"></div>
        <div class="meta">
          <h3>{title}</h3>
          <span class="tag">{motion} move</span>
          <ul>
            <li>Quality&nbsp;<b>{q:.2f}</b></li>
            <li>Sharpness&nbsp;<b>{s:.0f}</b></li>
          </ul>
        </div>
      </div>""".format(
            gif=gif_uri, theme=row["theme"], poster=poster_uri,
            title=row["theme"].title(), motion=row.get("motion", "cinematic"),
            q=row.get("perceptual_quality", 0.0), s=row.get("avg_sharpness", 0.0),
        ))

    page = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CineForge - Procedural Video Portfolio</title>
<style>
  :root { --bg:#0b0e14; --panel:#141926; --panel2:#1b2233; --fg:#e8ecf4;
           --mut:#9aa3b5; --acc:#7c6cf0; --acc2:#2ec4b6; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:'Inter',-apple-system,Segoe UI,Roboto,sans-serif;
          background:radial-gradient(1200px 500px at 50% -10%,#1b2340 0%,var(--bg) 55%);
          color:var(--fg); }
  header { text-align:center; padding:3rem 1rem 1.6rem; }
  header h1 { font-size:2.5rem; margin:0; letter-spacing:-.04em;
               background:linear-gradient(90deg,var(--acc),var(--acc2));
               -webkit-background-clip:text; background-clip:text; color:transparent; }
  header p { color:var(--mut); max-width:720px; margin:.8rem auto 0; }
  .stats { display:flex; gap:1.2rem; justify-content:center; flex-wrap:wrap; margin:2rem auto; }
  .stat { background:var(--panel); border:1px solid var(--panel2); border-radius:14px;
            padding:.7rem 1.4rem; min-width:120px; }
  .stat b { display:block; font-size:1.5rem; color:var(--acc2); }
  .stat span { color:var(--mut); font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
           gap:1.4rem; padding:1rem auto; }
  .card { background:var(--panel); border:1px solid var(--panel2); border-radius:16px;
           overflow:hidden; }
  .card:hover { transform:translateY(-4px); }
  .media { aspect-ratio:16/9; background:#000; }
  .media img { width:100%; height:100%; object-fit:cover; display:block; }
  .meta { padding:.9rem 1rem 1rem; }
  .meta h3 { margin:0 0 .3rem; font-size:1.05rem; }
  .tag { display:inline-block; background:var(--panel2); color:var(--mut);
          border-radius:999px; padding:.15rem .6rem; font-size:.72rem; }
  ul { list-style:none; margin:.7rem 0 0; padding:0; display:flex; gap:1rem;
        color:var(--mut); font-size:.82rem; }
  ul b { color:var(--fg); }
  footer { text-align:center; color:var(--mut); font-size:.78rem; padding:1rem 0 2.5rem;
            word-break:break-all; }
</style>
</head>
<body>
<header>
  <h1>CineForge &middot; Procedural Cinema</h1>
  <p>A self-contained AI/ML video-generation showcase. Every clip is rendered
     programmatically by a real-time procedural cinematography engine &mdash; no footage,
     no GPUs required on the showcase machine.</p>
</header>
<div class="stats">
  <div class="stat"><b>{themes}</b><span>Themes</span></div>
  <div class="stat"><b>{clips}</b><span>Clips</span></div>
  <div class="stat"><b>{frames}</b><span>Frames</span></div>
  <div class="stat"><b>{seconds:.1f}s</b><span>Footage</span></div>
</div>
<div class="grid">
{cards}
</div>
<footer>Generated at {outdir} &middot; CineForge</footer>
</body>
</html>"""
    page = (
        page.replace("{themes}", str(generated_stats["themes"]))
            .replace("{clips}", str(generated_stats["clips"]))
            .replace("{frames}", str(generated_stats["frames"]))
            .replace("{seconds:.1f}", "%.1f" % generated_stats["seconds"])
            .replace("{cards}", "\n".join(cards))
            .replace("{outdir}", str(Path(outdir).resolve()))
    )
    out = Path(outdir) / "index.html"
    out.write_text(page, encoding="utf-8")
    return str(out)


def build_portfolio(studio, outdir="media/portfolio", themes=None,
                    width=480, height=270, fps=15, duration=2.2,
                    seed=20240501):
    """Generate a full portfolio; returns a summary dict."""
    _banner()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    selected = _THEME_SHOTS if not themes else [t for t in _THEME_SHOTS if t[0] in themes]
    if not selected:
        raise ValueError("no valid themes selected from %r" % (themes,))

    rows = []
    total_frames = 0
    start = time.time()
    for i, (theme, prompt) in enumerate(selected):
        motion = _MOTIONS[i % len(_MOTIONS)]
        print("  [%d/%d] theme=%-10s motion=%-10s ..." % (i + 1, len(selected), theme, motion),
              end=" ", flush=True)
        t0 = time.time()
        clip = studio.text_to_video(
            prompt, backend="cinematic", width=width, height=height,
            fps=fps, duration=duration, seed=seed + i,
            extras={"motion": motion, "style": theme},
        )
        asset = _export_clip(clip, theme, out, gif_scale=0.4, gif_fps=8)
        asset["motion"] = motion
        rep = metrics.analyze(clip.frames, float(fps))
        asset["perceptual_quality"] = rep["metrics"]["perceptual_quality"]
        asset["avg_sharpness"] = rep["metrics"]["avg_sharpness"]
        asset["avg_colorfulness"] = rep["metrics"]["avg_colorfulness"]
        asset["frames"] = clip.T
        total_frames += clip.T
        rows.append(asset)
        print("ok (%.1fs)" % (time.time() - t0,))

    report = out / "metrics_report.txt"
    with open(report, "w") as fh:
        fh.write("CineForge Portfolio - Quality Analysis\n" + "=" * 46 + "\n\n")
        for row in rows:
            fh.write("%s  (%s move)\n" % (row["theme"].upper(), row["motion"]))
            fh.write("  perceptual_quality : %.3f\n" % row["perceptual_quality"])
            fh.write("  sharpness          : %.2f\n" % row["avg_sharpness"])
            fh.write("  colorfulness       : %.2f\n\n" % row["avg_colorfulness"])

    secs = time.time() - start
    gallery = _gallery_html(rows, out, {
        "themes": len(rows), "clips": len(rows),
        "frames": total_frames, "seconds": total_frames / float(fps),
    })

    print("\n" + "=" * 70)
    print("  Portfolio built in %.1fs at %s" % (secs, out.resolve()))
    print("  - %d clips, %d frames @ %dfps" % (len(rows), total_frames, fps))
    print("  - report       : %s" % report)
    print("  - gallery page : %s" % gallery)
    print("=" * 70)
    return {"clips": rows, "report": str(report), "gallery": gallery}