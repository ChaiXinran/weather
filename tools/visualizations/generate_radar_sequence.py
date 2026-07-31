"""Generate an interactive HTML fragment for BTH Radar sequence QA."""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openstl.datasets.dataloader_radar import BTHRadarDataset, FRAME_INTERVAL


SPLITS = {
    'Train random': ('2025-05-01', '2025-07-31'),
    'Val random': ('2025-08-01', '2025-08-15'),
    'Test random': ('2025-08-16', '2025-08-31'),
}
REF_BOUNDS = np.arange(0, 75, 5)
REF_COLORS = np.array([
    [0, 0, 246, 255], [1, 160, 246, 255], [0, 236, 236, 255],
    [1, 255, 0, 255], [0, 200, 0, 255], [1, 144, 0, 255],
    [255, 255, 0, 255], [231, 192, 0, 255], [255, 144, 0, 255],
    [255, 0, 0, 255], [214, 0, 0, 255], [192, 0, 0, 255],
    [255, 0, 240, 255], [120, 0, 132, 255], [173, 144, 240, 255],
], dtype=np.uint8)
AREA_THRESHOLDS = (20, 35, 45)


def _encode_png(display):
    buffer = io.BytesIO()
    display.save(buffer, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def _decode_dbz(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('L'), dtype=np.float32)
    return (255.0 - pixels) / 255.0 * 50.0


def _colorize_reflectivity(dbz):
    bins = np.clip(np.digitize(dbz, REF_BOUNDS, right=False) - 1,
                   0, len(REF_COLORS) - 1)
    rgba = REF_COLORS[bins].copy()
    rgba[dbz < 5, 3] = 0
    return Image.fromarray(rgba, mode='RGBA')


def _colorize_difference(delta):
    magnitude = np.clip(np.abs(delta) / 20.0, 0, 1)
    rgba = np.zeros((*delta.shape, 4), dtype=np.uint8)
    positive = delta > 0.5
    negative = delta < -0.5
    rgba[positive, 0] = 255
    rgba[positive, 1] = (180 * (1 - magnitude[positive])).astype(np.uint8)
    rgba[positive, 3] = (70 + 185 * magnitude[positive]).astype(np.uint8)
    rgba[negative, 2] = 255
    rgba[negative, 1] = (180 * (1 - magnitude[negative])).astype(np.uint8)
    rgba[negative, 3] = (70 + 185 * magnitude[negative]).astype(np.uint8)
    return Image.fromarray(rgba, mode='RGBA')


def _strong_area_score(dataset, index):
    timestamps = dataset.samples[index]
    score = 0
    for timestamp in timestamps:
        score += np.count_nonzero(_decode_dbz(dataset.frames[timestamp]) >= 35)
    return score


def _sample_payload(label, dataset, index):
    metadata = dataset.sample_metadata(index)
    start = dataset.samples[index][0]
    frames = []
    previous = None
    for offset, path in enumerate(metadata['files']):
        timestamp = start + offset * FRAME_INTERVAL
        lead = (offset - dataset.pre_seq_length + 1) * 6
        dbz = _decode_dbz(path)
        delta = np.zeros_like(dbz) if previous is None else dbz - previous
        frames.append({
            'src': _encode_png(_colorize_reflectivity(dbz)),
            'diffSrc': _encode_png(_colorize_difference(delta)),
            'time': timestamp.strftime('%Y-%m-%d %H:%M'),
            'role': 'Input' if offset < dataset.pre_seq_length else 'Target',
            'relative': f't{(offset - 9) * 6:+d} min'
            if offset < dataset.pre_seq_length else f'+{lead} min',
            'dbzMax': round(float(dbz.max()), 1),
            'areas': [
                round(float(np.count_nonzero(dbz >= threshold) / dbz.size * 100), 2)
                for threshold in AREA_THRESHOLDS
            ],
        })
        previous = dbz
    return {'label': label, 'frames': frames}


def build_payload(data_root, seed):
    rng = random.Random(seed)
    datasets = {
        label: BTHRadarDataset(data_root, start, end)
        for label, (start, end) in SPLITS.items()
    }
    payload = []
    for label, dataset in datasets.items():
        payload.append(_sample_payload(label, dataset, rng.randrange(len(dataset))))

    train = datasets['Train random']
    candidates = [round(i * (len(train) - 1) / 31) for i in range(32)]
    strongest = max(candidates, key=lambda index: _strong_area_score(train, index))
    payload.append(_sample_payload('Train ≥35 dBZ area candidate', train, strongest))
    return payload


def render_fragment(payload):
    data = json.dumps(payload, separators=(',', ':'))
    return f"""<div id="radar-sequence-qa">
  <div class="viz-controls">
    <label class="form-label">Sample
      <select id="radar-sample" class="form-select"></select>
    </label>
    <button id="radar-prev" type="button" class="btn">Previous</button>
    <button id="radar-next" type="button" class="btn btn-primary">Next</button>
    <label class="form-label">Layer
      <select id="radar-layer" class="form-select">
        <option value="ref">Reflectivity</option>
        <option value="diff">Change from previous frame</option>
      </select>
    </label>
  </div>
  <div class="viz-row text-small">
    <span id="radar-role" class="viz-badge"></span>
    <span id="radar-time"></span>
    <span id="radar-relative" class="text-muted"></span>
    <span id="radar-max"></span>
  </div>
  <label class="form-label" for="radar-frame">Frame <span id="radar-index"></span>/30</label>
  <input id="radar-frame" class="form-range" type="range" min="0" max="29" value="0">
  <div class="radar-stage">
    <div class="radar-image-wrap">
      <img id="radar-image" alt="Selected Radar frame">
      <span class="radar-row text-small">row 0</span>
      <span class="radar-col text-small">col 0</span>
    </div>
    <div id="radar-ref-scale" class="radar-discrete-scale text-small" aria-label="PyCINRAD reflectivity scale"></div>
    <div id="radar-diff-scale" class="radar-diff-scale text-small" hidden>
      <span>weakening −20</span><span class="radar-diff-gradient"></span><span>+20 dBZ strengthening</span>
    </div>
    <span class="text-small text-muted">Pixel orientation only; north/east direction is not yet verified.</span>
  </div>
  <svg id="radar-area-chart" class="radar-chart" viewBox="0 0 700 170" role="img"
       aria-label="Area fraction above 20, 35, and 45 dBZ over thirty frames"></svg>
  <div id="radar-strip" class="radar-strip" role="group" aria-label="Thirty-frame sequence"></div>
</div>
<style>
#radar-sequence-qa .radar-stage {{ display:grid; justify-items:center; gap:8px; margin:12px 0; }}
#radar-sequence-qa .radar-image-wrap {{ position:relative; width:min(100%,420px); }}
#radar-sequence-qa #radar-image {{ width:min(100%,420px); aspect-ratio:70/66; image-rendering:pixelated; }}
#radar-sequence-qa .radar-row {{ position:absolute; top:4px; left:6px; color:var(--foreground); background:var(--background); }}
#radar-sequence-qa .radar-col {{ position:absolute; bottom:4px; left:6px; color:var(--foreground); background:var(--background); }}
#radar-sequence-qa .radar-discrete-scale {{ display:flex; flex-wrap:wrap; justify-content:center; gap:2px; }}
#radar-sequence-qa .radar-swatch {{ display:grid; place-items:center; min-width:38px; height:24px; color:#111; }}
#radar-sequence-qa .radar-diff-scale {{ display:grid; grid-template-columns:auto minmax(100px,220px) auto; gap:8px; align-items:center; }}
#radar-sequence-qa [hidden] {{ display:none; }}
#radar-sequence-qa .radar-diff-gradient {{ height:12px; background:linear-gradient(90deg,#0000ff,transparent,#ff0000); border:1px solid var(--border); }}
#radar-sequence-qa .radar-chart {{ width:100%; height:auto; margin:8px 0 12px; }}
#radar-sequence-qa .radar-chart text {{ fill:var(--muted-foreground); }}
#radar-sequence-qa .radar-chart .axis {{ stroke:var(--border); stroke-width:1; }}
#radar-sequence-qa .radar-chart .split {{ stroke:var(--foreground); stroke-width:1; stroke-dasharray:4 3; }}
#radar-sequence-qa .radar-chart .line-0 {{ fill:none; stroke:var(--viz-series-1); stroke-width:2; }}
#radar-sequence-qa .radar-chart .line-1 {{ fill:none; stroke:var(--viz-series-2); stroke-width:2; }}
#radar-sequence-qa .radar-chart .line-2 {{ fill:none; stroke:var(--viz-series-3); stroke-width:2; }}
#radar-sequence-qa .radar-strip {{ display:grid; grid-template-columns:repeat(10,minmax(0,1fr)); gap:4px; }}
#radar-sequence-qa .radar-thumb {{ padding:2px; background:transparent; border:1px solid var(--border); color:var(--foreground); }}
#radar-sequence-qa .radar-thumb[aria-pressed="true"] {{ border-color:var(--primary); box-shadow:0 0 0 1px var(--primary); }}
#radar-sequence-qa .radar-thumb:nth-child(10) {{ margin-right:5px; }}
#radar-sequence-qa .radar-thumb img {{ display:block; width:100%; aspect-ratio:70/66; image-rendering:pixelated; }}
#radar-sequence-qa .radar-thumb span {{ display:block; }}
@media (max-width:520px) {{
  #radar-sequence-qa .radar-strip {{ grid-template-columns:repeat(5,minmax(0,1fr)); }}
  #radar-sequence-qa .radar-thumb:nth-child(10) {{ margin-right:0; }}
}}
</style>
<script>
(() => {{
  const root = document.getElementById('radar-sequence-qa');
  const samples = {data};
  const select = root.querySelector('#radar-sample');
  const range = root.querySelector('#radar-frame');
  const layer = root.querySelector('#radar-layer');
  const image = root.querySelector('#radar-image');
  const strip = root.querySelector('#radar-strip');
  let sampleIndex = 0;
  let frameIndex = 0;
  const refColors = {json.dumps([f'rgba({r},{g},{b},{a / 255:.2f})' for r, g, b, a in REF_COLORS[:11]])};
  const refBounds = {json.dumps(REF_BOUNDS[:11].tolist())};
  samples.forEach((sample, index) => {{
    const option = document.createElement('option');
    option.value = index;
    option.textContent = sample.label;
    select.appendChild(option);
  }});
  const scale = root.querySelector('#radar-ref-scale');
  refColors.forEach((color, index) => {{
    const swatch = document.createElement('span');
    swatch.className = 'radar-swatch';
    swatch.style.background = color;
    swatch.textContent = `${{refBounds[index]}}`;
    scale.appendChild(swatch);
  }});
  function drawChart() {{
    const svg = root.querySelector('#radar-area-chart');
    const frames = samples[sampleIndex].frames;
    const width = 700, height = 170, left = 42, right = 12, top = 18, bottom = 28;
    const plotW = width-left-right, plotH = height-top-bottom;
    const maximum = Math.max(1, ...frames.flatMap(frame => frame.areas));
    const x = index => left + index / 29 * plotW;
    const y = value => top + plotH - value / maximum * plotH;
    const path = series => frames.map((frame,index) =>
      `${{index ? 'L' : 'M'}}${{x(index).toFixed(1)}},${{y(frame.areas[series]).toFixed(1)}}`).join(' ');
    svg.innerHTML = `
      <line class="axis" x1="${{left}}" y1="${{top+plotH}}" x2="${{width-right}}" y2="${{top+plotH}}"/>
      <line class="axis" x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{top+plotH}}"/>
      <line class="split" x1="${{x(9.5)}}" y1="${{top}}" x2="${{x(9.5)}}" y2="${{top+plotH}}"/>
      <path class="line-0" d="${{path(0)}}"/><path class="line-1" d="${{path(1)}}"/>
      <path class="line-2" d="${{path(2)}}"/>
      <text x="${{left}}" y="12">Area fraction (%) · max ${{maximum.toFixed(1)}}</text>
      <text x="${{left}}" y="${{height-7}}">Input</text>
      <text x="${{x(10)}}" y="${{height-7}}">Target</text>
      <text x="${{width-230}}" y="12">20 dBZ · 35 dBZ · 45 dBZ</text>`;
  }}
  function rebuildStrip() {{
    strip.replaceChildren();
    samples[sampleIndex].frames.forEach((frame, index) => {{
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'radar-thumb';
      button.setAttribute('aria-label', `${{frame.role}} ${{frame.relative}}, ${{frame.time}}`);
      const thumb = document.createElement('img');
      thumb.src = frame.src;
      thumb.alt = '';
      const label = document.createElement('span');
      label.className = 'text-small';
      label.textContent = index < 10 ? `I${{index + 1}}` : `T${{index - 9}}`;
      button.append(thumb, label);
      button.addEventListener('click', () => {{ frameIndex = index; update(); }});
      strip.appendChild(button);
    }});
  }}
  function update() {{
    const frame = samples[sampleIndex].frames[frameIndex];
    image.src = layer.value === 'diff' ? frame.diffSrc : frame.src;
    image.alt = `${{frame.role}} Radar frame at ${{frame.time}}`;
    range.value = frameIndex;
    root.querySelector('#radar-index').textContent = frameIndex + 1;
    root.querySelector('#radar-role').textContent = frame.role;
    root.querySelector('#radar-time').textContent = frame.time;
    root.querySelector('#radar-relative').textContent = frame.relative;
    root.querySelector('#radar-max').textContent = `Max ${{frame.dbzMax}} dBZ`;
    [...strip.children].forEach((item, index) =>
      item.setAttribute('aria-pressed', String(index === frameIndex)));
  }}
  select.addEventListener('change', () => {{
    sampleIndex = Number(select.value); frameIndex = 0; rebuildStrip(); drawChart(); update();
  }});
  layer.addEventListener('change', () => {{
    root.querySelector('#radar-ref-scale').hidden = layer.value === 'diff';
    root.querySelector('#radar-diff-scale').hidden = layer.value !== 'diff';
    update();
  }});
  range.addEventListener('input', () => {{ frameIndex = Number(range.value); update(); }});
  root.querySelector('#radar-prev').addEventListener('click', () => {{
    frameIndex = (frameIndex + 29) % 30; update();
  }});
  root.querySelector('#radar-next').addEventListener('click', () => {{
    frameIndex = (frameIndex + 1) % 30; update();
  }});
  rebuildStrip();
  drawChart();
  update();
}})();
</script>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20250729)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_fragment(build_payload(args.data_root, args.seed)), encoding='utf-8')


if __name__ == '__main__':
    main()
