# UI system — "drafting sheet"

The house UI for `arabic-transcribe` players. It replaces the pipeline's default
rounded-card look with a technical drawing-sheet language borrowed from
[final-final-v1.com](https://final-final-v1.com/) (Swiss/Basel studio wall), then
adapted for a right-to-left Arabic transcript and a long recording.

Applied by `scripts/apply_ui.py`; it is a retrofit, so it works on both a fresh
pipeline output and an existing transcript HTML.

---

## 1. What was borrowed from the reference

| Reference trait | How it is used here |
|---|---|
| Warm paper `#eceae4` with a 40 px square grid | `body` background (two linear-gradients, 5.5 % ink) |
| Near-black ink instead of pure black | `--ink #141414`, `--body #3a3833` |
| One vermillion accent, used sparingly | `--accent #c8321e` for eyebrows, playhead, active clause, key-moment tick |
| Hard 1–2 px ink rules, no rounded corners | every card, table, section break and the sticky player |
| Mono technical labels, uppercase, `.08em` tracking | `font-family: JetBrains Mono`; `RECORDING`, `PROJECT`, `W01`, `02 / Summary` |
| Title block with label/value cells | `.titleblock` (brand + Project / Date / Speech / Clauses) |
| Enormous black display headline with tight leading | Arabic 900 headline, `line-height .94`, no tracking |
| Bordered fact list | `.facts` (Source / Model / Speakers / Windows) |
| Numbered sections (`01 / Write`, `02 / Elevation`) | `01 / Recording`, `02 / Summary`, `03 / Transcription` |
| Line-draw motion (logo strokes, rules) | SVG mark draw-in, section rules, hero rise |
| Focus rings + 44 px targets | `:focus-visible` 2 px ink/accent; custom player controls |
| Drafting-sheet metadata (project, scale, site) | recording metadata, "RECORDING 04 — …" eyebrow |

Nothing from the reference's content (letters, emoji rating, forms) is reused —
only the visual system.

## 2. Tokens

```css
--paper:#eceae4; --sheet:#f7f6f2; --ink:#141414; --body:#3a3833;
--muted:#5e5b55; --rule:#8a867d;
--accent:#c8321e; --accent-dark:#a02614;
--hair:rgba(20,20,20,.24); --hair-soft:rgba(20,20,20,.12); --grid:rgba(20,20,20,.055);
--sans:"Archivo","Cairo","Helvetica Neue",Helvetica,Arial,sans-serif;
--mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Consolas,monospace;
--gut:clamp(20px,4vw,40px); --maxw:1180px;
```

Speaker colours (kept from the transcript's diarization, deepened to sit on paper):

| Speaker | Colour |
|---|---|
| A | `#2f5d3a` |
| B | `#2c5a7d` |
| C | `#964a1f` |
| D | `#5f4376` |
| E | `#6f5c1c` |

## 3. Typography

- **Archivo** (variable, latin) — display + Latin words inside Arabic text.
- **Cairo** (variable, arabic + latin subsets) — Arabic headings and body.
- **JetBrains Mono** (variable, latin) — every technical label, timestamp, count,
  window number and metadata value.

The font stack is `Archivo, Cairo, …`: Latin glyphs resolve to Archivo, Arabic
glyphs fall through to Cairo. All three are shipped in `assets/fonts/` (OFL) and
inlined as base64 by `apply_ui.py`, so the HTML stays offline-capable and
self-contained. `--no-fonts` skips the embedding (system fallbacks only).

Scale:

| Role | Size |
|---|---|
| Hero headline | `clamp(44px, 8.6vw, 118px)`, weight 900, line-height .94 |
| Section heading | `clamp(30px, 4.6vw, 52px)`, weight 900 |
| Summary heading | `clamp(24px, 3vw, 34px)`, weight 900 |
| Summary / checklist body | `clamp(16px, 1.15vw, 17.5px)`, line-height 2.05 |
| Transcript body | `clamp(16px, 1.3vw, 18px)`, line-height 2.1 |
| Technical label | 11 px mono, uppercase, `.08em` |
| Window gutter / counts | 10.5 px mono, `.04em` |

**Arabic rules:** never apply `letter-spacing` to Arabic (it breaks the joins) —
keep tracking on Latin/mono labels only. Arabic needs a taller line box than
Latin: use 2.0–2.1.

## 4. Structure

```
body (dir=rtl, 40px grid)
├─ header.titleblock      brand + PROJECT / DATE / SPEECH / CLAUSES cells
├─ section.hero           eyebrow · h1 · lede | dl.facts
├─ section.playerbar      sticky; 01 / RECORDING + transport + rail + now-line
├─ section.summary        sheet card; 02 / SUMMARY, above the transcript
├─ section.sec-head       03 / TRANSCRIPTION + "التفريغ."
├─ main                   .win rows: counter gutter (W001…) + .txt
└─ footer.colophon        counts + model + shortcuts
```

The chrome (`titleblock` … `playerbar`) is rebuilt from `<!-- ui:chrome:start -->`
… `<!-- ui:chrome:end -->`, the style block is owned by `<style id="ui-theme">`,
the player by `<script id="ui-player-js">`. Re-running replaces each in place.

Section numbering is fixed: the summary sits above `<main>`, so the labels are
always `01 Recording / 02 Summary / 03 Transcription`. `body.summary-first` is
what flips the summary's number from its `03` fallback to `02`, and it is set by
`apply_ui.py` whenever the summary block precedes the transcript.

## 5. Key-moments rail

The "recording line" is a custom timeline, not the native control:

- `div.rail[role=slider]` — 46 px hit area, 2 px ink line, accent progress fill
  and a 15 px playhead.
- one `button.mk` per key moment, positioned with `--p: <percent>%` (recomputed
  from `audio.duration` on `loadedmetadata`), rendered as a 2 px tick with an
  8–9 px mono index above it.
- labels that would collide (< 3.2 % apart) drop to a second row (`.mk.low`)
  instead of overlapping; numbers are hidden below 720 px.
- `.nowline` under the rail always names the current moment:
  `Moment 05 · هامش الربح والتشغيل · 44:31`.
- states: upcoming = ink tick + muted index; past = accent tick; current =
  taller accent tick + accent index.

Interaction: click a tick to seek (and start playing if paused); click/drag the
rail to scrub; `←/→` ±5 s, `Shift+←/→` ±30 s, `PageUp/Down` ±60 s, `Home/End`;
`space` toggles playback when focus is not in a form control. A `1×` button
cycles 1 → 1.25 → 1.5 → 0.75. Transport and rail are `dir="ltr"` even in the
RTL page: time runs left → right, which is what people expect from a player.

Moment data:

```json
[{"t": "44:31", "label": "هامش الربح والتشغيل"}]
```

`t` may be seconds or `mm:ss`/`hh:mm:ss`. If `--moments` is omitted,
`apply_ui.py` derives moments from the summary markdown's section headings —
`**عنوان (12:30–20:00):**` becomes `{t: 750, label: "عنوان"}` — and skips
overview/action headings. Keep labels ≤ ~40 characters; the now-line truncates.

## 6. Components

| Class | Look |
|---|---|
| `.titleblock`, `.tb-brand`, `.tb-cells`, `.tb-cell` | hard-ruled label/value cells |
| `.hero`, `.hero-kicker`, `.facts` | oversized headline + bordered fact list |
| `.playerbar`, `.transport`, `.pp`, `.time`, `.rail`, `.mk`, `.nowline` | sticky player + moments rail |
| `.summary`, `.chk` | sheet card, square custom checkboxes (accent fill + white tick) |
| `.win`, `.meta`, `.txt`, `.chunk` | transcript index rows, `W###` counter, speaker-coloured underlines |
| `.colophon` | mono footer |

Chunk / window states: `.chunk:hover` = 6 % ink wash; `.chunk.on` = 11 % accent
wash + 2 px accent underline (speaker colour stays on `border-bottom`);
`.win.on` = 3.5 % ink wash + 2 px accent bar on the inline-start edge.

## 7. Motion

- Hero blocks rise 16 px, 700 ms, staggered (`--d`), `cubic-bezier(.2,.7,.2,1)`.
- Logo strokes draw with `stroke-dashoffset`, segment by segment.
- Rail fill/head are driven by `timeupdate` with no transition (scrubbing must
  feel exact); tick height/colour transitions are 160 ms.
- `prefers-reduced-motion: reduce` disables all of it, and the transcript stops
  smooth-scrolling (`scrollIntoView` uses `auto`).

## 8. Accessibility

- All targets ≥ 44 px: play/pause 46 px, ticks 26×46 px, summary checkboxes are
  full-row labels.
- Visible `:focus-visible` rings (ink on paper, accent on ink).
- Rail is `role="slider"` with `aria-valuemin/max/now/valuetext` kept in sync and
  full keyboard control.
- Moment buttons carry `aria-label="05 · هامش الربح والتشغيل — 44:31"`.
- Play button flips `aria-label` Play/Pause and `aria-pressed`.
- Contrast: ink/paper 14.6:1, body/paper 9.9:1, muted/paper 5.6:1, accent/paper
  4.9:1; speaker colours are decorative (meaning is also in the legend).
- The transcript body stays plain text; clauses are mouse conveniences, not tab
  stops (3 000+ tab stops would be hostile), while the rail and summary are the
  keyboard paths.

## 9. Do / don't

- **Do** keep the accent for one job at a time (playhead, active clause, key
  moment). Three red things on screen is already the limit.
- **Do** keep gutters and rules physical (left/right) for the sheet; use logical
  properties only for text-side insets (`inset-inline-start`).
- **Don't** round corners, add drop shadows, gradients (other than the paper
  grid), or emoji icons.
- **Don't** letter-space Arabic or set Arabic below 16 px.
- **Don't** put the native `<audio controls>` back on top of the custom rail —
  two progress bars confuse scrubbing.
- **Don't** hand-edit the CSS in a delivered file; change `apply_ui.py`, then
  re-run it (the style block is regenerated wholesale).
