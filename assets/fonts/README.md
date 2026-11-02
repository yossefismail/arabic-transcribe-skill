# Fonts

Inlined as base64 `@font-face` sources by `../scripts/apply_ui.py`, so transcripts
stay offline-capable and self-contained.

| File | Family | Weights | Upstream | Licence |
|---|---|---|---|---|
| `archivo-latin-wght.woff2` | Archivo (variable, latin) | 100–900 | [Omnibus-Type/Archivo](https://github.com/Omnibus-Type/Archivo) | SIL OFL 1.1 |
| `jetbrains-mono-latin-wght.woff2` | JetBrains Mono (variable, latin) | 100–800 | [JetBrains/JetBrainsMono](https://github.com/JetBrains/JetBrainsMono) | SIL OFL 1.1 |
| `cairo-arabic.woff2` | Cairo (variable, arabic subset) | 200–1000 | [googlefonts/cairo](https://github.com/Gue3bara/Cairo) | SIL OFL 1.1 |
| `cairo-latin.woff2` | Cairo (variable, latin subset) | 200–1000 | [googlefonts/cairo](https://github.com/Gue3bara/Cairo) | SIL OFL 1.1 |

The two Cairo files carry the `unicode-range` split Google Fonts serves; keep
those ranges when embedding or the arabic subset will shadow the latin one.

Roles: Archivo = display/Latin, Cairo = Arabic, JetBrains Mono = technical
labels and timestamps. See `../references/ui-system.md` §3.

## Licence

All four families are licensed under the SIL Open Font License 1.1. The full
licence text and the copyright notices for every family are in
[`OFL.txt`](OFL.txt) — it must accompany these files wherever they are
redistributed, including inside generated transcripts.
