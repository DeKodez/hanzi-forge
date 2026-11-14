# Hanzi Forge

Hanzi Forge turns an English phrase into one newly composed Chinese-character
form, renders it as an ink-on-paper image, and shows the closest standard
characters for comparison.

```text
"the feeling of coming home"
  -> closest characters: 归 (return), 家 (home), 安 (peace)
  -> components: 宀 (roof) + 回 (return)
  -> composition: ⿱ top-bottom
  -> output: 1024x1024 PNG
```

The generated glyph is not a standard Chinese character. The app also has a
mode that renders the closest real characters directly.

## Requirements

- Python 3.10 or newer (tested with 3.12)
- CPU only; no GPU or model training
- About 700 MB for the embedding model, fonts, and dictionaries
- Network access for the one-time asset download and index build

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_assets.py
python scripts/build_index.py
```

The first index build embeds 9,480 character glosses and takes roughly 15
minutes on an Apple Silicon CPU. Downloaded assets go to `assets/`; the built
index goes to `data/`. Both directories are gitignored.

The optional language-model mode is configured from a local `.env` file:

```bash
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env` if you want to use it. The app runs without a
key and without network access once the assets and index exist.

## Run

```bash
python app.py
# open http://127.0.0.1:7860
```

To listen on another interface or port:

```bash
FORGE_HOST=0.0.0.0 FORGE_PORT=7860 python app.py
```

Command line:

```bash
python cli.py "a fire that never goes out"
python cli.py "digital loneliness" -n 6 --style pen
python cli.py "the feeling of coming home" --mode real
python cli.py "quiet strength" --json
```

`--mode new` composes a character. `--mode real` renders the closest standard
characters. `--style brush|pen` selects Ma Shan Zheng or LXGW WenKai. `--seal`
adds the optional red seal. Images are written to `outputs/`; generated files
are gitignored.

## Optional language-model mode

The deterministic pipeline works without a key. With `DEEPSEEK_API_KEY` set,
the app uses the language model in stages: concept analysis, composition
planning, and final design. Plans cover repetition, semantic-plus-phonetic,
ideographic, and metaphorical approaches. Every plan passes the same novelty, renderability,
position, and redundancy checks as the deterministic pipeline before rendering.

```bash
python app.py                    # use the LLM checkbox in the UI
python cli.py "digital loneliness" --llm
```

Configuration:

```dotenv
DEEPSEEK_API_KEY=<your-key>
FORGE_LLM_MODEL=deepseek-flash
FORGE_LLM_BASE_URL=https://api.deepseek.com
FORGE_LLM_MAX_TOKENS=8192
FORGE_LLM_TIMEOUT=120
FORGE_LLM_REASONING=none
```

`FORGE_LLM_REASONING=none` uses the fast path. `low`, `medium`, or `high`
enable slower reasoning. Designs are cached in `data/llm_cache.json`. The cache
is not committed. To exercise the LLM path without a key, run:

```bash
python scripts/test_llm_mock.py
```

## How a character is built

1. The phrase is embedded with `BAAI/bge-base-en-v1.5` and matched against
   glosses for 9,480 characters. A frequency prior keeps rare dictionary-only
   characters from outranking common ones.
2. CC-CEDICT entries add keyword hints for modern or abstract vocabulary.
3. Retrieved characters vote for semantic components and phonetics. Components
   are placed with conventional IDS structures and positional variants such as
   水 → 氵, 心 → 忄, and 火 → 灬.
4. The engine checks the resulting decomposition against the CJKVI/CHISE IDS
   database so the result is not an existing character.
5. The mask is rendered with open-licensed calligraphy fonts on procedural
   rice paper, with an optional seal.

Two-part phono-semantic compounds are the default. Ideographic compounds may
use two to four components. The composer scores three- and four-part templates
for stroke-area balance, coverage, and stretch instead of taking the first
valid arrangement. The roof cap for 宀 adapts to the component below it, and
single-sided short components such as 灬 keep their natural height.

The IDS layer supports the standard operators ⿰ ⿱ ⿲ ⿳ ⿴ ⿵ ⿶ ⿷ ⿸ ⿹ ⿺ ⿻,
including overlay, plus the newer ⿼ ⿽ ⿾ ⿿ operators. Complete simple characters
can be used as components; the optional `forms` field preserves their full forms
instead of forcing a positional radical variant.

The engine also rejects duplicate radicals, repeated components in mixed
three-part designs, and full compound characters in three- or four-part
designs. Two complete compounds in a two-part design are reduced to one
radical plus one component when possible.

## Tests

```bash
python scripts/test_layouts.py
python scripts/smoke_test.py
python scripts/test_llm_mock.py
```

The layout test checks component positions and widths. The smoke test checks
validation, novelty normalization, ideographic layout, duplicate-radical
rejection, and deterministic readings.

## Demo recording

The interface supports a short screen recording. This is optional.

```bash
pip install -r requirements-demo.txt
playwright install chromium
python scripts/record_demo.py
# outputs/hanzi-forge-demo.mp4
```

The app must be running at `http://127.0.0.1:7860`, or set `FORGE_URL`.

## Project layout

```text
app.py                  Gradio interface
cli.py                  command line interface
forge/
  embed.py              embeddings and semantic index
  textdata.py           Make Me a Hanzi and CC-CEDICT parsing
  ids.py                IDS decomposition parsing
  radicals.py           positional shapes and placement rules
  compose.py            component-to-glyph mask composition
  render.py             ink, paper, seal, vignette
  engine.py             phrase-to-candidates pipeline
  llm.py                optional language-model client
  output.py             filename and save helpers
scripts/
  fetch_assets.py       download fonts and dictionaries
  build_index.py        build the embedding index
  test_layouts.py       layout regression checks
  smoke_test.py         end-to-end sanity checks
  test_llm_mock.py      language-model client check without a key
  record_demo.py        browser demo recording
assets/                 downloaded fonts and data (gitignored)
data/                   dictionaries and index (gitignored)
outputs/                generated images (gitignored)
```

## Data and licenses

| Resource | Use | License |
| --- | --- | --- |
| [Make Me a Hanzi](https://github.com/skishore/makemeahanzi) | definitions, pinyin, radicals, IDS decompositions | LGPL-3.0 code; Arphic Public License data and CC-CEDICT |
| [CC-CEDICT](https://www.mdbg.net/chinese/dictionary?page=cedict) | definitions and multi-character word hints | CC BY-SA 3.0 |
| [hanziDB](https://github.com/ruddfawcett/hanziDB.csv) | character frequency prior | MIT |
| [CJKVI IDS](https://github.com/cjkvi/cjkvi-ids) | decomposition database for novelty checks | derived from CHISE; GPLv2 terms |
| [LXGW WenKai](https://github.com/lxgw/LxgwWenKai) | pen-style rendering | SIL OFL 1.1 |
| [Ma Shan Zheng](https://fonts.google.com/specimen/Ma+Shan+Zheng) | brush rendering | SIL OFL 1.1 |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | English semantic embeddings | MIT |
| [fastembed](https://github.com/qdrant/fastembed) | ONNX embedding runtime | Apache-2.0 |

## Limitations

- English input only.
- Generated glyphs are compositions, not standard characters, and are not
  suitable for writing.
- Novelty checks are exact against the IDS database; rare variant spellings and
  unanalysed ideographs can still coincide with an existing character.
- Three- and four-part layouts are approximations. Two-part compositions
  currently have the most reliable proportions.
- The language-model mode needs a key and network access. The deterministic
  path does not.
- Abstract phrases can produce loose semantic matches; the interface warns when
  the top similarity score is low.
