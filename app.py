#!/usr/bin/env python3
"""Gradio interface for Hanzi Forge."""

from __future__ import annotations

import os
import random
from functools import lru_cache

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from forge import llm
from forge.engine import ForgeEngine, ForgeResult
from forge.output import save_images
from forge.paths import BRUSH_FONT, DATA, OUTPUTS
from forge.radicals import describe_structure

EXAMPLES = [
    "the ache of missing someone on a rainy night",
    "a fire that never goes out",
    "the feeling of coming home",
    "digital loneliness",
    "a dragon made of clouds",
    "quiet strength",
    "schadenfreude",
    "the melancholy of an empty train station",
]

@lru_cache(maxsize=1)
def _placeholder_image() -> Image.Image:
    image = Image.new("RGB", (1024, 1024), "#f4eee3")
    draw = ImageDraw.Draw(image)
    draw.rectangle((34, 34, 990, 990), outline="#e3d7c6", width=2)
    try:
        font = ImageFont.truetype(str(BRUSH_FONT), 430)
        small = ImageFont.truetype(str(BRUSH_FONT), 28)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.text((512, 470), "字", font=font, fill="#e8dfd0", anchor="mm")
    draw.text((512, 790), "AWAITING A PHRASE", font=small, fill="#9a8b78", anchor="mm")
    return image


@lru_cache(maxsize=1)
def _placeholder_path() -> str:
    path = DATA / "ui-placeholder.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _placeholder_image().save(path)
    return str(path)


CSS = """
:root {
  --paper: #f1ebdf;
  --surface: #fbf8f1;
  --ink: #201c17;
  --muted: #766957;
  --line: #d8cbb8;
  --accent: #a4382c;
  --accent-dark: #873027;
  --seal-ink: #f8f0e3;
}
html, body, .gradio-container { background: var(--paper) !important; }
body { color: var(--ink); }
.gradio-container .main { padding: 0 !important; }
.gradio-container {
  width: 100% !important;
  max-width: 1320px !important;
  flex: 1 1 auto !important;
  margin: 0 auto !important;
  padding: 12px 16px 28px !important;
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}
.gradio-container::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(circle at 12% 8%, rgba(255,255,255,.42), transparent 32%),
    radial-gradient(circle at 88% 72%, rgba(126,96,60,.06), transparent 38%);
  z-index: 0;
}
.masthead {
  position: relative;
  display: grid;
  grid-template-columns: 1fr minmax(220px, 360px) auto;
  gap: 28px;
  align-items: end;
  border-bottom: 1px solid var(--line);
  padding: 4px 0 20px;
  margin: 0 -12px 22px;
}
.brand-lockup { display: flex; align-items: center; gap: 15px; }
.seal {
  width: 56px; height: 56px; flex: 0 0 auto;
  display: grid; place-items: center;
  background: var(--accent); color: var(--seal-ink);
  font-family: "Songti SC", "STSong", "Noto Serif CJK SC", Georgia, serif;
  font-size: 32px; line-height: 1;
  border-radius: 4px;
  transform: rotate(-2deg);
  box-shadow: inset 0 0 0 3px rgba(255,255,255,.18);
}
.eyebrow {
  font: 600 10px/1.2 "SFMono-Regular", Consolas, monospace;
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 5px;
}
.masthead h1 {
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-size: clamp(2.4rem, 4.4vw, 3.7rem);
  font-weight: 500;
  letter-spacing: -.035em;
  line-height: .96;
  margin: 0;
  color: var(--ink);
}
.intro { color: var(--muted); font-size: .9rem; line-height: 1.5; text-align: right; padding-bottom: 3px; }
.intro p { margin: 0 0 5px; color: var(--ink); }
.intro span { display: block; }
.meta {
  justify-self: end;
  font: 600 10px/1.2 "SFMono-Regular", Consolas, monospace;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 10px;
  white-space: nowrap;
}
.section-label {
  font: 600 10px/1.2 "SFMono-Regular", Consolas, monospace;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 12px;
}
#studio-row { flex-wrap: nowrap !important; gap: 22px !important; align-items: flex-start !important; }
#compose-panel {
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: 18px !important;
  padding: 22px !important;
  box-shadow: 0 18px 45px rgba(74, 54, 32, .07);
  position: sticky;
  top: 16px;
  min-width: 0 !important;
}
#preview-panel { background: transparent !important; border: 0 !important; padding: 0 !important; min-width: 0 !important; }
#compose-panel .block, #preview-panel .block { background: transparent !important; border: 0 !important; box-shadow: none !important; }
#compose-panel textarea {
  background: #fffdf8 !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  font-size: 1rem !important;
  line-height: 1.55 !important;
  padding: 13px 14px !important;
  color: var(--ink) !important;
}
#compose-panel textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(164,56,44,.10) !important; }
#compose-panel label.container span { color: var(--muted) !important; font-size: .78rem !important; letter-spacing: .06em !important; text-transform: uppercase !important; }
#compose-panel .primary-action, #preview-panel .primary-action {
  width: 100% !important; min-height: 46px !important;
  border: 0 !important; border-radius: 9px !important;
  background: var(--accent) !important; color: #fff !important;
  font-weight: 650 !important; letter-spacing: .01em !important;
  box-shadow: 0 4px 0 var(--accent-dark) !important;
}
#compose-panel .primary-action:hover { background: var(--accent-dark) !important; }
#compose-panel .secondary-action {
  border: 1px solid var(--line) !important;
  border-radius: 9px !important;
  background: #fffdf8 !important;
  color: var(--ink) !important;
  min-height: 38px !important;
}
#compose-panel .secondary-action:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
.action-row { gap: 8px !important; margin-top: 8px !important; }
#llm-status {
  background: #f4eee3 !important;
  border: 1px solid var(--line) !important;
  border-radius: 9px !important;
  padding: 9px 11px !important;
  margin: 12px 0 0 !important;
  color: var(--muted) !important;
  font-size: .78rem !important;
  line-height: 1.4 !important;
}
#llm-status p { margin: 0 !important; }
#llm-status strong { color: var(--ink) !important; }
#compose-panel .label-wrap, #preview-panel .label-wrap {
  color: var(--muted) !important;
  font: 600 10px/1.2 "SFMono-Regular", Consolas, monospace !important;
  letter-spacing: .12em !important;
  text-transform: uppercase !important;
  border-top: 1px solid var(--line) !important;
  padding: 12px 0 !important;
  margin-top: 10px !important;
}
#compose-panel .examples { margin-top: 4px !important; }
#compose-panel .examples table { border-collapse: separate !important; border-spacing: 5px !important; }
#compose-panel .examples td { border: 1px solid var(--line) !important; border-radius: 7px !important; padding: 6px 8px !important; background: #fffdf8 !important; font-size: .78rem !important; color: var(--muted) !important; }
#compose-panel .examples td:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
#compose-panel .gallery-item {
  border: 1px solid var(--line) !important;
  border-radius: 7px !important;
  background: #fffdf8 !important;
  color: var(--muted) !important;
  font-size: .74rem !important;
  line-height: 1.25 !important;
  padding: 6px 8px !important;
  min-height: 30px !important;
  text-align: left !important;
}
#compose-panel .gallery-item:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
.gallery-heading { display: flex; justify-content: space-between; align-items: baseline; margin-top: 16px; }
.gallery-heading .section-label { margin: 0 !important; }
.gallery-hint { color: var(--muted); font-size: .72rem; }
#preview-panel .feature-image { border: 1px solid var(--line) !important; border-radius: 16px !important; overflow: hidden !important; background: var(--surface) !important; box-shadow: 0 18px 45px rgba(74, 54, 32, .08); }
#preview-panel .feature-image img { border-radius: 14px !important; }
#preview-panel .candidate-gallery { margin-top: 8px !important; background: transparent !important; }
#preview-panel .candidate-gallery img { border: 1px solid var(--line) !important; border-radius: 8px !important; background: var(--surface) !important; }
#preview-panel .gallery { gap: 7px !important; }
#preview-panel .label-wrap { margin-top: 4px !important; }
#preview-panel .accordion { border-top: 1px solid var(--line) !important; }
#preview-panel .accordion > .label-wrap { border-top: 0 !important; }
#build-notes { font-size: .9rem; line-height: 1.55; color: var(--muted); }
#build-notes h3 { font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif; font-size: 1.35rem; font-weight: 500; color: var(--ink); margin: 18px 0 6px; }
#build-notes h3:first-child { margin-top: 0; }
#build-notes table { font-size: .8rem; }
#build-notes td, #build-notes th { border-color: var(--line) !important; }
.colophon {
  display: flex; justify-content: space-between; gap: 20px;
  border-top: 1px solid var(--line);
  margin-top: 26px; padding-top: 12px;
  color: var(--muted); font-size: .75rem; line-height: 1.4;
}
.colophon .mark { font-family: "SFMono-Regular", Consolas, monospace; letter-spacing: .1em; text-transform: uppercase; }
footer { display: none !important; }
@media (max-width: 980px) {
  .gradio-container .main { padding: 0 !important; }
  .masthead { grid-template-columns: 1fr auto; gap: 16px; align-items: start; }
  .intro { grid-column: 1 / -1; text-align: left; max-width: none; padding: 0; }
  .meta { justify-self: end; }
  #studio-row { flex-direction: column !important; gap: 16px !important; }
  #compose-panel { position: static; width: 100% !important; }
  #preview-panel { width: 100% !important; }
}
@media (max-width: 720px) {
  #examples-block { display: none !important; }
}
@media (max-width: 540px) {
  .masthead h1 { font-size: 2.55rem; }
  .seal { width: 48px; height: 48px; font-size: 27px; }
  .meta { font-size: 9px; padding: 6px 8px; }
  #compose-panel { padding: 16px !important; border-radius: 13px !important; }
  .colophon { flex-direction: column; gap: 6px; }
}
"""



@lru_cache(maxsize=2)
def get_engine(style: str) -> ForgeEngine:
    return ForgeEngine(style=style)


def remix(
    phrase: str, count: float, mode: str, style: str, use_llm: bool, seal: bool
):
    if mode != "new" or not use_llm:
        yield from run(phrase, count, mode, style, use_llm, seal)
        return
    variation = random.randint(1, 1_000_000)
    yield from run(
        phrase, count, mode, style, use_llm, seal, variation=variation
    )


def surprise():
    return random.choice(EXAMPLES)


def _concept_block(analysis: dict | None) -> str:
    if not analysis:
        return ""
    definition = str(analysis.get("definition") or "").strip()
    literal = str(analysis.get("literal_meaning") or "").strip()
    concepts = analysis.get("key_concepts") or []
    terms = analysis.get("chinese_terms") or []
    components = analysis.get("component_candidates") or []
    transliteration = str(analysis.get("transliteration") or "").strip()
    reading = str(analysis.get("reading") or "").strip()
    confidence = str(analysis.get("confidence") or "").strip()

    lines = ["### Concept analysis"]
    if definition:
        lines.append(definition)
    if literal:
        lines.append(f"*Literal: {literal}*")
    meta: list[str] = []
    if concepts:
        meta.append("**Key concepts:** " + ", ".join(concepts[:6]))
    if terms:
        meta.append("**Chinese:** " + ", ".join(terms[:4]))
    if transliteration:
        meta.append(f"**Transliteration:** {transliteration}")
    if components:
        meta.append("**Radical candidates:** " + " ".join(components[:6]))
    if reading:
        meta.append(f"**Reading:** {reading}")
    if confidence:
        meta.append(f"**Confidence:** {confidence}")
    if meta:
        lines.append(" · ".join(meta))
    return "\n\n".join(lines)


def _details(result: ForgeResult) -> str:
    lines: list[str] = []
    concept = _concept_block(result.analysis)
    if concept:
        lines.append(concept)
        lines.append("")
    if result.warning:
        lines.append(f"> {result.warning}")
    lines.append("### How the glyph was built")
    for index, candidate in enumerate(result.candidates, start=1):
        badge = " · **Model-designed**" if candidate.origin == "llm" else ""
        if candidate.structure:
            header = structure_label(candidate.structure, candidate.parts)
            lines.append(
                f"**{index}. {header}** — "
                + " + ".join(candidate.shapes)
                + badge
            )
        else:
            lines.append(f"**{index}. closest real character**{badge}")
        lines.append(candidate.explanation)
        if candidate.rationale:
            lines.append(f"*{candidate.rationale}*")
        lines.append("")

    if result.matches:
        lines.append("### Closest real characters")
        lines.append("| | Pinyin | Meaning | Semantic match |")
        lines.append("|:---:|:---|:---|:---:|")
        for match in result.matches:
            meaning = match.record.get("definition", "").split(";")[0].replace("|", "\\|")
            lines.append(
                f"| <span class='cjk' style='font-size:1.5em'>{match.char}</span> "
                f"| {match.record.get('pinyin', '') or '—'} "
                f"| {meaning} | {match.score:.2f} |"
            )
        lines.append("")
        lines.append(
            "*Forged glyphs are original compositions, not standard Chinese characters. "
            "The real characters above are what the system matched the phrase to.*"
        )
    return "\n".join(lines)


def llm_status_text(result: ForgeResult | None = None, requested: bool = False) -> str:
    if not llm.is_available():
        return "**Model designs:** off. Add `DEEPSEEK_API_KEY` to `.env` to enable."
    if result is not None and requested:
        if result.llm_used:
            used = sum(1 for candidate in result.candidates if candidate.origin == "llm")
            suffix = (
                f" · {result.llm_rejected} rejected as existing/unconventional"
                if result.llm_rejected
                else ""
            )
            return (
                f"**Model designs:** on · {used} designs via "
                f"`{llm.model_name()}`{suffix}."
            )
        error = llm.last_error() or "no valid designs returned"
        return f"**Model designs:** failed — {error}. Using deterministic glyphs."
    if result is not None:
        return "**Model designs:** off for this run · deterministic glyphs."
    speed = "fast mode" if not llm.is_likely_slow() else "reasoning on"
    return f"**Model designs:** on · `{llm.model_name()}` · {speed}."


def structure_label(structure: str | None, parts: tuple[str, ...]) -> str:
    if not structure:
        return ""
    op = structure[0] if structure[0] in {"⿰", "⿱", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺"} else None
    if op and len(parts) == 2:
        return f"{op} {describe_structure(op)}"
    if op:
        return f"{structure} (nested)"
    return structure


def select_candidate(evt: gr.SelectData, items):
    if items and 0 <= evt.index < len(items):
        item = items[evt.index]
        return item[0] if isinstance(item, (tuple, list)) else item
    return None


def _present(result: ForgeResult):
    paths = save_images(result, OUTPUTS)
    main = result.candidates[0].image if result.candidates else None
    gallery = []
    for candidate in result.candidates:
        prefix = "LLM · " if candidate.origin == "llm" else ""
        reading = f" · {candidate.reading}" if candidate.reading else ""
        if candidate.structure:
            label = structure_label(candidate.structure, candidate.parts)
            caption = prefix + f"{label} · " + " + ".join(candidate.shapes) + reading
        else:
            caption = prefix + f"{candidate.parts[0]} · real character" + reading
        gallery.append((candidate.image, caption))
    return main, gallery, _details(result), [str(path) for path in paths]


def run(
    phrase: str,
    count: float,
    mode: str,
    style: str,
    use_llm: bool,
    seal: bool,
    variation: int = 0,
):
    if not phrase or not phrase.strip():
        yield None, [], "Enter a phrase first.", None, llm_status_text()
        return

    engine = get_engine(style)

    if (
        use_llm
        and mode == "new"
        and llm.is_available()
        and llm.is_likely_slow()
        and not llm.is_cached(phrase, 6, variation=variation)
    ):
        draft = engine.forge(
            phrase, count=int(count), mode=mode, use_llm=False, seal=seal
        )
        if draft.candidates:
            main, gallery, details, downloads = _present(draft)
            yield (
                main,
                gallery,
                details
                + "\n\n*Deterministic draft shown while the creative director works.*",
                downloads,
                f"**LLM:** draft shown; thinking with `{llm.model_name()}`…",
            )

    result = engine.forge(
        phrase,
        count=int(count),
        mode=mode,
        use_llm=use_llm,
        llm_variation=variation,
        seal=seal,
    )
    if not result.candidates:
        yield (
            None,
            [],
            "Could not build a glyph for that phrase. Try different wording.",
            None,
            llm_status_text(result, use_llm),
        )
        return

    main, gallery, details, downloads = _present(result)
    yield main, gallery, details, downloads, llm_status_text(result, use_llm)


with gr.Blocks(
    title="Hanzi Forge",
    theme=gr.themes.Base(primary_hue="red", neutral_hue="stone"),
    css=CSS,
) as demo:
    gr.HTML(
        """
        <header class="masthead">
          <div class="brand-lockup">
            <div class="seal">漢</div>
            <div>
              <div class="eyebrow">Composition studio</div>
              <h1>Hanzi Forge</h1>
            </div>
          </div>
          <div class="intro">
            <p>Compose a character form from an English phrase.</p>
            <span>Compare it with the closest standard characters.</span>
          </div>
          <div class="meta">Local / CPU</div>
        </header>
        """
    )

    with gr.Row(equal_height=False, elem_id="studio-row"):
        with gr.Column(scale=4, elem_id="compose-panel"):
            gr.HTML('<div class="section-label">01 / Prompt</div>')
            phrase = gr.Textbox(
                label="English phrase",
                placeholder="the ache of missing someone on a rainy night",
                lines=4,
            )
            gr.Markdown(
                "A feeling, a place, or a concept. Press Forge when the phrase is ready.",
                elem_classes=["hint"],
            )
            button = gr.Button(
                "Forge character",
                variant="primary",
                elem_classes=["primary-action"],
            )
            with gr.Row(elem_classes=["action-row"]):
                remix_button = gr.Button(
                    "Remix",
                    elem_classes=["secondary-action"],
                    interactive=llm.is_available(),
                )
                surprise_button = gr.Button(
                    "Surprise me",
                    elem_classes=["secondary-action"],
                )
            llm_status = gr.Markdown(llm_status_text(), elem_id="llm-status")

            with gr.Column(elem_id="examples-block"):
                gr.HTML('<div class="section-label" style="margin-top:18px">Try a phrase</div>')
                gr.Examples(EXAMPLES[:4], inputs=phrase, examples_per_page=4)

            with gr.Accordion("Settings", open=False):
                with gr.Row(elem_id="settings-row"):
                    count = gr.Slider(1, 6, value=4, step=1, label="Candidates")
                    style = gr.Radio(["brush", "pen"], value="brush", label="Stroke style")
                    mode = gr.Radio(
                        ["new", "real"],
                        value="new",
                        label="Mode",
                        info="new = compose a glyph; real = render the closest standard characters",
                    )
                with gr.Row(elem_id="settings-row"):
                    seal = gr.Checkbox(
                        value=False,
                        label="Red seal",
                        info="Decorative stamp of the closest real character.",
                    )
                    use_llm = gr.Checkbox(
                        value=llm.is_available(),
                        label="Use language-model designs",
                        info="Reads .env. Uncheck for deterministic glyphs.",
                        interactive=llm.is_available(),
                    )


        with gr.Column(scale=7, elem_id="preview-panel"):
            gr.HTML('<div class="section-label">02 / Result</div>')
            main_image = gr.Image(
                value=_placeholder_path(),
                show_label=False,
                type="pil",
                height=560,
                interactive=False,
                buttons=["download", "fullscreen"],
                elem_classes=["feature-image"],
            )
            gr.HTML(
                '<div class="gallery-heading">'
                '<span class="section-label" style="margin:14px 0 8px">Candidates</span>'
                '<span class="gallery-hint">Select one to enlarge</span>'
                '</div>'
            )
            gallery = gr.Gallery(
                show_label=False,
                columns=4,
                rows=1,
                height=180,
                object_fit="contain",
                interactive=False,
                elem_classes=["candidate-gallery"],
            )
            with gr.Accordion("Saved files", open=False):
                downloads = gr.File(
                    label=None,
                    file_count="multiple",
                    height=60,
                )
            with gr.Accordion("Build notes", open=False):
                details = gr.Markdown(
                    "Run a phrase to see its concept analysis and composition notes.",
                    elem_id="build-notes",
                )


    gr.HTML(
        """
        <div class="colophon">
          <span>Compositions are not standard Chinese characters. Readings are sound hints.</span>
          <span class="mark">Hanzi Forge · local / CPU</span>
        </div>
        """
    )

    event_inputs = [phrase, count, mode, style, use_llm, seal]
    event_outputs = [main_image, gallery, details, downloads, llm_status]
    button.click(run, inputs=event_inputs, outputs=event_outputs)
    phrase.submit(run, inputs=event_inputs, outputs=event_outputs)
    gallery.select(select_candidate, inputs=gallery, outputs=main_image)
    remix_button.click(remix, inputs=event_inputs, outputs=event_outputs)
    surprise_button.click(surprise, outputs=phrase).then(
        run, inputs=event_inputs, outputs=event_outputs
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("FORGE_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("FORGE_PORT", "7860")),
        show_error=True,
        footer_links=[],
    )
