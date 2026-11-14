#!/usr/bin/env python3
"""Gradio interface for Character Forge."""

from __future__ import annotations

import os
import random
from functools import lru_cache

import gradio as gr

from forge import llm
from forge.engine import ForgeEngine, ForgeResult
from forge.output import save_images
from forge.paths import OUTPUTS
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

CSS = """
body, .gradio-container { background: #f4eee2 !important; }
.gradio-container { max-width: 1240px !important; }
.hero { text-align: center; padding: 10px 0 2px; }
.hero h1 { font-family: Georgia, "Times New Roman", serif; font-size: 2.8rem;
  letter-spacing: .02em; color: #211c16; margin: 0; }
.hero p { color: #6d5f4f; font-size: 1.02rem; margin: .35rem 0 0; }
.panel { background: #fffaf0 !important; border: 1px solid #e6d9c5;
  border-radius: 18px; padding: 16px !important;
  box-shadow: 0 10px 30px rgba(70, 50, 30, .06); }
.primary-action, .primary-action button { background: #b23a2e !important;
  border: 0 !important; color: #fff !important; font-weight: 600; }
.secondary-action, .secondary-action button { background: #fffaf0 !important;
  border: 1px solid #d8c7ae !important; color: #4a3d30 !important; }
.hint { color: #8a7a66; font-size: .9rem; }
.hint p { margin: .15rem 0; }
.candidate-gallery img { border-radius: 10px; }
.feature-image img { border-radius: 12px; }
footer { display: none !important; }
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
    lines.append(
        "*This analysis is the bridge from the English term to usable Chinese "
        "components; the designer must build from these candidates.*"
    )
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
        badge = " · **LLM-designed**" if candidate.origin == "llm" else ""
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
        return "**Creative director:** off — add `DEEPSEEK_API_KEY` to `.env` for LLM designs."
    if result is not None and requested:
        if result.llm_used:
            used = sum(1 for candidate in result.candidates if candidate.origin == "llm")
            suffix = (
                f" · {result.llm_rejected} rejected as existing/unconventional"
                if result.llm_rejected
                else ""
            )
            return (
                f"**Creative director:** on · {used} designs via "
                f"`{llm.model_name()}`{suffix}."
            )
        error = llm.last_error() or "no valid designs returned"
        return f"**Creative director:** failed — {error}. Using deterministic glyphs."
    if result is not None:
        return "**Creative director:** off for this run · deterministic glyphs."
    speed = "fast mode" if not llm.is_likely_slow() else "reasoning on"
    return f"**Creative director:** on · `{llm.model_name()}` · {speed}."


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


with gr.Blocks(title="Character Forge") as demo:
    gr.HTML(
        """
        <div class="hero">
          <h1>Character Forge</h1>
          <p>An English phrase becomes a newly invented Chinese character —
          with a reading, a meaning, and a story.</p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, elem_classes=["panel"]):
            phrase = gr.Textbox(
                label="English phrase",
                placeholder="the ache of missing someone on a rainy night",
                lines=3,
            )
            gr.Markdown(
                "Try a feeling, a place, or an impossible concept.",
                elem_classes=["hint"],
            )
            with gr.Row():
                button = gr.Button(
                    "Forge character",
                    variant="primary",
                    scale=2,
                    elem_classes=["primary-action"],
                )
                remix_button = gr.Button(
                    "Remix",
                    scale=1,
                    elem_classes=["secondary-action"],
                    interactive=llm.is_available(),
                )
                surprise_button = gr.Button(
                    "Surprise me", scale=1, elem_classes=["secondary-action"]
                )
            llm_status = gr.Markdown(llm_status_text(), elem_classes=["hint"])

            with gr.Accordion("Options", open=False):
                count = gr.Slider(1, 6, value=4, step=1, label="Candidates")
                style = gr.Radio(["brush", "pen"], value="brush", label="Stroke style")
                mode = gr.Radio(
                    ["new", "real"],
                    value="new",
                    label="Mode",
                    info="new = forge a glyph; real = render the closest standard characters",
                )
                seal = gr.Checkbox(
                    value=False,
                    label="Red seal",
                    info="Decorative stamp of the closest real character. "
                    "It is not part of the forged glyph.",
                )
                use_llm = gr.Checkbox(
                    value=llm.is_available(),
                    label="Use the LLM creative director",
                    info="Reads .env. Uncheck for fully deterministic glyphs.",
                    interactive=llm.is_available(),
                )
            with gr.Accordion("Try an example", open=False):
                gr.Examples(EXAMPLES, inputs=phrase, examples_per_page=4)

        with gr.Column(scale=7, elem_classes=["panel"]):
            main_image = gr.Image(
                label="Featured glyph",
                type="pil",
                height=440,
                buttons=["download", "fullscreen"],
                elem_classes=["feature-image"],
            )
            gallery = gr.Gallery(
                label="Candidates — click one to feature it",
                columns=4,
                rows=1,
                height=220,
                object_fit="contain",
                elem_classes=["candidate-gallery"],
            )
            downloads = gr.File(
                label="Download PNGs", file_count="multiple", height=70
            )

    with gr.Accordion("Behind the glyph", open=False):
        details = gr.Markdown()

    gr.Markdown(
        "Forged glyphs are original compositions, not standard Chinese characters. "
        "The reading is borrowed from the phonetic component; the red seal is optional.",
        elem_classes=["hint"],
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
        theme=gr.themes.Soft(primary_hue="red", neutral_hue="stone"),
        css=CSS,
        footer_links=[],
    )
