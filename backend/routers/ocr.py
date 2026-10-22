"""OCR / document extraction via numind/NuExtract3 (a vision-language model
finetuned specifically for document understanding: structured extraction,
image-to-Markdown conversion, document-type detection, and template
generation) -- see https://huggingface.co/numind/NuExtract3.

NuExtract3's chat template takes several top-level template kwargs beyond the
usual OpenAI `messages` -- `mode`, `template`, `instructions`, `previous_output`
-- that llama-server forwards straight into Jinja rendering via
`chat_template_kwargs` (the same mechanism ChatCompletionRequest.chat_template_kwargs
in gateway.py already relies on for `enable_thinking`). This router builds
those kwargs from a purpose-built request shape instead of requiring callers to
know NuExtract3's template internals, and adds the "process a list of documents"
batch shape gateway.py doesn't have (that one's modeled closely on
media.py's ComfyUI batch runner -- same reasoning: one GPU, no benefit to
concurrent items, sequential + pollable progress).

Not covered here: few-shot examples (NuExtract3's `developer`-role message
convention for in-context examples). Power users who need that can still reach
it directly via POST /v1/chat/completions with a raw `chat_template_kwargs` and
`messages` including `role: "developer"` entries -- gateway.py's ChatCompletionRequest
is `extra="allow"` and forwards chat_template_kwargs verbatim, so nothing here
needs to special-case it.

Every page of a document is sent to the model as its OWN completion request,
never multiple images in one request, even though NuExtract3/llama.cpp
technically supports the latter. This was deliberate, not the original design:
a multi-page document sent as one request (numind's own reference approach --
all page images as separate image_url entries within a single completion) was
verified live, repeatedly (5 identical runs, byte-for-byte reproducible, so not
sampling noise), to merge adjacent pages' content together and hallucinate
specific text -- e.g. a real DOI silently replaced with a fabricated ISBN, and
~15 of ~60 names in a dense list garbled -- despite the images themselves being
individually correct and despite raising ctx_size well past what the content
needed. The exact same pages, sent one request per page and concatenated here,
came back byte-for-byte correct against the source document, every field. This
is evidently a genuine multi-image attention limitation of this model/quant,
not a context-length or sampling problem -- so per-page requests aren't an
optional mode, they're the only path.
"""
import asyncio
import base64
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import pymupdf as fitz  # PyMuPDF (its old import name `fitz` is deprecated)
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.model_manager import find_mmproj_path, get_or_load_model, get_upstream_error_detail, resolve_model_path
from backend.response_processing import process_completion_response
from backend.state import state

logger = logging.getLogger("ai-serve-ui.ocr")
router = APIRouter()

DEFAULT_OCR_MODEL = "NuExtract3-Q8_0.gguf"
# numind's own reference vLLM deployment uses --max-model-len 131072 -- but that
# figure is sized for THEIR reference approach (all pages of a document in one
# request, so a long document needs a correspondingly huge window). This
# endpoint never does that (see the module docstring on why: sending multiple
# pages in one request was verified to degrade accuracy, not just cost more
# context) -- every request here has exactly one image, so 131072 was wild
# overkill once that changed, and cost real VRAM for no benefit. Measured
# directly instead of guessing again: a single dense page at pdf_dpi=170
# (1406x1988px) needs ~3405 prompt tokens regardless of how much text is on it
# (vision tokenization is dominated by image resolution, not content -- two very
# differently-dense real pages measured identically), confirmed via llama-server's
# own usage.prompt_tokens on a direct /v1/chat/completions call replicating
# _run_ocr_page()'s exact payload. numind's own "memory-constrained" reference
# fallback, 16384, comfortably covers that (~3405 prompt + up to max_tokens=4096
# completion, ~7500 total, well under half of 16384) with headroom for a bigger
# max_tokens override, a larger/landscape page, or a longer instructions/template
# in structured mode -- and needs ~6.6GB VRAM for this model instead of ~18.6GB
# at 131072 (both measured live). Still just a default: ctx_size stays fully
# overridable per-request, and get_or_load_model()'s own VRAM check still fails
# clearly rather than degrading silently if a request genuinely needs more.
_DEFAULT_OCR_CTX_SIZE = 16384
# numind's reference: temperature 0.2 for non-thinking mode, 0.6 for thinking
# mode, 0 only for markdown conversion specifically (github.com/numindai/nuextract).
# This app previously defaulted temperature=0.0 unconditionally for every mode --
# right by coincidence for markdown (this endpoint's own default mode), wrong
# for structured/content/document-detection/template-generation.
_TEMPERATURE_MARKDOWN = 0.0
_TEMPERATURE_THINKING = 0.6
_TEMPERATURE_NON_THINKING = 0.2
_VALID_MODES = {"structured", "content", "markdown", "document-detection", "template-generation"}
# Modes whose output is JSON per NuExtract3's own template/design -- only these
# get a best-effort parse attempt; "content"/"markdown" are free text.
_JSON_MODES = {"structured", "document-detection", "template-generation"}

# NuExtract3 doesn't accept raw PDFs -- confirmed live: llama-server errors with
# "Failed to load image or audio file" when PDF bytes are sent as an image_url,
# since llama.cpp's mtmd image decoder (stb_image) is raster-only, no PDF
# renderer. numind's own model card gives their reference conversion approach
# for this exact model: PyMuPDF (fitz), one PNG per page at dpi=170, in page
# order -- matched here exactly (same library, same default DPI) rather than
# inventing our own settings, since 170 is what the model was actually
# validated against. Their reference then passes all pages in one request;
# this router doesn't (see the module docstring above for why).
_PDF_MAGIC = b"%PDF"
_DEFAULT_PDF_DPI = 170
# Safety cap, not a tuning knob (unlike pdf_dpi) -- a single runaway PDF
# shouldn't silently balloon into hundreds of page images blowing the context
# window or taking forever. Same spirit as _OCR_BATCH_MAX_ITEMS below; split a
# genuinely huge document into /api/ocr/batch calls instead.
_MAX_PDF_PAGES = 50


def _pdf_to_page_images(pdf_bytes: bytes, dpi: int) -> List[str]:
    """Render every page of a PDF to a base64 PNG at the given DPI, in page order."""
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if len(doc) > _MAX_PDF_PAGES:
            raise HTTPException(
                status_code=400,
                detail=f"PDF has {len(doc)} pages, over the {_MAX_PDF_PAGES}-page safety cap for a single "
                       f"document -- split it across multiple /api/ocr/batch items instead.",
            )
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            images.append(base64.b64encode(pix.tobytes("png")).decode())
    return images


def _expand_images_and_pdfs(images_base64: List[str], pdf_dpi: int) -> List[str]:
    """Each entry can be a real image OR a PDF -- content-sniffed via magic bytes,
    not any filename/extension the caller might imply (same convention the STT
    endpoint already uses for video: ffmpeg content-sniffs regardless of what
    the caller calls the file). A PDF expands in place into one image per page;
    a malformed PDF (magic bytes present, but fitz can't actually parse it) is
    left for the downstream image loader to reject with its own clear error
    rather than raising a second, different error here for the same root cause.
    """
    expanded: List[str] = []
    for img in images_base64:
        raw_b64 = img.split(",", 1)[1] if img.startswith("data:") else img
        try:
            raw_bytes = base64.b64decode(raw_b64)
        except Exception:
            expanded.append(img)
            continue
        if raw_bytes[:4] == _PDF_MAGIC:
            try:
                expanded.extend(_pdf_to_page_images(raw_bytes, dpi=pdf_dpi))
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to render PDF pages: {e}")
        else:
            expanded.append(img)
    return expanded


def _default_temperature(mode: str, enable_thinking: bool) -> float:
    """numind's own reference values (github.com/numindai/nuextract): markdown
    conversion gets 0 specifically; everything else is 0.2 in non-thinking mode
    or 0.6 in thinking mode. Only used when the caller doesn't set temperature
    explicitly -- an explicit value always wins."""
    if mode == "markdown":
        return _TEMPERATURE_MARKDOWN
    return _TEMPERATURE_THINKING if enable_thinking else _TEMPERATURE_NON_THINKING


# NuExtract3's chat template only auto-injects mode-specific guidance for
# mode="structured" (the template/instructions/previous_output block) -- every
# other mode gets nothing beyond the bare `【task】{mode}` label, confirmed by
# reading the template's own Jinja source. numind ships task_instructions_*.txt
# files in the model repo specifically to be used as the system message for the
# modes that need one; only "markdown" and "structured" have one. "structured"
# doesn't need it here (the template/instructions mechanism already carries
# equivalent guidance), so this is embedded only for "markdown" -- verbatim
# from numind/NuExtract3's task_instructions_markdown.txt (fetched from the
# model repo, not paraphrased) so a caller gets the model authors' own
# recommended framing by default, matching how they benchmarked it
# (.eval_results/olmocrbench.yaml: "markdown mode, non-thinking, greedy").
# "document-detection"/"content"/"template-generation" have no shipped
# instructions at all -- confirmed live that document-detection with no
# system message just falls back to describing/transcribing the document
# instead of classifying it, since nothing in the prompt tells it to classify.
# A caller who needs guidance for those (or wants to override this default)
# should pass their own `system_prompt`.
_MARKDOWN_TASK_INSTRUCTIONS = """\
**Role:** You are an advanced, specialized Document Parsing Assistant. Your task is to convert the provided document (titled `input`) into a high-fidelity, logically structured Markdown representation. The input may be an image or a multi-page PDF containing text, tables, spatial layouts, math, or graphic design elements.

**General Instructions:**
* **Logical Reading Order:** This is critical. Read the document as a human would. If the document has multiple columns or sidebars, extract the text column-by-column in its logical continuous flow. Do NOT read straight across multiple columns.
* **Maintain Hierarchy:** Use standard Markdown headers (`#`, `##`, `###`) to represent the visual importance and nesting of sections.
* **Transcribe Text Exactly:** Do not summarize, rephrase, or correct grammar. Maintain original spelling, capitalization, and punctuation.
* **Handling Obscured Text:** If a word or phrase is completely unreadable due to blur, stamps, or redactions, do not guess. Output `[ILLEGIBLE]` or `[REDACTED]`.

**Formatting Specifics:**
1. **Tables:**
   * For standard grid tables, use Markdown tables (`| Column |`).
   * For complex tables involving merged cells, multiple line-breaks within cells, or specific alignments, use standard HTML `<table>` tags utilizing `colspan` and `rowspan` to perfectly preserve the layout.
2. **Math and Equations:** Convert all mathematical formulas, equations, and scientific notation into LaTeX formatting. Use `$` for inline math (e.g., `$E=mc^2$`) and `$$` for block equations on their own lines.
3. **Visual Content & Figures:** For non-textual elements (logos, charts, photographs, floor plans):
   * Insert a Markdown image tag with a descriptive alt-text: `![Type: Brief Description](image_placeholder)`
   * Beneath it, describe the layout, data, or spatial relationships (e.g., *Top-left: Company Logo*, or *Floor plan detailing 3 rooms with dimensions*).
4. **Key-Value Clarity:** For forms or invoices, represent fields as bold keys followed by their values (e.g., **Invoice Date:** 2026-04-29).
5. **Footnotes & Citations:** Use standard Markdown footnote syntax (e.g., `[^1]`). Place the actual footnote text at the very bottom of the current section or page.
6. **Pagination:** If the input contains multiple pages, insert `<!-- PAGE BREAK -->` on a new line to separate the content of each page.
7. **Emphasis & Code:** Use `**bold**` for labels/headers, `*italics*` for fine print/captions, and backticks (`` ` ``) for raw code or technical strings.

**Output Constraint:** Provide ONLY the exact Markdown output. Do not include introductory remarks, explanations, or conclusions (e.g., do not say "Here is the converted document"). Start immediately with the markdown."""


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class OcrRequest(BaseModel):
    images_base64: List[str] = Field(
        ...,
        description="One or more page images (multi-page = one document). Any entry can also be a PDF "
                     "instead of an image -- content-sniffed via magic bytes, not filename/extension -- "
                     "which is expanded in place into one rendered page image per PDF page, in order.",
    )
    mode: str = Field("markdown", description="structured | content | markdown | document-detection | template-generation")
    template: Optional[str] = Field(None, description="JSON extraction template -- required when mode='structured'.")
    instructions: Optional[str] = Field(
        None,
        description="Extra guidance for the model. Only has an effect in mode='structured'. Worth setting "
                     "explicitly when extracting numbers/facts from a document that describes the same figure "
                     "both in a data table AND in narrative summary/discussion text elsewhere -- confirmed "
                     "live that a bare field-name-only template gives the model no signal to prefer the exact "
                     "table value over a rounded prose paraphrase of the same metric (e.g. table says '45', "
                     "a later discussion paragraph says 'knapp die Haelfte'/'just under half', and without "
                     "guidance the model can literalize the paraphrase into a wrong number like '50' instead "
                     "of the real table value). A blunt but effective instructions value: \"Extract numbers "
                     "ONLY from explicit data tables. Ignore approximate or rounded figures mentioned in "
                     "narrative/discussion text.\"",
    )
    previous_output: Optional[str] = Field(None, description="A prior structured output to refine/continue. Only has an effect in mode='structured'.")
    system_prompt: Optional[str] = Field(
        None,
        description="Guidance injected as a system message -- NuExtract3's template only auto-injects "
                     "mode-specific instructions for mode='structured'; every other mode gets nothing beyond "
                     "the bare task label unless you provide this. Defaults to numind's own shipped markdown "
                     "task instructions when mode='markdown' and this is left unset; no default for any other mode.",
    )
    enable_thinking: bool = Field(False, description="Only valid for mode='structured' or 'content' -- NuExtract3's template rejects it otherwise.")
    pdf_dpi: int = Field(
        _DEFAULT_PDF_DPI,
        description="Render resolution for any PDF pages in images_base64. Default (170) matches numind's "
                     "own reference conversion approach for this model -- only override if you know you need "
                     "higher fidelity for dense/small text, at the cost of more tokens per page.",
    )
    max_tokens: int = 4096
    temperature: Optional[float] = Field(
        None,
        description="Defaults to numind's own reference values when unset: 0 for mode='markdown', 0.2 for "
                     "other modes in non-thinking mode, 0.6 when enable_thinking is set. Only set this "
                     "yourself if you deliberately want to deviate.",
    )
    model: str = DEFAULT_OCR_MODEL
    gpu_id: Optional[int] = Field(
        None,
        description="Which GPU to load the OCR model onto -- passed straight through to "
                     "get_or_load_model(), the same auto-load mechanism every other gateway endpoint uses. "
                     "Left unset (default), the GPU with the most free VRAM is picked automatically. Set this "
                     "explicitly to pin OCR to a specific card, e.g. to keep it off a GPU that's also serving "
                     "a different model you don't want it contending with for compute (VRAM isn't the "
                     "concern there -- both models can sit in VRAM together fine -- it's that two models "
                     "sharing one GPU's compute serialize each other's requests instead of running in "
                     "parallel). Caveat: this only affects which GPU a *new* instance is spawned on -- the "
                     "existing-instance reuse check also matches on gpu_id, so if an instance is already "
                     "running on a different GPU than the one you specify here, it will NOT be moved; a "
                     "second, separate instance is spawned on the requested GPU instead, and the old one "
                     "sits idle until its own keep_alive expires. Set gpu_id consistently from the start if "
                     "you want to avoid that.",
    )
    keep_alive: Optional[int] = 300
    ctx_size: Optional[int] = Field(
        None,
        description=f"Defaults to {_DEFAULT_OCR_CTX_SIZE} (numind's own reference \"memory-constrained\" "
                     f"--max-model-len value -- comfortably covers one page: measured ~3405 prompt tokens "
                     f"for a single page at pdf_dpi=170 regardless of content density, plus up to max_tokens "
                     f"of completion). Every request here is one page, never multiple (see images_base64), "
                     f"so numind's much larger 131072 reference figure -- sized for their own approach of "
                     f"all pages in one request -- doesn't apply and would just cost unneeded VRAM. Raise it "
                     f"explicitly if you set a much bigger max_tokens or a higher pdf_dpi.",
    )


class OcrPageResult(BaseModel):
    page_index: int = Field(..., description="0-based index into the (PDF-expanded) images_base64 list.")
    status: str  # "completed" | "failed"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    parsed: Optional[Any] = None
    parse_error: Optional[str] = None
    error: Optional[str] = None
    runtime_seconds: Optional[float] = None


class OcrResult(BaseModel):
    id: Optional[str] = None
    status: str  # "completed" (every page ok) | "partial" (some pages failed) | "failed" (none did, or a setup error before any page ran)
    content: Optional[str] = Field(
        None,
        description="A single page: that page's own content. Multiple pages, text modes (markdown/content): "
                     "each page's content joined with an explicit '<!-- PAGE BREAK -->' inserted between them "
                     "by this endpoint (not requested from or trusted to the model -- see pages[] for exactly "
                     "what each page produced on its own). Multiple pages, JSON modes: null, use parsed instead.",
    )
    parsed: Optional[Any] = Field(
        None,
        description="A single page, JSON-producing mode: that page's own parsed object. Multiple pages, "
                     "JSON-producing modes: a list of one parsed object (or null) per page, in page order. "
                     "Text modes (markdown/content): always null.",
    )
    pages: List[OcrPageResult] = Field(
        default_factory=list,
        description="Full per-page breakdown, always populated (length 1 for a single-image document) -- the "
                     "source of truth content/parsed above are derived from. Check this when status='partial' "
                     "to see which page(s) failed and why.",
    )
    error: Optional[str] = None
    runtime_seconds: Optional[float] = None


class OcrBatchItem(BaseModel):
    images_base64: List[str]
    id: Optional[str] = Field(None, description="Caller-supplied label to match this item back in the batch results, e.g. a filename.")
    mode: Optional[str] = None
    template: Optional[str] = None
    instructions: Optional[str] = None
    previous_output: Optional[str] = None
    system_prompt: Optional[str] = None
    pdf_dpi: Optional[int] = None


class OcrBatchDefaults(BaseModel):
    mode: str = "markdown"
    template: Optional[str] = None
    instructions: Optional[str] = None
    system_prompt: Optional[str] = None
    enable_thinking: bool = False
    pdf_dpi: int = _DEFAULT_PDF_DPI
    max_tokens: int = 4096
    temperature: Optional[float] = None  # see OcrRequest.temperature for the default resolution rule
    model: str = DEFAULT_OCR_MODEL
    gpu_id: Optional[int] = None
    keep_alive: Optional[int] = 300
    ctx_size: Optional[int] = None  # see OcrRequest.ctx_size for the default resolution rule


class OcrBatchRequest(BaseModel):
    defaults: OcrBatchDefaults = OcrBatchDefaults()
    items: List[OcrBatchItem]


_OCR_BATCH_MAX_ITEMS = 500


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def _validate_mode(mode: str, template: Optional[str], enable_thinking: bool):
    if mode not in _VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Must be one of: {sorted(_VALID_MODES)}.")
    if mode == "structured" and not template:
        # NuExtract3's own chat template raises the same requirement (Jinja
        # exception at render time) -- caught here instead so it's a clean 400
        # with a clear message, not a 500 from a failed template render.
        raise HTTPException(status_code=400, detail="mode='structured' requires a 'template' (JSON extraction schema).")
    if enable_thinking and mode not in ("structured", "content"):
        raise HTTPException(
            status_code=400,
            detail=f"enable_thinking is only valid for mode='structured' or 'content', not '{mode}'.",
        )


def _build_ocr_messages(images_base64: List[str], system_prompt: Optional[str]) -> List[Dict[str, Any]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content = []
    for img in images_base64:
        # Tolerate either a raw base64 string or an already-prefixed data URI,
        # same convention media.py's _decode_image_base64 already uses. The
        # declared mime type is cosmetic here -- llama.cpp's image loader
        # auto-detects the real format from the decoded bytes regardless.
        raw = img.split(",", 1)[1] if img.startswith("data:") else img
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{raw}"}})
    messages.append({"role": "user", "content": content})
    return messages


async def _run_ocr_page(
    port: int,
    image_base64: str,
    mode: str,
    template: Optional[str],
    instructions: Optional[str],
    previous_output: Optional[str],
    system_prompt: Optional[str],
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    model: str,
) -> Dict[str, Any]:
    """One page, one completion request -- see the module docstring for why
    this is never batched with other pages into a single request."""
    page_start = time.time()
    try:
        chat_template_kwargs: Dict[str, Any] = {"mode": mode}
        if mode == "structured":
            chat_template_kwargs["template"] = template
            if instructions:
                chat_template_kwargs["instructions"] = instructions
            if previous_output:
                chat_template_kwargs["previous_output"] = previous_output
        # Always explicit, never omitted: llama-server defaults enable_thinking to
        # true for any template that declares support for it (server-context.cpp's
        # `enable_reasoning` defaults to -1/auto, i.e. enabled whenever the template
        # supports the kwarg -- confirmed against NuExtract3's, which does) unless a
        # request explicitly overrides it via chat_template_kwargs.enable_thinking.
        # Omitting the key here for the False case (the old behavior) left that
        # server-side auto-enabled default standing for modes other than
        # structured/content -- where it isn't just unwanted, it's a hard Jinja
        # exception ("'enable_thinking' can only be True for 'structured' and
        # 'content' modes"). _validate_mode() (called once by _run_ocr() before
        # any page runs) already guarantees enable_thinking is False here for
        # every mode but structured/content, so this is always the correct
        # value to send, never just the caller's intent.
        chat_template_kwargs["enable_thinking"] = enable_thinking

        # NuExtract3's template auto-injects mode guidance only for "structured"
        # (the template/instructions block above) -- every other mode gets
        # nothing beyond the bare task label unless a system message supplies it.
        # Confirmed live: mode="document-detection" with no system message just
        # transcribes the document instead of classifying it, since nothing in
        # the prompt tells it to classify. Default only exists for "markdown",
        # sourced verbatim from numind's own shipped task_instructions_markdown.txt.
        effective_system_prompt = system_prompt
        if not effective_system_prompt and mode == "markdown":
            effective_system_prompt = _MARKDOWN_TASK_INSTRUCTIONS

        payload = {
            "model": model,
            "messages": _build_ocr_messages([image_base64], effective_system_prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": chat_template_kwargs,
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        want_json = mode in _JSON_MODES
        parse_error = None
        try:
            raw_content, reasoning, parsed = process_completion_response(
                message, thinking=enable_thinking,
                response_format={"type": "object"} if want_json else None,
                finish_reason=finish_reason,
            )
        except Exception as e:
            raw_content, reasoning, parsed = process_completion_response(
                message, thinking=enable_thinking, response_format=None, finish_reason=finish_reason,
            )
            parse_error = str(e) if want_json else None

        # NuExtract3's own task instructions (still the default system prompt for
        # mode="markdown") tell it to insert "<!-- PAGE BREAK -->" between pages of
        # a multi-page input -- confirmed live that it sometimes emits this marker
        # on its own even for a single-image request (this endpoint never sends
        # more than one image per call -- see the module docstring), presumably
        # hedging on whatever pagination pattern it learned. Since _run_ocr() is now
        # the sole authority on where page breaks go (inserted deterministically
        # between pages, not requested from the model), any the model adds on its
        # own here would double up with that -- stripped so a page's content never
        # contributes stray break markers regardless of what triggered this one.
        if raw_content and "<!-- PAGE BREAK -->" in raw_content:
            raw_content = raw_content.replace("<!-- PAGE BREAK -->", "").rstrip()

        return {
            "status": "completed",
            "content": raw_content,
            "reasoning_content": reasoning or None,
            "parsed": parsed,
            "parse_error": parse_error,
            "error": None,
            "runtime_seconds": round(time.time() - page_start, 2),
        }
    except httpx.HTTPError as http_err:
        err_detail = get_upstream_error_detail(http_err)
        return {
            "status": "failed", "content": None, "reasoning_content": None, "parsed": None,
            "parse_error": None, "error": f"Upstream error: {err_detail}",
            "runtime_seconds": round(time.time() - page_start, 2),
        }


async def _run_ocr(
    images_base64: List[str],
    mode: str,
    template: Optional[str],
    instructions: Optional[str],
    previous_output: Optional[str],
    system_prompt: Optional[str],
    enable_thinking: bool,
    pdf_dpi: int,
    max_tokens: int,
    temperature: Optional[float],
    model: str,
    gpu_id: Optional[int],
    keep_alive: Optional[int],
    ctx_size: Optional[int],
) -> Dict[str, Any]:
    if not images_base64:
        raise HTTPException(status_code=400, detail="images_base64 must be a non-empty list.")
    _validate_mode(mode, template, enable_thinking)
    images_base64 = _expand_images_and_pdfs(images_base64, pdf_dpi=pdf_dpi)
    if temperature is None:
        temperature = _default_temperature(mode, enable_thinking)

    start_time = time.time()
    model_path = resolve_model_path(model)
    mmproj_path = find_mmproj_path(model_path)
    if not mmproj_path:
        raise HTTPException(
            status_code=400,
            detail=f"No mmproj sidecar found for '{model_path}' -- OCR needs a vision-capable model with a paired mmproj GGUF.",
        )

    port = await get_or_load_model(
        model_path=model_path, gpu_id=gpu_id, keep_alive=keep_alive or 300,
        ctx_size=ctx_size or _DEFAULT_OCR_CTX_SIZE, mmproj_path=mmproj_path, use_mtp=False,
    )
    state.managed_processes[port]["active_requests"] += 1
    state.managed_processes[port]["last_used"] = time.time()
    try:
        # Sequential, not concurrent, deliberately: this is the exact methodology
        # that was verified live to fix the accuracy problem (each page gets the
        # model's undivided attention), and concurrent slots against one instance
        # at this endpoint's now-large default ctx_size (_DEFAULT_OCR_CTX_SIZE)
        # would multiply KV-cache VRAM use per concurrent page -- an untested,
        # separate risk not worth taking on at the same time as the accuracy fix.
        pages: List[Dict[str, Any]] = []
        for idx, image_base64 in enumerate(images_base64):
            page_result = await _run_ocr_page(
                port=port, image_base64=image_base64, mode=mode, template=template,
                instructions=instructions, previous_output=previous_output,
                system_prompt=system_prompt, enable_thinking=enable_thinking,
                max_tokens=max_tokens, temperature=temperature, model=model,
            )
            page_result["page_index"] = idx
            pages.append(page_result)
            # Defensive, not just belt-and-suspenders: confirmed live that this
            # entry can genuinely disappear mid-document (a concurrent caller's
            # ctx_size-upgrade request evicting this same instance -- see the
            # active_requests guard added in model_manager.py's _load_locked()
            # existing-instance match -- was one real way this happened; a
            # human clicking Stop & Unload on the dashboard mid-request is
            # another). A missing entry here shouldn't crash the whole
            # document's remaining pages with an unhandled KeyError -- the
            # already-completed pages' results are still worth returning.
            if port in state.managed_processes:
                state.managed_processes[port]["last_used"] = time.time()

        succeeded = [p for p in pages if p["status"] == "completed"]
        if not succeeded:
            overall_status = "failed"
        elif len(succeeded) == len(pages):
            overall_status = "completed"
        else:
            overall_status = "partial"

        want_json = mode in _JSON_MODES
        if len(pages) == 1:
            # Single-page documents (the common case, and now also the
            # recommended shape for multi-page ones sent as separate items)
            # get an unwrapped content/parsed, unchanged from before pages[]
            # existed -- no reason to make every caller unwrap a length-1 list.
            content = pages[0]["content"]
            parsed = pages[0]["parsed"]
        elif want_json:
            content = None
            parsed = [p["parsed"] for p in pages]
        else:
            # Page breaks are inserted here, deterministically, rather than
            # asked of the model (its own task instructions mention a
            # <!-- PAGE BREAK --> convention, but nothing enforces it, and
            # relying on it was part of the original merging problem).
            content = "\n\n<!-- PAGE BREAK -->\n\n".join(p["content"] or "" for p in pages)
            parsed = None

        return {
            "status": overall_status,
            "content": content,
            "parsed": parsed,
            "pages": pages,
            # Aggregated only so a caller (or ocr_document()'s HTTP-502 path)
            # has something readable without digging into pages[] themselves --
            # the actual per-page reasons always live in pages[].error.
            "error": None if overall_status != "failed" else "; ".join(
                f"page {p['page_index']}: {p['error']}" for p in pages if p.get("error")
            ) or "All pages failed.",
            "runtime_seconds": round(time.time() - start_time, 2),
        }
    finally:
        if port in state.managed_processes:
            state.managed_processes[port]["active_requests"] = max(
                0, state.managed_processes[port].get("active_requests", 1) - 1
            )
            state.managed_processes[port]["last_used"] = time.time()


# ---------------------------------------------------------------------------
# Single document
# ---------------------------------------------------------------------------

@router.post("/api/ocr", summary="Extract/transcribe a single document via NuExtract3", response_model=OcrResult)
async def ocr_document(req: OcrRequest):
    result = await _run_ocr(
        images_base64=req.images_base64, mode=req.mode, template=req.template,
        instructions=req.instructions, previous_output=req.previous_output, system_prompt=req.system_prompt,
        enable_thinking=req.enable_thinking, pdf_dpi=req.pdf_dpi, max_tokens=req.max_tokens, temperature=req.temperature,
        model=req.model, gpu_id=req.gpu_id, keep_alive=req.keep_alive, ctx_size=req.ctx_size,
    )
    if result["status"] == "failed":
        raise HTTPException(status_code=502, detail=result["error"])
    return OcrResult(**result)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _merge_batch_item(defaults: OcrBatchDefaults, item: OcrBatchItem) -> Dict[str, Any]:
    merged = defaults.dict()
    merged.update({k: v for k, v in item.dict(exclude={"images_base64", "id"}).items() if v is not None})
    merged["images_base64"] = item.images_base64
    merged["id"] = item.id
    _validate_mode(merged["mode"], merged.get("template"), merged["enable_thinking"])
    if not item.images_base64:
        raise ValueError("images_base64 must be a non-empty list.")
    return merged


async def _run_ocr_batch(batch_id: str):
    """Sequential runner, same rationale as media.py's _run_comfyui_batch: one
    GPU, no benefit to running items concurrently against the same instance."""
    batch = state.ocr_batch
    if not batch or batch["id"] != batch_id:
        return
    items = batch["items"]

    for idx, item in enumerate(items):
        batch = state.ocr_batch
        if not batch or batch["id"] != batch_id or batch["status"] == "cancelled":
            return

        batch["current_index"] = idx
        item["status"] = "processing"
        req = item["request"]
        try:
            result = await _run_ocr(
                images_base64=req["images_base64"], mode=req["mode"], template=req.get("template"),
                instructions=req.get("instructions"), previous_output=req.get("previous_output"),
                system_prompt=req.get("system_prompt"),
                enable_thinking=req["enable_thinking"], pdf_dpi=req.get("pdf_dpi", _DEFAULT_PDF_DPI),
                max_tokens=req["max_tokens"], temperature=req["temperature"],
                model=req["model"], gpu_id=req.get("gpu_id"), keep_alive=req.get("keep_alive"), ctx_size=req.get("ctx_size"),
            )
            item["status"] = result["status"]
            item["result"] = result
        except Exception as e:
            item["status"] = "failed"
            item["result"] = {"status": "failed", "error": str(e)}

        batch = state.ocr_batch
        if not batch or batch["id"] != batch_id:
            return
        batch["completed_count"] = idx + 1
        if batch["status"] == "cancelled":
            for later in items[idx + 1:]:
                if later["status"] == "pending":
                    later["status"] = "cancelled"
            return

    batch = state.ocr_batch
    if batch and batch["id"] == batch_id and batch["status"] != "cancelled":
        batch["status"] = "completed"


@router.post("/api/ocr/batch", summary="Process a list of documents via NuExtract3")
async def start_ocr_batch(req: OcrBatchRequest):
    existing = state.ocr_batch
    if existing and existing["status"] in ("running",):
        raise HTTPException(
            status_code=409,
            detail="A batch is already running. Cancel it, or wait for it to finish, before starting another.",
        )
    if not req.items:
        raise HTTPException(status_code=400, detail="items must be a non-empty list.")
    if len(req.items) > _OCR_BATCH_MAX_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large ({len(req.items)} items) -- max {_OCR_BATCH_MAX_ITEMS} per batch.",
        )

    try:
        resolved_items = [_merge_batch_item(req.defaults, item) for item in req.items]
    except (ValueError, HTTPException) as e:
        detail = e.detail if isinstance(e, HTTPException) else str(e)
        raise HTTPException(status_code=400, detail=f"Invalid batch item: {detail}")

    batch_id = str(uuid.uuid4())
    state.ocr_batch = {
        "id": batch_id,
        "status": "running",
        "total": len(resolved_items),
        "completed_count": 0,
        "current_index": None,
        "created_at": time.time(),
        "items": [
            {"index": i, "request": resolved, "status": "pending", "result": None}
            for i, resolved in enumerate(resolved_items)
        ],
    }
    asyncio.create_task(_run_ocr_batch(batch_id))
    return {"batch_id": batch_id, "total": len(resolved_items)}


def _ocr_batch_item_view(it: Dict[str, Any]) -> Dict[str, Any]:
    result = it["result"] or {}
    return {
        "index": it["index"],
        "id": it["request"].get("id"),
        "status": it["status"],
        "content": result.get("content"),
        "parsed": result.get("parsed"),
        "pages": result.get("pages", []),
        "error": result.get("error"),
        "runtime_seconds": result.get("runtime_seconds"),
    }


@router.get("/api/ocr/batch/status", summary="Get the current/last OCR batch's progress and results")
async def get_ocr_batch_status():
    batch = state.ocr_batch
    if not batch:
        return {"status": "none"}
    return {
        "id": batch["id"],
        "status": batch["status"],
        "total": batch["total"],
        "completed_count": batch["completed_count"],
        "current_index": batch["current_index"],
        "items": [_ocr_batch_item_view(it) for it in batch["items"]],
    }


@router.post("/api/ocr/batch/cancel", summary="Abort a running OCR batch after its current item finishes")
async def cancel_ocr_batch():
    batch = state.ocr_batch
    if not batch or batch["status"] != "running":
        raise HTTPException(status_code=409, detail="No running batch to cancel.")
    batch["status"] = "cancelled"
    # Unlike ComfyUI's cancel, there's no llama-server "/interrupt" to reach for --
    # the in-flight item is left to finish; only not-yet-started items are skipped.
    return {"status": "cancelled"}
