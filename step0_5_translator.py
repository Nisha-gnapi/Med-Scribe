import re
import copy
from wordfreq import zipf_frequency
from lingua import Language, LanguageDetectorBuilder
from deep_translator import GoogleTranslator

# ---------------------------------------------------------------------------
# WHY THE OLD HARDCODED LISTS ARE GONE
#
# _PROTECTED_PATTERN (a regex of ~50 medical terms) and _ENGLISH_WORDS
# (a hardcoded set of ~80 English words) both had the same flaw: there are
# millions of medical terms and English words, so any hardcoded list will
# always be incomplete and will need endless manual upkeep.
#
# Both are replaced with statistics instead of enumeration:
#   - wordfreq tells us, per word, which language it is *most* at home in
#     (by real-world frequency), with no list to maintain.
#   - lingua tells us, per phrase, how confident it is about the language,
#     with a margin check so it only acts when genuinely sure.
#
# A word/phrase that doesn't confidently look foreign (whether because it's
# common English, a drug name, or ASR garbage like "diabetis melitus") is
# left untouched. That is what "protection" now means -- it falls out of
# the confidence check automatically, instead of needing to be listed.
# ---------------------------------------------------------------------------

# Restricted language set. This is a small, finite list of *languages* the
# pipeline knows how to look for -- a fundamentally different thing from a
# list of *words/terms*, which is what we removed. Both wordfreq and lingua
# work far more accurately on short ASR phrases when restricted to a
# plausible set, instead of being asked to weigh ~75 languages at once.
_CANDIDATE_LANGUAGES = [
    Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.SPANISH,
    Language.ITALIAN, Language.PORTUGUESE, Language.DUTCH, Language.RUSSIAN,
    Language.ARABIC, Language.HINDI, Language.CHINESE,
]

# wordfreq language codes corresponding to the same set (English first so
# ties default to "English", i.e. the safer/no-translate side).
_WORDFREQ_LANGS = ["en", "de", "fr", "es", "it", "pt", "nl", "ru", "ar", "hi", "zh"]

# lingua -> deep_translator language code mismatches (everything else lines
# up as a plain lowercase ISO 639-1 code).
_ISO_OVERRIDES = {"ZH": "zh-CN"}

_NUMERIC_RE = re.compile(r"^\d+\.?\d*$")
_UNIT_RE = re.compile(r"^(mg|mcg|ml|cc|iu|meq)$", re.IGNORECASE)
_HAS_LETTERS_RE = re.compile(r"[a-zA-Z]")

# How sure lingua needs to be before we trust a phrase-level language call.
# margin = gap between the top and second-best language scores.
_MIN_TOP_CONFIDENCE = 0.30
_MIN_MARGIN = 0.10

_detector = LanguageDetectorBuilder.from_languages(*_CANDIDATE_LANGUAGES).build()


def _lang_to_iso(language: Language) -> str:
    code = language.iso_code_639_1.name  # e.g. "DE"
    return _ISO_OVERRIDES.get(code, code.lower())


# ---------------------------------------------------------------------------
# WORD-LEVEL CLASSIFICATION
# Decides whether a single token should be considered a candidate for
# translation. This only decides span membership -- the actual translation
# always happens on the full phrase/span, never on an isolated word.
# ---------------------------------------------------------------------------

def _is_english_word(word: str) -> bool:
    """
    Input:  word -- single token string
    Output: True if this word should be left untouched (English, a number,
            a unit, punctuation, or simply not confidently non-English)
    """
    w = word.strip()
    if not w:
        return True
    if not _HAS_LETTERS_RE.search(w):          # punctuation / pure symbols
        return True
    if _NUMERIC_RE.match(w) or _UNIT_RE.match(w):
        return True

    w_lower = w.lower()
    scores = {}
    for lang in _WORDFREQ_LANGS:
        try:
            scores[lang] = zipf_frequency(w_lower, lang)
        except Exception:
            # e.g. optional tokenizer dependency (jieba for 'zh') not installed
            scores[lang] = 0.0
    best_lang, best_score = max(scores.items(), key=lambda kv: kv[1])

    if best_score == 0:
        # Word is unrecognized in every tracked language (e.g. ASR garbage
        # like "melitus" from "mellitus"). Treat as a translation candidate
        # so it can still group with neighboring foreign words -- the
        # phrase-level confidence check below decides whether to actually
        # translate it.
        return False

    return best_lang == "en"


# ---------------------------------------------------------------------------
# SPAN BUILDING
# Groups consecutive candidate (non-English) words into phrases so
# translation always has surrounding context, never a single token.
# ---------------------------------------------------------------------------

def _build_spans(tokens: list) -> list:
    """
    Input:  tokens -- list of token strings
    Output: list of (start_idx, end_idx_exclusive, is_candidate)
    """
    spans = []
    i, n = 0, len(tokens)
    while i < n:
        is_candidate = not _is_english_word(tokens[i])
        j = i + 1
        while j < n and (not _is_english_word(tokens[j])) == is_candidate:
            j += 1
        spans.append((i, j, is_candidate))
        i = j
    return spans


# ---------------------------------------------------------------------------
# PHRASE-LEVEL LANGUAGE DETECTION AND TRANSLATION
# ---------------------------------------------------------------------------

def _detect_span_language(phrase: str):
    """
    Input:  phrase -- joined span text
    Output: (iso_code, confidence) if confidently non-English,
            (None, top_confidence) otherwise (i.e. "protect this span")
    """
    try:
        confidence_values = _detector.compute_language_confidence_values(phrase)
    except Exception:
        return None, 0.0

    if not confidence_values:
        return None, 0.0

    top, second = confidence_values[0], confidence_values[1]
    margin = top.value - second.value

    if (top.language == Language.ENGLISH
            or top.value < _MIN_TOP_CONFIDENCE
            or margin < _MIN_MARGIN):
        return None, top.value

    return _lang_to_iso(top.language), top.value


def _translate_phrase(phrase: str, iso_lang: str) -> str:
    """Translate one phrase from iso_lang -> English. Returns original on failure."""
    try:
        result = GoogleTranslator(source=iso_lang, target="en").translate(phrase)
        return result.strip() if result and result.strip() else phrase
    except Exception:
        return phrase


# ---------------------------------------------------------------------------
# WORDS[] FIELD TRANSLATION
# Builds spans across the segment's words, translates each confidently
# foreign span as a whole phrase (never word-by-word), then maps the
# translated phrase back onto the original word positions/timestamps.
#
# Alignment rule when word counts don't match 1:1:
#   - translated phrase has fewer words than the span -> fill positions in
#     order, leave the trailing original position(s) untouched.
#   - translated phrase has more words than the span -> fill positions in
#     order, append any overflow words onto the last filled position.
# ---------------------------------------------------------------------------

def _translate_words_field(words: list):
    """
    Input:  words -- list of dicts with a "token" key
    Output: (translated_words, words_log, seg_lang, protected_terms)
    """
    tokens = [entry.get("token", "") for entry in words]
    spans = _build_spans(tokens)

    translated_words = [dict(entry) for entry in words]
    words_log = []
    protected_terms = []
    seg_lang = "en"

    for start, end, is_candidate in spans:
        if not is_candidate:
            continue

        span_tokens = tokens[start:end]
        phrase = " ".join(span_tokens)

        iso_lang, _confidence = _detect_span_language(phrase)

        if iso_lang is None:
            # Not confidently foreign -- this is the "protection" path that
            # used to require a hardcoded medical-term list.
            protected_terms.extend(span_tokens)
            continue

        if seg_lang == "en":
            seg_lang = iso_lang

        translated_phrase = _translate_phrase(phrase, iso_lang)
        translated_tokens = translated_phrase.split()
        span_len = end - start

        for k in range(span_len):
            if k >= len(translated_tokens):
                break  # translated phrase shorter -> leave remaining positions as-is

            if k == span_len - 1 and len(translated_tokens) > span_len:
                new_word = " ".join(translated_tokens[k:])  # overflow onto last token
            else:
                new_word = translated_tokens[k]

            pos = start + k
            original_token = tokens[pos]
            if new_word.lower() != original_token.lower():
                translated_words[pos]["token_original"] = original_token
                translated_words[pos]["token"] = new_word
                words_log.append({
                    "position": pos,
                    "original": original_token,
                    "translated": new_word,
                })

    return translated_words, words_log, seg_lang, protected_terms


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def translate_json_transcript(transcript_json: dict) -> tuple:
    """
    Input:  transcript_json -- {"medasr": [...], "whisperx": [...]}
            each segment has: text/transcript, words[{"token":...}]

    Output: (result, translation_log)
        result -- {
            "medasr":   [{"segment_id", "speaker", "words": [{"token",...}]}],
            "whisperx": [{"segment_id", "speaker", "words": [{"token",...}]}]
        }
        translation_log -- same shape as before
    """
    translated_json = copy.deepcopy(transcript_json)
    translation_log = []
    result = {"medasr": [], "whisperx": []}

    for source_name, segments in translated_json.items():
        if source_name not in result or not isinstance(segments, list):
            continue

        for seg_idx, segment in enumerate(segments):
            original_text = (
                segment.get("text") or
                segment.get("transcript") or ""
            ).strip()

            words = segment.get("words", [])
            translated_words, words_log, seg_lang, protected_terms = \
                _translate_words_field(words)

            if words_log or protected_terms:
                translation_log.append({
                    "source":           source_name,
                    "segment_index":    seg_idx,
                    "detected_lang":    seg_lang,
                    "original_text":    original_text,
                    "protected_terms":  protected_terms,
                    "words_translated": words_log,
                })

            result[source_name].append({
                "segment_id": seg_idx + 1,
                "speaker":    segment.get("speaker", ""),
                "words":      translated_words,
            })

    return result, translation_log


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    with open(
        r"record1.json",
        "r", encoding="utf-8"
    ) as f:
        transcript_json = json.load(f)

    translated, log = translate_json_transcript(transcript_json)

    print("=== TRANSLATION LOG ===\n")
    for entry in log:
        print(f"[{entry['source']}] seg {entry['segment_index']}"
              f"  lang={entry['detected_lang']!r}")
        if entry["protected_terms"]:
            print(f"  protected : {entry['protected_terms']}")
        if entry["words_translated"]:
            for w in entry["words_translated"]:
                print(f"  word pos {w['position']}: {w['original']!r} -> {w['translated']!r}")
        print()

    print("=== OUTPUT ===\n")
    print(json.dumps(translated, indent=2, ensure_ascii=False))