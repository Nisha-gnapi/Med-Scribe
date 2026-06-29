import re
import difflib
import json
from typing import Dict, List, Any, Tuple, Set
import spacy

# Try importing rapidfuzz, fallback to difflib if not present to ensure out-of-the-box stability
try:
    from rapidfuzz import fuzz
except ImportError:
    class MockFuzz:
        @staticmethod
        def ratio(s1: str, s2: str) -> float:
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100
    fuzz = MockFuzz()

# ---------------------------------------------------------------------------
# CONSTANTS & CONFIGURATIONS
# ---------------------------------------------------------------------------

import requests
import negspacy  # pip install negspacy
from negspacy.termsets import termset

CLINICAL_RISK_WEIGHTS: Dict[str, float] = {
    "DRUG":      10.0,
    "DOSAGE":     9.0,
    "ALLERGY":   10.0,
    "STRENGTH":   9.0,
    "NEGATION":   8.0,
    "DIAGNOSIS":  7.0,
    "SYMPTOM":    5.0,
    "ENTITY":     5.0,
    "UNKNOWN":    1.0,
}

SPEAKER_RISK_MULTIPLIER: Dict[str, float] = {
    "DOCTOR":  1.5,
    "PATIENT": 1.0,
    "UNKNOWN": 1.0,
}
RISK_LEVEL_ORDER       = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
RISK_LEVEL_SCORE_FLOOR = {"LOW": 0, "MEDIUM": 25, "HIGH": 50, "CRITICAL": 75}
# ---------------------------------------------------------------------------
# RUNTIME CLINICAL VOCABULARY — sourced entirely from packages/APIs, no hardcoding
# ---------------------------------------------------------------------------

# These sets are populated once at pipeline init via _build_clinical_vocab().
# Never written to manually anywhere in this file.
KNOWN_DRUG_NAMES:      set = set()
KNOWN_DIAGNOSIS_TERMS: set = set()
KNOWN_SYMPTOM_TERMS:   set = set()

# Populated per-drug on first encounter via RxNorm API.
# { "metformin": {"min": 500, "max": 2550, "unit": "mg"} }
DOSAGE_LIMITS: dict = {}

# Negation is handled by negspaCy — no trigger list needed here at all.
# NEGATION_SCOPE_WINDOW removed — negspaCy uses dependency parse scope instead.


def _fetch_rxnorm_dosage(drug_name: str) -> dict | None:
    """
    Queries the free RxNorm REST API for known dosage strengths for a drug.
    Returns {"min": float, "max": float, "unit": str} or None if unavailable.
    No API key required.
    """
    try:
        # Step 1: resolve name → RxCUI
        r = requests.get(
            "https://rxnav.nlm.nih.gov/REST/rxcui.json",
            params={"name": drug_name, "search": 2},
            timeout=3
        )
        rxcui = r.json().get("idGroup", {}).get("rxnormId", [None])[0]
        if not rxcui:
            return None

        # Step 2: get all related strength/dose concepts
        r2 = requests.get(
            f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json",
            params={"tty": "SCDC"},  # Semantic Clinical Drug Component = dose+unit
            timeout=3
        )
        concepts = r2.json().get("relatedGroup", {}).get("conceptGroup", [])
        strengths = []
        unit = "mg"
        for group in concepts:
            for concept in group.get("conceptProperties", []):
                name_str = concept.get("name", "")
                # Parse "500 MG" out of strings like "Metformin 500 MG"
                import re
                m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|meq|u\b)", name_str, re.IGNORECASE)
                if m:
                    strengths.append(float(m.group(1)))
                    unit = m.group(2).lower()

        if not strengths:
            return None

        return {"min": min(strengths), "max": max(strengths), "unit": unit}

    except Exception:
        return None  # Network unavailable or parse failed — caller handles gracefully


def _get_dosage_limits(drug_name: str) -> dict | None:
    """
    Lazy-loads dosage limits per drug from RxNorm on first encounter.
    Caches in DOSAGE_LIMITS to avoid repeated API calls per session.
    """
    if drug_name not in DOSAGE_LIMITS:
        result = _fetch_rxnorm_dosage(drug_name)
        if result:
            DOSAGE_LIMITS[drug_name] = result
    return DOSAGE_LIMITS.get(drug_name)


def _build_clinical_vocab(nlp) -> None:
    """
    For scispaCy models: pulls vocab from NER labels and KB aliases.
    For en_core_web_sm: NER labels are generic so we skip that path.
    Vocab is primarily grown per-doc via _collect_vocab_from_doc during inference,
    and per-drug via RxNorm lazy fetch in _get_dosage_limits.
    """
    _label_to_set = {
        "CHEMICAL":     KNOWN_DRUG_NAMES,
        "DRUG":         KNOWN_DRUG_NAMES,
        "DISEASE":      KNOWN_DIAGNOSIS_TERMS,
        "DISORDER":     KNOWN_DIAGNOSIS_TERMS,
        "SYMPTOM":      KNOWN_SYMPTOM_TERMS,
        "SIGN_SYMPTOM": KNOWN_SYMPTOM_TERMS,
    }

    for pipe_name, component in nlp.pipeline:
        if hasattr(component, "patterns"):
            for pattern in component.patterns:
                label = pattern.get("label", "").upper()
                text  = pattern.get("pattern", "")
                if isinstance(text, str):
                    target = _label_to_set.get(label)
                    if target is not None:
                        target.add(text.lower().strip())

    for pipe_name, component in nlp.pipeline:
        if hasattr(component, "kb"):
            try:
                for alias in component.kb.get_alias_strings():
                    a = alias.lower().strip()
                    if not a:
                        continue
                    if len(a.split()) == 1:
                        KNOWN_DRUG_NAMES.add(a)
                    else:
                        KNOWN_DIAGNOSIS_TERMS.add(a)
            except Exception:
                pass

    # Log what model we're using so we know NER coverage level
    model_name = nlp.meta.get("name", "unknown")
    has_clinical_labels = any(
        label in ("CHEMICAL", "DISEASE", "SYMPTOM", "DRUG", "DISORDER")
        for label in nlp.pipe_labels.get("ner", [])
    )
    if not has_clinical_labels:
        print(f"[VOCAB WARN] Model '{model_name}' has no clinical NER labels. "
              f"Drug/diagnosis vocab will be built from RxNorm lookups during inference only.")
        
def _seed_vocab_from_text(text: str) -> None:
    """
    For models without clinical NER (e.g. en_core_web_sm), scan raw text tokens
    and attempt RxNorm lookup on any unrecognised single-word noun-like tokens.
    Any token that resolves to a valid RxCUI gets added to KNOWN_DRUG_NAMES.
    Results are cached — each token is only looked up once per session.
    """
    _checked: set = getattr(_seed_vocab_from_text, "_checked", set())
    _seed_vocab_from_text._checked = _checked

    tokens = set(re.findall(r"\b[a-z]{4,}\b", text.lower()))
    candidates = tokens - _checked - KNOWN_DRUG_NAMES - KNOWN_DIAGNOSIS_TERMS - _GRAMMAR_STOPWORDS

    for tok in candidates:
        _checked.add(tok)
        try:
            r = requests.get(
                "https://rxnav.nlm.nih.gov/REST/rxcui.json",
                params={"name": tok, "search": 2},
                timeout=2
            )
            rxcui = r.json().get("idGroup", {}).get("rxnormId", [None])[0]
            if rxcui:
                KNOWN_DRUG_NAMES.add(tok)
        except Exception:
            pass

def _collect_vocab_from_doc(doc) -> None:
    """
    Self-expands runtime vocab from model NER output.
    For en_core_web_sm this is a no-op since labels are generic —
    vocab grows instead via RxNorm lazy fetch during dosage validation.
    """
    _label_to_set = {
        "CHEMICAL":     KNOWN_DRUG_NAMES,
        "DRUG":         KNOWN_DRUG_NAMES,
        "DISEASE":      KNOWN_DIAGNOSIS_TERMS,
        "DISORDER":     KNOWN_DIAGNOSIS_TERMS,
        "SYMPTOM":      KNOWN_SYMPTOM_TERMS,
        "SIGN_SYMPTOM": KNOWN_SYMPTOM_TERMS,
    }
    for ent in doc.ents:
        target = _label_to_set.get(ent.label_.upper())
        if target is not None:
            target.add(ent.text.lower().strip())

# ---------------------------------------------------------------------------
# TEXT PARSING INPUT LAYER
# ---------------------------------------------------------------------------

def parse_raw_text_to_turns(text: str) -> List[Dict[str, Any]]:
    """
    Parses flat string texts containing inline speaker tags line-by-line.
    Format expected: 'DOCTOR: text...' or 'PATIENT: text...'
    """
    turns = []
    lines = text.strip().split("\n")
    pattern = re.compile(r"^(DOCTOR|PATIENT|SPEAKER_00|SPEAKER_01):\s*(.*)$", re.IGNORECASE)
    
    current_index = 0
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        match = pattern.match(line_str)
        if match:
            speaker_tag = match.group(1).upper()
            # Unify tags
            if speaker_tag == "SPEAKER_00": speaker_tag = "DOCTOR"
            if speaker_tag == "SPEAKER_01": speaker_tag = "PATIENT"
            
            turns.append({
                "speaker": speaker_tag,
                "text": match.group(2).strip(),
                "turn_index": current_index
            })
            current_index += 1
    return turns
def parse_turns_from_dicts(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Accepts the canonical pipeline dict format:
    [{"speaker": "DOCTOR", "start": 0.0, "end": 0.0, "duration": 0.0, "text": "..."}]
    Empty speaker → UNKNOWN.
    """
    speaker_alias = {"SPEAKER_00": "DOCTOR", "SPEAKER_01": "PATIENT"}
    result = []
    for i, t in enumerate(turns):
        raw = t.get("speaker", "").strip().upper()
        speaker = speaker_alias.get(raw, raw) or "UNKNOWN"
        result.append({
            "speaker": speaker,
            "text": t.get("text", "").strip(),
            "turn_index": i,
            "start": t.get("start", 0.0),
            "end": t.get("end", 0.0),
        })
    return result

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

# ---------------------------------------------------------------------------
# METRICS & VALIDATION FUNCTIONS
# ---------------------------------------------------------------------------

def semantic_contrast(token_a: str, token_b: str) -> float:
    if token_a is None or token_b is None:
        return 1.0
    if token_a.lower().strip() == token_b.lower().strip():
        return 0.0
    ratio = fuzz.ratio(token_a.lower(), token_b.lower())
    return round(1.0 - (ratio / 100.0), 3)

# Sourced from NLTK stopwords corpus at runtime — no hardcoded list.
_GRAMMAR_STOPWORDS: Set[str] = set()

def _load_stopwords() -> None:
    """Pulls English stopwords from NLTK. Falls back to a minimal functional set."""
    global _GRAMMAR_STOPWORDS
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        _GRAMMAR_STOPWORDS = set(stopwords.words("english"))
    except Exception:
        # Absolute minimal fallback — only if NLTK is entirely unavailable
        _GRAMMAR_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "i", "my",
                               "to", "of", "in", "it", "and", "or", "but", "do", "did"}


def _clinical_token_weight(tok: str) -> float:
    """
    Returns penalty weight for a token based on runtime clinical vocab sets.
    Grammar/stopwords → 0.0 (ignored per spec).
    """
    t = tok.lower().strip()
    if t in _GRAMMAR_STOPWORDS:    return 0.0
    if t in KNOWN_DRUG_NAMES:      return CLINICAL_RISK_WEIGHTS["DRUG"]
    if t in KNOWN_DIAGNOSIS_TERMS: return CLINICAL_RISK_WEIGHTS["DIAGNOSIS"]
    for phrase in KNOWN_SYMPTOM_TERMS:
        if t in phrase.split():
            return CLINICAL_RISK_WEIGHTS["SYMPTOM"] * 0.5
    return CLINICAL_RISK_WEIGHTS["UNKNOWN"]


def _contrast_weighted_word_errors(ref_text: str, hyp_text: str) -> float:
    ref_tokens = tokenize(ref_text)
    hyp_tokens = tokenize(hyp_text)
    sm = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens)
    penalty = 0.0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            for a, b in zip(ref_tokens[i1:i2], hyp_tokens[j1:j2]):
                contrast = semantic_contrast(a, b)
                weight = max(_clinical_token_weight(a), _clinical_token_weight(b))
                penalty += weight * contrast
        elif tag in ("delete", "insert"):
            for tok in (ref_tokens[i1:i2] + hyp_tokens[j1:j2]):
                penalty += _clinical_token_weight(tok) * 1.0
    return penalty

def compute_wer(reference: str, hypothesis: str) -> Dict[str, Any]:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    n, m = len(ref), len(hyp)
    
    # CORRECTED INITIALIZATION HERE
    dp = [[0]*(m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1):  dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]: i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1: S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1: D += 1; i -= 1
        else: I += 1; j -= 1
    w_score = (S + D + I) / max(n, 1)
    return {"wer": round(w_score, 4), "detail": f"S={S} D={D} I={I} / {n} tokens"}

def compute_cer(reference: str, hypothesis: str) -> Dict[str, Any]:
    ref = list(reference.lower().replace(" ", ""))
    hyp = list(hypothesis.lower().replace(" ", ""))
    n, m = len(ref), len(hyp)
    
    # CORRECTED INITIALIZATION HERE
    dp = [ [0]*(m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1): dp[i][0] = i

    for j in range(m + 1):dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]: i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1: S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1: D += 1; i -= 1
        else: I += 1; j -= 1
    c_score = (S + D + I) / max(n, 1)
    return {"cer": round(c_score, 4), "detail": f"S={S} D={D} I={I} / {n} chars"}

def reclassify_entities(entities_by_type: dict) -> dict:
    result = {k: set(v) for k, v in entities_by_type.items()}
    flat_entities = set(result.get("ENTITY", set()))
    drug_found = set()
    diag_found = set()
    symptom_found = set()

    for ent_text in flat_entities:
        ent_lower = ent_text.lower().strip()
        for drug in KNOWN_DRUG_NAMES:
            if drug in ent_lower or ent_lower in drug:
                drug_found.add(ent_text)
                break
        for diag in KNOWN_DIAGNOSIS_TERMS:
            if diag in ent_lower or ent_lower in diag:
                diag_found.add(ent_text)
                break
        for symp in KNOWN_SYMPTOM_TERMS:
            if symp in ent_lower or ent_lower in symp:
                symptom_found.add(ent_text)
                break

    if drug_found:
        result.setdefault("DRUG", set()).update(drug_found)
    if diag_found:
        result.setdefault("DIAGNOSIS", set()).update(diag_found)
    if symptom_found:
        result.setdefault("SYMPTOM", set()).update(symptom_found)
        
    result["ENTITY"] = flat_entities - drug_found - diag_found - symptom_found
    return {k: v for k, v in result.items() if v}

def extract_entities(nlp, text: str) -> dict:
    doc = nlp(text)
    _collect_vocab_from_doc(doc)
    _seed_vocab_from_text(text)   # RxNorm fallback for non-clinical models   

    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, set()).add(ent.text.lower().strip())

    tokens_lower = tokenize(text)
    for tok in tokens_lower:
        if tok in KNOWN_DRUG_NAMES:      entities.setdefault("ENTITY", set()).add(tok)
        if tok in KNOWN_DIAGNOSIS_TERMS: entities.setdefault("ENTITY", set()).add(tok)

    text_lower = text.lower()
    for phrase in KNOWN_SYMPTOM_TERMS:
        if phrase in text_lower:
            entities.setdefault("ENTITY", set()).add(phrase)

    return reclassify_entities(entities)

def compute_ner_f1(nlp, reference: str, hypothesis: str) -> Dict[str, Any]:
    ref_ents = extract_entities(nlp, reference)
    hyp_ents = extract_entities(nlp, hypothesis)
    all_types = set(ref_ents) | set(hyp_ents)
    by_type = {}
    f1_scores = []

    for etype in sorted(all_types):
        ref_set = ref_ents.get(etype, set())
        hyp_set = hyp_ents.get(etype, set())
        correct = len(ref_set & hyp_set)
        p = correct / max(len(hyp_set), 1)
        r = correct / max(len(ref_set),  1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        by_type[etype] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "correct": correct, "ref_entities": sorted(ref_set), "hyp_entities": sorted(hyp_set),
        }
        f1_scores.append(f1)
    macro_f1 = round(sum(f1_scores) / max(len(f1_scores), 1), 4) if f1_scores else 1.0
    return {"by_type": by_type, "macro_f1": macro_f1, "detail": f"macro F1={macro_f1:.4f}"}

def _attach_negspacy(nlp):
    if "negex" not in nlp.pipe_names:
        try:
            # negspacy 1.1.0 requires explicit import to trigger factory registration
            from negspacy.negation import Negex
            from spacy.language import Language

            if not Language.has_factory("negex"):
                Language.factory("negex", func=lambda nlp, name: Negex(nlp, name=name))

            try:
                ts = termset("en_clinical_negations")
            except Exception:
                ts = termset("en_clinical")

            nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()}, last=True)

        except Exception as e:
            print(f"[NEGEX WARN] negspaCy could not be attached: {e}. Negation detection disabled.")


# Negation trigger tokens — used only for rule-based fallback
_NEG_TOKENS: set = set()

def _load_neg_tokens() -> None:
    """
    Pulls negation trigger tokens from negspaCy's built-in clinical termset.
    Introspects actual pattern structure at runtime — no assumed key names.
    """
    global _NEG_TOKENS
    try:
        try:
            ts = termset("en_clinical_negations")
        except Exception:
            ts = termset("en_clinical")

        patterns = ts.get_patterns()
        tokens = set()

        def extract_from_item(item):
            if isinstance(item, dict):
                for key in ("LOWER", "TEXT", "lower", "text"):
                    val = item.get(key)
                    if val and isinstance(val, str):
                        tokens.add(val.lower())
            elif isinstance(item, str):
                tokens.add(item.lower())

        for key, value in patterns.items():
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, list):
                        for item in entry:
                            extract_from_item(item)
                    else:
                        extract_from_item(entry)
            elif isinstance(value, dict):
                extract_from_item(value)

        if not tokens:
            # Last resort: dump raw pattern structure so we can see what's in it
            print(f"[NEGEX DEBUG] Raw termset keys: {list(patterns.keys())}")
            for k, v in patterns.items():
                print(f"[NEGEX DEBUG] {k}: {str(v)[:200]}")

        _NEG_TOKENS = tokens
        print(f"[NEGEX] Loaded {len(_NEG_TOKENS)} negation triggers from negspaCy termset")

    except Exception as e:
        print(f"[NEGEX WARN] Could not load negation triggers: {e}")
        _NEG_TOKENS = set()

def _rule_based_negation_map(text: str) -> Dict[str, bool]:
    """
    Fallback negation detector that works without NER.
    Scans for clinical content words following negation triggers within a 5-token window.
    Returns {content_word: is_negated}.
    """
    tokens = tokenize(text)
    result = {}
    for i, tok in enumerate(tokens):
        if tok in _GRAMMAR_STOPWORDS or tok in _NEG_TOKENS or len(tok) < 4:
            continue
        # Check if any negation trigger appears in the 5 tokens before this token
        window = tokens[max(0, i-5):i]
        is_negated = any(t in _NEG_TOKENS for t in window)
        result[tok] = is_negated
    return result


def compute_negation_accuracy(nlp, reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Primary: negspaCy entity-level negation detection.
    Fallback: rule-based token-window negation when NER yields no entities.
    """
    ref_doc = nlp(reference)
    hyp_doc = nlp(hypothesis)

    def negspacy_map(doc):
        return {
            ent.text.lower().strip(): ent._.negex
            for ent in doc.ents
            if hasattr(ent._, "negex")
        }

    ref_map = negspacy_map(ref_doc)
    hyp_map = negspacy_map(hyp_doc)

    # If negspaCy found no entities (non-clinical model), fall back to rule-based
    if not ref_map and not hyp_map:
        ref_map = _rule_based_negation_map(reference)
        hyp_map = _rule_based_negation_map(hypothesis)

    flipped = []
    preserved = []

    for entity, ref_negated in ref_map.items():
        hyp_negated = hyp_map.get(entity)
        if hyp_negated is None:
            continue
        if ref_negated == hyp_negated:
            preserved.append(entity)
        else:
            flipped.append(entity)

    spurious = [e for e in hyp_map if e not in ref_map]
    rate = len(preserved) / max(len(ref_map), 1)

    return {
        "preservation_rate": round(rate, 4),
        "preserved": preserved,
        "flipped": flipped,
        "spurious": spurious,
        "detail": f"preserved={len(preserved)}/{len(ref_map)} flipped={len(flipped)}"
    }

def extract_drug_dose_pairs(text: str) -> List[Tuple[str, float, str]]:
    """
    Finds all drug+dose+unit patterns in text.
    Drug names come from KNOWN_DRUG_NAMES (runtime vocab, not hardcoded).
    Dosage limits are fetched lazily from RxNorm per drug encountered.
    """
    lower = text.lower()
    pairs = []
    unit_pat = r"(?:mg|mcg|ml|g|u\b|meq)"

    for drug in KNOWN_DRUG_NAMES:          # runtime set, not a hardcoded list
        if drug not in lower:
            continue
        for m in re.finditer(
            rf"\b{re.escape(drug)}\s+(\d+(?:\.\d+)?)\s*({unit_pat})", lower
        ):
            pairs.append((drug, float(m.group(1)), m.group(2)))
    return pairs

def compute_dosage_validation(reference: str, hypothesis: str) -> Dict[str, Any]:
    ref_pairs = extract_drug_dose_pairs(reference)
    hyp_pairs = extract_drug_dose_pairs(hypothesis)
    hyp_by_drug = {p[0]: p for p in hyp_pairs}

    matched = []; mismatched = []; missing = []; violations = []
    for drug, ref_dose, ref_unit in ref_pairs:
        if drug not in hyp_by_drug:
            missing.append((drug, ref_dose, ref_unit))
            continue
        _, hyp_dose, hyp_unit = hyp_by_drug[drug]
        if abs(hyp_dose - ref_dose) < 0.01 and hyp_unit == ref_unit:
            matched.append((drug, hyp_dose, hyp_unit))
        else:
            mismatched.append({"drug": drug, "ref": (ref_dose, ref_unit), "hyp": (hyp_dose, hyp_unit)})

    for drug, dose, unit in hyp_pairs:
        lim = _get_dosage_limits(drug)   # lazy RxNorm fetch, cached after first call
        if lim and (dose > lim["max"] or dose < lim["min"]):
            violations.append({
                "drug": drug, "dose": dose, "unit": unit,
                "allowed": f"{lim['min']}-{lim['max']} {lim['unit']}"
            })
    return {
        "ref_pairs": ref_pairs, "hyp_pairs": hyp_pairs, "matched": matched, "mismatched": mismatched,
        "missing_in_hyp": missing, "limit_violations": violations,
        "detail": f"matched={len(matched)} mismatched={len(mismatched)} violations={len(violations)}"
    }



# ---------------------------------------------------------------------------
# RISK SCORING ENGINE (WITH CUMULATIVE HARD FLOORS)
# ---------------------------------------------------------------------------

def compute_clinical_risk_score(wer_result, ner_result, negation_result, dosage_result,
                                 ref_text, hyp_text, gt_speaker, hyp_speaker) -> Dict[str, Any]:
    multiplier = SPEAKER_RISK_MULTIPLIER.get(gt_speaker, 1.0)
    breakdown = {}
    raw_score = 0.0
    floor = "LOW"
    
    # Track critical / high anomalies inside individual turns
    critical_events_count = 0
    high_events_count = 0

    def escalate(level):
        nonlocal floor
        if RISK_LEVEL_ORDER[level] > RISK_LEVEL_ORDER[floor]: floor = level

    # --- Diarization Penalty Rule ---
    if gt_speaker != hyp_speaker:
        diarization_penalty = 20.0 * multiplier
        raw_score += diarization_penalty
        breakdown["SPEAKER_DIARIZATION_SWAP"] = {"penalty": round(diarization_penalty, 2), "type": "HIGH"}
        high_events_count += 1
        escalate("HIGH")

    # --- Negation Penalty Rule ---
    n_neg_errors = len(negation_result["flipped"]) + len(negation_result["spurious"])
    if n_neg_errors:
        penalty = n_neg_errors * CLINICAL_RISK_WEIGHTS["NEGATION"] * 1.5 * multiplier
        breakdown["NEGATION_FLIP"] = {"count": n_neg_errors, "penalty": round(penalty, 2)}
        raw_score += penalty
        critical_events_count += n_neg_errors
        escalate("CRITICAL")

    # --- Entity Category Mutation Check ---
    for etype, stats in ner_result["by_type"].items():
        weight = CLINICAL_RISK_WEIGHTS.get(etype, 5)
        missed = set(stats["ref_entities"]) - set(stats["hyp_entities"])
        spurious = set(stats["hyp_entities"]) - set(stats["ref_entities"])
        
        # Categorical cross-check to detect entity swaps
        pairs = list(zip(sorted(missed), sorted(spurious)))
        leftover_missed = list(missed)[len(pairs):]
        leftover_spurious = list(spurious)[len(pairs):]

        type_penalty = 0.0
        max_contrast_seen = 0.0

        for a, b in pairs:
            c = semantic_contrast(a, b)
            type_penalty += weight * c * multiplier
            max_contrast_seen = max(max_contrast_seen, c)
            # If substitution contrast is highly divergent, category replacement happened
            if c > 0.5:
                high_events_count += 1
                escalate("HIGH")
        
        for _ in leftover_missed + leftover_spurious:
            type_penalty += weight * 1.0 * multiplier
            max_contrast_seen = 1.0

        if type_penalty > 0:
            breakdown[f"NER_{etype}"] = {"penalty": round(type_penalty, 2), "weight": weight}
            raw_score += type_penalty
            if etype == "DRUG" and max_contrast_seen > 0.4:
                high_events_count += 1
                escalate("HIGH")

    # --- Directionality & Dosage Validation Rules ---
    dos_w = CLINICAL_RISK_WEIGHTS["DOSAGE"]
    dos_count = 0
    for m in dosage_result["mismatched"]:
        ref_d, hyp_d = m["ref"][0], m["hyp"][0]
        contrast = min(abs(ref_d - hyp_d) / max(ref_d, hyp_d, 1), 1.0)
        raw_score += dos_w * contrast * multiplier
        dos_count += 1
        if contrast > 0.3:  # Scale mutation rule (e.g. mg vs g)
            high_events_count += 1
            escalate("HIGH")
            
    if dosage_result["limit_violations"]:
        raw_score += dos_w * 1.0 * multiplier * len(dosage_result["limit_violations"])
        dos_count += len(dosage_result["limit_violations"])
        high_events_count += len(dosage_result["limit_violations"])
        escalate("HIGH")
        
    if dosage_result["missing_in_hyp"]:
        raw_score += dos_w * 0.8 * multiplier * len(dosage_result["missing_in_hyp"])
        dos_count += len(dosage_result["missing_in_hyp"])

    if dos_count:
        breakdown["DOSAGE"] = {"count": dos_count, "weight": dos_w}

    # General prose differences
    general_penalty = _contrast_weighted_word_errors(ref_text, hyp_text)
    if general_penalty > 0:
        breakdown["GENERAL_WORD_ERRORS"] = {"penalty": round(general_penalty, 2)}
    raw_score += general_penalty

    raw_pct = min(round((raw_score / 150) * 100, 2), 100.0)
    score_level = ("LOW" if raw_pct < 15 else "MEDIUM" if raw_pct < 35 else "HIGH" if raw_pct < 60 else "CRITICAL")
    final_level = max(score_level, floor, key=lambda l: RISK_LEVEL_ORDER[l])
    score = max(raw_pct, RISK_LEVEL_SCORE_FLOOR[final_level])

    return {
        "score": score, "raw_score": round(raw_score, 2), "risk_level": final_level,
        "breakdown": breakdown, "critical_count": critical_events_count, "high_count": high_events_count,
        "detail": f"Score={score:.1f}/100, Level={final_level}"
    }

# ---------------------------------------------------------------------------
# PIPELINE EXECUTION ENGINE
# ---------------------------------------------------------------------------

def validate_turn(ref_turn: dict, hyp_turn: dict, nlp) -> dict:
    ref_text = ref_turn["text"]
    hyp_text = hyp_turn["text"]

    wer = compute_wer(ref_text, hyp_text)
    cer = compute_cer(ref_text, hyp_text)
    ner = compute_ner_f1(nlp, ref_text, hyp_text)
    negation = compute_negation_accuracy(nlp, ref_text, hyp_text)
    dosage = compute_dosage_validation(ref_text, hyp_text)
    
    risk = compute_clinical_risk_score(
        wer, ner, negation, dosage, ref_text, hyp_text, ref_turn["speaker"], hyp_turn["speaker"]
    )

    return {
        "speaker": ref_turn["speaker"], "hyp_speaker": hyp_turn["speaker"],
        "ref_text": ref_text, "hyp_text": hyp_text,
        "wer": wer, "cer": cer, "ner": ner, "negation": negation, "dosage": dosage, "risk": risk,
    }

def aggregate_scores(turn_results: List[dict]) -> dict:
    n = max(len(turn_results), 1)
    avg_wer  = round(sum(t["wer"]["wer"] for t in turn_results) / n, 4)
    avg_cer  = round(sum(t["cer"]["cer"] for t in turn_results) / n, 4)
    avg_f1   = round(sum(t["ner"]["macro_f1"] for t in turn_results) / n, 4)
    avg_neg  = round(sum(t["negation"]["preservation_rate"] for t in turn_results) / n, 4)
    avg_risk = round(sum(t["risk"]["score"] for t in turn_results) / n, 2)

    levels = [t["risk"]["risk_level"] for t in turn_results]
    overall_level = max(levels, key=lambda l: RISK_LEVEL_ORDER[l]) if levels else "LOW"

    total_critical = sum(t["risk"]["critical_count"] for t in turn_results)
    total_high = sum(t["risk"]["high_count"] for t in turn_results)
    
    total_dos_issues = sum(
        len(t["dosage"]["mismatched"]) + len(t["dosage"]["missing_in_hyp"]) + len(t["dosage"]["limit_violations"])
        for t in turn_results
    )
    total_flipped = sum(len(t["negation"]["flipped"]) for t in turn_results)

    return {
        "avg_wer": avg_wer, "avg_cer": avg_cer, "avg_ner_macro_f1": avg_f1,
        "avg_negation_rate": avg_neg, "avg_risk_score": avg_risk, "risk_level": overall_level,
        "total_dosage_issues": total_dos_issues, "total_negation_flips": total_flipped,
        "n_turns": len(turn_results), "critical_events_count": total_critical, "high_events_count": total_high
    }

def _compute_final_score(aggregated: dict) -> float:
    ner_component  = aggregated["avg_ner_macro_f1"] * 100
    neg_component  = aggregated["avg_negation_rate"] * 100
    risk_component = max(0.0, 100.0 - aggregated["avg_risk_score"])
    wer_component  = max(0.0, 100.0 - aggregated["avg_wer"] * 100)
    cer_component  = max(0.0, 100.0 - aggregated["avg_cer"] * 100)
    n_turns        = max(aggregated["n_turns"], 1)
    dos_component  = max(0.0, 100.0 - (aggregated["total_dosage_issues"] / n_turns) * 10)

    weights = {"ner": 0.30, "neg": 0.25, "risk": 0.20, "wer": 0.15, "cer": 0.05, "dosage": 0.05}

    final = (
        weights["ner"]    * ner_component  +
        weights["neg"]    * neg_component  +
        weights["risk"]   * risk_component +
        weights["wer"]    * wer_component  +
        weights["cer"]    * cer_component  +
        weights["dosage"] * dos_component
    )

    crit_count = aggregated["critical_events_count"]
    high_count = aggregated["high_events_count"]

    if crit_count > 0:
        # Cap scales with critical density per turn, not raw count
        # 1 critical per turn → cap at 50. Each additional 0.5 per turn drops cap by 5.
        crit_density  = crit_count / n_turns
        cap           = max(10.0, 50.0 - (max(0.0, crit_density - 1.0) * 10.0))
        final         = min(final, cap)
    elif high_count > 0:
        high_density  = high_count / n_turns
        cap           = max(20.0, 69.0 - (max(0.0, high_density - 1.0) * 10.0))
        final         = min(final, cap)

    return round(final, 2)

# ---------------------------------------------------------------------------
# OUTPUT STDOUT VISUALISATION
# ---------------------------------------------------------------------------

def print_validation_report(turn_results: List[dict], aggregated: dict):
    SEP = "=" * 90
    THIN = "-" * 90
    print("\n" + SEP)
    print("                      CRITICAL CLINICAL VALIDATION METRIC REPORT")
    print(SEP)

    for i, turn in enumerate(turn_results):
        print(f"\n[TURN {i+1}] | GT Speaker: {turn['speaker']} -> Hyp Speaker: {turn['hyp_speaker']}")
        print(THIN)
        print(f"  GROUND TRUTH : {turn['ref_text']}")
        print(f"  TRANSCRIPTION: {turn['hyp_text']}")
        print(f"    WER/CER   : WER: {turn['wer']['wer']:.2%} ({turn['wer']['detail']}) | CER: {turn['cer']['cer']:.2%}")
        print(f"    NEGATION  : {turn['negation']['detail']}")
        if turn["negation"]["flipped"]:
            print(f"               ⚠️ FLIPPED DETECTED: {turn['negation']['flipped']}")
        print(f"    DOSAGE    : {turn['dosage']['detail']}")
        if turn["dosage"]["mismatched"]:
            print(f"               ⚠️ SCALE MISMATCH: {turn['dosage']['mismatched']}")
        if turn["dosage"]["limit_violations"]:
            print(f"               ⚠️ COMPLIANCE SAFETY VIOLATION: {turn['dosage']['limit_violations']}")
        print(f"    RISK VAL  : {turn['risk']['detail']}")
        for comp, info in turn["risk"]["breakdown"].items():
            penalty = info.get("penalty")

            if penalty is not None:
                print(
                    f"               -> {comp}: "
                    f"Penalty Inflicted: {penalty}"
                )
            else:
                print(
                    f"               -> {comp}: "
                    f"{info}"
                )

    print("\n" + SEP)
    print("                     AGGREGATED PERFORMANCE INTEGRATION SCORING")
    print(SEP)
    print(f"  Total Validated Dialogue Lines : {aggregated['n_turns']}")
    print(f"  Session Word Error Rate (WER)  : {aggregated['avg_wer']:.2%}")
    print(f"  Session Char Error Rate (CER)  : {aggregated['avg_cer']:.2%}")
    print(f"  Clinical Entity Macro-F1 Score : {aggregated['avg_ner_macro_f1']:.4f}")
    print(f"  Total Negation Fault Flips     : {aggregated['total_negation_flips']}")
    print(f"  Total Clinical Dosage Issues   : {aggregated['total_dosage_issues']}")
    print(f"  Cumulative Critical Penalties  : {aggregated['critical_events_count']} tracked")
    print(f"  Cumulative High-Risk Penalties : {aggregated['high_events_count']} tracked")
    print(f"  Aggregated Risk Category Level : {aggregated['risk_level']}")
    print(f"\n  =======================================================")
    print(f"  ║ FINAL QUALITY VALIDATION SCORE ASSESSMENT: {aggregated['final_score']}/100 ║")
    print(f"  =======================================================")
    print(SEP + "\n")

# ---------------------------------------------------------------------------
# MAIN PIPELINE PIPING PIPELINE
# ---------------------------------------------------------------------------

def run_validation_pipeline(
    ground_truth,
    final_transcript,
    verbose: bool = False
) -> float:
    """
    Accepts either:
      - raw string with 'DOCTOR: ...' lines
      - list of dicts: [{"speaker": ..., "text": ..., ...}]
    Returns a single float: score out of 100.
    Prints nothing unless verbose=True.
    """
    if isinstance(ground_truth, list):
        ref_turns = parse_turns_from_dicts(ground_truth)
        hyp_turns = parse_turns_from_dicts(final_transcript)
    else:
        ref_turns = parse_raw_text_to_turns(ground_truth)
        hyp_turns = parse_raw_text_to_turns(final_transcript)

    max_turns = min(len(ref_turns), len(hyp_turns))
    if len(ref_turns) != len(hyp_turns) and verbose:
        print(f"[METRIC WARN] Turn count divergence (GT: {len(ref_turns)} vs Transcript: {len(hyp_turns)}). Clipping to {max_turns}")

    nlp = None
    for model in ["en_core_sci_md", "en_core_sci_sm", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model)
            print(f"[NLP] Loaded: {model}")
            break
        except Exception:
            continue
    if nlp is None:
        print("[NLP] WARNING: No model found, using blank — NER will not function")
        nlp = spacy.blank("en")

    _load_stopwords()           # populate grammar stopword filter from NLTK
    _load_neg_tokens()
    _build_clinical_vocab(nlp)  # populate drug/diagnosis/symptom sets from model
    _attach_negspacy(nlp)       # attach negspaCy using its own clinical termset

    turn_results = [validate_turn(ref_turns[i], hyp_turns[i], nlp) for i in range(max_turns)]
    aggregated = aggregate_scores(turn_results)
    final_score = _compute_final_score(aggregated)
    aggregated["final_score"] = final_score

    if verbose:
        print_validation_report(turn_results, aggregated)

    return final_score

# ---------------------------------------------------------------------------
# SIMULATED LARGE DATASET (20-LINE TEST RIG WITH HIGHLY DIVERGENT NOISE)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("record.json", "r") as f:
        ip = json.load(f)
    with open("gt.json", "r") as f:
        gt = json.load(f)

    score = run_validation_pipeline(gt, ip, verbose=True)
    print(score)