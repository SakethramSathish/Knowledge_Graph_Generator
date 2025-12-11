"""
nlp_pipeline.py

Provides:
- extract_entities_with_spans(text)
- extract_relations_from_sentence(sentence)
- naive_coref_resolution(sentences)
"""

from typing import List, Tuple, Dict
import re

import spacy
from spacy.matcher import Matcher

try:
    nlp = spacy.load("en_core_web_trf")
except Exception:
    nlp = spacy.load("en_core_web_sm")

matcher = Matcher(nlp.vocab)
pattern_of = [
    [{"ENT_TYPE": {"NOT_IN": [""]}, "OP": "+"},
     {"LOWER": "of"},
     {"ENT_TYPE": {"NOT_IN": [""]}, "OP": "+"}]
]
try:
    matcher.add("OF_PATTERN", pattern_of)
except Exception:
    #if pattern fails due to model differences, ignore
    pass

PRONOUNS = set(["he", "she", "they", "it", "his", "her", "their", "its"])

def extract_entities_with_spans(text: str) -> List[Dict]:
    doc = nlp(text)
    return[
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]

def extract_relations_from_sentence(sentence: str) -> List[Tuple[str, str, str]]:
    """
    Heuristic relation extraction:
     - matches "X of Y" patterns
     - uses dependency parse to find verb connecting subj and obj
     - fallback: link sequential entities with related_to
    """
    doc = nlp(sentence)
    relations = []

    #1) matcher "X of Y"
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        text = span.text
        if " of " in text:
            left, right = text.split(" of ", 1)
            relations.append((left.strip(), "of", right.strip()))

    #2) dependency-based subj-verb-obj
    for token in doc:
        if token.pos_ == "VERB":
            subj = [w for w in token.lefts if w.dep_.endswith("subj")]
            objs = [w for w in token.rights if w.dep_.endswith("obj") or w.dep_ == "prep"]
            if subj and objs:
                subj_text = " ".join([w.text for w in subj])
                for o in objs:
                    obj_phrase = " ".join([tok.text for tok in o.subtree])
                    relations.append((subj_text, token.lemma_, obj_phrase))

    #3) Fallback: If Sentence has 2 or more entities, link them sequentially
    ents = [ent.text for ent in doc.ents]
    if len(ents) >= 2:
        for i in range(len(ents) - 1):
            relations.append((ents[i], "related_to", ents[i+1]))

    #Deduplicate Preserving Order
    seen = set()
    out = []
    for r in relations:
        key = tuple(r)
        if key not in seen:
            out.append(r)
            seen.add(key)
    return out

def naive_coref_resolution(sentences: List[str]) -> List[str]:
    """
    Very naive coreference: replace pronouns with the most recent PERSON/ORG/GPE mention.
    This is a lightweight heuristic to improve RE for MVP.
    """
    resolved = []
    last_entity = None
    for sent in sentences:
        doc = nlp(sent)
        ents = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPR", "NORP", "PRODUCT")]
        if ents:
            last_entity = ents[-1]
        def repl(m):
            if last_entity:
                return last_entity
            return m.group(0)
        sent2 = re.sub(r"\b(" + "|".join(PRONOUNS) + r")\b", repl, sent, flags=re.IGNORECASE)
        resolved.append(sent2)
    return resolved