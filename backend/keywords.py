"""Existing keyword assets and search semantics, independent of any UI."""

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz import process

MODES = ("default", "geopolitical", "thematic", "action", "category")


class KeywordIndex:
    def __init__(self, engine, assets: Path):
        self.words: dict[str, dict[str, set]] = {mode: {} for mode in MODES}
        self.subject_ids: dict[str, list[str]] = {}
        self.available = {mode: False for mode in MODES}
        ids = set(engine.resolution_table.undl_id)
        general = re.compile(r"[;,]")
        ignore = {"resolution", "general assembly"}

        def add(frame, column, mode, pattern=general, ignored=ignore):
            for rid, raw in frame[["undl_id", column]].itertuples(index=False, name=None):
                if rid not in ids or pd.isna(raw):
                    continue
                terms = {re.sub(r"\s+", " ", t.strip().lower()) for t in pattern.split(str(raw))}
                for term in terms - ignored - {""}:
                    self.words[mode].setdefault(term, set()).add(rid)
            self.available[mode] = True

        base = assets / "undlid_keywords.csv"
        if base.exists():
            add(pd.read_csv(base), "keywords", "default")
        three_d = assets / "undlid_keywords_3d_noun_fixed.csv"
        if three_d.exists():
            frame = pd.read_csv(three_d).rename(columns={"Original_ID": "undl_id"})
            for mode in ("geopolitical", "thematic", "action"):
                if mode.title() in frame:
                    add(frame, mode.title(), mode)
        subject = engine.subject_table
        links = engine.resolution_subject_table
        if not subject.empty and not links.empty:
            labels = subject.set_index("subject_id")["label_en"].to_dict()
            for sid, label in labels.items():
                term = re.sub(r"\s+", " ", str(label).strip().lower())
                self.subject_ids.setdefault(term, []).append(sid)
            frame = links.copy()
            frame["terms"] = frame.subject_id.map(labels).fillna(frame.subject_id)
            add(frame, "terms", "category", re.compile(r"\||--|;"), set())

    def search(self, token: str, exact: bool = False) -> set:
        words = self.words["default"]
        token = token.lower().strip()
        if not token:
            return set()
        if exact:
            return set(words.get(token, set()))
        matches = {key for key in words if token in key}
        matches.update(
            key for key, _, _ in process.extract(token, list(words), score_cutoff=80, limit=20)
        )
        return set().union(*(words[key] for key in matches))

    def matching_ids(self, frame: pd.DataFrame, expression: str) -> set:
        result = set()
        for clause in (c.strip() for c in expression.split(",") if c.strip()):
            clause_ids = None
            for term in (t.strip() for t in clause.split("&") if t.strip()):
                exact = len(term) >= 2 and term[0] == '"' and term[-1] == '"'
                token = term[1:-1].strip() if exact else term
                if not token:
                    continue
                title_match = frame.title.str.lower().str.contains(
                    token.lower(), regex=False, na=False
                )
                ids = set(frame.loc[title_match, "undl_id"]) | self.search(token, exact)
                clause_ids = ids if clause_ids is None else clause_ids & ids
                if not clause_ids:
                    break
            result.update(clause_ids or set())
        return result

    def frequencies(self, frame, mode):
        ids = set(frame.undl_id)
        scores = frame.set_index("undl_id").consensus_score.to_dict()
        counts = Counter({term: len(rids & ids) for term, rids in self.words[mode].items()})
        excluded = {"peoples", "states", "united nations"} if mode == "geopolitical" else set()
        result = []
        # Stable ties keep repeated requests and exported analyses reproducible.
        for term, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if not count or term in excluded:
                continue
            observed = [scores[r] for r in self.words[mode][term] & ids if pd.notna(scores[r])]
            result.append(
                {
                    "term": term,
                    "count": count,
                    "consensus": sum(observed) / len(observed) if observed else None,
                    "subject_ids": self.subject_ids.get(term, []) if mode == "category" else [],
                }
            )
        return result[:20]
