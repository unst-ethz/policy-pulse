"""Explanations and thresholds of the existing Policy Pulse methodology."""

METHODOLOGY = {
    "version": "1.0.0",
    "scope": "Adopted UN General Assembly resolutions in the UN Digital Library voting dataset.",
    "vote_encoding": {"Y": 1, "A": 0, "N": -1},
    "formula": "agreement(i, j, r) = 1 - |vote(i, r) - vote(j, r)| / 2",
    "definitions": {
        "bilateral": "Mean per-resolution agreement where both countries cast Y, N, or A. Identical votes score 1; Yes versus No scores 0; an abstention versus Yes or No scores 0.5.",
        "consensus": "Mean agreement over unique pairs of voting countries on a resolution, excluding self-comparisons. Undefined with fewer than two voters.",
        "multilateral": "For each resolution, mean agreement with every other voting country; then the unweighted mean of that country's valid resolution scores.",
        "vote_rates": "Yes, No, and abstention counts divided by participation (Y + N + A). X and missing votes are excluded from scoring and these denominators.",
        "timeline": "Mean agreement per General Assembly session, with at least 3 shared votes. The year is the year of the session's median resolution date. Uses full history; special/emergency sessions are excluded by default.",
        "subjects": "Mean bilateral agreement over resolutions assigned to a subject or its descendants, counting each resolution once per subject. Requires at least 30 shared votes. Date filters apply.",
        "profile": "Profile dates are clamped to the existing widest UN membership year range. Bilateral rankings require 100 shared votes. Multilateral ranks use descending scores and minimum rank for ties. Profile vote composition includes non-votes.",
        "resolution_filters": "AGREED and DISAGREED compare recorded vote codes after excluding missing values, preserving the existing list behavior (including X). STRONGLY_DISAGREED means Yes versus No. These list filters are distinct from agreement scoring.",
        "keywords": "Comma separates OR clauses; & combines terms with AND. Quoted terms match exact keyword entries; title matching remains a literal substring. Unquoted keyword lookup also uses RapidFuzz with cutoff 80 and limit 20.",
        "words": "Counts are the number of selected resolutions associated with each indexed term, once per resolution. Uses the existing keyword CSVs or UN subject labels; it does not generate new topics. Consensus coloring averages available resolution consensus scores.",
    },
    "thresholds": {
        "timeline_shared_votes": 3,
        "subject_shared_votes": 30,
        "profile_shared_votes": 100,
        "multilateral_votes": 10,
    },
    "limitations": [
        "Only adopted General Assembly resolutions are covered; rejected or withdrawn proposals and other UN bodies are outside the current scope.",
        "Scores describe recorded voting similarity, not political intent, policy quality, or diplomatic relationships.",
        "Resolutions are unweighted. Frequently recurring subjects can have an outsized influence on aggregate scores.",
        "X (did not vote) and missing/non-member data are distinct in resolution details; neither is an abstention or disagreement score.",
        "Membership filters preserve the existing widest membership interval; gaps in multi-period membership are not modeled.",
        "Historical entities and aliases follow the existing UN authority table. Geographic groupings use the existing M49 asset.",
        "Keyword assets are curated snapshots and may not cover newly ingested resolutions. No substitute keywords are inferred when coverage is missing.",
    ],
    "sources": [
        "https://digitallibrary.un.org/record/4060887",
        "https://digitallibrary.un.org/record/4075456",
        "https://digitallibrary.un.org/record/4082085",
    ],
}
