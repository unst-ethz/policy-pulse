"""Existing institutional era and comparison presets."""

# Country group presets for quick comparison selection
# Groups are based on official UN bodies and regional groupings
COUNTRY_PRESETS = {
    "p5": {
        "label": "P5 – Security Council Permanent Members",
        "countries": ["USA", "GBR", "FRA", "RUS", "CHN"],
    },
    "africa_group": {
        "label": "African Group (Representative)",
        "countries": ["NGA", "ZAF", "EGY", "KEN", "ETH"],
    },
    "asia_pacific": {
        "label": "Asia-Pacific Group (Representative)",
        "countries": ["IND", "JPN", "IDN", "BGD", "PAK"],
    },
    "grulac": {
        "label": "GRULAC – Latin America & Caribbean",
        "countries": ["BRA", "MEX", "ARG", "COL", "CHL"],
    },
    "sids": {
        "label": "Small Island Developing States (SIDS)",
        "countries": ["MDV", "FJI", "JAM", "TTO", "VUT"],
    },
}

# Era presets for quick year range selection
# Periods are anchored to UN institutional milestones, not geopolitical blocs
ERA_PRESETS = {
    "un_founding": {
        "label": "1945 to 1954 (UN Founding Era)",
        "start": 1945,
        "end": 1954,
    },
    "decolonization": {
        "label": "1955 to 1974 (Decolonization Era)",
        "start": 1955,
        "end": 1974,
    },
    "nieo_period": {
        "label": "1974 to 1991 (North–South Dialogue)",
        "start": 1974,
        "end": 1991,
    },
    "post_bipolarity": {
        "label": "1992 to 2000 (Post-Bipolarity Era)",
        "start": 1992,
        "end": 2000,
    },
    "mdg_era": {
        "label": "2001 to 2015 (Millennium Development Goals)",
        "start": 2001,
        "end": 2015,
    },
    "sdg_era": {
        "label": "2016 to present (Sustainable Development Goals)",
        "start": 2016,
        "end": None,  # will use latest year
    },
    "until_1991": {
        "label": "All years until 1991",
        "start": 1945,
        "end": 1991,
    },
    "since_1992": {
        "label": "All years since 1992",
        "start": 1992,
        "end": None,  # will use latest year
    },
}

# Ordered sequence of the six institutional eras for ◀ ▶ navigation
# Excludes the two cross-cutting presets (until_1991, since_1992)
ERA_SEQUENCE = [
    "un_founding",
    "decolonization",
    "nieo_period",
    "post_bipolarity",
    "mdg_era",
    "sdg_era",
]
