"""Shared country-level lookup utilities."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

_ASSETS = Path(__file__).resolve().parent.parent / "assets"


@lru_cache(maxsize=None)
def _load_joining_dates() -> pd.DataFrame:
    return pd.read_csv(_ASSETS / "joining_dates.csv", parse_dates=["min_date", "max_date"])


@lru_cache(maxsize=None)
def _load_m49() -> pd.DataFrame:
    return pd.read_csv(_ASSETS / "m49_regional_groupings.csv", sep=";")


@lru_cache(maxsize=None)
def _country_to_region() -> dict[str, str]:
    return {
        row["ISO-alpha3 Code"].strip(): row["Region Name"].strip()
        for _, row in _load_m49().iterrows()
        if isinstance(row.get("ISO-alpha3 Code"), str) and isinstance(row.get("Region Name"), str)
    }


@lru_cache(maxsize=None)
def _country_to_subregion() -> dict[str, str]:
    return {
        row["ISO-alpha3 Code"].strip(): row["Sub-region Name"].strip()
        for _, row in _load_m49().iterrows()
        if isinstance(row.get("ISO-alpha3 Code"), str) and isinstance(row.get("Sub-region Name"), str)
    }


def get_un_membership_years(country_alpha3: str) -> tuple[int, int] | None:
    """Return (first_year, last_year) of UN membership for a country.

    For countries with multiple membership periods, returns the widest range
    (earliest join to latest leave/current year). Returns None if not found.
    """
    jd = _load_joining_dates()
    rows = jd[jd["country"] == country_alpha3]
    if rows.empty:
        return None
    return rows["min_date"].min().year, rows["max_date"].max().year


def get_country_region(iso3: str) -> str:
    """Return M49 top-level region name for a country, or 'Other' if not found."""
    return _country_to_region().get(iso3, "Other")


def get_country_subregion(iso3: str) -> str | None:
    """Return M49 sub-region name for a country, or None if not found."""
    return _country_to_subregion().get(iso3)


def get_country_longitude(country_code: str) -> float:
    """Return the longitude of a country's centre point. Returns 0.0 if not found.

    Positive values denote eastern longitudes; negative values denote western longitudes.
    """

    country_lons = {
        "AFG": 67.71,  # Afghanistan
        "ALB": 20.17,  # Albania
        "DZA": 1.66,  # Algeria
        "AND": 1.52,  # Andorra
        "AGO": 17.87,  # Angola
        "ATG": -61.80,  # Antigua and Barbuda
        "ARG": -63.62,  # Argentina
        "ARM": 45.04,  # Armenia
        "AUS": 133.78,  # Australia
        "AUT": 14.55,  # Austria
        "AZE": 47.58,  # Azerbaijan
        "BHS": -77.40,  # Bahamas
        "BHR": 50.56,  # Bahrain
        "BGD": 90.36,  # Bangladesh
        "BRB": -59.54,  # Barbados
        "BLR": 27.95,  # Belarus
        "BEL": 4.47,  # Belgium
        "BLZ": -88.50,  # Belize
        "BEN": 2.32,  # Benin
        "BTN": 90.43,  # Bhutan
        "BOL": -63.59,  # Bolivia
        "BIH": 17.68,  # Bosnia and Herzegovina
        "BWA": 24.68,  # Botswana
        "BRA": -51.93,  # Brazil
        "BRN": 114.73,  # Brunei
        "BGR": 25.49,  # Bulgaria
        "BFA": -1.56,  # Burkina Faso
        "BDI": 29.92,  # Burundi
        "CPV": -24.01,  # Cabo Verde
        "KHM": 104.99,  # Cambodia
        "CMR": 12.35,  # Cameroon
        "CAN": -106.35,  # Canada
        "CAF": 20.94,  # Central African Republic
        "TCD": 18.73,  # Chad
        "CHL": -71.54,  # Chile
        "CHN": 104.20,  # China
        "COL": -74.30,  # Colombia
        "COM": 43.87,  # Comoros
        "COG": 15.83,  # Congo
        "COD": 21.76,  # Congo (DRC)
        "CRI": -84.09,  # Costa Rica
        "CIV": -5.55,  # Côte d'Ivoire
        "HRV": 15.20,  # Croatia
        "CUB": -77.78,  # Cuba
        "CYP": 33.43,  # Cyprus
        "CZE": 15.47,  # Czechia
        "DNK": 9.50,  # Denmark
        "DJI": 42.59,  # Djibouti
        "DMA": -61.37,  # Dominica
        "DOM": -70.16,  # Dominican Republic
        "ECU": -78.18,  # Ecuador
        "EGY": 30.80,  # Egypt
        "SLV": -88.90,  # El Salvador
        "GNQ": 10.27,  # Equatorial Guinea
        "ERI": 39.78,  # Eritrea
        "EST": 25.01,  # Estonia
        "SWZ": 31.47,  # Eswatini
        "ETH": 40.49,  # Ethiopia
        "FJI": 179.41,  # Fiji
        "FIN": 25.75,  # Finland
        "FRA": 2.21,  # France
        "GAB": 11.61,  # Gabon
        "GMB": -15.31,  # Gambia
        "GEO": 43.36,  # Georgia
        "DEU": 10.45,  # Germany
        "GHA": -1.02,  # Ghana
        "GRC": 21.82,  # Greece
        "GRD": -61.68,  # Grenada
        "GTM": -90.23,  # Guatemala
        "GIN": -9.70,  # Guinea
        "GNB": -15.18,  # Guinea-Bissau
        "GUY": -58.93,  # Guyana
        "HTI": -72.29,  # Haiti
        "HND": -86.24,  # Honduras
        "HUN": 19.50,  # Hungary
        "ISL": -19.02,  # Iceland
        "IND": 78.96,  # India
        "IDN": 113.92,  # Indonesia
        "IRN": 53.69,  # Iran
        "IRQ": 43.68,  # Iraq
        "IRL": -8.24,  # Ireland
        "ISR": 34.85,  # Israel
        "ITA": 12.57,  # Italy
        "JAM": -77.30,  # Jamaica
        "JPN": 138.25,  # Japan
        "JOR": 36.24,  # Jordan
        "KAZ": 66.92,  # Kazakhstan
        "KEN": 37.91,  # Kenya
        "KIR": -168.73,  # Kiribati
        "PRK": 127.51,  # North Korea
        "KOR": 127.77,  # South Korea
        "KWT": 47.48,  # Kuwait
        "KGZ": 74.77,  # Kyrgyzstan
        "LAO": 102.50,  # Laos
        "LVA": 24.60,  # Latvia
        "LBN": 35.86,  # Lebanon
        "LSO": 28.23,  # Lesotho
        "LBR": -9.43,  # Liberia
        "LBY": 17.23,  # Libya
        "LIE": 9.56,  # Liechtenstein
        "LTU": 23.88,  # Lithuania
        "LUX": 6.13,  # Luxembourg
        "MDG": 46.87,  # Madagascar
        "MWI": 34.30,  # Malawi
        "MYS": 101.98,  # Malaysia
        "MDV": 73.54,  # Maldives
        "MLI": -3.99,  # Mali
        "MLT": 14.38,  # Malta
        "MHL": 171.18,  # Marshall Islands
        "MRT": -10.94,  # Mauritania
        "MUS": 57.55,  # Mauritius
        "MEX": -102.55,  # Mexico
        "FSM": 158.29,  # Micronesia
        "MDA": 28.37,  # Moldova
        "MCO": 7.41,  # Monaco
        "MNG": 103.85,  # Mongolia
        "MNE": 19.37,  # Montenegro
        "MAR": -7.09,  # Morocco
        "MOZ": 35.53,  # Mozambique
        "MMR": 95.96,  # Myanmar
        "NAM": 18.49,  # Namibia
        "NRU": 166.93,  # Nauru
        "NPL": 84.12,  # Nepal
        "NLD": 5.29,  # Netherlands
        "NZL": 174.89,  # New Zealand
        "NIC": -85.21,  # Nicaragua
        "NER": 8.08,  # Niger
        "NGA": 8.68,  # Nigeria
        "MKD": 21.75,  # North Macedonia
        "NOR": 8.47,  # Norway
        "OMN": 55.92,  # Oman
        "PAK": 69.35,  # Pakistan
        "PLW": 134.58,  # Palau
        "PSE": 35.23,  # Palestine
        "PAN": -80.78,  # Panama
        "PNG": 143.96,  # Papua New Guinea
        "PRY": -58.44,  # Paraguay
        "PER": -75.02,  # Peru
        "PHL": 121.77,  # Philippines
        "POL": 19.15,  # Poland
        "PRT": -8.22,  # Portugal
        "QAT": 51.18,  # Qatar
        "ROU": 24.97,  # Romania
        "RUS": 105.32,  # Russia
        "RWA": 29.87,  # Rwanda
        "KNA": -62.78,  # Saint Kitts and Nevis
        "LCA": -60.98,  # Saint Lucia
        "VCT": -61.29,  # Saint Vincent and the Grenadines
        "WSM": -172.10,  # Samoa
        "SMR": 12.46,  # San Marino
        "STP": 6.61,  # Sao Tome and Principe
        "SAU": 45.08,  # Saudi Arabia
        "SEN": -14.45,  # Senegal
        "SRB": 21.01,  # Serbia
        "SYC": 55.49,  # Seychelles
        "SLE": -11.78,  # Sierra Leone
        "SGP": 103.82,  # Singapore
        "SVK": 19.70,  # Slovakia
        "SVN": 14.99,  # Slovenia
        "SLB": 160.16,  # Solomon Islands
        "SOM": 46.20,  # Somalia
        "ZAF": 22.94,  # South Africa
        "SSD": 31.31,  # South Sudan
        "ESP": -3.75,  # Spain
        "LKA": 80.77,  # Sri Lanka
        "SDN": 30.22,  # Sudan
        "SUR": -56.03,  # Suriname
        "SWE": 18.64,  # Sweden
        "CHE": 8.23,  # Switzerland
        "SYR": 38.99,  # Syria
        "TJK": 71.28,  # Tajikistan
        "TZA": 34.89,  # Tanzania
        "THA": 100.99,  # Thailand
        "TLS": 125.73,  # Timor-Leste
        "TGO": 0.82,  # Togo
        "TON": -175.20,  # Tonga
        "TTO": -61.52,  # Trinidad and Tobago
        "TUN": 9.54,  # Tunisia
        "TUR": 35.24,  # Turkey
        "TKM": 59.56,  # Turkmenistan
        "TUV": 179.19,  # Tuvalu
        "UGA": 32.29,  # Uganda
        "UKR": 31.17,  # Ukraine
        "ARE": 53.85,  # United Arab Emirates
        "GBR": -3.44,  # United Kingdom
        "USA": -95.71,  # United States
        "URY": -55.77,  # Uruguay
        "UZB": 64.59,  # Uzbekistan
        "VUT": 166.96,  # Vanuatu
        "VAT": 12.45,  # Vatican City
        "VEN": -66.59,  # Venezuela
        "VNM": 108.28,  # Vietnam
        "YEM": 48.52,  # Yemen
        "ZMB": 27.85,  # Zambia
        "ZWE": 29.15,  # Zimbabwe
    }

    return country_lons.get(country_code, 0.0)
