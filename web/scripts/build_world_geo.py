"""Build public/geo/world.json, the country shapes the maps draw, from Natural Earth.

Source: Natural Earth 1:50m Admin 0 – Countries (public domain), as GeoJSON from
https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson

1:50m rather than 1:110m because the coarser file drops most small island states, which are a
sizeable share of the 193 UN members. Properties are cut down to `iso3` and `name`, and
coordinates are rounded to 0.01° (~1 km), which keeps the file small without visible loss at the
zoom levels the app uses.

    python3 -I web/scripts/build_world_geo.py <downloaded.geojson> web/public/geo/world.json
"""

import json
import sys


def _round(coords):
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], 2), round(coords[1], 2)]
    return [_round(c) for c in coords]


def main(source: str, target: str) -> None:
    with open(source, encoding="utf-8") as fh:
        collection = json.load(fh)

    features = []
    for feature in collection["features"]:
        props = feature["properties"]
        # ISO_A3 is "-99" for a few states with disputed parts (France, Norway); the "EH" variant
        # and ADM0_A3 carry the code they are known by.
        iso3 = next(
            (
                c
                for c in (props.get("ISO_A3"), props.get("ISO_A3_EH"), props.get("ADM0_A3"))
                if c and c != "-99"
            ),
            None,
        )
        if iso3 is None:
            continue
        geometry = feature["geometry"]
        features.append(
            {
                "type": "Feature",
                "properties": {"iso3": iso3, "name": props.get("NAME")},
                "geometry": {
                    "type": geometry["type"],
                    "coordinates": _round(geometry["coordinates"]),
                },
            }
        )

    with open(target, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh, separators=(",", ":"))
    print(f"{len(features)} features -> {target}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
