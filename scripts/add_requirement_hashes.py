"""Attach PyPI's published artifact digests to already resolved, exact lockfiles.

Avoids downloading every platform's wheel just to hash it. Resolution remains
pip-compile's responsibility. Run after compiling both production and dev locks.
"""

import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LINE = re.compile(r"^([a-zA-Z0-9_.-]+)==([^\s;\\]+)$")


def main(paths):
    contents = {Path(path): Path(path).read_text() for path in paths}
    packages = {
        match.groups()
        for text in contents.values()
        for line in text.splitlines()
        if (match := LINE.fullmatch(line.strip()))
    }

    def fetch(package):
        name, version = package
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{name}/{version}/json", timeout=60
        ) as response:
            payload = json.load(response)
        hashes = sorted(
            {entry["digests"]["sha256"] for entry in payload["urls"] if not entry.get("yanked")}
        )
        if not hashes:
            raise ValueError(f"No published artifact hashes for {name}=={version}")
        return package, hashes

    with ThreadPoolExecutor(max_workers=6) as pool:
        hashes = dict(pool.map(fetch, sorted(packages)))
    for path, content in contents.items():
        output = []
        for line in content.splitlines():
            match = LINE.fullmatch(line.strip())
            if match:
                output.append(line + " \\")
                entries = hashes[match.groups()]
                output.extend(
                    f"    --hash=sha256:{digest}" + (" \\" if i < len(entries) - 1 else "")
                    for i, digest in enumerate(entries)
                )
            else:
                output.append(line)
        path.write_text("\n".join(output) + "\n")
        print(f"Hashed {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
