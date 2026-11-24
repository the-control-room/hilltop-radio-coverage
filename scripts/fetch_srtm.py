"""Downloads the six SRTM3 tiles this project uses into data/srtm and checks them.

Usage: python scripts/fetch_srtm.py

The tiles are SRTM3 version 2.1 (NASA / USGS, public domain), three arc-seconds. The
USGS server that used to publish them (dds.cr.usgs.gov) is retired, so they are fetched
from its Internet Archive copy. SHA-256 sums of the unzipped files are checked.
"""

import hashlib
import io
import os
import urllib.request
import zipfile

BASE = "https://web.archive.org/web/2017id_/https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/North_America/"
SUMS = {
    "N25W100.hgt": "3036b64718bd150baef168b36c93646529711daf7104521cafd4aef2585b06d2",
    "N25W101.hgt": "38ee90682a403a330c3a42c28e9e2834e0ac5c5cdf7b6a3dceac3125bc4ba945",
    "N26W100.hgt": "39058bf7c4007296435b7711470b8e737e888f160396878f9f3a248bcfe4353a",
    "N26W101.hgt": "ee6e85623b5cb63f4da1188041bf6d8d3d2be528dd6075d2b49e31d21649a31f",
    "N27W100.hgt": "11df6c8e47a0486cc7a248e0e89984c78017144783591f90fa8ce9ddbf803ebd",
    "N27W101.hgt": "9ac35c7ee280e91961556adf70c8fac42ae1e0b54ab716011b72b5394e73e7a3",
}


def main() -> None:
    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "srtm")
    os.makedirs(folder, exist_ok=True)
    for name, expected in SUMS.items():
        target = os.path.join(folder, name)
        if not os.path.exists(target):
            request = urllib.request.Request(BASE + name + ".zip", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            with open(target, "wb") as handle:
                handle.write(archive.read(name))
        with open(target, "rb") as handle:
            actual = hashlib.sha256(handle.read()).hexdigest()
        print(("ok       " if actual == expected else "MISMATCH ") + name)
        if actual != expected:
            raise SystemExit(f"{name}: checksum {actual} does not match {expected}")


if __name__ == "__main__":
    main()
