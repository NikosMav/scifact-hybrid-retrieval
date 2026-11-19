"""Fetch the checksum-verified SciFact release and extract its evaluation inputs."""

import argparse
import hashlib
import io
import tarfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# Hugging Face datasets 2.5.0, datasets/scifact/dataset_infos.json
SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
URLS = (
    "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
    "https://raw.githubusercontent.com/marwasayed72/Scientific-Claim-Verifier/"
    "a3ffd2a17d15e3a3cf0478f8deda3720cf33a7b2/data.tar.gz",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    destination = parser.parse_args().destination
    for url in URLS:
        try:
            with urlopen(url, timeout=30) as response:
                archive = response.read()
            break
        except URLError:
            if url == URLS[-1]:
                raise
    if hashlib.sha256(archive).hexdigest() != SHA256:
        raise ValueError("SciFact archive checksum mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for filename in ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"]:
            payload = bundle.extractfile("data/" + filename).read()
            path = destination / filename
            if path.exists():
                if path.read_bytes() != payload:
                    raise FileExistsError(f"Existing file differs from release: {path}")
            else:
                path.write_bytes(payload)
    print(f"Verified SciFact corpus and judgments in {destination}")


if __name__ == "__main__":
    main()
