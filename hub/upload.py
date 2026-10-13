"""Push the built repositories to the Hugging Face Hub.

    python -m hub.upload [--family llada|nemotron] [--message "..."] [--dist /ssd1/.../pig_hub_dist]

Uploads hub/dist/<name>/ to the repository named in hub.build.REPOS, replacing the files of the same
names and deleting files that are no longer part of the build (only inside that repository). It
needs a token with write access to the UT-IFML organization (hf auth login).
"""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from hub.build import DIST, REPOS, build


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(REPOS), action="append")
    ap.add_argument("--message", default="Rebuild from PIG-DLM-release")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--dist", help="upload this build directory (hub.build --out DIR) instead of building")
    a = ap.parse_args()
    api = HfApi()
    me = api.whoami()
    print("acting as", me["name"], "| orgs:", [o["name"] for o in me.get("orgs", [])])
    for fam in a.family or sorted(REPOS):
        spec = REPOS[fam]
        d = Path(a.dist) / spec["name"] if a.dist else Path(DIST / spec["name"]) if a.no_build else build(fam)
        api.upload_folder(folder_path=str(d), repo_id=spec["repo"], repo_type="model",
                          commit_message=a.message, delete_patterns=["*"])
        for p in sorted(d.iterdir()):
            local = hashlib.sha256(p.read_bytes()).hexdigest()   # every file, shards included
            remote = hashlib.sha256(Path(hf_hub_download(spec["repo"], p.name, force_download=True)).read_bytes()).hexdigest()
            print(f"  {spec['repo']}/{p.name}: {'ok' if local == remote else 'MISMATCH'}")


if __name__ == "__main__":
    main()
