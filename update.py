from pathlib import Path
import dotenv
import obsidian_to_hugo
import os
import time
import subprocess

dotenv.load_dotenv(Path(__file__).parent / ".env")

OBSIDIAN_DIR = os.getenv("OBSIDIAN_DIR",'')
HUGO_DIR = os.getenv("HUGO_DIR",'')

assert OBSIDIAN_DIR !='', "OBSIDIAN_DIR is not set"
assert HUGO_DIR !='', "HUGO_DIR is not set"

check_interval = 60
hash_file = Path(__file__).parent / "obsidian_to_hugo.hash"

def get_hash(dir: str) -> str:
    import hashlib
    import os
    hash = hashlib.sha256()
    for root, _, files in os.walk(dir):
        # exclude hidden directories and files
        if "/." in root or "\\." in root:
            continue
        for file in files:
            with open(os.path.join(root, file), "rb") as f:
                hash.update(f.read())
    return hash.hexdigest()

def push_to_git():
    print("Pushing to git")
    subprocess.run(["git", "add", "."], cwd=HUGO_DIR)
    subprocess.run(["git", "commit", "-m", "Auto update"], cwd=HUGO_DIR)
    subprocess.run(["git", "push"], cwd=HUGO_DIR)
    print("Push complete")

def check(o2h: obsidian_to_hugo.ObsidianToHugo):
    current_hash = get_hash(OBSIDIAN_DIR)
    last_hash = open(hash_file).read() if os.path.exists(hash_file) else ""
    if current_hash != last_hash:
        print("Obsidian directory has changed, updating...")
        o2h.run()
        with open(hash_file, "w") as f:
            f.write(current_hash)

        push_to_git()
        print("Update complete")
    else:
        print("No changes detected")

if __name__ == "__main__":
    o2h = obsidian_to_hugo.ObsidianToHugo(OBSIDIAN_DIR, HUGO_DIR)
    while True:
        check(o2h)
        time.sleep(check_interval)

