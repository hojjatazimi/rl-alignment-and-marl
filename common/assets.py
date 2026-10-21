"""Immutable HF revisions for known assets; custom models may specify their own ID@revision."""
import json
from pathlib import Path
LOCK=Path(__file__).resolve().parents[1]/'assets.lock.json'
ASSETS=json.loads(LOCK.read_text())

def asset(repo):
    if '@' in repo:
        name,revision=repo.rsplit('@',1)
        return dict(path=name,revision=revision)
    return dict(path=repo,revision=ASSETS.get(repo,{}).get('revision','main'))

def hf(repo):
    info=asset(repo)
    return info['path'], {'revision':info['revision']}
