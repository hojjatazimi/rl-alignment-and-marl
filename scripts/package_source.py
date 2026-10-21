"""Small source zip for Colab upload; excludes virtualenv, caches, weights and logs."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
root=Path(__file__).resolve().parents[1]
out=root/'rl-alignment-and-marl-source.zip'
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for p in root.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(root)
        if any(part in {'.venv','.cache','.git','__pycache__','.pytest_cache','runs','reruns','wandb'} for part in relative.parts):continue
        if p.name=='MANIFEST.sha256.json':continue
        if p.suffix in {'.zip','.pt','.safetensors','.pyc'}:continue
        z.write(p,Path(root.name)/relative)
print(out)
