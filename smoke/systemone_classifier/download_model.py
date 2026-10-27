import json
from pathlib import Path
from huggingface_hub import snapshot_download

revision = 'f0b7d3cd65296abfe459e6c228b618f5d8b26d2f'
path = snapshot_download('Cloudflare/clef-flash', revision=revision,
                         allow_patterns=['*.json','*.jinja','*.safetensors'],max_workers=4)
Path('/work/model-download.json').write_text(json.dumps({'revision':revision,'snapshot_path':path},indent=2)+'\n')
print(path,flush=True)
