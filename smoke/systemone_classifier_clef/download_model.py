import os,json
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ["HF_HOME"]=str(root/"cache/hf")
os.environ["HF_XET_CACHE"]=str(root/"cache/xet")
from huggingface_hub import snapshot_download
revision="a30162b99d68a481d8bd75d38df069a0cf83df40"
path=snapshot_download("Cloudflare/clef",revision=revision,cache_dir=str(root/"cache/hf/hub"),allow_patterns=["*.json","*.jinja","*.safetensors"],max_workers=2)
(root/"model-download.json").write_text(json.dumps({"revision":revision,"snapshot_path":path},indent=2)+"\n")
print(path,flush=True)
