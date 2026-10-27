import pathlib,shutil,json,importlib.metadata
root=pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
for name in ["entrypoints/openai/decisions/strategies.py","model_executor/models/clef_flash.py","model_executor/models/clef_joint_schema.py"]:
 source=root/name
 shutil.copyfile(source,pathlib.Path("/work/docs")/("runtime-"+source.name))
print(json.dumps({x:importlib.metadata.version(x) for x in ["vllm","torch","transformers"]}))
