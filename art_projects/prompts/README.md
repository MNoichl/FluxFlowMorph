# Art-loop prompt files

Put project-specific JSON prompt plans in this folder. They are ignored by Git so unfinished prompts and private project material stay local. The single tracked exception is `example_still_life_loop.json`, which documents and exercises the schema.

In `notebooks/StillLife_Loop_Runware_Art.ipynb`, set `PROMPT_SPEC_PATH` to the JSON file available to the active kernel. When a VS Code notebook is attached to a remote Colab kernel, copy or upload the JSON into `/content/FlowMorphKlein9B/art_projects/prompts/` first.

Each transition must follow the listed mainframe order and contain exactly 20 ordered bridge prompts. For a closed loop, include the final transition from the last mainframe back to the first.

The example pins the RIJKSOIL LoRA repository, revision, weight filename, scale, and `RIJKSOIL` trigger prefix. Its checkpoint metadata names `flux2_klein_9b` without explicitly saying `base`, so the example records the deliberate experimental provenance override; runtime tensor shapes are still checked against the loaded Base-9B transformer.
