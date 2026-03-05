"""
Standalone Chinese board evaluation runner.
Zero dependency on Japanese evaluation modules.
"""
import os, sys
from pathlib import Path
from argparse import ArgumentParser
import wandb, weave, openai, httpx
from omegaconf import OmegaConf
from dotenv import load_dotenv
from config_singleton import WandbConfigSingleton

load_dotenv()

parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
args = parser.parse_args()

config_dir = Path("configs")
custom_cfg_path = config_dir / args.config
if custom_cfg_path.suffix != ".yaml":
    custom_cfg_path = custom_cfg_path.with_suffix(".yaml")
assert custom_cfg_path.exists(), f"Config not found: {custom_cfg_path}"

base_cfg = OmegaConf.load(config_dir / "base_config.yaml")
custom_cfg = OmegaConf.load(custom_cfg_path)
cfg = OmegaConf.merge(base_cfg, custom_cfg)
cfg_dict = OmegaConf.to_container(cfg, resolve=True)

wandb.login()
run = wandb.init(
    entity=cfg_dict["wandb"]["entity"],
    project=cfg_dict["wandb"]["project"],
    name=cfg_dict["wandb"]["run_name"],
    config=cfg_dict,
    job_type="evaluation",
)
weave.init(cfg_dict["wandb"]["entity"] + "/" + cfg_dict["wandb"]["project"])

api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", os.environ.get("VLLM_API_KEY", "EMPTY"))

class LLM:
    def __init__(self):
        self.client = openai.OpenAI(api_key=api_key, base_url=cfg.get("base_url", "http://localhost:8000/v1"), timeout=httpx.Timeout(connect=10, read=300, write=300, pool=30), max_retries=3)
        self.model = cfg.model.pretrained_model_name_or_path
        self.gen = OmegaConf.to_container(cfg.generator, resolve=True)
    def invoke(self, messages, max_tokens=None, **kwargs):
        p = {"model": self.model, "messages": messages, "temperature": self.gen.get("temperature", 0.1), "top_p": self.gen.get("top_p", 1.0)}
        if max_tokens: p["max_tokens"] = max_tokens
        r = self.client.chat.completions.create(**p)
        class R: pass
        o = R(); o.content = r.choices[0].message.content or ""; return o

llm = LLM()
WandbConfigSingleton.initialize(run=run, llm=llm)

print(f"\n{'='*60}\nZH Board | Model: {cfg.model.pretrained_model_name_or_path}\n{'='*60}\n")

if cfg.run.get("tmmluplus", False):
    print("[TMMLU+] starting...")
    from evaluator.tmmluplus import evaluate as f; f()
    print("[TMMLU+] done\n")

if cfg.run.get("cbbq_zh", False):
    print("[CBBQ] starting...")
    from evaluator.cbbq import evaluate as f; f()
    print("[CBBQ] done\n")

if cfg.run.get("taiwan_zh_tw", False):
    print("[TW] starting...")
    from evaluator.taiwan_zh_tw import evaluate as f; f()
    print("[TW] done\n")

print(f"\nRun URL: {run.get_url()}\n")
run.finish()
