# Lazy imports to avoid pulling heavy deps (comet, sacrebleu, etc.) when not needed
import importlib as _il

def __getattr__(name):
    _submodules = {
        'formatter': '.formatter',
        'metrics': '.metrics',
        'prompt': '.prompt',
        'LLMAsyncProcessor': '.llm_async_processor',
        'get_openai_judge_client': '.llm_judge_client',
        'evaluate_robustness': '.robustness',
        'answer_parser': '.answer_parser',
        'Sample': '.to_be_deprecated',
    }
    for attr, mod_path in _submodules.items():
        if name == attr:
            mod = _il.import_module(mod_path, __name__)
            return getattr(mod, name, mod)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
