# Evaluator modules - lazy imports to avoid pulling unused dependencies
import importlib

_MODULE_NAMES = [
    'jaster', 'jbbq', 'mtbench', 'jaster_translation', 'toxicity',
    'jtruthfulqa', 'aggregate', 'bfcl', 'swe_bench', 'hallulens',
    'arc_agi', 'hle', 'm_ifeval', 'taiwan_zh_tw', 'tmmluplus', 'cbbq',
]

__all__ = list(_MODULE_NAMES)

def __getattr__(name):
    if name in _MODULE_NAMES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
