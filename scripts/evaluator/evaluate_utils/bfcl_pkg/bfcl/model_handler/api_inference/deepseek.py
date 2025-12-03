import json
import os
import time

from omegaconf import OmegaConf
from .openai_completion import OpenAICompletionsHandler
from ..model_style import ModelStyle
from ..utils import (
    combine_consecutive_user_prompts,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from overrides import override

try:
    from scripts.wandb_singleton import WandbConfigSingleton
except ImportError:
    WandbConfigSingleton = None


class DeepSeekAPIHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Completions
        self.client = OpenAI(
            base_url="https://api.deepseek.com", api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY")
        )

    # The deepseek API is unstable at the moment, and will frequently give empty responses, so retry on JSONDecodeError is necessary
    @retry_with_backoff(error_type=[RateLimitError, json.JSONDecodeError])
    def generate_with_backoff(self, **kwargs):
        """
        Per the DeepSeek API documentation:
        https://api-docs.deepseek.com/quick_start/rate_limit

        DeepSeek API does NOT constrain user's rate limit. We will try out best to serve every request.
        But please note that when our servers are under high traffic pressure, you may receive 429 (Rate Limit Reached) or 503 (Server Overloaded). When this happens, please wait for a while and retry.

        Thus, backoff is still useful for handling 429 and 503 errors.
        """
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        # Source https://api-docs.deepseek.com/quick_start/pricing
        # This will need to be updated if newer models are released.
        if "DeepSeek-V3" in self.model_name:
            api_model_name = "deepseek-chat"
        elif "DeepSeek-R1" in self.model_name:
            api_model_name = "deepseek-reasoner"
        else:
            raise ValueError(
                f"Model name {self.model_name} not yet supported in this method"
            )

        if len(tools) > 0:
            return self.generate_with_backoff(
                model=api_model_name,
                messages=message,
                tools=tools,
                temperature=self.temperature,
            )
        else:
            return self.generate_with_backoff(
                model=api_model_name,
                messages=message,
                temperature=self.temperature,
            )

    @override
    def _query_prompting(self, inference_data: dict):
        """
        This method is intended to be used by the `DeepSeek-R1` models. If used for other models, you will need to modify the code accordingly.

        Reasoning models don't support temperature parameter
        https://api-docs.deepseek.com/guides/reasoning_model

        `DeepSeek-R1` should use `deepseek-reasoner` as the model name in the API
        https://api-docs.deepseek.com/quick_start/pricing
        """
        message: list[dict] = inference_data["message"]
        inference_data["inference_input_log"] = {"message": repr(message)}

        if "DeepSeek-R1" in self.model_name:
            api_model_name = "deepseek-reasoner"
        else:
            raise ValueError(
                f"Model name {self.model_name} not yet supported in this method"
            )

        return self.generate_with_backoff(
            model=api_model_name,
            messages=message,
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        # 'deepseek-reasoner does not support successive user messages, so we need to combine them
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        return {"message": []}

    @override
    def _parse_query_response_prompting(self, api_response: any) -> dict:
        response_data = super()._parse_query_response_prompting(api_response)
        self._add_reasoning_content_if_available(api_response, response_data)
        return response_data


class DeepSeekV32APIHandler(DeepSeekAPIHandler):
    """
    Handler for DeepSeek V3.2 API with reasoning support.
    
    DeepSeek V3.2 supports the `reasoning` parameter to control thinking mode.
    See: https://api-docs.deepseek.com/guides/reasoning_model
    """
    
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.thinking_param = None
        
        # Load reasoning config from WandbConfigSingleton if available
        if WandbConfigSingleton is not None:
            try:
                instance = WandbConfigSingleton.get_instance()
                cfg = instance.config
                gen_cfg = getattr(cfg, "generator", {})
                
                # OmegaConf.DictConfig を dict に変換
                if hasattr(OmegaConf, "to_container") and not isinstance(gen_cfg, dict):
                    gen_cfg = OmegaConf.to_container(gen_cfg)
                
                extra_body = gen_cfg.get("extra_body", {}) if isinstance(gen_cfg, dict) else {}
                # DeepSeek uses 'thinking' parameter
                thinking = extra_body.get("thinking") if isinstance(extra_body, dict) else None
                
                if thinking is not None:
                    self.thinking_param = thinking
            except Exception:
                pass
    
    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        
        # DeepSeek V3.2 Thinking Mode uses "deepseek-reasoner"
        # Non-thinking Mode uses "deepseek-chat"
        # See: https://api-docs.deepseek.com/quick_start/pricing
        if self.thinking_param is not None:
            api_model_name = "deepseek-reasoner"
        else:
            api_model_name = "deepseek-chat"
        
        kwargs = {
            "model": api_model_name,
            "messages": message,
        }
        
        if len(tools) > 0:
            kwargs["tools"] = tools
        
        # Add thinking parameter if configured
        if self.thinking_param is not None:
            kwargs["extra_body"] = {"thinking": self.thinking_param}
        
        # deepseek-reasoner doesn't support temperature parameter
        # Only add temperature for non-reasoning mode
        if api_model_name == "deepseek-chat":
            kwargs["temperature"] = self.temperature
        
        return self.generate_with_backoff(**kwargs)
