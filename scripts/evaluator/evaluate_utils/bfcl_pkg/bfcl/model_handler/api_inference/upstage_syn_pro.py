import json
import os
import time

from ..base_handler import BaseHandler
from ...constants.type_mappings import GORILLA_TO_OPENAPI
from ..model_style import ModelStyle
from ..utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from config_singleton import WandbConfigSingleton


class UpstageSynProHandler(BaseHandler):
    # デバッグ出力フラグ（Trueでデバッグ出力を有効化）
    DEBUG_OUTPUT = True

    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Completions  # Upstage uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
            base_url="http://solar-pro-vllm-cluster-alb-3-1453891318.us-east-2.elb.amazonaws.com/v1"
        )
        
        # 設定ファイルからchat_template_kwargsを取得
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        self.chat_template_kwargs = None
        
        # generator.chat_template_kwargsから取得
        try:
            if hasattr(cfg, 'generator') and hasattr(cfg.generator, 'chat_template_kwargs'):
                from omegaconf import OmegaConf
                # DictConfigをPythonの辞書に変換
                self.chat_template_kwargs = OmegaConf.to_container(cfg.generator.chat_template_kwargs)
        except Exception:
            pass

    def decode_ast(self, result, language="Python"):
        if "FC" in self.model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language)

    def decode_execute(self, result):
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()

        # デバッグ出力: リクエスト内容
        if self.DEBUG_OUTPUT:
            print("=" * 80)
            print("[DEBUG] Request to LLM:")
            print(f"Model: {kwargs.get('model', 'N/A')}")
            print(f"Temperature: {kwargs.get('temperature', 'N/A')}")
            print(f"Messages ({len(kwargs.get('messages', []))} messages):")
            for i, msg in enumerate(kwargs.get('messages', [])):
                try:
                    print(f"  [{i}] role={msg.get('role', 'N/A')}")
                    print(f"      content={msg.get('content', '')}")
                except Exception:
                    print(f"  [{i}] role={msg.role}")
                    print(f"      content={msg.content}")
            if 'tools' in kwargs:
                print(f"Tools: {len(kwargs['tools'])} functions")
                for i, tool in enumerate(kwargs['tools']):
                    func_name = tool.get('function', {}).get('name', 'N/A')
                    print(f"  [{i}] {func_name}")
            print("=" * 80)

        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        # デバッグ出力: レスポンス内容
        if self.DEBUG_OUTPUT:
            print("=" * 80)
            print("[DEBUG] Response from LLM:")
            print(f"Time taken: {end_time - start_time:.2f}s")
            print(f"Model: {api_response.model}")
            print(f"Usage: prompt_tokens={api_response.usage.prompt_tokens}, completion_tokens={api_response.usage.completion_tokens}")
            print(f"Finish reason: {api_response.choices[0].finish_reason}")

            # レスポンスの内容を表示
            message = api_response.choices[0].message
            print(f"Response role: {message.role}")

            if message.content:
                print(f"Response content: {message.content}")

            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"Tool calls: {len(message.tool_calls)}")
                for i, tool_call in enumerate(message.tool_calls):
                    print(f"  [{i}] id={tool_call.id}")
                    print(f"      function={tool_call.function.name}")
                    print(f"      arguments={tool_call.function.arguments}")

            # 生のレスポンステキスト
            try:
                raw_response = api_response.model_dump_json(indent=2)
                print(f"Raw response: {raw_response}")
            except:
                print("Raw response: (unable to serialize)")

            print("=" * 80)

        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        # 基本パラメータ
        base_params = {
            "messages": message,
            "model": self.model_name.replace("-FC", ""),
            "temperature": self.temperature,
        }
        
        # chat_template_kwargsが設定されている場合は追加
        if self.chat_template_kwargs:
            base_params["extra_body"] = {
                "chat_template_kwargs": self.chat_template_kwargs
            }
        
        if len(tools) > 0:
            # Upstage Solar models support temperature parameter
            base_params.update({
                "tools": tools,
                "tool_choice": "auto"  # Upstage supports tool_choice parameter
            })
            return self.generate_with_backoff(**base_params)
        else:
            return self.generate_with_backoff(**base_params)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: any) -> dict:
        try:
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in api_response.choices[0].message.tool_calls
            ]
            tool_call_ids = [
                func_call.id for func_call in api_response.choices[0].message.tool_calls
            ]
        except:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.choices[0].message

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        # 基本パラメータ
        base_params = {
            "messages": inference_data["message"],
            "model": self.model_name,
            "temperature": self.temperature,
        }
        
        # chat_template_kwargsが設定されている場合は追加
        if self.chat_template_kwargs:
            base_params["extra_body"] = {
                "chat_template_kwargs": self.chat_template_kwargs
            }
        
        # Upstage Solar models support temperature parameter
        return self.generate_with_backoff(**base_params)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        return {
            "model_responses": api_response.choices[0].message.content,
            "model_responses_message_for_chat_history": api_response.choices[0].message,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data

    # Upstage Solar models may support reasoning content in the future
    # This method is included for compatibility
    def _add_reasoning_content_if_available(
        self, api_response: any, response_data: dict
    ) -> None:
        message = api_response.choices[0].message
        if hasattr(message, "reasoning_content"):
            response_data["reasoning_content"] = message.reasoning_content
            # Reasoning content should not be included in the chat history
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": str(response_data["model_responses"]),
            }