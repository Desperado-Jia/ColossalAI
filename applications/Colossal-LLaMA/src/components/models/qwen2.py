from typing import ContextManager, List, Tuple, Union, Dict, Any

import json
import re
from contextlib import nullcontext

from transformers.models.auto import AutoTokenizer, AutoModelForCausalLM

from .base import ChatLLM, ChatLLMConfig, Language
from ..data.schema import (
    Content, ContentType,
    Message, Role,
    RawSample, TokenizedSample,
    Tool, ToolCall, ToolType
)


class Qwen2ChatLLM(ChatLLM):
    """A class used to represent Qwen2-based LLMs."""

    _BOM_TOKEN = "<|im_start|>"  # The token at the beginning of the message
    _EOM_TOKEN = "<|im_end|>"  # The token at the ending of the message
    _PAD_TOKEN = "<|endoftext|>"

    # See: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/function_calling.py#L351
    _FN_NAME = '✿FUNCTION✿'
    _FN_ARGS = '✿ARGS✿'
    _FN_RESULT = '✿RESULT✿'
    _FN_EXIT = '✿RETURN✿'
    _FN_STOP_WORDS = [_FN_RESULT, _FN_EXIT]

    # Language-related settings.
    _FN_DESC_TPL = {
        Language.CHINESE: "### {name_for_human}\n\n"
                          "{name_for_model}: {description_for_model}\n输入参数：\n{parameters}\n{args_format}",
        Language.ENGLISH: "### {name_for_human}\n\n"
                          "{name_for_model}: {description_for_model}\nParameters：\n{parameters}\n{args_format}"
    }  # Template for function description.
    # See: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/function_calling.py#L453-L483
    _FN_PREFIX = {
        Language.CHINESE: "\n\n# 工具\n\n# 你拥有如下工具：",
        Language.ENGLISH: "\n\n# Tools\n\n## You have access to the following tools:"
    }
    _FN_SUFFIX_TPL = {
        Language.CHINESE: "## 你可以在回复中插入零次、一次或多次以下命令以调用工具：\n\n"
                          "%s: 工具名称，必须是[{names}]之一\n"
                          "%s: 工具输入\n"
                          "%s: 工具结果\n"
                          "%s: 根据工具结果进行回复，需将图片用![](url)渲染出来"
                          % (_FN_NAME, _FN_ARGS, _FN_RESULT, _FN_EXIT),
        Language.ENGLISH: "## When you need to call a tool, please insert the following command in your reply, "
                          "which can be called zero or multiple times according to your needs:\n\n"
                          "%s: The tool to use, should be one of [{names}]\n"
                          "%s: The input of the tool\n"
                          "%s: Tool results\n"
                          "%s: Reply based on tool results. Images need to be rendered as ![](url)"
                          % (_FN_NAME, _FN_ARGS, _FN_RESULT, _FN_EXIT),
    }  # Template for function,
    # see: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/function_calling.py#L369-L391
    # _FN_CALL_TPL = {
    #     Language.CHINESE: "\n\n工具 \"{name}\" 被调用时使用了以下参数：\n{arguments}",
    #     Language.ENGLISH: "\n\nThe tool \"{name}\" was called with the following arguments:\n{arguments}"
    # }  # Template for function calling. Note that it only used in inference mode,
    _FN_CALL_TPL = "%s: {name}\n%s: {arguments}" % (_FN_NAME, _FN_ARGS)
    _ARGS_FORMAT = {
        Language.CHINESE: "此工具的输入应为JSON对象。",
        Language.ENGLISH: "Format the arguments as a JSON object."
    }  # See: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/function_calling.py#L473-L476
    _DEFAULT_SYSTEM = {
        Language.CHINESE: "你是一个名为 Qwen2 的人工智能助手，你的任务是针对用户的问题和要求提供适当的答复和支持。",
        Language.ENGLISH: "You are an AI assistant named Qwen2, and your task is to provide appropriate "
                          "responses and support to users' questions and requests."
    }
    _DEFAULT_LANGUAGE = Language.CHINESE

    def __init__(self, config: ChatLLMConfig) -> None:
        super().__init__(config=config)

    def init_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            trust_remote_code=self.config.trust_remote_code
        )
        if self.tokenizer.pad_token != self._PAD_TOKEN:
            raise RuntimeError(
                f"Invalid pad_token ({self.tokenizer.pad_token}), expected to be \"{self._PAD_TOKEN}\"."
            )

    def init_model(self, context: ContextManager = nullcontext()) -> None:
        with context:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
                trust_remote_code=self.config.trust_remote_code
            )

    def parse_response(self, text: str) -> Message:
        pattern = fr"{self._FN_NAME}:(\w+)\n{self._FN_ARGS}(.*?)"
        raise NotImplementedError()

    def _verify(self, sample: Union[Dict[str, Any], RawSample], training: bool) -> None:
        return

    def _tokenize(self, sample: RawSample, training: bool, **kwargs) -> TokenizedSample:
        """
        See: https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/tokenizer_config.json#L31
        """
        language = Language(kwargs.get("language", Language.CHINESE))  # `Union[str, Language]` -> `Language`
        suffix_token_ids = self.tokenizer.encode(text="\n", add_special_tokens=False)  # `List[int]`, i.e., [198]

        input_ids = []  # `List[int]`
        labels = []  # `List[int]`
        for i, msg in enumerate(sample.messages):
            signal = self._get_signal(role=msg.role)
            # Indicates whether the current message needs to calculate loss,
            # which is used for efficient training of multi-turn conversation samples.
            trainable = self._is_trainable_msg(msg=msg, training=training)  # `bool`

            if msg.tools:  # Must appear in system, see: `ChatLLM._base_verify`
                content = self._textify_content(msg=msg)
                content += self._get_tools_prompt(tools=msg.tools, language=language)
                content += f"{self._EOM_TOKEN}\n"  # Add eos suffix tokens.

                tokenized = self.tokenizer.encode(
                    text=signal + content, add_special_tokens=False
                )  # `List[int]`, create a new latest system.
                input_ids.extend(tokenized)
                labels.extend([self.ignore_index for _ in range(len(tokenized))])
                continue

            if msg.tool_calls:  # Must appear in assistant, see: `ChatLLM._base_verify`
                content = self._textify_tool_calls(msg=msg)
                content += f"{self._EOM_TOKEN}\n"  # Add eos suffix tokens.

                tokenized = [
                    self.tokenizer.encode(text=t, add_special_tokens=False)
                    for t in (signal, signal + content)
                ]  # `List[List[int]]`
                input_ids.extend(tokenized[1])
                if trainable:
                    labels.extend(
                        [self.ignore_index for _ in range(len(tokenized[0]))] +
                        tokenized[1][len(tokenized[0]): -len(suffix_token_ids)] +
                        [self.ignore_index for _ in range(len(suffix_token_ids))]
                    )
                else:
                    labels.extend([self.ignore_index for _ in range(len(tokenized[1]))])
                continue

            content = self._textify_content(msg=msg)
            content += f"{self._EOM_TOKEN}\n"  # Add eos suffix tokens.
            tokenized = [
                self.tokenizer.encode(text=t, add_special_tokens=False)
                for t in (signal, signal + content)
            ]  # `List[List[int]]`
            input_ids.extend(tokenized[1])
            if trainable:
                labels.extend(
                    [self.ignore_index for _ in range(len(tokenized[0]))] +
                    tokenized[1][len(tokenized[0]): -len(suffix_token_ids)] +
                    [self.ignore_index for _ in range(len(suffix_token_ids))]
                )
            else:
                labels.extend([self.ignore_index for _ in range(len(tokenized[1]))])

        if training:
            return TokenizedSample(input_ids=input_ids, labels=labels)
        # Add generation prompt for inference mode.
        if not (len(sample.messages) > 0 and sample.messages[-1].role == Role.USER):
            raise RuntimeError(
                "Unexpected conversation for chat completion inference, "
                "the latest message must be user."
            )
        signal = self._get_signal(role=Role.ASSISTANT)
        input_ids.extend(
            self.tokenizer.encode(text=signal, add_special_tokens=False)
        )
        return TokenizedSample(input_ids=input_ids)

    @classmethod
    def _textify_tool_calls(cls, msg: Message) -> str:
        if msg.role != Role.ASSISTANT:
            raise RuntimeError(
                f"Invalid role for tool calling, it expected to be \"{Role.ASSISTANT.value}\"."
            )
        if msg.content is not None:
            raise RuntimeError("When assistant responses tool calls, content is expected to be None.")
        if not msg.tool_calls:
            raise RuntimeError(f"Invalid or empty `{Message.__name__}.tool_calls`.")

        tool_calls = msg.tool_calls
        if isinstance(tool_calls, ToolCall):
            tool_calls = [tool_calls]  # `ToolCall` -> `List[ToolCall]`
        texts = []  # `List[str]`
        for tool_call in tool_calls:
            texts.append(cls._parse_tool_call(tool_call=tool_call))
        return "\n\n".join(texts)

    @classmethod
    def _textify_content(cls, msg: Message) -> str:
        if not msg.content:
            raise RuntimeError("Invalid or empty content of the message.")
        # `Union[str, Content, List[Content]]`
        if isinstance(msg.content, str):
            return msg.content
        if not isinstance(msg.content, Content):
            raise RuntimeError(
                f"Unsupported type of `{Message.__name__}.content`, "
                f"it expected to be a `str` or `{Content.__name__}`."
            )
        if msg.content.type != ContentType.TEXT:
            raise RuntimeError(
                f"Invalid content type, it expected to be \"{ContentType.TEXT.value}\"."
            )
        if not msg.content.value:
            raise RuntimeError(f"Invalid or empty `{Content}.value`.")
        return msg.content.value

    @classmethod
    def _get_tools_prompt(cls, tools: Union[Tool, List[Tool]], language: Language) -> str:
        """Get the prompt of a series of tool definitions."""
        if not isinstance(tools, List):
            tools = [tools]
        content = cls._FN_PREFIX.get(language, cls._DEFAULT_LANGUAGE)
        names = []  # `List[str]`, a list of candidate function names.
        for tool in tools:
            tool_name, tool_desc = cls._parse_tool(tool=tool, language=language)
            content += f"\n\n{tool_desc}"
            names.append(tool_name)
        content += "\n\n"
        content += cls._FN_SUFFIX_TPL.get(language, cls._DEFAULT_LANGUAGE).format_map(
            {"names": ",".join(names)}
        )
        return content

    @classmethod
    def _parse_tool_call(cls, tool_call: ToolCall) -> str:
        if tool_call.type != ToolType.FUNCTION:
            raise RuntimeError(
                f"Unexpected tool call type ({tool_call.type.value}), "
                f"Only support \"{ToolType.FUNCTION.value}\" tool now."
            )
        if not tool_call.function:
            raise RuntimeError(
                f"Invalid function calling, must be a valid one."
            )
        text = cls._FN_CALL_TPL.format_map(
            {
                "name": tool_call.function.name,
                "arguments": json.dumps(tool_call.function.arguments, ensure_ascii=False)
            }
        )
        return text

    @classmethod
    def _parse_tool(cls, tool: Tool, language: Language) -> Tuple[str, str]:
        if tool.type != ToolType.FUNCTION:
            raise RuntimeError(f"Invalid tool, expected to be with \"{ToolType.FUNCTION.value}\" type.")
        if not tool.function:
            raise RuntimeError(f"Invalid tool, expected to be a valid function.")
        tpl = cls._FN_DESC_TPL.get(language, cls._DEFAULT_LANGUAGE)
        args_fmt = cls._ARGS_FORMAT.get(language, cls._DEFAULT_LANGUAGE)

        name = tool.function.name
        desc = tpl.format_map(
            {
                "name_for_human": name,
                "name_for_model": name,
                "description_for_model": tool.function.description,
                "parameters": tool.function.parameters.model_dump_json(indent=4, exclude_none=True),
                "args_format": args_fmt
            }
        ).rstrip()
        return name, desc

    @classmethod
    def _get_signal(cls, role: Union[str, Role]) -> str:
        return f"{cls._BOM_TOKEN}{Role(role).value}\n"
