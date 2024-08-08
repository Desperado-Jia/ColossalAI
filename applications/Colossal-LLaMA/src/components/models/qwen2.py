from typing import ContextManager, List, Tuple, Union

import json
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
    _FN_PREFIX_TPL = {
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
    _FN_CALL_TPL = {
        Language.CHINESE: "\n\n工具 \"{name}\" 被调用时使用了以下参数：\n{arguments}",
        Language.ENGLISH: "\n\nThe tool \"{name}\" was called with the following arguments:\n{arguments}"
    }  # Template for function calling. Note that it only used in inference mode,
    _ARGS_FORMAT = {
        Language.CHINESE: "此工具的输入应为JSON对象。",
        Language.ENGLISH: "Format the arguments as a JSON object."
    }  # See: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/function_calling.py#L473-L476
    _DEFAULT_SYSTEM = {
        Language.CHINESE: "你是一个名为 Qwen2 的人工智能助手，你的任务是针对用户的问题和要求提供适当的答复和支持。",
        Language.ENGLISH: "You are an AI assistant named Qwen2, and your task is to provide appropriate "
                          "responses and support to users' questions and requests."
    }

    def __init__(self, config: ChatLLMConfig) -> None:
        super().__init__(config=config)
        self.language = Language(self.config.language)
        self.default_language = Language.ENGLISH

    def _tokenize(self, sample: RawSample, training: bool) -> TokenizedSample:
        """
        See: https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/tokenizer_config.json#L31
        """
        suffix_token_ids = self.tokenizer.encode(text="\n", add_special_tokens=False)  # `List[int]`, i.e., [198]

        input_ids = []  # `List[int]`
        labels = []  # `List[int]`
        for i, msg in enumerate(sample.messages):
            signal = self._get_signal(role=msg.role)
            # Indicates whether the current message needs to calculate loss,
            # which is used for efficient training of multi-turn conversation samples.
            trainable = self._is_trainable_msg(msg=msg, training=training)  # `bool`

            if msg.tools:
                if msg.role not in {Role.SYSTEM, Role.USER}:
                    raise RuntimeError(
                        "Invalid message with tools, only user & system can be the message sender with tools."
                    )
                _, content = self._fetch_sys_content(msgs=sample.messages[: i + 1])  # Get the latest system up to now.
                content += self._textify_msg_tools(msg=msg)
                content += f"{self._EOM_TOKEN}\n"  # Add eos suffix tokens.

                # Update the system message based on the latest tools.
                tokenized = self.tokenizer.encode(
                    text=self._get_signal(role=Role.SYSTEM) + content, add_special_tokens=False
                )  # `List[int]`, create a new latest system.
                input_ids.extend(tokenized)
                labels.extend([self.ignore_index for _ in range(len(tokenized))])

                if msg.role == Role.USER:
                    content = self._textify_msg_content(msg=msg)
                    content += f"{self._EOM_TOKEN}\n"  # Add eos suffix tokens.
                    tokenized = self.tokenizer.encode(
                        text=signal + content, add_special_tokens=False
                    )  # `List[int]`
                    input_ids.extend(tokenized)
                    labels.extend([self.ignore_index for _ in range(len(tokenized))])
                continue

            if msg.tool_calls:
                content = f"{self._textify_msg_tool_calls(msg=msg, trainable=trainable)}{self._EOM_TOKEN}\n"
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

            # Pure content.
            content = f"{self._textify_msg_content(msg=msg)}{self._EOM_TOKEN}\n"
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

    def _textify_msg_tools(self, msg: Message) -> str:
        if not msg.tools:
            raise RuntimeError("Invalid tools within argument ``msg``.")
        content = self._FN_PREFIX_TPL.get(self.language, self.default_language)
        tools = msg.tools
        if not isinstance(tools, List):
            tools = [tools]  # `Tool` -> `List[Tool]`
        tool_names = []  # `List[str]`, containing all tool names.
        for tool in tools:
            tool_name, tool_desc = self._parse_tool(tool=tool)
            content += f"\n\n{tool_desc}"
            tool_names.append(tool_name)
        content += "\n\n"
        content += self._FN_SUFFIX_TPL.get(self.language, self.default_language).format_map(
            {"names": ",".join(tool_names)}
        )
        return content

    def _textify_msg_tool_calls(self, msg: Message, trainable: bool) -> str:
        if msg.role != Role.ASSISTANT:
            raise RuntimeError(
                f"Invalid role for tool calling, it expected to be \"{Role.ASSISTANT.value}\"."
            )
        if msg.content is not None:
            raise RuntimeError("When assistant responses tool calls, content is expected to be None.")
        if not msg.tool_calls:
            raise RuntimeError(f"Invalid or empty `{Message.__name__}.tool_calls`.")

        tpl = self._FN_CALL_TPL.get(self.language, self.default_language)
        tool_calls = msg.tool_calls
        if isinstance(tool_calls, ToolCall):
            tool_calls = [tool_calls]  # `ToolCall` -> `List[ToolCall]`
        texts = []  # `List[str]`
        for tool_call in tool_calls:
            if tool_call.type != ToolType.FUNCTION:
                raise RuntimeError(
                    f"Unexpected tool call type ({tool_call.type.value}), "
                    f"Only support \"{ToolType.FUNCTION.value}\" tool now."
                )
            if not tool_call.function:
                raise RuntimeError(
                    "Invalid function calling."
                )
            if not trainable:
                t = tpl.format_map(
                    {
                        "name": tool_call.function.name,
                        "arguments": json.dumps(tool_call.function.arguments, ensure_ascii=False, indent=4)
                    }
                )
            else:
                t = tool_call.model_dump_json(exclude_none=True)  # Do not set indent when need loss.
            texts.append(t)
        if not trainable:
            return "".join(texts)
        # TODO: Maybe better for multi tool calls for training?
        return "\n".join([item.strip() for item in texts])

    @staticmethod
    def _textify_msg_content(msg: Message) -> str:
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

    def _parse_tool(self, tool: Tool) -> Tuple[str, str]:
        """Get the tool name and description.
        See: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/function_calling.py
        """
        if tool.type != ToolType.FUNCTION:
            raise RuntimeError(f"Invalid tool, expected to be with \"{ToolType.FUNCTION.value}\" type.")
        if not tool.function:
            raise RuntimeError(f"Invalid tool, expected to be a valid function.")
        tpl = self._FN_DESC_TPL.get(self.language, self.default_language)
        args_format = self._ARGS_FORMAT.get(self.language, self.default_language)

        name = tool.function.name
        desc = tpl.format_map(
            {
                "name_for_human": name,
                "name_for_model": name,
                "description_for_model": tool.function.description,
                "parameters": tool.function.parameters.model_dump_json(indent=4, exclude_none=True),
                "args_format": args_format
            }
        ).rstrip()
        return name, desc

    @classmethod
    def _get_signal(cls, role: Union[str, Role]) -> str:
        return f"{cls._BOM_TOKEN}{Role(role).value}\n"

    def _fetch_sys_content(self, msgs: List[Message]) -> Tuple[bool, str]:
        """Get the system prompt state of the conversation.

        Note that the purpose of this function is to facilitate the support of
        custom system prompt in tool calling scenarios,
        thereby fixing the defects of the original GLM4 release
        https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/tokenization_chatglm.py#L172

        Args:
            msgs (`List[Message]`): A sequence of chat messages representing a
                multi-turn conversation.

        Returns:
            (`bool`): Whether the current conversation contains a system prompt.
            (`str`): The system prompt obtained or to be added.
        """
        if len(msgs) == 0:
            return False, self._DEFAULT_SYSTEM[self.language]
        i = len(msgs) - 1
        while i >= 0:
            if msgs[i].role == Role.SYSTEM:
                break
            i -= 1
        if i == -1:  # There is no system message in conversation.
            return False, self._DEFAULT_SYSTEM[self.language]
        # Find the latest system message.
        msg = msgs[i]
        if not isinstance(msg.content, (str, Content)):
            raise RuntimeError(
                f"Invalid content type for system message, "
                f"expected to be `{Content.__name__}` or `str`, "
                f"but `{type(msg.content).__name__}` now."
            )
        if isinstance(msg.content, str):
            if not msg.content:
                return False, self._DEFAULT_SYSTEM[self.language]
            return True, msg.content
        if msg.content.type != ContentType.TEXT:
            raise RuntimeError(
                "Invalid content value type for system message, "
                f"expect to be \"{ContentType.TEXT.value}\", "
                f"but \"{msg.content.type}\" now."
            )
        if not msg.content.value:
            return False, self._DEFAULT_SYSTEM[self.language]
        return True, msg.content.value

    def prepare_response_message(self, text: str) -> Message:
        raise NotImplementedError()

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
