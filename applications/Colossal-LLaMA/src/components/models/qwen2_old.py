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

    _FN_NAME = '✿FUNCTION✿'
    _FN_ARGS = '✿ARGS✿'
    _FN_RESULT = '✿RESULT✿'
    _FN_EXIT = '✿RETURN✿'
    _FN_STOP_WORDS = [_FN_RESULT, _FN_EXIT]

    _BOM_TOKEN = "<|im_start|>"  # Token represent begin of a message.
    _EOM_TOKEN = "<|im_end|>"  # Token represent end of a message.
    _PAD_TOKEN = "<|endoftext|>"
    _DEFAULT_SYSTEM_CONTENT = {
        Language.CHINESE: "你是一个名为 Qwen2 的人工智能助手，你的任务是针对用户的问题和要求提供适当的答复和支持。",
        Language.ENGLISH: "You are an AI assistant named Qwen2, and your task is to provide appropriate "
                          "responses and support to users' questions and requests."
    }

    def __init__(self, config: ChatLLMConfig) -> None:
        super().__init__(config=config)

    def _tokenize(self, sample: RawSample, training: bool) -> TokenizedSample:
        """
        See: https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/tokenizer_config.json#L31
        """
        text = ""  # For logging.
        input_ids = []  # `List[int]`
        labels = []  # `List[int]`

        for i, msg in enumerate(sample.messages):
            signal = self._get_signal(role=msg.role)  # `str`

            # Indicates whether the current message needs to calculate loss,
            # which is used for efficient training of multi-turn conversation samples.
            need = training and msg.role == Role.ASSISTANT
            if need is True and msg.loss is False:
                need = False

            if msg.tools:
                if msg.role not in {Role.SYSTEM, Role.USER}:
                    raise RuntimeError(
                        "Invalid message with tools, only user & system can be the message sender with tools."
                    )
                # Exist tool definition in current chat message.
                _, content = self._system_content_state(msgs=sample.messages[: i + 1])  # Get the latest system
                # content found in the conversation sequence up to the current msg.
                # If not found, a default system content is returned.
                content += self._textify_msg_tools(msg=msg)
                content_with_signal = f"{self._get_signal(role=Role.SYSTEM)}{content}{self._EOM_TOKEN}\n"
                text += content_with_signal

                tokenized = self.tokenizer.encode(text=content_with_signal, add_special_tokens=False)  # `List[int]`
                input_ids.extend(tokenized)
                labels.extend([self.config.ignore_index for _ in range(len(tokenized))])

                if msg.role == Role.USER:
                    content = f"{self._textify_msg_content(msg=msg)}{self._EOM_TOKEN}\n"
                    text += (signal + content)
                    tokenized = self.tokenizer.encode(text=signal + content, add_special_tokens=False)
                    input_ids.extend(tokenized)
                    labels.extend([self.config.ignore_index for _ in range(len(tokenized))])
                continue

            if msg.tool_calls:
                content = f"{self._textify_msg_tool_calls(msg=msg)}{self._EOM_TOKEN}\n"
                text += (signal + content)
                tokenized = [
                    self.tokenizer.encode(text=t, add_special_tokens=False)
                    for t in (signal, signal + content)
                ]  # `List[List[int]]`
                input_ids.extend(tokenized[1])
                if need:  # TODO: need to train '\n' ?
                    labels.extend(
                        [self.config.ignore_index for _ in range(len(tokenized[0]))] + tokenized[1][len(tokenized[0]):]
                    )
                else:
                    labels.extend([self.config.ignore_index for _ in range(len(tokenized[1]))])
                continue

            content = f"{self._textify_msg_content(msg=msg)}{self._EOM_TOKEN}\n"
            text += (signal + content)
            tokenized = [
                self.tokenizer.encode(text=t, add_special_tokens=False)
                for t in (signal, signal + content)
            ]  # `List[List[int]]`
            input_ids.extend(tokenized[1])
            if need:  # TODO: need to train '\n' ?
                labels.extend(
                    [self.config.ignore_index for _ in range(len(tokenized[0]))] + tokenized[1][len(tokenized[0]):]
                )
            else:
                labels.extend([self.config.ignore_index for _ in range(len(tokenized[1]))])

        if training:
            return TokenizedSample(input_ids=input_ids, labels=labels)
        # Add generation prompt for inference mode.
        if not (len(sample.messages) > 0 and sample.messages[-1].role == Role.USER):
            raise RuntimeError(
                "Unexpected conversation for chat completion inference, "
                "the latest message must be user."
            )
        signal = self._get_signal(role=Role.ASSISTANT)
        text += signal
        input_ids.extend(
            self.tokenizer.encode(text=signal, add_special_tokens=False)
        )
        return TokenizedSample(input_ids=input_ids)

    def process_response_message(self, text: str) -> Message:
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

    @classmethod
    def _get_signal(cls, role: Union[str, Role]) -> str:
        return f"{cls._BOM_TOKEN}{Role(role).value}\n"

    def _system_content_state(self, msgs: List[Message]) -> Tuple[bool, str]:
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
            return False, self._DEFAULT_SYSTEM_CONTENT[self.config.language]
        i = len(msgs) - 1
        while i >= 0:
            if msgs[i].role == Role.SYSTEM:
                break
            i -= 1
        if i == -1:  # There is no system message in conversation.
            return False, self._DEFAULT_SYSTEM_CONTENT[self.config.language]
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
                return False, self._DEFAULT_SYSTEM_CONTENT[self.config.language]
            return True, msg.content
        if msg.content.type != ContentType.TEXT:
            raise RuntimeError(
                "Invalid content value type for system message, "
                f"expect to be \"{ContentType.TEXT.value}\", "
                f"but \"{msg.content.type}\" now."
            )
        if not msg.content.value:
            return False, self._DEFAULT_SYSTEM_CONTENT[self.config.language]
        return True, msg.content.value

    def _textify_msg_tools(self, msg: Message) -> str:
        candidate_content = {
            Language.CHINESE: "\n\n# 工具\n\n# 你拥有如下工具：",
            Language.ENGLISH: "\n\n# Tools\n\n## You have access to the following tools:"
        }
        candidate_fn_call_tpl = {
            Language.CHINESE: "## 你可以在回复中插入零次、一次或多次以下命令以调用工具：\n\n"
                              "%s: 工具名称，必须是[{tool_names}]之一\n"
                              "%s: 工具输入\n"
                              "%s: 工具结果\n"
                              "%s: 根据工具结果进行回复，需将图片用![](url)渲染出来"
                              % (self._FN_NAME, self._FN_ARGS, self._FN_RESULT, self._FN_EXIT),
            Language.ENGLISH: "## When you need to call a tool, please insert the following command in your reply, "
                              "which can be called zero or multiple times according to your needs:\n\n"
                              "%s: The tool to use, should be one of [{tool_names}]\n"
                              "%s: The input of the tool\n"
                              "%s: Tool results\n"
                              "%s: Reply based on tool results. Images need to be rendered as ![](url)"
                              % (self._FN_NAME, self._FN_ARGS, self._FN_RESULT, self._FN_EXIT),
        }

        content = candidate_content[self.config.language]
        tools = msg.tools
        if isinstance(tools, Tool):
            tools = [tools]  # `Tool` -> `List[Tool]`
        tool_names = []
        for tool in tools:
            name, detail = self._parse_tool(tool=tool)
            # Add tool description information to the latest system message.
            content += f"\n\n{detail}"
            tool_names.append(name)
        content += "\n\n"
        content += candidate_fn_call_tpl[self.config.language].format_map(
            {"tool_names": ",".join(tool_names)}
        )
        return content

    def _textify_msg_tool_calls(self, msg: Message) -> str:
        if msg.role != Role.ASSISTANT:
            raise RuntimeError(
                f"Invalid role for tool calling, it expected to be \"{Role.ASSISTANT.value}\"."
            )
        if msg.content is not None:
            raise RuntimeError("When assistant responses tool calls, content is expected to be None.")
        if not msg.tool_calls:
            raise RuntimeError(f"Invalid or empty `{Message.__name__}.tool_calls`.")

        tpl = {
            Language.CHINESE: "\n\n工具 \"{name}\" 被调用时使用了以下参数：\n{arguments}",
            Language.ENGLISH: "\n\nThe tool \"{name}\" was called with the following arguments:\n{arguments}"
        }

        tool_calls = msg.tool_calls
        if isinstance(tool_calls, ToolCall):
            tool_calls = [tool_calls]  # `ToolCall` -> `List[ToolCall]`

        text = ""
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
            text += tpl[self.config.language].format_map(
                {
                    "name": tool_call.function.name,
                    "arguments": json.dumps(tool_call.function.arguments, ensure_ascii=False, indent=4)
                }
            )
        return text.lstrip()

    @classmethod
    def _textify_msg_content(cls, msg: Message) -> str:
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
        """Get tool name and textual instruction.

        See: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/function_calling.py
        """
        if tool.type != ToolType.FUNCTION:
            raise RuntimeError(f"Invalid tool, expected to be with \"{ToolType.FUNCTION.value}\" type.")
        if not tool.function:
            raise RuntimeError(f"Invalid tool, expected to be a valid func.")
        tpl = {
            Language.CHINESE: ("### {name_for_human}\n\n"
                               "{name_for_model}: {description_for_model}\n输入参数：\n{parameters}\n{args_format}"),
            Language.ENGLISH: ("### {name_for_human}\n\n"
                               "{name_for_model}: {description_for_model}\nParameters：\n{parameters}\n{args_format}"),
        }
        tpl_args_format = {
            Language.CHINESE: "此工具的输入应为JSON对象。",
            Language.ENGLISH: "Format the arguments as a JSON object."
        }
        name = tool.function.name
        instruction = tpl[self.config.language].format_map(
            {
                "name_for_human": name,
                "name_for_model": name,
                "description_for_model": tool.function.description,
                "parameters": tool.function.parameters.model_dump_json(indent=4, exclude_none=True),
                "args_format": tpl_args_format[self.config.language]
            }
        ).rstrip()
        return name, instruction
