from typing import List, Tuple, Union, Dict, Any

import json
import re
from contextlib import nullcontext
from typing import ContextManager
from pydantic import ValidationError

from transformers.models.auto import AutoModel, AutoTokenizer

from .base import ChatLLM, ChatLLMConfig, Language
from ..data.schema.raw import Content, ContentType, RawSample, Message, Role
from ..data.schema.tool import FunctionCall, Tool, ToolCall, ToolType
from ..data.schema.tokenized import TokenizedSample


class Llama31ChatLLM(ChatLLM):
    """A class used to represent LLaMA-3.1 base LLMs.
    See: https://ai.meta.com/blog/meta-llama-3-1/
    """

    # https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/
    # Specifies the start of the prompt.
    _BOS_TOKEN = "<|begin_of_text|>"
    # Model will cease to generate more tokens. This token is generated only by the base models.
    _EOS_TOKEN = "<|end_of_text|>"
    # This token is used for padding text sequences.
    _PAD_TOKEN = "<|finetune_right_pad_id|>"
    # These tokens enclose the role for a particular message.
    # The possible roles are: [system, user, assistant and ipython]
    # 'ipython' originally comes from `Role.EXECUTOR`.
    _BOH_TOKEN = "<|start_header_id|>"  # Begin of the header.
    _EOH_TOKEN = "<|end_header_id|>"  # End of the header.
    # End of turn. Represents when the model has determined that it has finished interacting with
    # the user message that initiated its response. This is used in two scenarios:
    # (a). at the end of a direct interaction between the model and the user
    # (b). at the end of multiple interactions between the model and any available tools
    # This token signals to the executor that the model has finished generating a response.
    _EOT_TOKEN = "<|eot_id|>"
    # End of message. A message represents a possible stopping point for execution where the model can
    # inform the executor that a tool call needs to be made. This is used for multi-step interactions
    # between the model and any available tools. This token is emitted by the model when
    # the 'Environment: ipython' instruction is used in the system prompt,
    # or if the model calls for a built-in tool.
    _EOM_TOKEN = "<|eom_id|>"
    # Is a special tag used in the model’s response to signify a tool call.
    _TOOL_TAG_TOKEN = "<|python_tag|>"

    _DEFAULT_SYSTEM = "You are a helpful Assistant."
    _FN_CALL_INFERENCE_TPL = "The tool '{name}' was called with the following arguments:\n{arguments}"

    def __init__(self, config: ChatLLMConfig) -> None:
        super().__init__(config=config)

    def init_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            trust_remote_code=self.config.trust_remote_code
        )
        # Verification for tokenizer.
        if self.tokenizer.bos_token != self._BOS_TOKEN:
            raise AssertionError(
                f"Invalid bos token of for `{self.__class__.__name__}.tokenizer`, "
                f"expected to be \"{self._BOS_TOKEN}\" but got \"{self.tokenizer.bos_token}\"."
            )
        # No verification is performed on eos token because the base model and
        # the instruction model use different.
        self.tokenizer.pad_token = self._PAD_TOKEN  # Explicitly, manually set padding tokens.

    def init_model(self, context: ContextManager = nullcontext()) -> None:
        with context:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
                trust_remote_code=self.config.trust_remote_code
            )

    def _tokenize(self, sample: RawSample, training: bool, **kwargs) -> TokenizedSample:
        input_ids = [self.tokenizer.bos_token_id]  # `List[int]`
        labels = [self.ignore_index]  # `List[int]`

        for i, msg in enumerate(sample.messages):
            signal = self._get_signal(role=msg.role)
            trainable = self._is_trainable_msg(msg=msg, training=training)
            if msg.tools:  # Case of the message with tool definition.
                # See: https://docs.together.ai/docs/llama-3-function-calling
                # Note that the basic verification has been defined in the base class and pre-executed.
                content = self._get_tools_prompt(tools=msg.tools)
                content += self._textify_content(msg=msg)
                # https://github.com/meta-llama/llama-agentic-system/blob/main/llama_agentic_system/system_prompt.py#L50
                content += self._EOT_TOKEN

                tokenized = self.tokenizer.encode(text=signal + content, add_special_tokens=False)  # `List[int]`
                input_ids.extend(tokenized)
                labels.extend([self.ignore_index for _ in range(len(tokenized))])
                continue

            if msg.tool_calls:  # Case of the message with tool calling.
                content = self._textify_tool_calls(msg=msg, trainable=trainable)
                content += self._EOT_TOKEN

                tokenized = [
                    self.tokenizer.encode(text=t, add_special_tokens=False) for t in (signal, signal + content)
                ]  # `List[List[int]]`
                input_ids.extend(tokenized[1])
                if trainable:
                    labels.extend(
                        [self.ignore_index for _ in range(len(tokenized[0]))] + tokenized[1][len(tokenized[0]):]
                    )
                else:
                    labels.extend([self.ignore_index for _ in range(len(tokenized[1]))])
                continue

            # Case of normal message.
            content = self._textify_content(msg=msg)
            content += self._EOS_TOKEN
            tokenized = [
                self.tokenizer.encode(text=t, add_special_tokens=False) for t in (signal, signal + content)
            ]  # `List[List[int]]`
            input_ids.extend(tokenized[1])
            if trainable:
                labels.extend(
                    [self.ignore_index for _ in range(len(tokenized[0]))] + tokenized[1][len(tokenized[0]):]
                )
            else:
                labels.extend([self.ignore_index for _ in range(len(tokenized[1]))])

        if training:
            return TokenizedSample(input_ids=input_ids, labels=labels)
        # Add generation prompt (inference mode).
        if not (len(sample.messages) > 0 and sample.messages[-1].role == Role.USER):
            raise RuntimeError(
                "Unexpected conversation for chat completion inference, "
                "the latest message must be user."
            )
        signal = self._get_signal(role=Role.ASSISTANT)
        input_ids.extend(self.tokenizer.encode(text=signal, add_special_tokens=False))
        return TokenizedSample(input_ids=input_ids)

    @classmethod
    def _get_signal(cls, role: Union[str, Role]) -> str:
        r = Role(role).value
        if Role(r) == Role.EXECUTOR:
            r = "ipython"  # See: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
        return f"{cls._BOH_TOKEN}{r}{cls._EOH_TOKEN}\n\n"

    @classmethod
    def _textify_content(cls, msg: Message) -> str:
        """Get the normal text (or prompt) from the message."""
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
    def _textify_tool_calls(cls, msg: Message, trainable: bool) -> str:
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
        # Note: It is allowed to call the same function simultaneously.
        # e.g., '<function=func>{"p":"a"}</function>\n<function=func>{"p":"b"}</function><|eot_id|>'

        texts = []  # `List[str]`
        func_names = set()
        for tool_call in tool_calls:
            texts.append(
                cls._get_tool_call_text(tool_call=tool_call, trainable=trainable)
            )
            func_names.add(tool_call.function.name)
        if len(func_names) != 1:
            raise RuntimeError("Only allow call one function at once.")
        sep = "\n" if trainable else "\n\n"
        return sep.join(texts)

    @classmethod
    def _get_tool_call_text(cls, tool_call: ToolCall, trainable: bool) -> str:
        if tool_call.type != ToolType.FUNCTION:
            raise RuntimeError(
                f"Unexpected tool call type ({tool_call.type.value}), "
                f"Only support \"{ToolType.FUNCTION.value}\" tool now."
            )
        if not tool_call.function:
            raise RuntimeError(
                f"Invalid function calling, must be a valid one."
            )
        if not trainable:
            return cls._FN_CALL_INFERENCE_TPL.format_map(
                {
                    "name": tool_call.function.name,
                    "arguments": json.dumps(tool_call.function.arguments, ensure_ascii=False)
                }
            )
        else:
            # Must be consistent with the descriptive constraints in the tool prompt.
            func_name = tool_call.function.name
            func_args = tool_call.function.arguments
            return f"<function={func_name}>{json.dumps(func_args, ensure_ascii=False)}</function>"

    def parse_response(self, text: str) -> Message:
        """Need to be consistent with the tool prompt."""
        func_pattern = r"<function=(\w+)>(.*?)</function>"
        items = re.findall(pattern=func_pattern, string=text)
        if not items:
            return Message(
                role=Role.ASSISTANT, content=text
            )
        tool_calls = []  # `List[ToolCall]`
        func_names = set()
        for item in items:
            func_name, func_args_str = item
            try:
                func_args = json.loads(func_args_str)
            except json.JSONDecodeError:
                # TODO: warning
                return Message(
                    role=Role.ASSISTANT, content=text
                )
            tc = ToolCall(
                type=ToolType.FUNCTION,
                function=FunctionCall(name=func_name, arguments=func_args)
            )
            tool_calls.append(tc)
            func_names.add(func_name)
        if len(func_names) != 1:
            # Only call one function at a time.
            # TODO: warning
            return Message(
                role=Role.ASSISTANT, content=text
            )
        if len(tool_calls) == 1:
            tool_calls = tool_calls[0]
        return Message(
            role=Role.ASSISTANT, content=None,
            tool_calls=tool_calls
        )

    def _verify(self, sample: Union[Dict[str, Any], RawSample], training: bool) -> None:
        names = set()
        for i, msg in enumerate(sample.messages):
            if msg.tool_calls:
                tool_calls = msg.tool_calls
                if isinstance(tool_calls, ToolCall):
                    tool_calls = [tool_calls]
                for tool_call in tool_calls:
                    if tool_call.type != ToolType.FUNCTION:
                        raise ValidationError(
                            f"Unexpected tool call type ({tool_call.type.value}), "
                            f"Only support \"{ToolType.FUNCTION.value}\" tool now."
                        )
                    if not tool_call.function:
                        raise ValidationError(
                            f"Invalid function calling, must be a valid one."
                        )
                    names.add(tool_call.function.name)
        if len(names) != 1:
            raise ValidationError(f"Only support call one function at once for `{self.__class__.__name__}`.")

    @staticmethod
    def _get_tools_prompt(tools: Union[Tool, List[Tool]]) -> str:
        """Get the prompt for custom tools, add it before normal system content.

        See:
            https://docs.together.ai/docs/llama-3-function-calling
            https://github.com/meta-llama/llama-agentic-system/tree/main/llama_agentic_system
            https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
        """
        if not isinstance(tools, List):
            tools = [tools]
        content = "You have access to the following functions:\n\n"
        for tool in tools:
            if tool.type != ToolType.FUNCTION:
                raise RuntimeError(f"Invalid tool, expected to be with \"{ToolType.FUNCTION.value}\" type.")
            if not tool.function:
                raise RuntimeError(f"Invalid tool, expected to be a valid function.")
            content += (f"Use the function '{tool.function.name}' to '{tool.function.description}':\n"
                        f"{tool.function.model_dump_json(exclude_none=True)}\n\n")  # TODO: indent?
        content += ("If you choose to call a function ONLY reply in the following format with no prefix or suffix:\n\n"
                    "<function=example_function_name>{{\"example_name\": \"example_value\"}}</function>\n\n"
                    "Reminder:\n"
                    "- Function calls MUST follow the specified format, start with <function= "
                    "and end with </function>\n"
                    "- Required parameters MUST be specified\n"
                    "- Only call one function at a time\n"
                    "- Put the entire function call reply on one line\n"
                    "- If there is no function call available, answer the question like normal with your current "
                    "knowledge and do not tell the user about function calls\n\n")
        return content
