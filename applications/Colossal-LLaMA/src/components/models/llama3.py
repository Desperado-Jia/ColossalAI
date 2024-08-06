import json
from contextlib import nullcontext
from typing import ContextManager

from transformers.models.auto import AutoModel, AutoTokenizer

from .base import ChatLLM, ChatLLMConfig
from ..data.schema.raw import Content, ContentType, RawSample, Message, Role
from ..data.schema.tool import Tool, ToolCall, ToolType
from ..data.schema.tokenized import TokenizedSample


class Llama3ChatLLM(ChatLLM):
    """A class used to represent LLaMA-based LLMs."""

    def __init__(self, config: ChatLLMConfig) -> None:
        super().__init__(config=config)
        self.bos_token = "<|begin_of_text|>"
        self.eos_token = "<|eot_id|>"
        self.signal_tokens = ("<|start_header_id|>", "<|end_header_id|>")
        # Affix pair tokens to add to the role token. These tokens enclose the role for a particular message.

    def init_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            trust_remote_code=self.config.trust_remote_code
        )
        # Verification for tokenizer.
        if self.tokenizer.bos_token != self.bos_token or self.tokenizer.eos_token != self.eos_token:
            raise AssertionError(
                f"Invalid bos and eos special tokens of for `{self.__class__.__name__}.tokenizer`, "
                f"expected to be \"{self.bos_token}\" and \"{self.eos_token}\", "
                f"but got \"{self.tokenizer.bos_token}\" and \"{self.tokenizer.eos_token}\"."
            )
        if self.tokenizer.unk_token or self.tokenizer.pad_token:
            raise AssertionError(
                "The original unk & pad tokens should not be non-empty, "
                f"which need to be set manually in the post-initialization strategy of `{self.__class__.__name__}`."
            )
        # Explicitly, manually set unknown & padding tokens.
        self.tokenizer.unk_token = self.eos_token
        self.tokenizer.pad_token = self.eos_token

    def init_model(self, context: ContextManager = nullcontext()) -> None:
        with context:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
                trust_remote_code=self.config.trust_remote_code
            )

    def _tokenize(self, sample: RawSample, training: bool) -> TokenizedSample:
        if not self.tokenizer:
            raise RuntimeError(
                f"Uninstantiated `{self.__class__.__name__}.tokenizer`, "
                f"please execute 'init_tokenizer' method first."
            )
        input_ids = [self.tokenizer.bos_token_id]  # `List[int]`
        labels = [self.config.ignore_index]  # `List[int]`

        for i, msg in enumerate(sample.messages):
            signal = self._get_signal(role=msg.role)

            # Indicates whether the current message needs to calculate loss,
            # which is used for efficient training of multi-turn conversation samples.
            need = training and msg.role == Role.ASSISTANT
            if need is True and msg.loss is False:
                need = False

            if msg.tools:
                # Exist tool definition in current chat message.
                content = ("Given the following functions, please respond with a JSON for a function call "
                           "with its proper arguments that best answers the given prompt.\n\n"
                           "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument "
                           "name and its value}. Do not use variables.\n\n")
                content += self._textify_msg_tools(msg=msg)
                content += f"Question: {self._textify_msg_content(msg=msg)}{self.tokenizer.eos_token}"

                tokenized = [
                    self.tokenizer.encode(text=t, add_special_tokens=False)
                    for t in (signal, signal + content)
                ]  # `List[List[int]]`
                input_ids.extend(tokenized[1])
                if need:
                    labels.extend(
                        [self.config.ignore_index for _ in range(len(tokenized[0]))] + tokenized[1][len(tokenized[0]):]
                    )
                else:
                    labels.extend([self.config.ignore_index for _ in range(len(tokenized[1]))])
                continue

            if msg.tool_calls:
                content = f"{self._textify_msg_tool_calls(msg=msg)}{self.tokenizer.eos_token}"
                tokenized = [
                    self.tokenizer.encode(text=t, add_special_tokens=False)
                    for t in (signal, signal + content)
                ]  # `List[List[int]]`
                input_ids.extend(tokenized[1])
                if need:
                    labels.extend(
                        [self.config.ignore_index for _ in range(len(tokenized[0]))] + tokenized[1][len(tokenized[0]):]
                    )
                else:
                    labels.extend([self.config.ignore_index for _ in range(len(tokenized[1]))])
                continue

            content = f"{self._textify_msg_content(msg=msg)}{self.tokenizer.eos_token}"
            tokenized = [
                self.tokenizer.encode(text=t, add_special_tokens=False)
                for t in (signal, signal + content)
            ]  # `List[List[int]]`
            input_ids.extend(tokenized[1])
            if need:
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
        input_ids.extend(
            self.tokenizer.encode(text=signal, add_special_tokens=False)
        )
        return TokenizedSample(input_ids=input_ids)

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

    @classmethod
    def _textify_msg_tools(cls, msg: Message) -> str:
        if msg.role != Role.USER:
            raise RuntimeError(
                f"Unexpected tool definitions within non-user message for {cls.__name__}. "
                f"See: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1"
            )
        tools = msg.tools
        if not tools:
            raise RuntimeError("Invalid or empty tools in message.")
        if isinstance(tools, Tool):
            tools = [tools]
        text = ""
        for tool in tools:
            text += f"{cls._textify_tool(tool=tool)}\n\n"
        return text

    @classmethod
    def _textify_msg_tool_calls(cls, msg: Message) -> str:
        if msg.role != Role.ASSISTANT:
            raise RuntimeError(
                f"Invalid role for tool calling, it expected to be \"{Role.ASSISTANT.value}\"."
            )
        if msg.content is not None:
            raise RuntimeError("When assistant responses tool calls, content is expected to be None.")
        if not msg.tool_calls:
            raise RuntimeError(f"Invalid or empty `{Message.__name__}.tool_calls`.")
        tool_call = msg.tool_calls
        if not isinstance(tool_call, ToolCall):
            raise RuntimeError("Only support one tool calling at once.")
        if tool_call.type != ToolType.FUNCTION:
            raise RuntimeError(
                f"Unexpected tool call type ({tool_call.type.value}), "
                f"Only support \"{ToolType.FUNCTION.value}\" tool now."
            )
        if not tool_call.func:
            raise RuntimeError("Invalid or empty function call.")
        text = (f"The tool \"{tool_call.function.name}\" was called with the following arguments:\n"
                f"{json.dumps(tool_call.function.arguments, ensure_ascii=False, indent=4)}")
        return text

    @staticmethod
    def _textify_tool(tool: Tool) -> str:
        if tool.type != ToolType.FUNCTION:
            raise RuntimeError(
                f"Unexpected tool type ({tool.type.value}), Only support \"{ToolType.FUNCTION.value}\" tool now."
            )
        if not tool.function:
            raise RuntimeError("Invalid or empty function field.")
        return tool.model_dump_json(indent=4, exclude_none=True)

    def _get_signal(self, role: Role) -> str:
        """Get signal tokens from specific role."""
        r = Role(role).value
        if Role(r) == Role.EXECUTOR:
            r = "ipython"  # See: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
        return f"{self.signal_tokens[0]}{r}{self.signal_tokens[1]}\n\n"
