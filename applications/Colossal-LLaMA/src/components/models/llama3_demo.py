import json
from contextlib import nullcontext
from typing import ContextManager

from transformers.models.auto import AutoModel, AutoTokenizer

from .base import ChatLLM, ChatLLMConfig, Language
from ..data.schema.raw import Content, ContentType, RawSample, Message, Role
from ..data.schema.tool import Tool, ToolCall, ToolType
from ..data.schema.tokenized import TokenizedSample


class Llama3ChatLLM(ChatLLM):
    """A class used to represent LLaMA3-based LLMs."""

    def __init__(self, config: ChatLLMConfig) -> None:
        super().__init__(config=config)
        # Special tokens, see: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/
        self.bos_token = "<|begin_of_text|>"  # Specifies the start of the prompt.
        self.eos_token = "<|end_of_text|>"  # Model will cease to generate more tokens.
        # This token is generated only by the base models.
        self.pad_token = "<|finetune_right_pad_id|>"  # This token is used for padding text sequences.
        # to the same length in a batch.
        self.header_tokens = ("<|start_header_id|>", "<|end_header_id|>")  # These tokens enclose the role for
        # a particular message. The possible roles are: [system, user, assistant and ipython].
        self.eot_token = "<|eot_id|>"  # End of turn. Represents when the model has determined that
        # it has finished interacting with the user message that initiated its response. This is used in two scenarios:
        # a. at the end of a direct interaction between the model and the user;
        # b. at the end of multiple interactions between the model and any available tools
        # This token signals to the executor that the model has finished generating a response.
        self.eom_token = "<|eom_id|>"  # A message represents a possible stopping point for
        # execution where the model can inform the executor that a tool call needs to be made.
        # This is used for multi-stepped interactions between the model and any available tools.
        # This token is emitted by the model when the 'Environment: ipython' instruction is used
        # in the system prompt, or if the model calls for a built-in tool.
        self.python_tag_token = "<|python_tag|>"  # Usually not used.

        self.in_system_tool_call_tag = "Environment: ipython"

    def init_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            trust_remote_code=self.config.trust_remote_code
        )
        # Verification for tokenizer.
        if self.tokenizer.bos_token != self.bos_token or self.tokenizer.eos_token != self.eot_token:
            raise AssertionError(
                f"Invalid bos and eos special tokens of for `{self.__class__.__name__}.tokenizer`, "
                f"expected to be \"{self.bos_token}\" and \"{self.eot_token}\", "
                f"but got \"{self.tokenizer.bos_token}\" and \"{self.tokenizer.eos_token}\"."
            )
        if self.tokenizer.unk_token or self.tokenizer.pad_token:
            raise AssertionError(
                "The original unk & pad tokens should not be non-empty, "
                f"which need to be set manually in the post-initialization strategy of `{self.__class__.__name__}`."
            )
        # Explicitly, manually set padding tokens.
        self.tokenizer.pad_token = self.pad_token

    def init_model(self, context: ContextManager = nullcontext()) -> None:
        with context:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
                trust_remote_code=self.config.trust_remote_code
            )

    def _tokenize(self, sample: RawSample, training: bool) -> TokenizedSample:
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
                content += f"Question: {self._textify_msg_content(msg=msg)}{self.eot_token}"

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
                content = f"{self._textify_msg_tool_calls(msg=msg)}{self.eot_token}"
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

            content = f"{self._textify_msg_content(msg=msg)}{self.eot_token}"
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

    def prepare_response_message(self, text: str) -> Message:
        raise NotImplementedError()

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
                f"Unexpected custom tool definitions within non-user message for {cls.__name__}. "
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
        if isinstance(tool_call, ToolCall):
            tool_call = [tool_call]
        if len(tool_call) != 1:
            raise RuntimeError("Only support one tool calling at once.")
        tool_call = tool_call[0]
        if tool_call.type != ToolType.FUNCTION:
            raise RuntimeError(
                f"Unexpected tool call type ({tool_call.type.value}), "
                f"Only support \"{ToolType.FUNCTION.value}\" tool now."
            )
        if not tool_call.func:
            raise RuntimeError("Invalid or empty function call.")
        text = (f"The tool \"{tool_call.func.name}\" was called with the following arguments:\n"
                f"{json.dumps(tool_call.func.args, ensure_ascii=False, indent=4)}")
        return text

    @staticmethod
    def _textify_tool(tool: Tool) -> str:
        if tool.type != ToolType.FUNCTION:
            raise RuntimeError(
                f"Unexpected tool type ({tool.type.value}), Only support \"{ToolType.FUNCTION.value}\" tool now."
            )
        if not tool.func:
            raise RuntimeError("Invalid or empty function field.")
        return tool.model_dump_json(indent=4, exclude_none=True)

    def _get_signal(self, role: Role) -> str:
        """Get signal tokens from specific role."""
        r = Role(role).value
        if Role(r) == Role.EXECUTOR:
            r = "ipython"  # See: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
        return f"{self.header_tokens[0]}{r}{self.header_tokens[1]}\n\n"
