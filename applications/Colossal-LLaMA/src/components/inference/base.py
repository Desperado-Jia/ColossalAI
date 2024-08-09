from typing import List, Optional, Tuple, Union

from abc import ABC, abstractmethod
from pydantic import Field
from pydantic_settings import BaseSettings

from ..data.schema.raw import Role, RawSample, Message
from ..data.schema.tokenized import TokenizedSample
from ..data.schema.base import Schema
from ..models import ChatLLM
from ..utils.generic import DeviceType


class Usage(Schema):
    """A class used to represent the usage of LLMs/VLMs inference."""

    num_prompt_tokens: Optional[int] = Field(
        default=None,
        description="Number of prompt (i.e., input) tokens."
    )
    num_response_tokens: Optional[int] = Field(
        default=None,
        description="Number of response (i.e., output) tokens."
    )


class InferenceEngineConfig(BaseSettings):
    """A class used to represent the configurations of inference engine."""

    device: Optional[Union[str, DeviceType]] = Field(
        default=None,
        description="Device type that the current model process needs to deploy on."
    )
    tensor_parallel_size: Optional[int] = Field(
        default=None, description="Tensor parallel size "
                                  "this parameter takes effect only when vLLM is used as the inference engine."
    )
    pipeline_parallel_size: Optional[int] = Field(
        default=None, description="Pipeline parallel size, "
                                  "this parameter takes effect only when vLLM is used as the inference engine."
                                  "Note that we can run inference and serving on multiple machines "
                                  "by launching the vLLM process on the head node by setting "
                                  "``tensor_parallel_size`` multiplied by ``pipeline_parallel_size`` "
                                  "to the number of GPUs to be the total number of GPUs across all machines. "
    )
    model_parallel_size: Optional[int] = Field(
        default=None, description="Model parallel size, it's used used when your model is split on several GPUs "
                                  "is naive and not optimized, meaning that only one GPU works at a given time "
                                  "and the other sits idle."
    )
    gpu_memory_utilization: float = Field(
        default=0.95,
        description="Ratio of the GPU memory you want to allow for vLLM. "
                    "Note that this parameter is only for the vllm engine "
                    "and will not take effect on the transformer-based inference engine."
    )


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""

    def __init__(self, llm: ChatLLM, config: InferenceEngineConfig) -> None:
        self.llm = llm
        self.config = config
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the inference engine. Different engines have distinct workflows."""
        pass

    @abstractmethod
    def _generate(self,
                  prompts: Union[List[str], List[TokenizedSample]],
                  **kwargs
                  ) -> List[Tuple[str, Usage]]:
        pass

    def generate(self,
                 prompts: Union[Union[str, List[str]], Union[TokenizedSample, List[TokenizedSample]]],
                 **kwargs
                 ) -> Union[Tuple[str, Usage], List[Tuple[str, Usage]]]:
        batched = True
        if not isinstance(prompts, List):
            batched = False
            prompts = [prompts]  # `List[RawSample]`
        if len(prompts) == 0:
            raise RuntimeError("Invalid argument ``prompts``, it must contain at least one valid prompt sample.")
        responses = self._generate(prompts=prompts, **kwargs)
        if not batched:
            return responses[0]
        return responses

    def chat(self,
             conversations: Union[RawSample, List[RawSample]],
             **kwargs
             ) -> Union[Tuple[Message, Usage], List[Tuple[Message, Usage]]]:
        batched = True
        if isinstance(conversations, RawSample):
            batched = False
            conversations = [conversations]  # `List[RawSample]`
        if len(conversations) == 0:
            raise RuntimeError(
                "Invalid argument ``conversations``, "
                "it must contain at least one valid prompt conversation sample."
            )

        # Verification of each prompt conversation.
        _ = [self._verify_prompt(msgs=sample.messages) for sample in conversations]
        prompts = [
            self.llm.tokenize(sample=sample, training=False, return_dict=False)
            for sample in conversations
        ]  # `List[TokenizedSample]`
        responses = self._generate(prompts=prompts, **kwargs)  # `List[Tuple[str, Usage]]`
        # Postprocess of response.
        outputs = []  # `List[Tuple[Message, Usage]]`
        for content, usage in responses:
            msg = self.llm.parse_response(text=content)  # Contains the situation handling logic
            # for complex scenarios (e.g., tool calling)
            self._verify_response(resp=msg)
            outputs.append((msg, usage))

        if not batched:
            return outputs[0]
        return outputs

    @classmethod
    def _verify_prompt(cls, msgs: List[Message]) -> None:
        """Verify the correctness of the prompt conversation."""
        if len(msgs) == 0:
            raise RuntimeError("Invalid prompt conversation, it must contain at least one valid message.")
        if msgs[-1].role != Role.USER:
            raise RuntimeError(
                f"Invalid prompt conversation, the role of the latest prompt message must be \"{Role.USER.value}\", "
                f"now \"{msgs[-1].role.value}\"."
            )
        num_tool_call = 0
        num_tool_exec = 0
        for i, msg in enumerate(msgs):
            cls._verify_msg(msg=msg)
            if msg.tool_calls:
                num_tool_call += 1
            if msg.role == Role.EXECUTOR:
                num_tool_exec += 1
        if num_tool_call != num_tool_exec:
            raise RuntimeError("Invalid prompt conversation, tool call and execution times are different.")

    @classmethod
    def _verify_response(cls, resp: Message, prompt: List[Message] = None) -> None:
        if resp.role != Role.ASSISTANT:
            raise ValueError(f"Invalid response role, must be \"{Role.ASSISTANT.value}\".")

    @staticmethod
    def _verify_msg(msg: Message) -> None:
        # Verification of tools.
        if msg.tools and msg.role not in {Role.SYSTEM, Role.USER}:
            raise RuntimeError(
                f"Invalid `{Message.__name__}`, only {Role.USER.value} or {Role.SYSTEM.value} "
                f"can be the sender of tools."
            )
        # Verification of tool calls.
        if msg.tool_calls and msg.role != Role.ASSISTANT:
            raise RuntimeError(
                f"Invalid `{Message.__name__}`, only {Role.ASSISTANT.value} "
                f"can be the sender of tool calls."
            )
        if msg.tool_calls and msg.content:
            raise RuntimeError(
                f"Invalid `{Message.__name__}`, tool_calls and content cannot be effective at the same time."
            )
