"""OpenAI agent worker."""

import asyncio
import json
import logging
import uuid
from functools import partial
from threading import Thread
from typing import Any, Dict, List, Callable, Optional, Tuple, Union, cast, get_args
import re

from llama_index.agent.openai.utils import resolve_tool_choice
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.agent.utils import add_user_step_to_memory
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import OpenAIToolCall

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

DEFAULT_MAX_FUNCTION_CALLS = 5


def get_function_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    """Get function by name."""
    name_to_tool = {tool.metadata.name: tool for tool in tools}
    if name not in name_to_tool:
        raise ValueError(f"Tool with name {name} not found")
    return name_to_tool[name]


def call_tool_with_error_handling(
    tool: BaseTool,
    input_dict: Dict,
    error_message: Optional[str] = None,
    raise_error: bool = False,
) -> ToolOutput:
    """Call tool with error handling.

    Input is a dictionary with args and kwargs

    """
    try:
        return tool(**input_dict)
    except Exception as e:
        if raise_error:
            raise
        error_message = error_message or f"Error: {e!s}"
        return ToolOutput(
            content=error_message,
            tool_name=tool.metadata.name,
            raw_input={"kwargs": input_dict},
            raw_output=e,
        )


def default_tool_call_parser(tool_call: OpenAIToolCall):
    try:
        return json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Error in calling tool {tool_call.function.name}: The input json block is malformed:\n```json\n{tool_call.function.arguments}\n```"
        )


def advanced_tool_call_parser(tool_call: OpenAIToolCall) -> Dict:
    r"""Parse tool calls that are not standard json.

    Also parses tool calls of the following forms:
    variable = \"\"\"Some long text\"\"\"
    variable = "Some long text"'
    variable = '''Some long text'''
    variable = 'Some long text'
    """
    arguments_str = tool_call.function.arguments
    if len(arguments_str.strip()) == 0:
        # OpenAI returns an empty string for functions containing no args
        return {}
    try:
        tool_call = json.loads(arguments_str)
        if not isinstance(tool_call, dict):
            raise ValueError(
                f"Error in calling tool {tool_call.function.name}: The input json block is malformed:\n```json\n{tool_call.function.arguments}\n```"
            )
        return tool_call
    except json.JSONDecodeError as e:
        # pattern to match variable names and content within quotes
        pattern = r'([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*["\']+(.*?)["\']+'
        match = re.search(pattern, arguments_str)

        if match:
            variable_name = match.group(1)  # This is the variable name
            content = match.group(2)  # This is the content within the quotes
            return {variable_name: content}
        raise ValueError(
            f"Error in calling tool {tool_call.function.name}: The input json block is malformed:\n```json\n{tool_call.function.arguments}\n```"
        )


def call_function(
    tools: List[BaseTool],
    tool_call: OpenAIToolCall,
    verbose: bool = False,
    tool_call_parser: Optional[Callable[[OpenAIToolCall], Dict]] = None,
) -> Tuple[ChatMessage, ToolOutput]:
    """Call a function and return the output as a string."""
    # validations to get passed mypy
    assert tool_call.id is not None
    assert tool_call.function is not None
    assert tool_call.function.name is not None
    assert tool_call.function.arguments is not None
    tool_call_parser = tool_call_parser or default_tool_call_parser

    id_ = tool_call.id
    _function_call = tool_call.function
    name = tool_call.function.name
    arguments_str = tool_call.function.arguments
    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = get_function_by_name(tools, name)
    error_message: Optional[str] = None
    try:
        argument_dict = tool_call_parser(tool_call)
    except ValueError as e:
        error_message = str(e)
        return (
            ChatMessage(
                content=error_message,
                role=MessageRole.TOOL,
                additional_kwargs={
                    "name": name,
                    "tool_call_id": id_,
                },
            ),
            ToolOutput(
                content=error_message,
                tool_name=name,
                raw_input={"args": arguments_str},
                raw_output=error_message,
                is_error=True,
            ),
        )

    # Call tool
    # Use default error message except if json parsing fails
    output = call_tool_with_error_handling(
        tool, argument_dict, error_message=error_message
    )
    if verbose:
        print(f"Got output: {output!s}")
        print("========================\n")
    return (
        ChatMessage(
            content=str(output),
            role=MessageRole.TOOL,
            additional_kwargs={
                "name": name,
                "tool_call_id": id_,
            },
        ),
        output,
    )


async def acall_function(
    tools: List[BaseTool],
    tool_call: OpenAIToolCall,
    verbose: bool = False,
    tool_call_parser: Optional[Callable[[OpenAIToolCall], Dict]] = None,
) -> Tuple[ChatMessage, ToolOutput]:
    """Call a function and return the output as a string."""
    # validations to get passed mypy
    assert tool_call.id is not None
    assert tool_call.function is not None
    assert tool_call.function.name is not None
    assert tool_call.function.arguments is not None
    tool_call_parser = tool_call_parser or default_tool_call_parser

    id_ = tool_call.id
    _function_call = tool_call.function
    name = tool_call.function.name
    arguments_str = tool_call.function.arguments
    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = get_function_by_name(tools, name)
    async_tool = adapt_to_async_tool(tool)
    error_message: Optional[str] = None
    try:
        argument_dict = tool_call_parser(tool_call)
    except ValueError as e:
        error_message = str(e)
        return (
            ChatMessage(
                content=error_message,
                role=MessageRole.TOOL,
                additional_kwargs={
                    "name": name,
                    "tool_call_id": id_,
                },
            ),
            ToolOutput(
                content=error_message,
                tool_name=name,
                raw_input={"args": arguments_str},
                raw_output=error_message,
                is_error=True,
            ),
        )

    output = await async_tool.acall(**argument_dict)
    if verbose:
        print(f"Got output: {output!s}")
        print("========================\n")
    return (
        ChatMessage(
            content=str(output),
            role=MessageRole.TOOL,
            additional_kwargs={
                "name": name,
                "tool_call_id": id_,
            },
        ),
        output,
    )


class OllamaRamaAgentWorker(BaseAgentWorker):
    """OpenAI Agent agent worker."""

    def __init__(
        self,
        tools: List[BaseTool],
        llm: OpenAI,
        prefix_messages: List[ChatMessage],
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        tool_call_parser: Optional[Callable[[OpenAIToolCall], Dict]] = None,
    ):
        self._llm = llm
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.prefix_messages = prefix_messages
        self.callback_manager = callback_manager or self._llm.callback_manager
        self.tool_call_parser = tool_call_parser or default_tool_call_parser

        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            # no tools
            self._get_tools = lambda _: []

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        tool_call_parser: Optional[Callable[[OpenAIToolCall], Dict]] = None,
        **kwargs: Any,
    ) -> "OpenAIAgentWorker":
        """Create an OpenAIAgent from a list of tools.

        Similar to `from_defaults` in other classes, this method will
        infer defaults for a variety of parameters, including the LLM,
        if they are not specified.

        """
        tools = tools or []

        llm = llm or Settings.llm
        # if not isinstance(llm, OpenAI):
        #     raise ValueError("llm must be a OpenAI instance")

        if callback_manager is not None:
            llm.callback_manager = callback_manager

        if not llm.metadata.is_function_calling_model:
            raise ValueError(
                f"Model name {llm.model} does not support function calling API. "
            )

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        return cls(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            tool_call_parser=tool_call_parser,
        )

    def get_all_messages(self, task: Task) -> List[ChatMessage]:
        return (
            self.prefix_messages
            + task.memory.get()
            + task.extra_state["new_memory"].get_all()
        )

    def get_latest_tool_calls(self, task: Task) -> Optional[List[OpenAIToolCall]]:
        chat_history: List[ChatMessage] = task.extra_state["new_memory"].get_all()
        return (
            chat_history[-1].additional_kwargs.get("tool_calls", None)
            if chat_history
            else None
        )

    def _get_llm_chat_kwargs(
        self,
        task: Task,
        openai_tools: List[dict],
        tool_choice: Union[str, dict] = "auto",
    ) -> Dict[str, Any]:
        llm_chat_kwargs: dict = {"messages": self.get_all_messages(task)}
        if openai_tools:
            llm_chat_kwargs.update(
                tools=openai_tools, tool_choice=resolve_tool_choice(tool_choice)
            )
        return llm_chat_kwargs

    def _process_message(
        self, task: Task, chat_response: ChatResponse
    ) -> AgentChatResponse:
        ai_message = chat_response.message
        task.extra_state["new_memory"].put(ai_message)
        return AgentChatResponse(
            response=str(ai_message.content), sources=task.extra_state["sources"]
        )

    def _get_stream_ai_response(
        self, task: Task, **llm_chat_kwargs: Any
    ) -> StreamingAgentChatResponse:
        chat_stream_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(**llm_chat_kwargs),
            sources=task.extra_state["sources"],
        )
        # Get the response in a separate thread so we can yield the response
        thread = Thread(
            target=chat_stream_response.write_response_to_history,
            args=(task.extra_state["new_memory"],),
            kwargs={"on_stream_end_fn": partial(self.finalize_task, task)},
        )
        thread.start()
        # Wait for the event to be set
        chat_stream_response.is_function_not_none_thread_event.wait()
        # If it is executing an openAI function, wait for the thread to finish
        if chat_stream_response.is_function:
            thread.join()

        # if it's false, return the answer (to stream)
        return chat_stream_response

    async def _get_async_stream_ai_response(
        self, task: Task, **llm_chat_kwargs: Any
    ) -> StreamingAgentChatResponse:
        chat_stream_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(**llm_chat_kwargs),
            sources=task.extra_state["sources"],
        )
        # create task to write chat response to history
        asyncio.create_task(
            chat_stream_response.awrite_response_to_history(
                task.extra_state["new_memory"],
                on_stream_end_fn=partial(self.finalize_task, task),
            )
        )
        chat_stream_response._ensure_async_setup()

        # wait until openAI functions stop executing
        await chat_stream_response.is_function_false_event.wait()

        # return response stream
        return chat_stream_response

    def _get_agent_response(
        self, task: Task, mode: ChatResponseMode, **llm_chat_kwargs: Any
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        if mode == ChatResponseMode.WAIT:
            chat_response: ChatResponse = self._llm.chat(**llm_chat_kwargs)
            return self._process_message(task, chat_response)
        elif mode == ChatResponseMode.STREAM:
            return self._get_stream_ai_response(task, **llm_chat_kwargs)
        else:
            raise NotImplementedError

    async def _get_async_agent_response(
        self, task: Task, mode: ChatResponseMode, **llm_chat_kwargs: Any
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        if mode == ChatResponseMode.WAIT:
            chat_response: ChatResponse = await self._llm.achat(**llm_chat_kwargs)
            return self._process_message(task, chat_response)
        elif mode == ChatResponseMode.STREAM:
            return await self._get_async_stream_ai_response(task, **llm_chat_kwargs)
        else:
            raise NotImplementedError

    def _call_function(
        self,
        tools: List[BaseTool],
        tool_call: OpenAIToolCall,
        memory: BaseMemory,
        sources: List[ToolOutput],
    ) -> bool:
        function_call = tool_call.function
        # validations to get passed mypy
        assert function_call is not None
        assert function_call.name is not None
        assert function_call.arguments is not None

        tool = get_function_by_name(tools, function_call.name)

        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: function_call.arguments,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            function_message, tool_output = call_function(
                tools,
                tool_call,
                verbose=self._verbose,
                tool_call_parser=self.tool_call_parser,
            )
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        sources.append(tool_output)
        memory.put(function_message)

        return tool.metadata.return_direct and not tool_output.is_error

    async def _acall_function(
        self,
        tools: List[BaseTool],
        tool_call: OpenAIToolCall,
        memory: BaseMemory,
        sources: List[ToolOutput],
    ) -> bool:
        function_call = tool_call.function
        # validations to get passed mypy
        assert function_call is not None
        assert function_call.name is not None
        assert function_call.arguments is not None

        tool = get_function_by_name(tools, function_call.name)

        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: function_call.arguments,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            function_message, tool_output = await acall_function(
                tools,
                tool_call,
                verbose=self._verbose,
                tool_call_parser=self.tool_call_parser,
            )
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        sources.append(tool_output)
        memory.put(function_message)

        return tool.metadata.return_direct and not tool_output.is_error

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()
        # initialize task state
        task_state = {
            "sources": sources,
            "n_function_calls": 0,
            "new_memory": new_memory,
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
        )

    def _should_continue(
        self, tool_calls: Optional[List[OpenAIToolCall]], n_function_calls: int
    ) -> bool:
        if n_function_calls > self._max_function_calls:
            return False
        if not tool_calls:
            return False
        return True

    def get_tools(self, input: str) -> List[BaseTool]:
        """Get tools."""
        return self._get_tools(input)

    def _run_step(
        self,
        step: TaskStep,
        task: Task,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        tool_choice: Union[str, dict] = "auto",
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_memory(
                step, task.extra_state["new_memory"], verbose=self._verbose
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)
        openai_tools = [tool.metadata.to_openai_tool() for tool in tools]

        llm_chat_kwargs = self._get_llm_chat_kwargs(task, openai_tools, tool_choice)
        agent_chat_response = self._get_agent_response(
            task, mode=mode, **llm_chat_kwargs
        )

        # TODO: implement _should_continue
        latest_tool_calls = self.get_latest_tool_calls(task) or []
        if not self._should_continue(
            latest_tool_calls, task.extra_state["n_function_calls"]
        ):
            is_done = True
            new_steps = []
            # TODO: return response
        else:
            is_done = False
            for tool_call in latest_tool_calls:
                # Some validation
                if not isinstance(tool_call, get_args(OpenAIToolCall)):
                    raise ValueError("Invalid tool_call object")

                if tool_call.type != "function":
                    raise ValueError("Invalid tool type. Unsupported by OpenAI")
                # TODO: maybe execute this with multi-threading
                return_direct = self._call_function(
                    tools,
                    tool_call,
                    task.extra_state["new_memory"],
                    task.extra_state["sources"],
                )
                # change function call to the default value, if a custom function was given
                # as an argument (none and auto are predefined by OpenAI)
                if tool_choice not in ("auto", "none"):
                    tool_choice = "auto"
                task.extra_state["n_function_calls"] += 1

                if return_direct and len(latest_tool_calls) == 1:
                    is_done = True
                    response_str = task.extra_state["sources"][-1].content
                    chat_response = ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT, content=response_str
                        )
                    )
                    agent_chat_response = self._process_message(task, chat_response)
                    agent_chat_response.is_dummy_stream = (
                        mode == ChatResponseMode.STREAM
                    )
                    break

            new_steps = (
                [
                    step.get_next_step(
                        step_id=str(uuid.uuid4()),
                        # NOTE: input is unused
                        input=None,
                    )
                ]
                if not is_done
                else []
            )

        # attach next step to task

        return TaskStepOutput(
            output=agent_chat_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    async def _arun_step(
        self,
        step: TaskStep,
        task: Task,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        tool_choice: Union[str, dict] = "auto",
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_memory(
                step, task.extra_state["new_memory"], verbose=self._verbose
            )

        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)
        openai_tools = [tool.metadata.to_openai_tool() for tool in tools]

        llm_chat_kwargs = self._get_llm_chat_kwargs(task, openai_tools, tool_choice)
        agent_chat_response = await self._get_async_agent_response(
            task, mode=mode, **llm_chat_kwargs
        )

        # TODO: implement _should_continue
        latest_tool_calls = self.get_latest_tool_calls(task) or []
        if not self._should_continue(
            latest_tool_calls, task.extra_state["n_function_calls"]
        ):
            is_done = True

        else:
            is_done = False
            for tool_call in latest_tool_calls:
                # Some validation
                if not isinstance(tool_call, get_args(OpenAIToolCall)):
                    raise ValueError("Invalid tool_call object")

                if tool_call.type != "function":
                    raise ValueError("Invalid tool type. Unsupported by OpenAI")
                # TODO: maybe execute this with multi-threading
                return_direct = await self._acall_function(
                    tools,
                    tool_call,
                    task.extra_state["new_memory"],
                    task.extra_state["sources"],
                )
                # change function call to the default value, if a custom function was given
                # as an argument (none and auto are predefined by OpenAI)
                if tool_choice not in ("auto", "none"):
                    tool_choice = "auto"
                task.extra_state["n_function_calls"] += 1

                if return_direct and len(latest_tool_calls) == 1:
                    is_done = True
                    response_str = task.extra_state["sources"][-1].content
                    chat_response = ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT, content=response_str
                        )
                    )
                    agent_chat_response = self._process_message(task, chat_response)
                    agent_chat_response.is_dummy_stream = (
                        mode == ChatResponseMode.STREAM
                    )
                    break

        # generate next step, append to task queue
        new_steps = (
            [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    # NOTE: input is unused
                    input=None,
                )
            ]
            if not is_done
            else []
        )

        return TaskStepOutput(
            output=agent_chat_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        tool_choice = kwargs.get("tool_choice", "auto")
        return self._run_step(
            step, task, mode=ChatResponseMode.WAIT, tool_choice=tool_choice
        )

    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        tool_choice = kwargs.get("tool_choice", "auto")
        return await self._arun_step(
            step, task, mode=ChatResponseMode.WAIT, tool_choice=tool_choice
        )

    @trace_method("run_step")
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        # TODO: figure out if we need a different type for TaskStepOutput
        tool_choice = kwargs.get("tool_choice", "auto")
        return self._run_step(
            step, task, mode=ChatResponseMode.STREAM, tool_choice=tool_choice
        )

    @trace_method("run_step")
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        tool_choice = kwargs.get("tool_choice", "auto")
        return await self._arun_step(
            step, task, mode=ChatResponseMode.STREAM, tool_choice=tool_choice
        )

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # add new messages to memory
        task.memory.set(
            task.memory.get_all() + task.extra_state["new_memory"].get_all()
        )
        # reset new memory
        task.extra_state["new_memory"].reset()

    def undo_step(self, task: Task, **kwargs: Any) -> Optional[TaskStep]:
        """Undo step from task.

        If this cannot be implemented, return None.

        """
        raise NotImplementedError("Undo is not yet implemented")
        # if len(task.completed_steps) == 0:
        #     return None

        # # pop last step output
        # last_step_output = task.completed_steps.pop()
        # # add step to the front of the queue
        # task.step_queue.appendleft(last_step_output.task_step)

        # # undo any `step_state` variables that have changed
        # last_step_output.step_state["n_function_calls"] -= 1

        # # TODO: we don't have memory pop capabilities yet
        # # # now pop the memory until we get to the state
        # # last_step_response = cast(AgentChatResponse, last_step_output.output)
        # # while last_step_response != task.memory.:
        # #     last_message = last_step_output.task_step.memory.pop()
        # #     if last_message == cast(AgentChatResponse, last_step_output.output).response:
        # #         break

        # # while cast(AgentChatResponse, last_step_output.output).response !=

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: make this abstractmethod (right now will break some agent impls)
        self.callback_manager = callback_manager
