import json
import os
from collections import deque

import openhands.agenthub.deepcoder_agent.function_calling as deepcoder_function_calling
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import (
    Action,
    AgentFinishAction,
)
from openhands.llm.llm import LLM
from openhands.memory.condenser import Condenser
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager


class DeepCoderAgent(Agent):
    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        super().__init__(llm, config)
        self.pending_actions: deque[Action] = deque()
        self.reset()

        self.tools = deepcoder_function_calling.get_tools()
        logger.debug(
            f'TOOLS loaded for DeepCoderAgent: {json.dumps(self.tools, indent=2, ensure_ascii=False).replace("\\n", "\n")}'
        )
        self.prompt_manager = PromptManager(
            microagent_dir=None,
            prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
            disabled_microagents=[],
            enable_world_info=self.config.enable_world_info,
        )

        self.conversation_memory = ConversationMemory(self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser)
        logger.debug(f'Using condenser: {type(self.condenser)}')

    def reset(self) -> None:
        """Resets the DeepCoder Agent."""
        super().reset()
        self.pending_actions.clear()

    def step(self, state: State) -> Action:
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # prepare what we want to send to the LLM
        messages = self._get_messages(state)
        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
        }
        params['tools'] = self.tools
        response = self.llm.completion(**params)
        actions = deepcoder_function_calling.response_to_actions(response)
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _get_messages(self, state: State) -> list[Message]:
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        messages = self.conversation_memory.process_initial_messages(
            with_caching=self.llm.is_caching_prompt_active()
        )

        events = self.condenser.condensed_history(state)

        logger.debug(
            f'Processing {len(events)} events from a total of {len(state.history)} events'
        )

        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_messages=messages,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
            enable_som_visual_browsing=self.config.enable_som_visual_browsing,
        )

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def _take_my_shoes(self, messages: list[Message]) -> list[Message]:
        return [
            Message(
                role='user',
                content='Take my shoes off',
            )
        ]
