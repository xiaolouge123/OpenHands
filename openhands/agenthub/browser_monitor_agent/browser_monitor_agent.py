import json
import os
from collections import deque

import openhands.agenthub.browser_monitor_agent.function_calling as browser_monitor_function_calling
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
from openhands.utils.prompt import PromptManager


class BrowserMonitorAgent(Agent):
    VERSION = '1.0'
    """
    The Browser Monitor Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.
    """

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        super().__init__(llm, config)
        self.pending_actions: deque[Action] = deque()
        self.reset()

        self.mock_function_calling = False
        if not self.llm.is_function_calling_active():
            logger.info(
                f'Function calling not enabled for model {self.llm.config.model}. '
                'Mocking function calling via prompting.'
            )
            self.mock_function_calling = True

        self.tools = browser_monitor_function_calling.get_tools(
            browser_enable_listening=self.config.browser_enable_listening,
        )
        logger.debug(
            f'TOOLS loaded for BrowserMonitorAgent: {json.dumps(self.tools, indent=2, ensure_ascii=False).replace("\\n", "\n")}'
        )
        self.prompt_manager = PromptManager(
            microagent_dir=None,
            prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
            disabled_microagents=[],
            enable_world_info=self.config.enable_world_info,
        )
        self.conversation_memory = ConversationMemory(self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser)
        logger.debug(f'Using condenser in BrowserMonitorAgent: {self.condenser}')

    def reset(self) -> None:
        """Resets the Browser Monitor Agent."""
        super().reset()
        self.pending_actions.clear()

    def step(self, state: State) -> Action:
        """Performs one step using the Browser Monitor Agent.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - Action: the next action to take
        """
        # Continue with pending actions if any
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
        if self.mock_function_calling:
            params['mock_function_calling'] = True

        response = self.llm.completion(**params)
        actions = browser_monitor_function_calling.response_to_actions(response)
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _get_messages(self, state: State) -> list[Message]:
        """Gets the messages to pass to the model.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - list[Message]: the messages to pass to the model
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        messages = self.conversation_memory.process_initial_messages(
            with_caching=self.llm.is_caching_prompt_active()
        )

        # Condense the events from the state.
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
