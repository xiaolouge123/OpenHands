import json
import os
from collections import deque

import openhands.agenthub.deepplanner_agent.function_calling as deepplanner_function_calling
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import (
    Action,
    AgentFinishAction,
)
from openhands.events.observation import AgentDelegateObservation
from openhands.llm.llm import LLM
from openhands.memory.condenser import Condenser
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager


class DeepPlannerAgent(Agent):
    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, llm: LLM, config: AgentConfig):
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

        self.tools = deepplanner_function_calling.get_tools()
        logger.debug(
            f'TOOLS loaded for DeepPlannerAgent: {json.dumps(self.tools, indent=2, ensure_ascii=False).replace("\\n", "\n")}'
        )

        self.prompt_manager = PromptManager(
            microagent_dir=None,
            prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
            disabled_microagents=[],
            enable_world_info=self.config.enable_world_info,
        )

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser)
        logger.debug(f'Using condenser in DeepPlannerAgent: {self.condenser}')

        # hacking some important information sharing between agents, like workspace directory
        self.runtime_auxiliary_info = None

    def reset(self) -> None:
        """Resets the DeepPlanner Agent."""
        super().reset()
        self.pending_actions.clear()

    def step(self, state: State) -> Action:
        """Performs one step using the DeepPlanner Agent.
        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - CmdRunAction(command) - bash command to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
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
        actions = deepplanner_function_calling.response_to_actions(response)
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _get_messages(self, state: State) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand. It handles both regular
        message flow and function-calling scenarios.

        The method performs the following steps:
        1. Initializes with system prompt,
        2. Processes events (Actions and Observations) into messages
        3. Handles tool calls and their responses in function-calling mode
        4. Manages message role alternation (user/assistant/tool)
        5. Applies caching for specific LLM providers (e.g., Anthropic)
        6. Adds environment reminders for non-function-calling mode

        Args:
            state (State): The current state object containing conversation history and other metadata

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption, including:
                - System message with prompt
                - Initial user message (if configured)
                - Action messages (from both user and assistant)
                - Observation messages (including tool responses)
                - Environment reminders (in non-function-calling mode)

        Note:
            - In function-calling mode, tool calls and their responses are carefully tracked
              to maintain proper conversation flow
            - Messages from the same role are combined to prevent consecutive same-role messages
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        messages = self.conversation_memory.process_initial_messages(
            with_caching=self.llm.is_caching_prompt_active()
        )
        mark = False
        mark = self._check_delegate_observation(state)
        # Condense the events from the state.
        events = self.condenser.condensed_history(state)

        if mark:
            for event in events:
                if isinstance(event, AgentDelegateObservation):
                    logger.info(f'Found delegate observation after condense: {event}')

        logger.info(
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

    def _take_my_shoes(self):
        """
        To improve the agent reasoning ability, inject the reasoning-content to the messages, working as the same purpose as the thinker agent, but no delegation and no thinkaction in the history events.
        Before the original agent generation completion, inject a user content of context reasoning-content to the messages.
        """
        pass

    def _check_delegate_observation(self, state: State) -> bool:
        events = state.history
        for event in events:
            if isinstance(event, AgentDelegateObservation):
                logger.info(f'Found delegate observation: {event}')
                return True
        return False
