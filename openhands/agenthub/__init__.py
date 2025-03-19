from dotenv import load_dotenv

from openhands.agenthub.micro.agent import MicroAgent
from openhands.agenthub.micro.registry import all_microagents
from openhands.controller.agent import Agent

load_dotenv()


from openhands.agenthub import (  # noqa: E402
    browser_monitor_agent,
    browsing_agent,
    codeact_agent,
    deepcoder_agent,
    deepplanner_agent,
    delegator_agent,
    dummy_agent,
    visualbrowsing_agent,
)

__all__ = [
    'codeact_agent',
    'delegator_agent',
    'dummy_agent',
    'browsing_agent',
    'visualbrowsing_agent',
    'deepplanner_agent',
    'browser_monitor_agent',
    'deepcoder_agent',
]

for agent in all_microagents.values():
    name = agent['name']
    prompt = agent['prompt']

    anon_class = type(
        name,
        (MicroAgent,),
        {
            'prompt': prompt,
            'agent_definition': agent,
        },
    )

    Agent.register(name, anon_class)
