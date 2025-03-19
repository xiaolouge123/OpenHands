from openhands.agenthub.browser_monitor_agent.browser_monitor_agent import (
    BrowserMonitorAgent,
)
from openhands.controller.agent import Agent

Agent.register('BrowserMonitorAgent', BrowserMonitorAgent)
