from __future__ import annotations

from openhands.core.config.condenser_config import BrowserOutputCondenserConfig
from openhands.events.event import Event
from openhands.events.observation import BrowserOutputObservation
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.memory.condenser.condenser import Condenser


class BrowserOutputCondenser(Condenser):
    """A condenser that masks the observations from browser outputs outside of a recent attention window.

    The intent here is to mask just the browser outputs and leave everything else untouched. This is important because currently we provide screenshots and accessibility trees as input to the model for browser observations. These are really large and consume a lot of tokens without any benefits in performance. So we want to mask all such observations from all previous timesteps, and leave only the most recent one in context.
    """

    def __init__(self, attention_window: int = 1):
        self.attention_window = attention_window
        super().__init__()

    def condense(self, events: list[Event]) -> list[Event]:
        """Replace the content of browser observations outside of the attention window with a placeholder."""
        results: list[Event] = []
        cnt: int = 0
        for event in reversed(events):
            if (
                isinstance(event, BrowserOutputObservation)
                and cnt >= self.attention_window
            ):
                obs = AgentCondensationObservation(
                    f'Current URL: {event.url}\nContent Omitted'
                )
                if event.tool_call_metadata is not None:
                    obs.tool_call_metadata = event.tool_call_metadata  # 如果这里不添加tool_call_metadata，那么经过condenser的observation event事件对应的action事件，在后续conversation_memory处理时会被丢掉，因为tool call的action找不到与之tool call id对应的observation了，所以context上会出现，只有tool的observation结果，却没有assistant发起tool call的那个回答轮次。
                results.append(obs)
            else:
                results.append(event)
                if isinstance(event, BrowserOutputObservation):
                    cnt += 1

        return list(reversed(results))

    @classmethod
    def from_config(
        cls, config: BrowserOutputCondenserConfig
    ) -> BrowserOutputCondenser:
        return BrowserOutputCondenser(**config.model_dump(exclude=['type']))


BrowserOutputCondenser.register_config(BrowserOutputCondenserConfig)
