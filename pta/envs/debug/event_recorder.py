"""EventRecorder -- Logs timestamped events during episode rollout.

Useful for post-hoc debugging of failure modes (e.g. when did spill
start, when did contact force spike, when did the tool lose grip).
"""

from __future__ import annotations

from typing import Any, Dict, List


class EventRecorder:
    """Record and query timestamped events during an episode.

    Usage::

        recorder = EventRecorder()
        recorder.record("spill_start", step=42, data={"count": 5})
        events = recorder.get_events("spill_start")
        recorder.save("logs/events_ep_001.json")
    """

    def __init__(self) -> None:
        """Initialise an empty event log."""
        self._events: List[Dict[str, Any]] = []

    def record(self, event_type: str, step: int, data: Dict[str, Any] | None = None) -> None:
        """Record a single event.

        Parameters
        ----------
        event_type:
            Category string (e.g. ``"spill_start"``, ``"contact_spike"``).
        step:
            Simulation step at which the event occurred.
        data:
            Optional payload with event-specific details.
        """
        raise NotImplementedError

    def get_events(self, event_type: str | None = None) -> List[Dict[str, Any]]:
        """Return recorded events, optionally filtered by *event_type*.

        Parameters
        ----------
        event_type:
            If provided, return only events of this type.

        Returns
        -------
        list[dict]
            List of event dicts with keys ``"type"``, ``"step"``,
            ``"data"``.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Serialise events to a JSON file at *path*.

        Parameters
        ----------
        path:
            Output file path.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all recorded events."""
        raise NotImplementedError
