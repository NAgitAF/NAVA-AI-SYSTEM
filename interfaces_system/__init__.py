"""Interfaces System — user and external channels for NAVA."""


class InterfaceAdapter:
    """Generic adapter for CLI, web, mobile or API frontends."""

    def __init__(self, channel_name: str):
        self.channel_name = channel_name

    def send(self, payload):
        return {"channel": self.channel_name, "payload": payload}
