"""Generic SeisComP client: event listening, pick retrieval and publishing.

This package owns every transaction with SeisComP (messaging + database).
It is science-agnostic: EventListenerApp delegates ready events to an
injected handler callable. Copied from REHASH (shared scclient design).
"""

SUBSCRIPTIONS = ("EVENT", "LOCATION")
OUTPUT_GROUP = "FOCMECH"
LOAD_INVENTORY = True
