"""
Feast entity definitions
"""

from feast import Entity, ValueType

aircraft = Entity(
    name="aircraft_id",
    description="ICA0 24-bit hex address uniquely identifying an aircraft.",
    value_type=ValueType.STRING,
    join_keys=["aircraft_id"],
)

airport = Entity(
    name="airport_code",
    description="ICA0 airport code (e.g YMML,EGLL)",
    value_type=ValueType.STRING,
    join_keys=["airport_code"],
)

route = Entity(
    name="route_id",
    description="Route identifier derived from callsign prefix (airline + route number)",
    value_type=ValueType.STRING,
    join_keys=["route_id"],
)
