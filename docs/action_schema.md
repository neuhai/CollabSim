## Action Schema Overview

Actions are structured payloads emitted by agents and validated by the
controller. The action envelope includes common fields and action-specific
payloads.

Common fields:
- `type`: action primitive (`message`, `decide`, `propose`, `respond`, `transfer`, `do_nothing`)
- `actor_id`: agent identifier
- `timestamp`: ISO-8601 string or step index
- `payload`: action-specific payload
- `meta`: optional metadata

Notes:
- Action payload constraints are enforced by the action validator.
- `do_nothing` supports an optional `reason` string in the payload.
