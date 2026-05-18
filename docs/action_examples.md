## Action Examples

Representative action payloads (envelope + payload):

### Communicate
```json
{
  "type": "message",
  "actor_id": "agent_A",
  "timestamp": "2026-03-09T12:00:00Z",
  "payload": {
    "channel": "broadcast",
    "recipients": [],
    "content": "Status update: ready to proceed.",
    "content_type": "text"
  },
  "meta": {"turn_id": "t1"}
}
```

### Decide
```json
{
  "type": "decide",
  "actor_id": "agent_B",
  "timestamp": 3,
  "payload": {
    "decision_id": "allocate_resource",
    "choice": "option_A",
    "reveal": "aggregated"
  }
}
```

### Transfer
```json
{
  "type": "transfer",
  "actor_id": "agent_A",
  "timestamp": 4,
  "payload": {
    "resource_id": "credits",
    "amount": 3,
    "to": "agent_B"
  }
}
```

### Do nothing
```json
{
  "type": "do_nothing",
  "actor_id": "agent_A",
  "timestamp": 5,
  "payload": {
    "reason": "No new information to act on."
  }
}
```
