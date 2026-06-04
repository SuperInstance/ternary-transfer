# Future Integration: ternary-transfer

## Current State
Provides transfer learning for ternary agents: `KnowledgeMatrix` represents learned feature weights, `TransferStrategy` defines how to move knowledge (DirectCopy, WeightedBlend, SelectiveTransfer, Progressive), `DomainGap` measures distance between source and target tasks, `TransferScore` evaluates transfer benefit, and `NegativeTransferDetector` prevents harmful transfers.

## Integration Opportunities

### With ternary-memory (Knowledge Persistence and Transfer)
ternary-memory's `LongTermMemory` stores learned weights. ternary-transfer's `KnowledgeMatrix` represents those weights for transfer. The bridge: `LongTermMemory::weights()` → `KnowledgeMatrix::new(weights)`, then apply `TransferStrategy::WeightedBlend { alpha }` to combine source room knowledge with target room baseline. `NegativeTransferDetector` checks whether the transfer helps or hurts before committing.

### With ternary-cell (Cross-Room Skill Sharing)
When an agent moves from Room A to Room B, it carries learned cell behavior. ternary-transfer defines how: `DomainGap::compute(room_a, room_b)` measures how similar the rooms are. High similarity → `DirectCopy` (full transfer). Moderate similarity → `SelectiveTransfer` (only shared features). Low similarity → `Progressive` (gradual transfer over multiple ticks). The agent's room transition IS a transfer learning event.

### With construct-core (Skill Transfer Across Layers)
construct-core's Layer 0/1/2 skill system needs knowledge transfer when a skill optimized for one layer is adapted for another. A Layer 2 async skill's knowledge matrix transfers to a Layer 1 sync skill via `WeightedBlend` — keeping the core decision logic but losing async-specific features. `NegativeTransferDetector` prevents transferring Layer 2 async patterns to ESP32's Layer 0 lookup table where they'd cause errors.

## Potential in Mature Systems
In room-as-codespace, rooms share knowledge through PLATO's tile store. When Room A learns something valuable (high-surprise discovery consolidated into long-term memory), PLATO evaluates which other rooms would benefit. `DomainGap` between Room A and every other room determines transfer candidates. `TransferScore` prioritizes transfers. `Progressive` transfer ensures knowledge diffuses gradually without overwhelming recipient rooms. `NegativeTransferDetector` prevents knowledge that helps Room A from hurting Room B.

## Cross-Pollination Ideas
- **ternary-federated**: Federated transfer — aggregate knowledge from multiple source rooms before transferring to the target, improving transfer quality.
- **ternary-curriculum**: Curriculum-guided transfer — transfer knowledge in curriculum order (easy→hard), so the target room builds foundations before receiving complex knowledge.
- **ternary-ensemble**: Ensemble transfer — combine knowledge from multiple source rooms using ensemble methods for more robust transfer.

## Dependencies for Next Steps
- Define `RoomTransfer` mapping rooms to source/target task pairs
- Add `DomainGap` computation based on room capability vectors
- Implement `NegativeTransferDetector` for room-to-room knowledge sharing
- Build progressive transfer pipeline for agent room transitions
