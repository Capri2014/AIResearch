# Contingency Planning - Implementation TODO

## Phase 1: Environment Setup (Week 1)
- [x] 1.1 Create requirements.txt with CARLA, numpy, torch, osqp
- [x] 1.2 Define contingency scenarios (pedestrian, cut-in, occlusion, sensor)
- [x] 1.3 Implement Metrics dataclass
- [x] 1.4 Setup CARLA client connection (ready for integration)

## Phase 2: Classical Tree-Based (Week 2-3)
- [x] 2.1 Implement TreeNode and ControlTree data structures
- [x] 2.2 Implement TreeQPOptimizer (OSQP integration)
- [x] 2.3 Implement BeliefTracker for discrete uncertainties
- [ ] 2.4 Integrate with existing waypoint model
- [ ] 2.5 Basic test in simulation

## Phase 3: Model-Based (Week 4-5)
- [ ] 3.1 Implement ContingencyNetwork (neural branching)
- [ ] 3.2 Implement ControlBarrierFilter (CBF-QP)
- [ ] 3.3 Create training loop for contingency network
- [ ] 3.4 Integrate safety filter with planning
- [ ] 3.5 Basic test in simulation

## Phase 4: Simulation & Comparison (Week 6)
- [ ] 4.1 Implement CARLA scenario runner
- [ ] 4.2 Implement comparison benchmark
- [ ] 4.3 Create visualization dashboard
- [ ] 4.4 Run full comparison and generate report

## Status
Current: Phase 1 - Environment Setup
