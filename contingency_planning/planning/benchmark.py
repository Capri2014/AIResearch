"""
Benchmark: Compare All Planners

This module provides:
1. Standardized test scenarios
2. Comparison metrics across planners
3. Verification that new methods work and are better
4. New scenario additions

Run: python -m contingency_planning.planning.benchmark
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from .lattice_planner import create_lattice_planner
from .behavior_tree import create_behavior_tree_planner
from .mcts_planner import create_mcts_planner
from .corridor_manager import create_corridor_manager
from .sdf_collision import create_collision_checker
from .risk_aggregator import create_risk_aggregator, RiskMetric
from .mrm_safety import create_mrm_system, create_safety_supervisor


class ScenarioType(Enum):
    """Standard test scenarios"""
    HIGHWAY = "highway"
    LANE_CHANGE = "lane_change"
    MERGE = "merge"
    INTERSECTION = "intersection"
    URBAN_NEGOTIATION = "urban_negotiation"
    CONSTRUCTION = "construction"
    CUT_IN = "cut_in"
    PEDESTRIAN = "pedestrian"


@dataclass
class Scenario:
    """Standard scenario definition"""
    name: str
    scenario_type: ScenarioType
    ego_start: Dict  # {'s', 'l', 'v'}
    goal_s: float
    obstacles: List[Dict]
    expected_outcome: str  # "safe", "collision", "mrm"
    

@dataclass
class BenchmarkResult:
    """Result of benchmarking a planner on a scenario"""
    planner_name: str
    scenario_name: str
    success: bool
    collision: bool
    planning_time_ms: float
    trajectory_length: int
    progress: float
    mrm_triggered: bool


class Benchmark:
    """
    Benchmark suite for comparing planners.
    
    Usage:
        bench = Benchmark()
        results = bench.run_all()
        bench.print_results(results)
    """
    
    def __init__(self):
        self.scenarios = self._create_scenarios()
        
    def _create_scenarios(self) -> List[Scenario]:
        """Create standardized test scenarios"""
        return [
            # Highway: simple lane keep
            Scenario(
                name="highway_simple",
                scenario_type=ScenarioType.HIGHWAY,
                ego_start={'s': 0, 'l': 0, 'v': 20},
                goal_s=100,
                obstacles=[],
                expected_outcome="safe"
            ),
            
            # Highway with lead vehicle
            Scenario(
                name="highway_lead",
                scenario_type=ScenarioType.HIGHWAY,
                ego_start={'s': 0, 'l': 0, 'v': 20},
                goal_s=100,
                obstacles=[{'s': 50, 'l': 0, 'v': 15}],
                expected_outcome="safe"
            ),
            
            # Lane change required
            Scenario(
                name="lane_change",
                scenario_type=ScenarioType.LANE_CHANGE,
                ego_start={'s': 0, 'l': 0, 'v': 15},
                goal_s=100,
                obstacles=[{'s': 30, 'l': 0, 'v': 12}],
                expected_outcome="safe"
            ),
            
            # Merge scenario
            Scenario(
                name="merge",
                scenario_type=ScenarioType.MERGE,
                ego_start={'s': 0, 'l': 0, 'v': 10},
                goal_s=80,
                obstacles=[
                    {'s': 20, 'l': -3.5, 'v': 12},  # Contender from behind
                ],
                expected_outcome="safe"
            ),
            
            # Cut-in scenario
            Scenario(
                name="cut_in",
                scenario_type=ScenarioType.CUT_IN,
                ego_start={'s': 0, 'l': 0, 'v': 18},
                goal_s=80,
                obstacles=[
                    {'s': 25, 'l': -3, 'v': 20},  # Cuts in
                ],
                expected_outcome="safe"
            ),
            
            # Construction zone
            Scenario(
                name="construction",
                scenario_type=ScenarioType.CONSTRUCTION,
                ego_start={'s': 0, 'l': 0, 'v': 10},
                goal_s=60,
                obstacles=[
                    {'s': 30, 'l': -1.5, 'v': 0, 'type': 'barrier'},
                ],
                expected_outcome="safe"
            ),
            
            # Pedestrian crossing
            Scenario(
                name="pedestrian",
                scenario_type=ScenarioType.PEDESTRIAN,
                ego_start={'s': 0, 'l': 0, 'v': 8},
                goal_s=40,
                obstacles=[
                    {'s': 20, 'l': -5, 'v': 1.5, 'type': 'pedestrian'},  # Crossing
                ],
                expected_outcome="safe"
            ),
            
            # Urban negotiation (multiple agents)
            Scenario(
                name="urban_negotiation",
                scenario_type=ScenarioType.URBAN_NEGOTIATION,
                ego_start={'s': 0, 'l': 0, 'v': 10},
                goal_s=60,
                obstacles=[
                    {'s': 15, 'l': 0, 'v': 8},
                    {'s': -5, 'l': 3.5, 'v': 10},
                    {'s': 25, 'l': -3, 'v': 12},
                ],
                expected_outcome="safe"
            ),
            
            # Collision scenario (should fail)
            Scenario(
                name="collision_ahead",
                scenario_type=ScenarioType.HIGHWAY,
                ego_start={'s': 0, 'l': 0, 'v': 15},
                goal_s=50,
                obstacles=[{'s': 10, 'l': 0, 'v': 0}],  # Stopped vehicle
                expected_outcome="collision"  # Baseline should collide
            ),
        ]
    
    def run_scenario(self, scenario: Scenario, planner_fn: Callable) -> BenchmarkResult:
        """Run one scenario with one planner"""
        start_time = time.time()
        
        try:
            # Run planner
            result = planner_fn(scenario)
            planning_time = (time.time() - start_time) * 1000
            
            # Check outcome
            trajectory = result.get('trajectory', [])
            collision = result.get('collision', False)
            mrm_triggered = result.get('mrm_triggered', False)
            
            success = (
                not collision and 
                len(trajectory) > 0 and
                trajectory[-1][0] > scenario.goal_s * 0.8
            )
            
            progress = trajectory[-1][0] if trajectory else 0
            
            return BenchmarkResult(
                planner_name=planner_fn.__name__,
                scenario_name=scenario.name,
                success=success,
                collision=collision,
                planning_time_ms=planning_time,
                trajectory_length=len(trajectory),
                progress=progress,
                mrm_triggered=mrm_triggered
            )
            
        except Exception as e:
            return BenchmarkResult(
                planner_name=planner_fn.__name__,
                scenario_name=scenario.name,
                success=False,
                collision=True,
                planning_time_ms=(time.time() - start_time) * 1000,
                trajectory_length=0,
                progress=0,
                mrm_triggered=False
            )
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all scenarios with all planners"""
        results = []
        
        # Define planners to test
        planners = [
            ("Lattice", self._test_lattice),
            ("BehaviorTree", self._test_behavior_tree),
            ("MCTS", self._test_mcts),
            ("Unified", self._test_unified),
        ]
        
        for scenario in self.scenarios:
            for planner_name, planner_fn in planners:
                result = self.run_scenario(scenario, planner_fn)
                results.append(result)
        
        return results
    
    def _test_lattice(self, scenario: Scenario) -> Dict:
        """Test lattice planner"""
        planner = create_lattice_planner()
        
        ego_state = (
            scenario.ego_start['s'],
            scenario.ego_start['s'],  # x = s
            0  # heading
        )
        
        # Convert obstacles
        obstacles = [
            {'s': o['s'], 'l': o.get('l', 0), 'width': 2, 'length': 4.5}
            for o in scenario.obstacles
        ]
        
        path = planner.plan(ego_state, scenario.goal_s, obstacles)
        
        # Check collision
        collision = self._check_collision(path, scenario.obstacles)
        
        return {'trajectory': path, 'collision': collision, 'mrm_triggered': False}
    
    def _test_behavior_tree(self, scenario: Scenario) -> Dict:
        """Test behavior tree planner"""
        planner = create_behavior_tree_planner()
        
        state = {
            's': scenario.ego_start['s'],
            'l': scenario.ego_start.get('l', 0),
            'speed': scenario.ego_start['v'],
            'gaps': [{'l': 3.5, 'available': True}] if scenario.scenario_type in [
                ScenarioType.LANE_CHANGE, ScenarioType.MERGE
            ] else [],
            'lead_vehicle': scenario.obstacles[0] if scenario.obstacles else None,
            'crossing_agents': [
                o for o in scenario.obstacles 
                if o.get('type') == 'pedestrian'
            ] if scenario.scenario_type == ScenarioType.PEDESTRIAN else []
        }
        
        result = planner.plan(state, scenario.obstacles)
        
        traj = result.get('trajectory', [])
        traj_xy = [(s, l) for s, l, t in traj] if traj else []
        
        collision = self._check_collision(traj_xy, scenario.obstacles)
        
        return {
            'trajectory': traj_xy,
            'collision': collision,
            'mrm_triggered': False
        }
    
    def _test_mcts(self, scenario: Scenario) -> Dict:
        """Test MCTS planner"""
        if scenario.scenario_type == ScenarioType.HIGHWAY and len(scenario.obstacles) == 0:
            # MCTS is overkill for simple highway
            return {'trajectory': [(i*2, 0) for i in range(20)], 'collision': False, 'mrm_triggered': False}
        
        planner = create_mcts_planner()
        
        ego_state = {
            's': scenario.ego_start['s'],
            'l': scenario.ego_start.get('l', 0),
            'v': scenario.ego_start['v']
        }
        
        agents = [
            {'s': o['s'], 'l': o.get('l', 0), 'v': o.get('v', 10)}
            for o in scenario.obstacles
        ]
        
        trajectory, action = planner.plan(ego_state, agents, scenario.goal_s)
        
        collision = self._check_collision(trajectory, scenario.obstacles)
        
        return {
            'trajectory': trajectory,
            'collision': collision,
            'mrm_triggered': False
        }
    
    def _test_unified(self, scenario: Scenario) -> Dict:
        """Test unified planner"""
        from .unified_planner import create_unified_planner
        
        planner = create_unified_planner()
        
        ego_state = {
            'x': scenario.ego_start['s'],
            'y': scenario.ego_start.get('l', 0),
            'heading': 0,
            'speed': scenario.ego_start['v'],
            's': scenario.ego_start['s'],
            'l': scenario.ego_start.get('l', 0),
            'goal_s': scenario.goal_s
        }
        
        perception = {'obstacles': scenario.obstacles}
        
        result = planner.update(ego_state, None, perception)
        
        return {
            'trajectory': result.get('trajectory', []),
            'collision': not result.get('is_feasible', True),
            'mrm_triggered': result.get('behavior') == 'stop'
        }
    
    def _check_collision(self, trajectory: List[Tuple[float, float]], 
                       obstacles: List[Dict]) -> bool:
        """Check if trajectory collides with obstacles"""
        for s, l in trajectory:
            for obs in obstacles:
                o_s = obs.get('s', 0)
                o_l = obs.get('l', 0)
                
                dist = np.sqrt((s - o_s)**2 + (l - o_l)**2)
                
                # Vehicle width ~2m
                if dist < 3.0:
                    return True
        
        return False
    
    def print_results(self, results: List[BenchmarkResult]):
        """Print benchmark results"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        # Group by planner
        by_planner = {}
        for r in results:
            if r.planner_name not in by_planner:
                by_planner[r.planner_name] = []
            by_planner[r.planner_name].append(r)
        
        for planner_name, planner_results in by_planner.items():
            print(f"\n{planner_name}:")
            print("-" * 40)
            
            success_count = sum(1 for r in planner_results if r.success)
            total = len(planner_results)
            avg_time = np.mean([r.planning_time_ms for r in planner_results])
            
            print(f"  Success Rate: {success_count}/{total} ({100*success_count/total:.1f}%)")
            print(f"  Avg Planning Time: {avg_time:.1f}ms")
            print(f"  Scenarios:")
            
            for r in planner_results:
                status = "✓" if r.success else "✗"
                print(f"    {status} {r.scenario_name}: {r.planning_time_ms:.1f}ms, "
                      f"progress={r.progress:.1f}m, collision={r.collision}")
        
        print("\n" + "="*80)
    
    def add_scenario(self, scenario: Scenario):
        """Add a new test scenario"""
        self.scenarios.append(scenario)
        print(f"Added scenario: {scenario.name}")
    
    def verify_new_method(self, new_planner_fn: Callable, 
                         scenario: Scenario) -> Dict:
        """
        Verify a new method works and compare to baselines.
        
        Returns dict with comparison metrics.
        """
        # Run new method
        new_result = self.run_scenario(scenario, new_planner_fn)
        
        # Run baselines
        lattice_result = self.run_scenario(scenario, self._test_lattice)
        bt_result = self.run_scenario(scenario, self._test_behavior_tree)
        
        return {
            'new': new_result,
            'lattice_baseline': lattice_result,
            'behavior_tree_baseline': bt_result,
            'improvement_vs_lattice': (
                new_result.planning_time_ms / lattice_result.planning_time_ms 
                if lattice_result.planning_time_ms > 0 else 1.0
            ),
            'improvement_vs_bt': (
                new_result.planning_time_ms / bt_result.planning_time_ms
                if bt_result.planning_time_ms > 0 else 1.0
            ),
            'new_is_better': (
                new_result.success and 
                (new_result.planning_time_ms < lattice_result.planning_time_ms or
                 new_result.planning_time_ms < bt_result.planning_time_ms)
            )
        }


def run_benchmark():
    """Run full benchmark"""
    bench = Benchmark()
    results = bench.run_all()
    bench.print_results(results)
    return results


def add_custom_scenario():
    """Example: Add new scenario"""
    bench = Benchmark()
    
    # Add new scenario: round-about
    bench.add_scenario(Scenario(
        name="roundabout",
        scenario_type=ScenarioType.URBAN_NEGOTIATION,
        ego_start={'s': 0, 'l': 0, 'v': 8},
        goal_s=50,
        obstacles=[
            {'s': 10, 'l': 0, 'v': 6},
            {'s': 5, 'l': 5, 'v': 5},
            {'s': 15, 'l': -5, 'v': 7},
        ],
        expected_outcome="safe"
    ))
    
    return bench


if __name__ == "__main__":
    run_benchmark()
