"""CARLA ScenarioRunner Integration.

Driving-first plan: Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval

This module provides the execution layer that runs scenarios defined in scenarios.py
via CARLA ScenarioRunner. It bridges scenario definitions with the CARLA simulator.

Usage
-----
Run a single scenario:
  python -m sim.driving.carla_srunner.runner --scenario straight_clear

Run a scenario suite:
  python -m sim.driving.carla_srunner.runner --suite smoke
  python -m sim.driving.carla_srunner.runner --suite weather

List available options:
  python -m sim.driving.carla_srunner.runner --list-scenarios
  python -m sim.driving.carla_srunner.runner --list-suites
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

from sim.driving.carla_srunner.scenarios import (
    SCENARIO_CATALOG,
    SCENARIO_SUITES,
    ROUTE_CATALOG,
    ScenarioDef,
    RouteDef,
    get_scenario,
    get_route,
    list_scenarios,
    list_routes,
    list_suites,
    get_suite,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default CARLA host/port
DEFAULT_CARLA_HOST = os.environ.get("CARLA_HOST", "localhost")
DEFAULT_CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))


@dataclass
class RunnerConfig:
    """Configuration for scenario runner execution."""

    # CARLA connection
    carla_host: str = DEFAULT_CARLA_HOST
    carla_port: int = DEFAULT_CARLA_PORT
    
    # Scenario specification
    scenario_id: Optional[str] = None
    route_id: Optional[str] = None
    suite_name: Optional[str] = None
    
    # Model/policy to evaluate
    checkpoint: Optional[Path] = None
    
    # Execution settings
    headless: bool = False
    timeout: int = 300
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("out/srunner_eval"))
    
    # Waypoint inference settings
    model_type: str = "waypoint_bc"  # waypoint_bc, delta_waypoint, rl_refined
    inference_host: str = "localhost"
    inference_port: int = 8000
    
    # Recording
    record_sensor_data: bool = True
    traffic_manager_port: int = 8000


def load_srunner_config() -> Dict[str, Any]:
    """Load ScenarioRunner configuration from environment or defaults."""
    return {
        "carla_host": DEFAULT_CARLA_HOST,
        "carla_port": DEFAULT_CARLA_PORT,
        "carla_version": os.environ.get("CARLA_VERSION", "0.9.15"),
        "srunner_version": os.environ.get("SRUNNER_VERSION", "v2.3"),
        "record_sensors": os.environ.get("RECORD_SENSORS", "true").lower() == "true",
    }


def build_srunner_command(
    scenario_def: ScenarioDef,
    config: RunnerConfig,
) -> List[str]:
    """Build the ScenarioRunner command for a scenario.
    
    Returns the full command as a list of arguments.
    """
    # Base ScenarioRunner command
    cmd = [
        "python",
        "-m scenario_runner",
        "--scenario", scenario_def.scenario_id,
        "--town", scenario_def.town,
        "--host", config.carla_host,
        "--port", str(config.carla_port),
    ]
    
    # Add weather
    weather = scenario_def.weather
    if weather != "clear":
        cmd.extend(["--weather", weather])
    
    # Add vehicle model
    if scenario_def.vehicle_model:
        cmd.extend(["--vehicle", scenario_def.vehicle_model])
    
    # Add timeout
    cmd.extend(["--timeout", str(scenario_def.timeout)])
    
    # Add headless mode
    if config.headless:
        cmd.append("--headless")
    
    # Add output directory
    cmd.extend(["--output", str(config.output_dir)])
    
    return cmd


def build_srunner_command_for_route(
    route_def: RouteDef,
    config: RunnerConfig,
) -> List[str]:
    """Build the ScenarioRunner command for a route-based evaluation."""
    cmd = [
        "python",
        "-m scenario_runner",
        "--route", route_def.route_id,
        "--town", route_def.town,
        "--host", config.carla_host,
        "--port", str(config.carla_port),
    ]
    
    # Add weather
    if route_def.weather:
        cmd.extend(["--weather", route_def.weather])
    
    # Add timeout
    cmd.extend(["--timeout", "120"])  # Default
    
    # Add headless
    if config.headless:
        cmd.append("--headless")
    
    # Add output
    cmd.extend(["--output", str(config.output_dir)])
    
    return cmd


class ScenarioRunner:
    """Main class for running CARLA scenarios via ScenarioRunner."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_carla_connection(self) -> bool:
        """Check if CARLA is running and accessible."""
        import socket
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            result = sock.connect_ex((
                self.config.carla_host,
                self.config.carla_port,
            ))
            sock.close()
            return result == 0
        except Exception:
            return False

    def run_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """Run a single scenario and return results."""
        scenario_def = get_scenario(scenario_id)
        if scenario_def is None:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        logger.info(f"Running scenario: {scenario_id}")
        
        # Build command
        cmd = build_srunner_command(scenario_def, self.config)
        
        # Execute
        result = self._execute_command(cmd)
        
        # Parse results
        return self._parse_results(scenario_id, result)

    def run_route(self, route_id: str) -> Dict[str, Any]:
        """Run a route-based evaluation."""
        route_def = get_route(route_id)
        if route_def is None:
            raise ValueError(f"Unknown route: {route_id}")
        
        logger.info(f"Running route: {route_id}")
        
        cmd = build_srunner_command_for_route(route_def, self.config)
        
        result = self._execute_command(cmd)
        
        return self._parse_route_results(route_id, result)

    def run_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a scenario suite (batch evaluation)."""
        suite = get_suite(suite_name)
        if suite is None:
            raise ValueError(f"Unknown suite: {suite_name}")
        
        logger.info(f"Running suite '{suite_name}': {suite['description']}")
        
        results = {
            "suite": suite_name,
            "description": suite["description"],
            "scenarios": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        for scenario_id in suite["scenarios"]:
            try:
                result = self.run_scenario(scenario_id)
                results["scenarios"].append(result)
            except Exception as e:
                logger.error(f"Scenario {scenario_id} failed: {e}")
                results["scenarios"].append({
                    "scenario_id": scenario_id,
                    "status": "failed",
                    "error": str(e),
                })
        
        # Compute aggregate metrics
        results["aggregate"] = self._compute_aggregate(results["scenarios"])
        
        return results

    def _execute_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Execute a ScenarioRunner command."""
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=self.config.output_dir,
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {self.config.timeout}s")
            raise
        except Exception as e:
            logger.error(f"Command failed: {e}")
            raise

    def _parse_results(
        self,
        scenario_id: str,
        result: subprocess.CompletedProcess,
    ) -> Dict[str, Any]:
        """Parse results from a scenario execution."""
        output_file = self.output_dir / f"{scenario_id}_results.json"
        
        # Try to load results from file
        if output_file.exists():
            with open(output_file) as f:
                return json.load(f)
        
        # Fallback: parse from return code
        status = "success" if result.returncode == 0 else "failed"
        
        return {
            "scenario_id": scenario_id,
            "status": status,
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }

    def _parse_route_results(
        self,
        route_id: str,
        result: subprocess.CompletedProcess,
    ) -> Dict[str, Any]:
        """Parse results from a route execution."""
        output_file = self.output_dir / f"{route_id}_results.json"
        
        if output_file.exists():
            with open(output_file) as f:
                return json.load(f)
        
        status = "success" if result.returncode == 0 else "failed"
        
        return {
            "route_id": route_id,
            "status": status,
            "returncode": result.returncode,
        }

    def _compute_aggregate(
        self,
        scenario_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute aggregate metrics from scenario results."""
        successful = [
            r for r in scenario_results
            if r.get("status") == "success"
        ]
        
        if not successful:
            return {
                "num_scenarios": len(scenario_results),
                "num_success": 0,
                "success_rate": 0.0,
            }
        
        # Compute means
        route_completions = []
        collisions = []
        
        for r in successful:
            if "route_completion" in r:
                route_completions.append(r["route_completion"])
            if "collisions" in r:
                collisions.append(r["collisions"])
        
        aggregate = {
            "num_scenarios": len(scenario_results),
            "num_success": len(successful),
            "success_rate": len(successful) / len(scenario_results),
        }
        
        if route_completions:
            aggregate["mean_route_completion"] = sum(route_completions) / len(route_completions)
        
        if collisions:
            aggregate["mean_collisions"] = sum(collisions) / len(collisions)
        
        return aggregate


def main() -> None:
    p = argparse.ArgumentParser(
        description="CARLA ScenarioRunner Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # CARLA connection
    p.add_argument("--host", default=DEFAULT_CARLA_HOST, help="CARLA host")
    p.add_argument("--port", type=int, default=DEFAULT_CARLA_PORT, help="CARLA port")
    
    # Scenario/route/suite
    p.add_argument("--scenario", type=str, help="Run single scenario")
    p.add_argument("--route", type=str, help="Run single route")
    p.add_argument("--suite", type=str, help="Run scenario suite")
    
    # Model
    p.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    
    # Execution
    p.add_argument("--headless", action="store_true", help="Run headless")
    p.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    
    # Output
    p.add_argument("--output-dir", type=str, default="out/srunner_eval", help="Output directory")
    
    # List options
    p.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    p.add_argument("--list-routes", action="store_true", help="List available routes")
    p.add_argument("--list-suites", action="store_true", help="List available suites")
    
    a = p.parse_args()

    # Build config
    config = RunnerConfig(
        carla_host=a.host,
        carla_port=a.port,
        scenario_id=a.scenario,
        route_id=a.route,
        suite_name=a.suite,
        checkpoint=Path(a.checkpoint) if a.checkpoint else None,
        headless=a.headless,
        timeout=a.timeout,
        output_dir=Path(a.output_dir),
    )
    
    # List operations
    if a.list_scenarios:
        print("Available scenarios:")
        for s in list_scenarios():
            sd = SCENARIO_CATALOG[s]
            print(f"  {s}: town={sd.town}, weather={sd.weather}")
        return

    if a.list_routes:
        print("Available routes:")
        for r in list_routes():
            rd = ROUTE_CATALOG[r]
            print(f"  {r}: town={rd.town}, weather={rd.weather}, length={rd.length_m}m")
        return

    if a.list_suites:
        print("Available suites:")
        for name, suite in SCENARIO_SUITES.items():
            print(f"  {name}: {suite['description']} ({len(suite['scenarios'])} scenarios)")
        return

    # Create runner
    runner = ScenarioRunner(config)
    
    # Check CARLA connection
    if not runner.check_carla_connection():
        logger.warning(f"Carla not available at {config.carla_host}:{config.carla_port}")
        logger.info("Use --host/--port to specify CARLA server, or set CARLA_HOST/CARLA_PORT env vars")
    
    # Execute
    results = None
    
    if a.scenario:
        results = runner.run_scenario(a.scenario)
        print(json.dumps(results, indent=2))
        
    elif a.route:
        results = runner.run_route(a.route)
        print(json.dumps(results, indent=2))
        
    elif a.suite:
        results = runner.run_suite(a.suite)
        print(json.dumps(results, indent=2))
        
        # Save aggregate
        if "aggregate" in results:
            output_file = config.output_dir / f"suite_{a.suite}_aggregate.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved aggregate results to {output_file}")
    
    else:
        p.print_help()
        return


if __name__ == "__main__":
    main()