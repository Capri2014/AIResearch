#!/usr/bin/env python3
"""
Pipeline Executor - Unified Entry Point for Driving-First Pipeline

Connects all pipeline components:
- Data loading (Waymo episodes)
- Pipeline checkpoint loader (PR #2)
- Evaluation runner (PR #1)
- Pipeline evaluation reporter (PR #3)

Usage:
    # Full pipeline with all stages
    python training/pipeline_executor.py run --episodes data/waymo/episodes/*.json

    # Run specific stage
    python training/pipeline_executor.py run --stage pretrain
    python training/pipeline_executor.py run --stage bc
    python training/pipeline_executor.py run --stage rl
    python training/pipeline_executor.py run --stage eval

    # Resume pipeline from checkpoint
    python training/pipeline_executor.py run --stage bc --checkpoint out/pretrain/final.pt

    # List available checkpoints
    python training/pipeline_executor.py checkpoints

    # Show pipeline status
    python training/pipeline_executor.py status

    # Generate pipeline report
    python training/pipeline_executor.py report --stages bc,rl
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch

# Add workspace root and training dir to path
_WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_WORKSPACE_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from training.pipeline_checkpoint_loader import PipelineCheckpointLoader, CheckpointMetadata
from training.pipeline_orchestrator import PipelineConfig, PipelineOrchestrator
try:
    from sim.driving.carla_srunner.run_waypoint_eval import WaypointEvalRunner, EvalConfig
except ImportError:
    WaypointEvalRunner = None
    EvalConfig = None
try:
    from sim.driving.carla_srunner.pipeline_eval_reporter import PipelineReport, ReportConfig
except ImportError:
    PipelineReport = None
    ReportConfig = None


# Pipeline stages
STAGE_PRETRAIN = "pretrain"
STAGE_BC = "bc"
STAGE_RL = "rl"
STAGE_EVAL = "eval"
STAGE_FULL = "full"


STAGE_NAMES = {
    STAGE_PRETRAIN: "SSL Pretrain",
    STAGE_BC: "Waypoint BC",
    STAGE_RL: "RL Refinement",
    STAGE_EVAL: "Evaluation",
    STAGE_FULL: "Full Pipeline",
}


@dataclass
class ExecutorConfig:
    """Configuration for pipeline executor."""
    stage: str = STAGE_FULL
    episodes_glob: str = "data/waymo/episodes/*.json"
    checkpoint: Optional[str] = None
    output_dir: str = "out/pipeline"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dry_run: bool = False
    verbose: bool = True


class PipelineExecutor:
    """
    Unified executor for driving-first pipeline.
    
    Coordinates between:
    - pipeline_checkpoint_loader (checkpoint discovery/loading)
    - pipeline_orchestrator (training stages)
    - run_waypoint_eval (evaluation)
    - pipeline_eval_reporter (reporting)
    """
    
    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.checkpoint_loader = PipelineCheckpointLoader()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self) -> Dict[str, Any]:
        """Execute pipeline based on config.stage."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Pipeline Executor - {STAGE_NAMES.get(self.config.stage, self.config.stage)}")
            print(f"{'='*60}")
        
        if self.config.stage == STAGE_FULL:
            return self._run_full_pipeline()
        elif self.config.stage == STAGE_PRETRAIN:
            return self._run_pretrain()
        elif self.config.stage == STAGE_BC:
            return self._run_bc()
        elif self.config.stage == STAGE_RL:
            return self._run_rl()
        elif self.config.stage == STAGE_EVAL:
            return self._run_eval()
        else:
            raise ValueError(f"Unknown stage: {self.config.stage}")
    
    def _run_full_pipeline(self) -> Dict[str, Any]:
        """Run full pipeline: pretrain → BC → RL → eval."""
        results = {}
        
        # Stage 1: SSL Pretrain
        if self.config.verbose:
            print("\n[Stage 1/4] SSL Pretrain")
        results['pretrain'] = self._run_pretrain()
        
        # Stage 2: Waypoint BC
        if self.config.verbose:
            print("\n[Stage 2/4] Waypoint BC")
        checkpoint = results['pretrain'].get('checkpoint')
        results['bc'] = self._run_bc(checkpoint=checkpoint)
        
        # Stage 3: RL Refinement
        if self.config.verbose:
            print("\n[Stage 3/4] RL Refinement")
        checkpoint = results['bc'].get('checkpoint')
        results['rl'] = self._run_rl(checkpoint=checkpoint)
        
        # Stage 4: Evaluation
        if self.config.verbose:
            print("\n[Stage 4/4] Evaluation")
        results['eval'] = self._run_eval()
        
        # Generate report
        if self.config.verbose:
            print("\n[Generating Pipeline Report]")
        results['report'] = self._generate_report(results)
        
        return results
    
    def _run_pretrain(self, checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Run SSL pretrain stage."""
        import glob
        
        episodes = sorted(glob.glob(self.config.episodes_glob))
        if not episodes:
            return {'status': 'skipped', 'reason': 'no episodes found'}
        
        if self.config.dry_run:
            print(f"  [DRY RUN] Would run pretrain on {len(episodes)} episodes")
            return {'status': 'dry_run', 'num_episodes': len(episodes)}
        
        # Use orchestrator for pretrain
        config = PipelineConfig(
            stage=STAGE_PRETRAIN,
            episodes_glob=self.config.episodes_glob,
            output_dir=str(self.output_dir / "pretrain"),
            device=self.config.device,
        )
        
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()
        
        checkpoint_path = self.output_dir / "pretrain" / "final.pt"
        return {
            'status': 'success',
            'checkpoint': str(checkpoint_path) if checkpoint_path.exists() else None,
            'num_episodes': len(episodes),
        }
    
    def _run_bc(self, checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Run Waypoint BC stage."""
        import glob
        
        episodes = sorted(glob.glob(self.config.episodes_glob))
        if not episodes:
            return {'status': 'skipped', 'reason': 'no episodes found'}
        
        if self.config.dry_run:
            print(f"  [DRY RUN] Would run BC on {len(episodes)} episodes")
            return {'status': 'dry_run', 'num_episodes': len(episodes)}
        
        # Checkpoint from pretrain
        ckpt = checkpoint or self.config.checkpoint
        if not ckpt:
            # Try to find latest pretrain checkpoint
            try:
                checkpoints = self.checkpoint_loader.list_checkpoints(stage="pretrain")
                if checkpoints:
                    ckpt = checkpoints[0].path
            except Exception:
                pass
        
        config = PipelineConfig(
            stage=STAGE_WAYPOINT_BC,
            episodes_glob=self.config.episodes_glob,
            pretrain_checkpoint=ckpt,
            output_dir=str(self.output_dir / "bc"),
            device=self.config.device,
        )
        
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()
        
        checkpoint_path = self.output_dir / "bc" / "final.pt"
        return {
            'status': 'success',
            'checkpoint': str(checkpoint_path) if checkpoint_path.exists() else None,
            'num_episodes': len(episodes),
        }
    
    def _run_rl(self, checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Run RL refinement stage."""
        import glob
        
        episodes = sorted(glob.glob(self.config.episodes_glob))
        if not episodes:
            return {'status': 'skipped', 'reason': 'no episodes found'}
        
        if self.config.dry_run:
            print(f"  [DRY RUN] Would run RL on {len(episodes)} episodes")
            return {'status': 'dry_run', 'num_episodes': len(episodes)}
        
        # Checkpoint from BC
        ckpt = checkpoint or self.config.checkpoint
        if not ckpt:
            # Try to find latest BC checkpoint
            try:
                checkpoints = self.checkpoint_loader.list_checkpoints(stage="bc")
                if checkpoints:
                    ckpt = checkpoints[0].path
            except Exception:
                pass
        
        config = PipelineConfig(
            stage=STAGE_RL_REFINEMENT,
            episodes_glob=self.config.episodes_glob,
            sft_checkpoint=ckpt,
            output_dir=str(self.output_dir / "rl"),
            device=self.config.device,
        )
        
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()
        
        checkpoint_path = self.output_dir / "rl" / "final.pt"
        return {
            'status': 'success',
            'checkpoint': str(checkpoint_path) if checkpoint_path.exists() else None,
            'num_episodes': len(episodes),
        }
    
    def _run_eval(self) -> Dict[str, Any]:
        """Run evaluation stage."""
        if self.config.dry_run:
            print("  [DRY RUN] Would run evaluation")
            return {'status': 'dry_run'}
        
        # Find latest checkpoint
        try:
            checkpoints = self.checkpoint_loader.list_checkpoints()
            if not checkpoints:
                return {'status': 'skipped', 'reason': 'no checkpoints found'}
            
            latest = checkpoints[0]
        except Exception as e:
            return {'status': 'error', 'reason': str(e)}
        
        # Run evaluation
        eval_config = EvalConfig(
            policy_type=latest.stage,
            checkpoint_path=latest.path,
            suite="smoke",
            num_runs=3,
        )
        
        runner = WaypointEvalRunner(eval_config)
        summary = runner.run()
        
        return {
            'status': 'success',
            'checkpoint': latest.path,
            'metrics': summary.to_dict(),
        }
    
    def _generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pipeline evaluation report."""
        report_config = ReportConfig(
            output_dir=str(self.output_dir / "report"),
            checkpoint_dirs={
                "pretrain": str(self.output_dir / "pretrain"),
                "bc": str(self.output_dir / "bc"),
                "rl": str(self.output_dir / "rl"),
            },
        )
        
        reporter = PipelineEvalReporter(report_config)
        report = reporter.generate_report()
        
        return {'status': 'success', 'report': report}
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        try:
            checkpoints = self.checkpoint_loader.list_checkpoints()
            return [ck.to_dict() for ck in checkpoints]
        except Exception as e:
            return [{'error': str(e)}]
    
    def show_status(self) -> Dict[str, Any]:
        """Show pipeline status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'stage': self.config.stage,
            'output_dir': str(self.output_dir),
        }
        
        # Try to get checkpoint info
        try:
            checkpoints = self.checkpoint_loader.list_checkpoints()
            status['checkpoints'] = [ck.to_dict() for ck in checkpoints[:5]]
        except Exception as e:
            status['checkpoint_error'] = str(e)
        
        return status


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Pipeline Executor - Unified Entry Point for Driving-First Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python training/pipeline_executor.py run --episodes data/waymo/episodes/*.json

  # Run BC stage only
  python training/pipeline_executor.py run --stage bc --checkpoint out/pretrain/final.pt

  # List checkpoints
  python training/pipeline_executor.py checkpoints

  # Show status
  python training/pipeline_executor.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # run command
    run_parser = subparsers.add_parser('run', help='Run pipeline')
    run_parser.add_argument('--stage', default=STAGE_FULL,
                         choices=[STAGE_FULL, STAGE_PRETRAIN, STAGE_BC, STAGE_RL, STAGE_EVAL],
                         help='Pipeline stage to run')
    run_parser.add_argument('--episodes', dest='episodes_glob',
                         default='data/waymo/episodes/*.json',
                         help='Episodes glob pattern')
    run_parser.add_argument('--checkpoint', help='Checkpoint to resume from')
    run_parser.add_argument('--output', default='out/pipeline',
                         help='Output directory')
    run_parser.add_argument('--device', default='cuda',
                         help='Device (cuda/cpu)')
    run_parser.add_argument('--dry-run', action='store_true',
                         help='Dry run (verify config only)')
    
    # checkpoints command
    subparsers.add_parser('checkpoints', help='List available checkpoints')
    
    # status command
    subparsers.add_parser('status', help='Show pipeline status')
    
    # report command
    report_parser = subparsers.add_parser('report', help='Generate pipeline report')
    report_parser.add_argument('--stages', default='bc,rl',
                         help='Stages to include (comma-separated)')
    report_parser.add_argument('--output', default='out/pipeline/report',
                         help='Output directory')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config = ExecutorConfig(
        stage=getattr(args, 'stage', STAGE_FULL),
        episodes_glob=getattr(args, 'episodes_glob', 'data/waymo/episodes/*.json'),
        checkpoint=getattr(args, 'checkpoint', None),
        output_dir=getattr(args, 'output', 'out/pipeline'),
        device=getattr(args, 'device', 'cuda'),
        dry_run=getattr(args, 'dry_run', False),
    )
    
    executor = PipelineExecutor(config)
    
    if args.command == 'run':
        result = executor.run()
        print(f"\n[Result] {result.get('status', 'unknown')}")
    elif args.command == 'checkpoints':
        checkpoints = executor.list_checkpoints()
        print(f"\nAvailable Checkpoints ({len(checkpoints)}):")
        for ck in checkpoints:
            print(f"  - {ck.get('stage', '?')}: {ck.get('path', ck.get('error', '?'))}")
    elif args.command == 'status':
        status = executor.show_status()
        print(f"\nPipeline Status:")
        print(f"  Stage: {status.get('stage')}")
        print(f"  Output: {status.get('output_dir')}")
        if 'checkpoints' in status:
            print(f"  Checkpoints: {len(status.get('checkpoints', []))}")
    elif args.command == 'report':
        print("\n[Pipeline Report] Use subcommand 'run --stage eval' to generate")


if __name__ == '__main__':
    main()