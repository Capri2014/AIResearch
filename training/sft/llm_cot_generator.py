"""
LLM-Based CoT Generator

Uses LLM models to generate high-quality reasoning traces for driving.
Supports:
- Local models (Llama, Mistral)
- API models (OpenAI, Anthropic)
- Learned/finetuned models

Usage:
    from training.sft.llm_cot_generator import LLMCoTGenerator
    
    generator = LLMCoTGenerator(
        model_type="local",
        model_name="meta-llama/Llama-2-7b-chat",
    )
    
    cot = generator.generate(state, images, objects)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import time
from pathlib import Path


# ============================================================================
# Base CoT Generator
# ============================================================================

class CoTGeneratorBase(ABC):
    """Base class for CoT generators."""
    
    @abstractmethod
    def generate(
        self,
        state: Dict,
        objects: List[Dict],
        context: Optional[Dict] = None
    ) -> "CoTTrace":
        """Generate CoT trace for a driving scenario."""
        pass
    
    @abstractmethod
    def batch_generate(
        self,
        scenarios: List[Tuple[Dict, List[Dict], Optional[Dict]]]
    ) -> List["CoTTrace"]:
        """Generate CoT traces for multiple scenarios."""
        pass


@dataclass
class CoTTrace:
    """Chain of Thought reasoning trace."""
    perception: str
    prediction: str
    planning: str
    justification: str
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "perception": self.perception,
            "prediction": self.prediction,
            "planning": self.planning,
            "justification": self.justification,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# ============================================================================
# Rule-Based Generator (Fast, Baseline)
# ============================================================================

class RuleBasedCoTGenerator(CoTGeneratorBase):
    """
    Fast rule-based CoT generator.
    
    Uses heuristics and templates to generate reasoning traces.
    Good baseline, low quality but fast.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def generate(
        self,
        state: Dict,
        objects: List[Dict],
        context: Optional[Dict] = None
    ) -> CoTTrace:
        """Generate CoT trace using rules."""
        
        # Perception
        perception = self._generate_perception(state, objects)
        
        # Prediction
        prediction = self._generate_prediction(state, objects)
        
        # Planning
        planning = self._generate_planning(state, objects, perception, prediction)
        
        # Justification
        justification = self._generate_justification(perception, prediction, planning)
        
        # Confidence
        confidence = self._estimate_confidence(state, objects)
        
        return CoTTrace(
            perception=perception,
            prediction=prediction,
            planning=planning,
            justification=justification,
            confidence=confidence,
            metadata={"source": "rule_based"},
        )
    
    def batch_generate(
        self,
        scenarios: List[Tuple[Dict, List[Dict], Optional[Dict]]]
    ) -> List[CoTTrace]:
        """Generate CoT traces for multiple scenarios."""
        return [self.generate(state, objects, ctx) for state, objects, ctx in scenarios]
    
    def _generate_perception(self, state: Dict, objects: List[Dict]) -> str:
        """Generate perception description."""
        parts = []
        
        # Speed and heading
        speed = state.get("speed", 0)
        heading = state.get("heading", 0)
        parts.append(f"Ego vehicle traveling at {speed:.1f} m/s, heading {heading:.1f}°")
        
        # Objects
        n_vehicles = sum(1 for o in objects if o.get("type") == "vehicle")
        n_pedestrians = sum(1 for o in objects if o.get("type") == "pedestrian")
        n_cyclists = sum(1 for o in objects if o.get("type") == "cyclist")
        
        if n_vehicles > 0:
            parts.append(f"{n_vehicles} vehicles detected")
        if n_pedestrians > 0:
            parts.append(f"{n_pedestrians} pedestrians detected")
        if n_cyclists > 0:
            parts.append(f"{n_cyclists} cyclists detected")
        
        if not any([n_vehicles, n_pedestrians, n_cyclists]):
            parts.append("No critical objects detected")
        
        # Road context
        if "road_type" in state:
            parts.append(f"Road type: {state['road_type']}")
        
        return ". ".join(parts) + "."
    
    def _generate_prediction(self, state: Dict, objects: List[Dict]) -> str:
        """Generate prediction description."""
        predictions = []
        
        for obj in objects[:5]:  # Top 5 objects
            obj_type = obj.get("type", "unknown")
            distance = obj.get("distance", float("inf"))
            
            if distance < 50:  # Only relevant objects
                velocity = obj.get("velocity", 0)
                
                if obj_type == "vehicle":
                    if velocity < -1:
                        predictions.append(f"Vehicle ahead slowing ({velocity:.1f} m/s)")
                    elif velocity > 1:
                        predictions.append(f"Vehicle ahead accelerating (+{velocity:.1f} m/s)")
                    else:
                        predictions.append(f"Vehicle ahead stable")
                
                elif obj_type == "pedestrian":
                    intent = obj.get("intent", "standing")
                    predictions.append(f"Pedestrian: {intent}")
        
        if not predictions:
            predictions.append("No immediate threats detected")
        
        return " | ".join(predictions)
    
    def _generate_planning(
        self,
        state: Dict,
        objects: List[Dict],
        perception: str,
        prediction: str
    ) -> str:
        """Generate planning description."""
        actions = []
        
        speed = state.get("speed", 0)
        
        # Speed adjustment
        lead_vehicle = self._find_lead_vehicle(objects, state)
        if lead_vehicle and lead_vehicle.get("distance", float("inf")) < 20:
            actions.append("Reduce speed to maintain safe distance")
        elif speed < 2:
            actions.append("Accelerate to cruising speed")
        else:
            actions.append("Maintain current speed")
        
        # Lane keeping
        if "lane_offset" in state and abs(state["lane_offset"]) > 0.5:
            direction = "left" if state["lane_offset"] < 0 else "right"
            actions.append(f"Steer {direction} to center lane")
        else:
            actions.append("Maintain lane position")
        
        # Special conditions
        if state.get("traffic_light_red", False):
            actions.append("Prepare to stop at traffic light")
        
        return ". ".join(actions) + "."
    
    def _generate_justification(
        self,
        perception: str,
        prediction: str,
        planning: str
    ) -> str:
        """Generate justification."""
        return (
            f"Based on perception: {perception[:50]}... "
            f"Predicted changes: {prediction[:50]}... "
            f"Action: {planning[:50]}..."
        )
    
    def _estimate_confidence(self, state: Dict, objects: List[Dict]) -> float:
        """Estimate confidence based on scene complexity."""
        base = 1.0
        
        # Reduce confidence for complex scenes
        if len(objects) > 10:
            base -= 0.2
        elif len(objects) > 5:
            base -= 0.1
        
        # Reduce for poor conditions
        if state.get("weather", "clear") != "clear":
            base -= 0.1
        
        if state.get("time_of_day", "day") == "night":
            base -= 0.1
        
        return max(0.3, min(1.0, base))
    
    def _find_lead_vehicle(
        self,
        objects: List[Dict],
        state: Dict
    ) -> Optional[Dict]:
        """Find the lead vehicle in front of ego."""
        lead = None
        min_dist = float("inf")
        
        for obj in objects:
            if obj.get("type") != "vehicle":
                continue
            
            # Check if in front
            rel_x = obj.get("x", 0) - state.get("x", 0)
            rel_y = obj.get("y", 0) - state.get("y", 0)
            
            heading = state.get("heading", 0)
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            
            forward = rel_x * cos_h + rel_y * sin_h
            lateral = -rel_x * sin_h + rel_y * cos_h
            
            if forward > 0 and abs(lateral) < 2.0:  # Same lane
                dist = np.sqrt(forward**2 + lateral**2)
                if dist < min_dist:
                    min_dist = dist
                    lead = obj
        
        return lead


# ============================================================================
# LLM-Based Generator
# ============================================================================

class LLMBasedCoTGenerator(CoTGeneratorBase):
    """
    LLM-based CoT generator.
    
    Uses local or API models for high-quality reasoning traces.
    Higher quality but slower and requires compute.
    """
    
    def __init__(
        self,
        config: Dict,
        model_type: str = "local",
        model_name: str = "meta-llama/Llama-2-7b-chat"
    ):
        self.config = config
        self.model_type = model_type
        self.model_name = model_name
        
        # Initialize model
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
    
    def _load_model(self):
        """Load LLM model."""
        if self.model_type == "local":
            try:
                from transformers import AutoModelForCausalLM
                import torch
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                return model
            except ImportError:
                print("Warning: transformers not installed, falling back to rule-based")
                return None
        elif self.model_type == "api":
            # API-based model (OpenAI, Anthropic)
            return {"type": self.model_name}
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_tokenizer(self):
        """Load tokenizer."""
        if self.model_type == "local":
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(self.model_name)
            except ImportError:
                return None
        return None
    
    def generate(
        self,
        state: Dict,
        objects: List[Dict],
        context: Optional[Dict] = None
    ) -> CoTTrace:
        """Generate CoT trace using LLM."""
        
        # Build prompt
        prompt = self._build_prompt(state, objects, context)
        
        # Get LLM response
        response = self._query_model(prompt)
        
        # Parse response
        trace = self._parse_response(response)
        
        return trace
    
    def batch_generate(
        self,
        scenarios: List[Tuple[Dict, List[Dict], Optional[Dict]]]
    ) -> List[CoTTrace]:
        """Generate CoT traces for multiple scenarios."""
        traces = []
        
        for state, objects, ctx in scenarios:
            try:
                trace = self.generate(state, objects, ctx)
                traces.append(trace)
            except Exception as e:
                print(f"Error generating CoT: {e}")
                # Fallback to rule-based
                fallback = RuleBasedCoTGenerator()
                traces.append(fallback.generate(state, objects, ctx))
        
        return traces
    
    def _build_prompt(
        self,
        state: Dict,
        objects: List[Dict],
        context: Optional[Dict]
    ) -> str:
        """Build prompt for LLM."""
        
        prompt = f"""You are an expert autonomous driving assistant. Analyze the following driving scenario and provide a Chain of Thought reasoning trace.

## Current State
- Speed: {state.get('speed', 0):.1f} m/s
- Heading: {state.get('heading', 0):.1f}°
- Position: ({state.get('x', 0):.1f}, {state.get('y', 0):.1f})
- Road type: {state.get('road_type', 'unknown')}
- Traffic light: {state.get('traffic_light', 'unknown')}

## Detected Objects
"""
        
        for obj in objects[:10]:  # Top 10 objects
            obj_type = obj.get('type', 'unknown')
            distance = obj.get('distance', 'unknown')
            velocity = obj.get('velocity', 0)
            
            prompt += f"- {obj_type} at {distance}m, velocity: {velocity:.1f} m/s\n"
        
        prompt += """
## Task
Provide your reasoning in the following format:
1. PERCEPTION: What do you see? (brief)
2. PREDICTION: What will happen? (brief)
3. PLANNING: What action will you take? (brief)
4. JUSTIFICATION: Why this action? (brief)

Respond in valid JSON format:
```json
{
    "perception": "...",
    "prediction": "...",
    "planning": "...",
    "justification": "..."
}
```
"""
        
        return prompt
    
    def _query_model(self, prompt: str) -> str:
        """Query LLM model."""
        
        if self.model is None:
            # Fallback for demo
            return self._mock_response(prompt)
        
        if self.model_type == "api":
            return self._query_api(prompt)
        
        # Local model
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (after the prompt)
        if "```json" in response:
            response = response.split("```json")[-1]
            if "```" in response:
                response = response.split("```")[0]
        
        return response
    
    def _query_api(self, prompt: str) -> str:
        """Query API-based model."""
        # Placeholder for API integration
        # Would implement OpenAI/Anthropic API calls here
        return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Mock response for demo/fallback."""
        return json.dumps({
            "perception": "Ego vehicle traveling at 15.0 m/s on urban road. 3 vehicles and 1 pedestrian detected.",
            "prediction": "Lead vehicle maintaining speed. Pedestrian waiting at crosswalk.",
            "planning": "Maintain lane, slight speed reduction for safety margin.",
            "justification": "Traffic is stable. Pedestrian is not crossing. Maintain efficient trajectory."
        })
    
    def _parse_response(self, response: str) -> CoTTrace:
        """Parse LLM response into CoTTrace."""
        
        try:
            # Try to parse JSON
            data = json.loads(response)
            
            return CoTTrace(
                perception=data.get("perception", ""),
                prediction=data.get("prediction", ""),
                planning=data.get("planning", ""),
                justification=data.get("justification", ""),
                confidence=self._estimate_confidence(response),
                metadata={"source": "llm", "model": self.model_name},
            )
        except json.JSONDecodeError:
            # Fallback: try to extract from text
            return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> CoTTrace:
        """Parse text response (fallback)."""
        
        perception = ""
        prediction = ""
        planning = ""
        justification = ""
        
        lines = response.split("\n")
        current_section = ""
        
        for line in lines:
            line = line.strip().lower()
            if "perception" in line:
                current_section = "perception"
            elif "prediction" in line:
                current_section = "prediction"
            elif "planning" in line:
                current_section = "planning"
            elif "justification" in line:
                current_section = "justification"
            elif current_section and line and not line.startswith("-"):
                text = line.split(":", 1)[-1].strip() if ":" in line else line
                if current_section == "perception":
                    perception += text + " "
                elif current_section == "prediction":
                    prediction += text + " "
                elif current_section == "planning":
                    planning += text + " "
                elif current_section == "justification":
                    justification += text + " "
        
        return CoTTrace(
            perception=perception.strip() or "Unable to parse perception",
            prediction=prediction.strip() or "Unable to parse prediction",
            planning=planning.strip() or "Unable to parse planning",
            justification=justification.strip() or "Unable to parse justification",
            confidence=0.7,
            metadata={"source": "llm_fallback", "model": self.model_name},
        )
    
    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence based on response quality."""
        # Simple heuristics
        confidence = 0.8  # Base confidence for LLM
        
        # Check response length
        if len(response) < 100:
            confidence -= 0.1
        
        # Check for completeness
        if "perception" in response and "prediction" in response:
            confidence += 0.1
        
        return min(1.0, max(0.5, confidence))


# ============================================================================
# Learned CoT Generator (Fine-tuned)
# ============================================================================

class LearnedCoTGenerator(CoTGeneratorBase):
    """
    Learned/Finetuned CoT generator.
    
    Uses a fine-tuned model for CoT generation.
    Best quality but requires training data and compute.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def generate(
        self,
        state: Dict,
        objects: List[Dict],
        context: Optional[Dict] = None
    ) -> CoTTrace:
        """Generate CoT trace using learned model."""
        # Implementation would use fine-tuned model
        # For now, fallback to rule-based
        fallback = RuleBasedCoTGenerator()
        return fallback.generate(state, objects, context)
    
    def batch_generate(
        self,
        scenarios: List[Tuple[Dict, List[Dict], Optional[Dict]]]
    ) -> List[CoTTrace]:
        """Generate CoT traces for multiple scenarios."""
        # Implementation for batch inference
        return [self.generate(s, o, c) for s, o, c in scenarios]
    
    def fine_tune(self, dataset: List[CoTTrace]):
        """
        Fine-tune the model on CoT traces.
        
        Would implement:
        - Prepare training data from CoT traces
        - Fine-tune language model
        - Evaluate on held-out set
        """
        print("Fine-tuning not yet implemented")
        pass


# ============================================================================
# Hybrid Generator (Best of Both Worlds)
# ============================================================================

class HybridCoTGenerator(CoTGeneratorBase):
    """
    Hybrid CoT generator combining rule-based and LLM.
    
    Strategy:
    - Use rule-based for simple scenarios
    - Use LLM for complex scenarios
    - Cascade for quality and speed
    """
    
    def __init__(
        self,
        llm_generator: LLMBasedCoTGenerator,
        rule_generator: Optional[RuleBasedCoTGenerator] = None
    ):
        self.llm = llm_generator
        self.rule = rule_generator or RuleBasedCoTGenerator()
        self.complexity_detector = ComplexityDetector()
    
    def generate(
        self,
        state: Dict,
        objects: List[Dict],
        context: Optional[Dict] = None
    ) -> CoTTrace:
        """Generate CoT trace using hybrid approach."""
        
        # Check complexity
        complexity = self.complexity_detector.estimate(state, objects)
        
        if complexity < 0.3:
            # Simple scenario - use rule-based (fast)
            trace = self.rule.generate(state, objects, context)
            trace.metadata["generator"] = "rule_based"
            trace.metadata["complexity"] = complexity
            return trace
        elif complexity < 0.7:
            # Medium complexity - try rule-based first, verify with LLM
            rule_trace = self.rule.generate(state, objects, context)
            
            # Could add LLM verification here
            rule_trace.metadata["generator"] = "rule_based_verified"
            rule_trace.metadata["complexity"] = complexity
            return rule_trace
        else:
            # Complex scenario - use LLM (high quality)
            trace = self.llm.generate(state, objects, context)
            trace.metadata["generator"] = "llm"
            trace.metadata["complexity"] = complexity
            return trace
    
    def batch_generate(
        self,
        scenarios: List[Tuple[Dict, List[Dict], Optional[Dict]]]
    ) -> List[CoTTrace]:
        """Generate CoT traces using hybrid approach."""
        traces = []
        
        for state, objects, ctx in scenarios:
            try:
                trace = self.generate(state, objects, ctx)
                traces.append(trace)
            except Exception as e:
                print(f"Error: {e}")
                fallback = self.rule.generate(state, objects, ctx)
                fallback.metadata["error"] = str(e)
                traces.append(fallback)
        
        return traces


class ComplexityDetector:
    """Detect scene complexity to route to appropriate generator."""
    
    def estimate(self, state: Dict, objects: List[Dict]) -> float:
        """
        Estimate scene complexity.
        
        Returns:
            float between 0 (simple) and 1 (complex)
        """
        complexity = 0.0
        
        # Object count
        if len(objects) > 15:
            complexity += 0.3
        elif len(objects) > 8:
            complexity += 0.2
        elif len(objects) > 3:
            complexity += 0.1
        
        # Pedestrians (high complexity)
        n_ped = sum(1 for o in objects if o.get("type") == "pedestrian")
        if n_ped > 2:
            complexity += 0.2
        elif n_ped > 0:
            complexity += 0.1
        
        # Speed (higher speed = higher complexity)
        speed = state.get("speed", 0)
        if speed > 25:
            complexity += 0.2
        elif speed > 15:
            complexity += 0.1
        
        # Weather/conditions
        if state.get("weather", "clear") != "clear":
            complexity += 0.1
        
        if state.get("time_of_day") == "night":
            complexity += 0.1
        
        # Intersection
        if state.get("is_intersection", False):
            complexity += 0.2
        
        return min(1.0, complexity)


# ============================================================================
# Factory and Usage
# ============================================================================

class CoTGeneratorFactory:
    """Factory for creating CoT generators."""
    
    @staticmethod
    def create(config: Dict) -> CoTGeneratorBase:
        """
        Create CoT generator based on configuration.
        
        Config:
        {
            "type": "rule_based" | "llm" | "learned" | "hybrid",
            "model_type": "local" | "api",
            "model_name": "...",
            "llm_api_key": "...",
        }
        """
        gen_type = config.get("type", "rule_based")
        
        if gen_type == "rule_based":
            return RuleBasedCoTGenerator(config)
        
        elif gen_type == "llm":
            return LLMBasedCoTGenerator(
                config,
                model_type=config.get("model_type", "local"),
                model_name=config.get("model_name", "meta-llama/Llama-2-7b-chat"),
            )
        
        elif gen_type == "learned":
            return LearnedCoTGenerator(config)
        
        elif gen_type == "hybrid":
            llm = LLMBasedCoTGenerator(
                config,
                model_type=config.get("model_type", "local"),
                model_name=config.get("model_name", "meta-llama/Llama-2-7b-chat"),
            )
            return HybridCoTGenerator(llm)
        
        else:
            raise ValueError(f"Unknown generator type: {gen_type}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example configuration
    config = {
        "type": "hybrid",  # Use hybrid generator
        "model_type": "local",
        "model_name": "meta-llama/Llama-2-7b-chat",
    }
    
    # Create generator
    generator = CoTGeneratorFactory.create(config)
    
    # Example driving scenario
    state = {
        "speed": 15.0,
        "heading": 90.0,
        "x": 100.0,
        "y": 200.0,
        "road_type": "urban",
        "traffic_light": "green",
    }
    
    objects = [
        {"type": "vehicle", "x": 120.0, "y": 200.0, "distance": 20.0, "velocity": -0.5},
        {"type": "vehicle", "x": 80.0, "y": 195.0, "distance": 25.0, "velocity": 0.0},
        {"type": "pedestrian", "x": 110.0, "y": 180.0, "distance": 30.0, "intent": "waiting"},
    ]
    
    # Generate CoT
    trace = generator.generate(state, objects)
    
    print("Generated CoT Trace:")
    print(f"  Perception: {trace.perception}")
    print(f"  Prediction: {trace.prediction}")
    print(f"  Planning: {trace.planning}")
    print(f"  Justification: {trace.justification}")
    print(f"  Confidence: {trace.confidence:.2f}")
    print(f"  Metadata: {trace.metadata}")
