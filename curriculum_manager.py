# import time
# import numpy as np
# import yaml
# import os
# from datetime import datetime
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Optional
# from enum import Enum
# from collections import deque
#
# # Robust imports for RL components
# try:
#     from agents.training.ppo_trainer import MultiAgentTrainer
#     from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
# except ImportError:
#     print("Warning: RL libraries (MultiAgentTrainer/StableBaselines3) not found. Curriculum Manager running in standalone mode.")
#     # Dummy classes for standalone compilation
#     MultiAgentTrainer = object
#     BaseCallback = object
#     CheckpointCallback = object
#     EvalCallback = object
#     CallbackList = object
#
# # Try importing environment for type hinting/reference if needed
# try:
#     from environments.carla_gym.multi_agent_env import WaymoStyleCarlaEnv
# except ImportError:
#     pass
#
#
# class DifficultyLevel(Enum):
#     BASIC = 1
#     MEDIUM = 2
#     ADVANCED = 3
#     EXPERT = 4
#
#
# @dataclass
# class CurriculumStage:
#     """Configuration for a curriculum learning stage"""
#     name: str
#     level: DifficultyLevel
#     num_vehicles: int
#     pedestrian_count: int  # Renamed from 'pedestrians' to match env
#     weather: str
#     traffic_density: float
#     success_threshold: float
#     min_episodes: int
#     max_episodes: int
#
#
# class CurriculumManager:
#     """Advanced curriculum learning manager for progressive training"""
#
#     def __init__(self, config_path: str = None):
#         self.current_stage_idx = 0
#         self.stages = []
#
#         # Performance tracking with rolling windows for stability
#         self.window_size = 20
#         self.performance_history = deque(maxlen=self.window_size)
#
#         # Statistics storage
#         self.stage_statistics = {}
#
#         # Episode tracking
#         self.episodes_in_current_stage = 0
#         self.total_episodes = 0
#
#         if config_path and os.path.exists(config_path):
#             self.load_curriculum(config_path)
#         else:
#             self._create_default_curriculum()
#
#     def _create_default_curriculum(self):
#         """
#         Create default curriculum based on the 'Brutal Review' feedback.
#         Starts easy (Safety First) to build confidence, then ramps up difficulty.
#         """
#         self.stages = [
#             # Stage 1: Basic Traffic (No pedestrians, few cars, easy passing requirement)
#             CurriculumStage(
#                 name="Basic Handling",
#                 level=DifficultyLevel.BASIC,
#                 num_vehicles=3,
#                 pedestrian_count=0,  # FIX: 0 Pedestrians prevents early segfaults/crashes
#                 weather="ClearNoon",
#                 traffic_density=0.2,
#                 success_threshold=0.5,  # FIX: Lowered to 50% to prevent stalling
#                 min_episodes=20,
#                 max_episodes=200
#             ),
#             # Stage 2: Moderate Traffic (More cars, stricter success requirement)
#             CurriculumStage(
#                 name="Urban Navigation",
#                 level=DifficultyLevel.MEDIUM,
#                 num_vehicles=10,
#                 pedestrian_count=0,  # Still safe (no jaywalkers)
#                 weather="CloudyNoon",
#                 traffic_density=0.5,
#                 success_threshold=0.7,  # 70% success rate
#                 min_episodes=30,
#                 max_episodes=300
#             ),
#             # Stage 3: Complex Scenarios (Pedestrians introduced, high traffic)
#             CurriculumStage(
#                 name="Complex Urban",
#                 level=DifficultyLevel.ADVANCED,
#                 num_vehicles=15,
#                 pedestrian_count=10,  # Jaywalkers active!
#                 weather="WetNoon",
#                 traffic_density=0.7,
#                 success_threshold=0.8,  # 80% success rate
#                 min_episodes=50,
#                 max_episodes=400
#             ),
#             # Stage 4: Expert (Chaos mode)
#             CurriculumStage(
#                 name="Fleet Management",
#                 level=DifficultyLevel.EXPERT,
#                 num_vehicles=25,
#                 pedestrian_count=20,
#                 weather="HardRainNoon",
#                 traffic_density=1.0,
#                 success_threshold=0.9,  # 90% success rate
#                 min_episodes=100,
#                 max_episodes=500
#             )
#         ]
#
#     def get_current_config(self) -> Dict:
#         """Get configuration for current curriculum stage"""
#         if self.current_stage_idx >= len(self.stages):
#             # Return final stage config if beyond curriculum
#             return self._stage_to_config(self.stages[-1])
#
#         return self._stage_to_config(self.stages[self.current_stage_idx])
#
#     def _stage_to_config(self, stage: CurriculumStage) -> Dict:
#         """Convert curriculum stage to environment configuration"""
#         return {
#             'num_vehicles': stage.num_vehicles,
#             'pedestrian_count': stage.pedestrian_count, # Key matches enhanced_env init
#             'weather': stage.weather,
#             'traffic_density': stage.traffic_density,
#             'town': 'Town03',  # Fixed typo in original code (Town03)
#             'stage_name': stage.name,
#             'difficulty_level': stage.level.value
#         }
#
#     def record_episode_performance(self, metrics: Dict):
#         """Record metrics from a completed episode"""
#         # Normalize data for history
#         episode_data = {
#             'success': 1.0 if metrics.get('success', False) else 0.0,
#             'reward': metrics.get('episode_reward', 0.0),
#             'collision_rate': metrics.get('collision_rate', 0.0),
#             'stage': self.current_stage_idx,
#             'episode_global': self.total_episodes
#         }
#
#         self.performance_history.append(episode_data)
#         self.episodes_in_current_stage += 1
#         self.total_episodes += 1
#
#     def should_advance_stage(self, performance_metrics: Dict = None) -> bool:
#         """Determine if ready to advance to next curriculum stage"""
#         # 1. Check if we are at the last stage
#         if self.current_stage_idx >= len(self.stages) - 1:
#             return False
#
#         current_stage = self.stages[self.current_stage_idx]
#
#         # 2. Check minimum episodes constraint
#         if self.episodes_in_current_stage < current_stage.min_episodes:
#             return False
#
#         # 3. Check performance (Average Success Rate over window)
#         if not self.performance_history:
#             return False
#
#         # Calculate rolling averages
#         avg_success = np.mean([ep['success'] for ep in self.performance_history])
#         avg_collision = np.mean([ep['collision_rate'] for ep in self.performance_history])
#         avg_reward = np.mean([ep['reward'] for ep in self.performance_history])
#
#         # Debug print to track progress in logs
#         print(f"   [Curriculum Check] Stage: {current_stage.name} | "
#               f"Avg Success: {avg_success:.2f}/{current_stage.success_threshold} | "
#               f"Avg Reward: {avg_reward:.1f}")
#
#         # Advancement Criteria
#         criteria_met = (
#             avg_success >= current_stage.success_threshold and
#             avg_collision <= 0.1 and  # Less than 10% collision rate tolerance
#             avg_reward > 0.0  # Positive reward
#         )
#
#         # Stability check: Coefficient of Variation
#         if criteria_met:
#             recent_rewards = [ep['reward'] for ep in self.performance_history]
#             reward_std = np.std(recent_rewards)
#             reward_mean = np.mean(recent_rewards)
#
#             # If performance is stable (CV < 0.5) or simply very good, allow pass
#             if reward_mean > 0:
#                 cv = reward_std / reward_mean
#                 if cv < 0.5 or avg_success > (current_stage.success_threshold + 0.1):
#                     print(f"\n🎉 CONGRATULATIONS! Criteria met for Stage {self.current_stage_idx + 1}")
#                     return True
#
#         # Force advancement if max episodes reached (prevent eternal stalling)
#         if self.episodes_in_current_stage >= current_stage.max_episodes:
#             print(f"⚠️ Force advancing from stage {self.current_stage_idx} due to max episodes limit.")
#             return True
#
#         return False
#
#     def advance_stage(self) -> bool:
#         """Advance to next curriculum stage"""
#         if self.current_stage_idx < len(self.stages) - 1:
#             # Save statistics for completed stage
#             self._save_stage_statistics()
#
#             self.current_stage_idx += 1
#
#             # Reset for new stage
#             self.episodes_in_current_stage = 0
#             self.performance_history.clear()
#
#             new_stage = self.stages[self.current_stage_idx]
#             print(f"   >>> Advancing to Stage: {new_stage.name}")
#             print(f"   >>> Config: {new_stage.num_vehicles} Vehicles, {new_stage.pedestrian_count} Pedestrians\n")
#             return True
#
#         return False
#
#     def _save_stage_statistics(self):
#         """Save statistics for completed stage"""
#         if not self.performance_history:
#             return
#
#         stage = self.stages[self.current_stage_idx]
#
#         stats = {
#             'stage_name': stage.name,
#             'episodes_completed': self.episodes_in_current_stage,
#             'average_reward': float(np.mean([ep['reward'] for ep in self.performance_history])),
#             'success_rate': float(np.mean([ep['success'] for ep in self.performance_history])),
#             'collision_rate': float(np.mean([ep['collision_rate'] for ep in self.performance_history])),
#             'final_performance': float(np.mean([ep['reward'] for ep in list(self.performance_history)[-10:]])),
#             'learning_progress': self._calculate_learning_progress()
#         }
#
#         self.stage_statistics[self.current_stage_idx] = stats
#
#     def _calculate_learning_progress(self) -> float:
#         """Calculate learning progress during current stage"""
#         if len(self.performance_history) < 10:
#             return 0.0
#
#         # Compare early vs late performance in the current window
#         history_list = list(self.performance_history)
#         early_performance = np.mean([ep['reward'] for ep in history_list[:5]])
#         late_performance = np.mean([ep['reward'] for ep in history_list[-5:]])
#
#         if early_performance == 0:
#             return 1.0 if late_performance > 0 else 0.0
#
#         improvement = (late_performance - early_performance) / abs(early_performance)
#         return max(0.0, min(improvement, 2.0))  # Cap at 200% improvement
#
#     def get_curriculum_progress(self) -> Dict:
#         """Get overall curriculum progress information"""
#         return {
#             'current_stage': self.current_stage_idx,
#             'total_stages': len(self.stages),
#             'progress_percentage': (self.current_stage_idx / len(self.stages)) * 100,
#             'current_stage_name': self.stages[self.current_stage_idx].name if self.current_stage_idx < len(
#                 self.stages) else "Completed",
#             'episodes_in_current_stage': self.episodes_in_current_stage,
#             'stage_statistics': self.stage_statistics
#         }
#
#     def save_curriculum(self, filepath: str):
#         """Save curriculum configuration and progress"""
#         data = {
#             'stages': [
#                 {
#                     'name': stage.name,
#                     'level': stage.level.value,
#                     'num_vehicles': stage.num_vehicles,
#                     'pedestrian_count': stage.pedestrian_count,
#                     'weather': stage.weather,
#                     'traffic_density': stage.traffic_density,
#                     'success_threshold': stage.success_threshold,
#                     'min_episodes': stage.min_episodes,
#                     'max_episodes': stage.max_episodes
#                 }
#                 for stage in self.stages
#             ],
#             'current_stage': self.current_stage_idx,
#             'stage_statistics': self.stage_statistics
#         }
#
#         try:
#             with open(filepath, 'w') as f:
#                 yaml.dump(data, f, default_flow_style=False)
#         except Exception as e:
#             print(f"Error saving curriculum: {e}")
#
#     def load_curriculum(self, filepath: str):
#         """Load curriculum configuration and progress"""
#         try:
#             with open(filepath, 'r') as f:
#                 data = yaml.safe_load(f)
#
#             # Recreate curriculum stages
#             self.stages = []
#             for stage_data in data['stages']:
#                 stage = CurriculumStage(
#                     name=stage_data['name'],
#                     level=DifficultyLevel(stage_data['level']),
#                     num_vehicles=stage_data['num_vehicles'],
#                     pedestrian_count=stage_data.get('pedestrian_count', stage_data.get('pedestrians', 0)), # Handle both keys
#                     weather=stage_data['weather'],
#                     traffic_density=stage_data['traffic_density'],
#                     success_threshold=stage_data['success_threshold'],
#                     min_episodes=stage_data['min_episodes'],
#                     max_episodes=stage_data['max_episodes']
#                 )
#                 self.stages.append(stage)
#
#             self.current_stage_idx = data.get('current_stage', 0)
#             self.stage_statistics = data.get('stage_statistics', {})
#         except Exception as e:
#             print(f"Error loading curriculum: {e}, falling back to default.")
#             self._create_default_curriculum()
#
#
# # ==============================================================================
# # CURRICULUM TRAINER WRAPPER
# # ==============================================================================
#
# class CurriculumTrainer(MultiAgentTrainer):
#     """Enhanced trainer with curriculum learning support"""
#
#     def __init__(self, config, curriculum_config=None):
#         super().__init__(config)
#         self.curriculum_manager = CurriculumManager(curriculum_config)
#
#         # Trackers
#         self.episode_rewards = []
#         self.episode_steps = []
#         self.collision_counts = []
#         self.success_flags = []
#
#     def train_with_curriculum(self, total_timesteps=2000000):
#         """Train using curriculum learning approach"""
#         print("🎓 Starting Curriculum Learning Training")
#         print("=" * 50)
#
#         remaining_timesteps = total_timesteps
#
#         while remaining_timesteps > 0 and self.curriculum_manager.current_stage_idx < len(self.curriculum_manager.stages):
#             # Get current curriculum configuration
#             curriculum_config = self.curriculum_manager.get_current_config()
#             stage_name = curriculum_config['stage_name']
#
#             print(f"\n📚 Training Stage: {stage_name}")
#             print(f"   Vehicles: {curriculum_config['num_vehicles']}")
#             print(f"   Pedestrians: {curriculum_config['pedestrian_count']}")
#             print(f"   Difficulty: {curriculum_config['difficulty_level']}/4")
#
#             # Update environment configuration
#             self.config.update(curriculum_config)
#
#             # Setup training for current stage
#             self.setup_training()
#
#             # Train for current stage with curriculum tracking
#             # Train in smaller blocks to allow frequent checks
#             stage_block_steps = 25000
#             steps_to_train = min(stage_block_steps, remaining_timesteps)
#
#             self.train_agents_with_curriculum(steps_to_train)
#             remaining_timesteps -= steps_to_train
#
#             # Note: The advancement check is handled inside the callback,
#             # but we check here in the outer loop if we need to break out or reload config.
#             if self.curriculum_manager.current_stage_idx >= len(self.curriculum_manager.stages):
#                 print("All stages completed!")
#                 break
#
#         print("\n🎉 Curriculum Learning Completed!")
#
#         # Final training with all features enabled
#         if remaining_timesteps > 0:
#             print(f"\n🚀 Final Training Phase ({remaining_timesteps} timesteps)")
#             final_model = self.train_agents(remaining_timesteps)
#             return final_model
#
#         return self.model
#
#     def train_agents_with_curriculum(self, total_timesteps):
#         """Train agents with curriculum progression tracking"""
#         if not self.eval_env:
#             self.setup_evaluation()
#
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#         # Create custom callback that tracks episodes for curriculum
#         curriculum_callback = CurriculumCallback(self.curriculum_manager, self)
#
#         # Checkpoint callback - save every 25k timesteps
#         checkpoint_callback = CheckpointCallback(
#             save_freq=25000,
#             save_path=f"./experiments/models/checkpoints_{timestamp}/",
#             name_prefix="waymo_driver",
#             verbose=1
#         )
#
#         # Evaluation callback (optional, can be removed if slowing down training)
#         eval_callback = EvalCallback(
#             self.eval_env,
#             best_model_save_path=f"./experiments/models/best_model_{timestamp}/",
#             log_path=f"./experiments/eval_logs/eval_{timestamp}/",
#             eval_freq=50000, # Evaluate less frequently
#             n_eval_episodes=5,
#             deterministic=True,
#             render=False,
#             verbose=1
#         )
#
#         callback_list = CallbackList([curriculum_callback, checkpoint_callback, eval_callback])
#
#         self.model.learn(
#             total_timesteps=total_timesteps,
#             callback=callback_list,
#             tb_log_name=f"waymo_curriculum_{timestamp}",
#             reset_num_timesteps=False,
#             progress_bar=True
#         )
#
#         return self.model
#
#     def _evaluate_current_stage(self) -> Dict:
#         """Evaluate performance in current curriculum stage (Helper)"""
#         # Use recent history from trainer
#         if self.episode_rewards:
#             window = 20
#             recent_rewards = self.episode_rewards[-window:]
#             recent_collisions = self.collision_counts[-window:]
#             recent_success = self.success_flags[-window:]
#
#             return {
#                 'episode_reward': np.mean(recent_rewards),
#                 'success': np.mean(recent_success) > 0.5,
#                 'collision_rate': np.mean(recent_collisions),
#                 'completion_rate': np.mean(recent_success),
#                 'efficiency': np.mean(recent_rewards) / 100.0,
#                 'timestamp': time.time()
#             }
#         return {}
#
#
# # ==============================================================================
# # CALLBACK
# # ==============================================================================
#
# class CurriculumCallback(BaseCallback):
#     """Custom callback to handle curriculum learning progression"""
#
#     def __init__(self, curriculum_manager, trainer, verbose=0):
#         super().__init__(verbose)
#         self.curriculum_manager = curriculum_manager
#         self.trainer = trainer
#         self.current_episode_reward = 0.0
#         self.current_episode_steps = 0
#
#     def _on_step(self) -> bool:
#         # Accumulate rewards (assuming 1 env)
#         rewards = self.locals['rewards']
#         self.current_episode_reward += rewards[0]
#         self.current_episode_steps += 1
#
#         # Check for episode end (dones)
#         dones = self.locals['dones']
#         if dones[0]:
#             # Get info to check success/collision
#             infos = self.locals['infos'][0]
#
#             # Determine success based on environment info
#             # Using the logic: Collision = Failure, Max Steps without collision = Potential Success?
#             # Better: Env should explicitly set a 'success' key or positive reward threshold.
#             collision = infos.get('collisions', 0) > 0
#
#             # Define success threshold (e.g., > 50 total reward and no collision)
#             success = not collision and self.current_episode_reward > 50.0
#
#             collision_rate = 1.0 if collision else 0.0
#
#             # Record metrics in Curriculum Manager
#             episode_metrics = {
#                 'episode_reward': self.current_episode_reward,
#                 'success': success,
#                 'collision_rate': collision_rate,
#                 'timestamp': time.time()
#             }
#             self.curriculum_manager.record_episode_performance(episode_metrics)
#
#             # Update trainer's internal trackers
#             self.trainer.episode_rewards.append(self.current_episode_reward)
#             self.trainer.episode_steps.append(self.current_episode_steps)
#             self.trainer.collision_counts.append(collision_rate)
#             self.trainer.success_flags.append(success)
#
#             # Check for advancement
#             if self.curriculum_manager.should_advance_stage():
#                 self.curriculum_manager.advance_stage()
#                 # Stop training to force a reload of the environment config in the outer loop
#                 return False
#
#             # Reset episode counters
#             self.current_episode_reward = 0.0
#             self.current_episode_steps = 0
#
#         return True




import time
import numpy as np
import yaml
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
from collections import deque

# Robust imports for RL components
try:
    from agents.training.ppo_trainer import MultiAgentTrainer
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
except ImportError:
    print("Warning: RL libraries (MultiAgentTrainer/StableBaselines3) not found. Curriculum Manager running in standalone mode.")
    MultiAgentTrainer = object
    BaseCallback = object
    CheckpointCallback = object
    EvalCallback = object
    CallbackList = object

# Try importing environment for type hinting/reference if needed
try:
    from environments.carla_gym.multi_agent_env import WaymoStyleCarlaEnv
except ImportError:
    pass


class DifficultyLevel(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class CurriculumStage:
    """Configuration for a curriculum learning stage"""
    name: str
    level: DifficultyLevel
    town: str
    num_vehicles: int
    pedestrian_count: int
    weather: str
    traffic_density: float
    success_threshold: float         # success rate threshold
    min_episodes: int
    max_episodes: int
    reward_success_min: float        # reward threshold for success (stage-specific)


class CurriculumManager:
    """
    Town-aware curriculum:
    - Runs 4 stages per town, then restarts at EASY in next town.
    - Uses both performance-based advancement AND per-town timestep budgets.
    """

    def __init__(
        self,
        config_path: str = None,
        total_timesteps: int = 1_000_000,
        towns: Optional[List[str]] = None,
        town_budgets: Optional[Dict[str, int]] = None,
    ):
        self.current_stage_idx = 0
        self.stages: List[CurriculumStage] = []

        self.window_size = 20
        self.performance_history = deque(maxlen=self.window_size)

        self.stage_statistics = {}

        self.episodes_in_current_stage = 0
        self.total_episodes = 0

        # --- NEW: town progression + timestep budgets ---
        self.total_timesteps = total_timesteps
        self.towns = towns or ["Town01", "Town02", "Town03", "Town10"]

        self.town_budgets = town_budgets or {
            "Town01": 150_000,
            "Town02": 150_000,
            "Town03": 300_000,
            "Town10": 400_000,
        }

        # Track timesteps inside a town/stage
        self.timesteps_in_current_stage = 0
        self.timesteps_in_current_town = 0

        # Derive per-stage budgets per town
        self.per_stage_budget = {
            town: max(1, self.town_budgets[town] // 4)
            for town in self.towns
        }

        if config_path and os.path.exists(config_path):
            self.load_curriculum(config_path)
        else:
            self._create_default_curriculum()

    def _create_default_curriculum(self):
        """
        Create the full town-aware curriculum:
        For each town: EASY -> MEDIUM -> HARD -> EXPERT
        with vehicle/ped ramp and stage reward thresholds.
        """
        self.stages = []

        # stage templates shared across towns
        stage_templates = [
            dict(
                name="Easy",
                level=DifficultyLevel.EASY,
                num_vehicles=5,
                pedestrian_count=0,
                weather="ClearNoon",
                traffic_density=0.2,
                success_threshold=0.5,
                min_episodes=20,
                max_episodes=200,
                reward_success_min=50.0
            ),
            dict(
                name="Medium",
                level=DifficultyLevel.MEDIUM,
                num_vehicles=10,
                pedestrian_count=2,
                weather="CloudyNoon",
                traffic_density=0.5,
                success_threshold=0.7,
                min_episodes=30,
                max_episodes=300,
                reward_success_min=75.0
            ),
            dict(
                name="Hard",
                level=DifficultyLevel.HARD,
                num_vehicles=15,
                pedestrian_count=5,
                weather="WetNoon",
                traffic_density=0.7,
                success_threshold=0.8,
                min_episodes=50,
                max_episodes=400,
                reward_success_min=90.0
            ),
            dict(
                name="Expert",
                level=DifficultyLevel.EXPERT,
                num_vehicles=25,
                pedestrian_count=10,
                weather="HardRainNoon",
                traffic_density=1.0,
                success_threshold=0.9,
                min_episodes=100,
                max_episodes=500,
                reward_success_min=110.0
            ),
        ]

        for town in self.towns:
            for tmpl in stage_templates:
                self.stages.append(
                    CurriculumStage(
                        town=town,
                        name=f"{town}-{tmpl['name']}",
                        level=tmpl["level"],
                        num_vehicles=tmpl["num_vehicles"],
                        pedestrian_count=tmpl["pedestrian_count"],
                        weather=tmpl["weather"],
                        traffic_density=tmpl["traffic_density"],
                        success_threshold=tmpl["success_threshold"],
                        min_episodes=tmpl["min_episodes"],
                        max_episodes=tmpl["max_episodes"],
                        reward_success_min=tmpl["reward_success_min"],
                    )
                )

    def _current_stage(self) -> CurriculumStage:
        return self.stages[min(self.current_stage_idx, len(self.stages)-1)]

    def get_current_config(self) -> Dict:
        """Get env configuration for current stage"""
        stage = self._current_stage()
        return self._stage_to_config(stage)

    def _stage_to_config(self, stage: CurriculumStage) -> Dict:
        return {
            "town": stage.town,
            "num_vehicles": stage.num_vehicles,
            "pedestrian_count": stage.pedestrian_count,
            "weather": stage.weather,
            "traffic_density": stage.traffic_density,
            "stage_name": stage.name,
            "difficulty_level": stage.level.value,
            "reward_success_min": stage.reward_success_min,  # pass to callback if wanted
        }

    def record_episode_performance(self, metrics: Dict):
        """Record episode metrics + update episode counters"""
        episode_data = {
            "success": 1.0 if metrics.get("success", False) else 0.0,
            "reward": metrics.get("episode_reward", 0.0),
            "collision_rate": metrics.get("collision_rate", 0.0),
            "stage": self.current_stage_idx,
            "town": self._current_stage().town,
            "episode_global": self.total_episodes
        }
        self.performance_history.append(episode_data)
        self.episodes_in_current_stage += 1
        self.total_episodes += 1

    # --- NEW: called by callback each step ---
    def record_timesteps(self, n_steps: int = 1):
        self.timesteps_in_current_stage += n_steps
        self.timesteps_in_current_town += n_steps

    def should_advance_stage(self) -> bool:
        """Advance if performance ok OR stage-budget exhausted."""
        if self.current_stage_idx >= len(self.stages) - 1:
            return False

        stage = self._current_stage()
        town = stage.town

        # 0) Budget-based advance (hard guarantee)
        if self.timesteps_in_current_stage >= self.per_stage_budget[town]:
            print(f"⏱ Stage budget reached for {stage.name} "
                  f"({self.timesteps_in_current_stage}/{self.per_stage_budget[town]}). Advancing.")
            return True

        # 1) Must complete minimum episodes
        if self.episodes_in_current_stage < stage.min_episodes:
            return False

        if not self.performance_history:
            return False

        avg_success = np.mean([ep["success"] for ep in self.performance_history])
        avg_collision = np.mean([ep["collision_rate"] for ep in self.performance_history])
        avg_reward = np.mean([ep["reward"] for ep in self.performance_history])

        print(f"   [Curriculum Check] {stage.name} | "
              f"Town={town} | "
              f"AvgSuccess={avg_success:.2f}/{stage.success_threshold} | "
              f"AvgReward={avg_reward:.1f} | "
              f"StageSteps={self.timesteps_in_current_stage}/{self.per_stage_budget[town]}")

        criteria_met = (
            avg_success >= stage.success_threshold and
            avg_collision <= 0.1 and
            avg_reward > 0.0
        )

        if criteria_met:
            rewards = [ep["reward"] for ep in self.performance_history]
            reward_std = np.std(rewards)
            reward_mean = np.mean(rewards)
            if reward_mean > 0:
                cv = reward_std / reward_mean
                if cv < 0.5 or avg_success > (stage.success_threshold + 0.1):
                    print(f"\n🎉 Criteria met for {stage.name}. Advancing.")
                    return True

        if self.episodes_in_current_stage >= stage.max_episodes:
            print(f"⚠️ Force advancing {stage.name} due to max episodes.")
            return True

        return False

    def advance_stage(self) -> bool:
        """Advance to next stage. If town changes, reset town-specific counters."""
        if self.current_stage_idx < len(self.stages) - 1:
            prev_stage = self._current_stage()
            prev_town = prev_stage.town

            self._save_stage_statistics()

            self.current_stage_idx += 1
            new_stage = self._current_stage()
            new_town = new_stage.town

            # reset stage counters
            self.episodes_in_current_stage = 0
            self.performance_history.clear()
            self.timesteps_in_current_stage = 0

            # if town changed, reset town timestep counter
            if new_town != prev_town:
                print(f"\n🏙 Switching town: {prev_town} → {new_town}")
                self.timesteps_in_current_town = 0

            print(f"   >>> Advancing to Stage: {new_stage.name}")
            print(f"   >>> Config: Town={new_town}, "
                  f"{new_stage.num_vehicles} Vehicles, "
                  f"{new_stage.pedestrian_count} Pedestrians, "
                  f"Budget={self.per_stage_budget[new_town]} steps\n")
            return True

        return False

    def _save_stage_statistics(self):
        if not self.performance_history:
            return
        stage = self._current_stage()
        stats = {
            "stage_name": stage.name,
            "town": stage.town,
            "episodes_completed": self.episodes_in_current_stage,
            "average_reward": float(np.mean([ep["reward"] for ep in self.performance_history])),
            "success_rate": float(np.mean([ep["success"] for ep in self.performance_history])),
            "collision_rate": float(np.mean([ep["collision_rate"] for ep in self.performance_history])),
            "final_performance": float(np.mean([ep["reward"] for ep in list(self.performance_history)[-10:]])),
            "learning_progress": self._calculate_learning_progress(),
            "timesteps_in_stage": int(self.timesteps_in_current_stage),
        }
        self.stage_statistics[self.current_stage_idx] = stats

    def _calculate_learning_progress(self) -> float:
        if len(self.performance_history) < 10:
            return 0.0
        hist = list(self.performance_history)
        early = np.mean([ep["reward"] for ep in hist[:5]])
        late = np.mean([ep["reward"] for ep in hist[-5:]])
        if early == 0:
            return 1.0 if late > 0 else 0.0
        improvement = (late - early) / abs(early)
        return max(0.0, min(improvement, 2.0))

    def get_curriculum_progress(self) -> Dict:
        stage = self._current_stage()
        return {
            "current_stage": self.current_stage_idx,
            "total_stages": len(self.stages),
            "progress_percentage": (self.current_stage_idx / len(self.stages)) * 100,
            "current_stage_name": stage.name,
            "current_town": stage.town,
            "episodes_in_current_stage": self.episodes_in_current_stage,
            "timesteps_in_current_stage": self.timesteps_in_current_stage,
            "timesteps_in_current_town": self.timesteps_in_current_town,
            "stage_statistics": self.stage_statistics
        }

    def save_curriculum(self, filepath: str):
        data = {
            "towns": self.towns,
            "town_budgets": self.town_budgets,
            "stages": [
                {
                    "name": s.name,
                    "level": s.level.value,
                    "town": s.town,
                    "num_vehicles": s.num_vehicles,
                    "pedestrian_count": s.pedestrian_count,
                    "weather": s.weather,
                    "traffic_density": s.traffic_density,
                    "success_threshold": s.success_threshold,
                    "min_episodes": s.min_episodes,
                    "max_episodes": s.max_episodes,
                    "reward_success_min": s.reward_success_min
                } for s in self.stages
            ],
            "current_stage": self.current_stage_idx,
            "stage_statistics": self.stage_statistics
        }
        try:
            with open(filepath, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving curriculum: {e}")

    def load_curriculum(self, filepath: str):
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)

            self.towns = data.get("towns", self.towns)
            self.town_budgets = data.get("town_budgets", self.town_budgets)
            self.per_stage_budget = {
                town: max(1, self.town_budgets[town] // 4)
                for town in self.towns
            }

            self.stages = []
            for sd in data["stages"]:
                self.stages.append(
                    CurriculumStage(
                        name=sd["name"],
                        level=DifficultyLevel(sd["level"]),
                        town=sd.get("town", "Town03"),
                        num_vehicles=sd["num_vehicles"],
                        pedestrian_count=sd.get("pedestrian_count", 0),
                        weather=sd["weather"],
                        traffic_density=sd["traffic_density"],
                        success_threshold=sd["success_threshold"],
                        min_episodes=sd["min_episodes"],
                        max_episodes=sd["max_episodes"],
                        reward_success_min=sd.get("reward_success_min", 50.0),
                    )
                )

            self.current_stage_idx = data.get("current_stage", 0)
            self.stage_statistics = data.get("stage_statistics", {})
        except Exception as e:
            print(f"Error loading curriculum: {e}, falling back to default.")
            self._create_default_curriculum()


# ==============================================================================
# CURRICULUM TRAINER WRAPPER
# ==============================================================================

class CurriculumTrainer(MultiAgentTrainer):
    """Enhanced trainer with town-aware curriculum learning support"""

    def __init__(self, config, curriculum_config=None):
        super().__init__(config)
        self.curriculum_manager = CurriculumManager(curriculum_config, total_timesteps=1_000_000)

        self.episode_rewards = []
        self.episode_steps = []
        self.collision_counts = []
        self.success_flags = []

    def train_with_curriculum(self, total_timesteps=1_000_000):
        print("🎓 Starting Town-Aware Curriculum Training")
        print("=" * 60)

        remaining_timesteps = total_timesteps

        while remaining_timesteps > 0 and self.curriculum_manager.current_stage_idx < len(self.curriculum_manager.stages):
            cfg = self.curriculum_manager.get_current_config()
            stage_name = cfg["stage_name"]

            print(f"\n📚 Training Stage: {stage_name}")
            print(f"   Town: {cfg['town']}")
            print(f"   Vehicles: {cfg['num_vehicles']}")
            print(f"   Pedestrians: {cfg['pedestrian_count']}")
            print(f"   Difficulty: {cfg['difficulty_level']}/4")
            print(f"   Stage Budget: {self.curriculum_manager.per_stage_budget[cfg['town']]} steps")

            self.config.update(cfg)

            self.setup_training()

            stage_block_steps = 25_000
            steps_to_train = min(stage_block_steps, remaining_timesteps)

            self.train_agents_with_curriculum(steps_to_train)
            remaining_timesteps -= steps_to_train

            if self.curriculum_manager.current_stage_idx >= len(self.curriculum_manager.stages):
                print("All stages completed!")
                break

        print("\n🎉 Curriculum Learning Completed!")

        if remaining_timesteps > 0:
            print(f"\n🚀 Final Training Phase ({remaining_timesteps} timesteps)")
            final_model = self.train_agents(remaining_timesteps)
            return final_model

        return self.model

    def train_agents_with_curriculum(self, total_timesteps):
        if not getattr(self, "eval_env", None):
            self.setup_evaluation()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage = self.curriculum_manager._current_stage()
        town = stage.town

        curriculum_callback = CurriculumCallback(self.curriculum_manager, self)

        checkpoint_callback = CheckpointCallback(
            save_freq=25_000,
            save_path=f"./experiments/models/checkpoints_{town}_{timestamp}/",
            name_prefix="waymo_driver",
            verbose=1
        )

        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"./experiments/models/best_model_{town}_{timestamp}/",
            log_path=f"./experiments/eval_logs/eval_{town}_{timestamp}/",
            eval_freq=50_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1
        )

        callback_list = CallbackList([curriculum_callback, checkpoint_callback, eval_callback])

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            tb_log_name=f"waymo_curriculum_{town}_{timestamp}",
            reset_num_timesteps=False,
            progress_bar=True
        )

        return self.model


# ==============================================================================
# CALLBACK
# ==============================================================================

class CurriculumCallback(BaseCallback):
    """Custom callback to handle town-aware curriculum progression"""

    def __init__(self, curriculum_manager, trainer, verbose=0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.trainer = trainer
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0

    def _on_step(self) -> bool:
        # count timesteps for stage budgets
        self.curriculum_manager.record_timesteps(1)

        rewards = self.locals["rewards"]
        self.current_episode_reward += float(rewards[0])
        self.current_episode_steps += 1

        dones = self.locals["dones"]
        if dones[0]:
            infos = self.locals["infos"][0]
            collision = infos.get("collisions", 0) > 0

            stage = self.curriculum_manager._current_stage()
            success_reward_min = stage.reward_success_min

            success = (not collision) and (self.current_episode_reward > success_reward_min)
            collision_rate = 1.0 if collision else 0.0

            episode_metrics = {
                "episode_reward": self.current_episode_reward,
                "success": success,
                "collision_rate": collision_rate,
                "timestamp": time.time()
            }

            self.curriculum_manager.record_episode_performance(episode_metrics)

            self.trainer.episode_rewards.append(self.current_episode_reward)
            self.trainer.episode_steps.append(self.current_episode_steps)
            self.trainer.collision_counts.append(collision_rate)
            self.trainer.success_flags.append(success)

            if self.curriculum_manager.should_advance_stage():
                self.curriculum_manager.advance_stage()
                return False  # stop learn() so outer loop reloads env config

            self.current_episode_reward = 0.0
            self.current_episode_steps = 0

        return True
