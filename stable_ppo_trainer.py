# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
# from stable_baselines3.common.monitor import Monitor
# import yaml
# import os
# from datetime import datetime
#
# # INTEGRATION: Import the Curriculum Manager and Environment
# try:
#     from agents.training.curriculum_manager import CurriculumManager
#     from validation_metrics import ValidationMetrics
#     from enhanced_multi_agent_env import StableMultiAgentCarlaEnv
# except ImportError:
#     print("CRITICAL: Could not import required modules. Check your python path.")
#     raise
#
#
# class ValidationCallback(BaseCallback):
#     """
#     Custom callback that tracks metrics for theoretical validation and curriculum advancement.
#     Integrates real telemetry from the environment instead of mock data.
#     """
#
#     def __init__(self, validation_metrics, curriculum_manager, verbose=0):
#         super().__init__(verbose)
#         self.validation_metrics = validation_metrics
#         self.curriculum_manager = curriculum_manager
#
#         # Buffer to accumulate stats over an episode
#         self.episode_buffer = {
#             'reward': 0.0,
#             'length': 0,
#             'min_distance': float('inf'),
#             'collisions': 0,
#             'emergency_stops': 0,
#             'conflicts_resolved': 0
#         }
#
#     def _on_step(self) -> bool:
#         """Called at each step"""
#         # Accumulate step reward
#         self.episode_buffer['reward'] += self.locals['rewards'][0]
#         self.episode_buffer['length'] += 1
#
#         # Extract real-time info from environment
#         if 'infos' in self.locals and len(self.locals['infos']) > 0:
#             info = self.locals['infos'][0]
#
#             # Track Safety Metrics
#             if 'min_distance' in info:
#                 self.episode_buffer['min_distance'] = min(
#                     self.episode_buffer['min_distance'],
#                     info['min_distance']
#                 )
#             if 'collisions' in info:
#                 self.episode_buffer['collisions'] = info['collisions']
#
#             # Track Fleet Manager Metrics
#             if 'emergency_stops' in info:
#                 self.episode_buffer['emergency_stops'] = info['emergency_stops']
#             if 'conflicts_resolved' in info:
#                 # Info gives cumulative conflicts for the episode
#                 self.episode_buffer['conflicts_resolved'] = info['conflicts_resolved']
#
#         # Check if episode is done
#         if self.locals['dones'][0]:
#             # Determine Success (e.g. positive reward)
#             success = self.episode_buffer['reward'] > 0
#
#             # Compile REAL episode data
#             episode_data = {
#                 'reward': self.episode_buffer['reward'],
#                 'length': self.episode_buffer['length'],
#                 'success': 1.0 if success else 0.0,
#                 'min_distance': self.episode_buffer['min_distance'],
#                 'collision': 1 if self.episode_buffer['collisions'] > 0 else 0,
#                 'safety_violation': 1 if self.episode_buffer['min_distance'] < 5.0 else 0,
#
#                 # INTELLIGENT METRICS (No longer random)
#                 'emergency_stops': self.episode_buffer['emergency_stops'],
#                 'conflicts_resolved': self.episode_buffer['conflicts_resolved'],
#                 'stage': self.curriculum_manager.current_stage
#             }
#
#             # 1. Record for Validation Report
#             self.validation_metrics.record_episode(episode_data)
#
#             # 2. Update Curriculum Logic
#             self.curriculum_manager.record_episode_performance(episode_data)
#
#             # 3. Check for Promotion
#             if self.curriculum_manager.should_advance_stage():
#                 self.curriculum_manager.advance_stage()
#                 new_config = self.curriculum_manager.get_current_config()
#                 print(f"\n🎓 CURRICULUM PROMOTION! Advanced to Stage: {new_config['stage_name']}")
#                 print(f"   Difficulty Level: {new_config['difficulty_level']}/4")
#
#             # Reset buffer
#             self.episode_buffer = {
#                 'reward': 0.0,
#                 'length': 0,
#                 'min_distance': float('inf'),
#                 'collisions': 0,
#                 'emergency_stops': 0,
#                 'conflicts_resolved': 0
#             }
#
#         return True
#
#
# class StableMultiAgentTrainer:
#     """
#     Enhanced trainer with stability improvements, validation, and curriculum learning.
#     """
#
#     def __init__(self, config):
#         self.config = config
#         self.model = None
#         self.training_env = None
#         self.eval_env = None
#         self.validation_metrics = ValidationMetrics()
#
#         # INTEGRATION: Initialize Curriculum Manager
#         self.curriculum_manager = CurriculumManager()
#
#     def create_training_env(self):
#         """Create training environment"""
#
#         def make_env():
#             # Pass config parameters that might change via curriculum
#             # Note: For simple resizing, we rely on the environment's internal handling or restarts
#             # Here we init with the base config
#             return Monitor(StableMultiAgentCarlaEnv(
#                 num_vehicles=self.config['num_vehicles'],
#                 town=self.config['town'],
#                 port=self.config.get('carla_port', 2000),
#                 max_episode_steps=self.config.get('max_episode_steps', 1000),
#                 pedestrian_count=15  # Default high count for randomness
#             ))
#
#         return make_env
#
#     def setup_training(self):
#         """Setup training with optimized parameters"""
#         env_fn = self.create_training_env()
#         self.training_env = VecMonitor(DummyVecEnv([env_fn]))
#
#         # Determine policy type
#         if len(self.training_env.observation_space.shape) == 1:
#             policy_type = "MlpPolicy"
#         else:
#             policy_type = "MultiInputPolicy"
#
#         # PPO Hyperparameters
#         self.model = PPO(
#             policy_type,
#             self.training_env,
#             learning_rate=3e-4,
#             n_steps=512,
#             batch_size=64,
#             n_epochs=10,
#             gamma=0.99,
#             gae_lambda=0.95,
#             clip_range=0.2,
#             ent_coef=0.01,
#             vf_coef=0.5,
#             max_grad_norm=0.5,
#             verbose=1,
#             tensorboard_log="./experiments/tensorboard_logs/",
#             device="cuda"
#         )
#
#     def setup_evaluation(self):
#         """Setup evaluation environment"""
#         eval_env_fn = self.create_training_env()
#         self.eval_env = eval_env_fn()
#
#     def train_agents(self, total_timesteps=1000000):
#         """Train with validation tracking and curriculum"""
#         if not self.eval_env:
#             self.setup_evaluation()
#
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#         # Callback Stack
#         validation_callback = ValidationCallback(
#             self.validation_metrics,
#             self.curriculum_manager
#         )
#
#         checkpoint_callback = CheckpointCallback(
#             save_freq=50000,
#             save_path=f"./experiments/models/checkpoints_{timestamp}/",
#             name_prefix="stable_model",
#             verbose=1
#         )
#
#         callback_list = CallbackList([validation_callback, checkpoint_callback])
#
#         print(f"🚀 Starting Training: {total_timesteps} steps")
#         print(f"   Device: {self.model.device}")
#         print(f"   Curriculum: Enabled")
#
#         # Train
#         self.model.learn(
#             total_timesteps=total_timesteps,
#             callback=callback_list,
#             tb_log_name=f"waymo_training_{timestamp}",
#             progress_bar=True
#         )
#
#         # Save final model
#         final_model_path = f"./experiments/models/final_model_{timestamp}/"
#         os.makedirs(final_model_path, exist_ok=True)
#         self.model.save(f"{final_model_path}/final_model")
#
#         # Generate validation reports
#         print("\nGenerating validation reports...")
#         self.validation_metrics.plot_collision_avoidance_validation()
#         self.validation_metrics.plot_deadlock_freedom_validation()
#         self.validation_metrics.plot_training_progress()
#         self.validation_metrics.generate_validation_report()
#         self.validation_metrics.export_to_csv()
#
#         return self.model
# stable_ppo_trainer.py


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
import time
import numpy as np
from datetime import datetime

# INTEGRATION: Import the Curriculum Manager and Environment
try:
    # IMPORTANT: this should point to your NEW town-aware curriculum manager file
    # If your path is different, adjust the import.
    from curriculum_manager import CurriculumManager
    from validation_metrics import ValidationMetrics
    from enhanced_multi_agent_env import StableMultiAgentCarlaEnv
except ImportError as e:
    print("CRITICAL: Could not import required modules. Check your python path.")
    raise e


# ==============================================================================
# CALLBACKS
# ==============================================================================

class CurriculumCallback(BaseCallback):
    """
    Town-aware curriculum callback.
    - Records timesteps for stage budgets.
    - Computes per-episode success/collision.
    - Triggers advancement by returning False (forces outer loop to reload env).
    """

    def __init__(self, curriculum_manager: CurriculumManager, trainer, verbose=0):
        super().__init__(verbose)
        self.cm = curriculum_manager
        self.trainer = trainer

        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        self.current_episode_collisions = 0

    def _on_step(self) -> bool:
        # 1) Count timesteps for budgets
        self.cm.record_timesteps(1)

        # 2) Track reward / steps
        r = float(self.locals["rewards"][0])
        self.current_episode_reward += r
        self.current_episode_steps += 1

        # 3) Track collisions from env info
        infos = self.locals.get("infos", None)
        if infos and len(infos) > 0:
            self.current_episode_collisions = max(
                self.current_episode_collisions,
                int(infos[0].get("collisions", 0))
            )

        dones = self.locals["dones"]
        if dones[0]:
            stage = self.cm._current_stage()
            success_reward_min = stage.reward_success_min
            collision = self.current_episode_collisions > 0

            success = (not collision) and (self.current_episode_reward > success_reward_min)
            collision_rate = 1.0 if collision else 0.0

            episode_metrics = {
                "episode_reward": self.current_episode_reward,
                "success": success,
                "collision_rate": collision_rate,
                "timestamp": time.time()
            }

            self.cm.record_episode_performance(episode_metrics)

            # trainer-side logs
            self.trainer.episode_rewards.append(self.current_episode_reward)
            self.trainer.episode_steps.append(self.current_episode_steps)
            self.trainer.collision_counts.append(collision_rate)
            self.trainer.success_flags.append(success)

            # 4) Advance stage if ready
            if self.cm.should_advance_stage():
                self.cm.advance_stage()
                # Stop learn() so outer loop rebuilds env with new town/stage
                return False

            # reset per-episode accumulators
            self.current_episode_reward = 0.0
            self.current_episode_steps = 0
            self.current_episode_collisions = 0

        return True


class ValidationCallback(BaseCallback):
    """
    Records real episode metrics into ValidationMetrics.
    NOTE: This callback is independent of curriculum, just logs.
    """

    def __init__(self, validation_metrics: ValidationMetrics, curriculum_manager: CurriculumManager, verbose=0):
        super().__init__(verbose)
        self.vm = validation_metrics
        self.cm = curriculum_manager

        self.buffer = {
            "reward": 0.0,
            "length": 0,
            "min_distance": float("inf"),
            "collisions": 0,
            "emergency_stops": 0,
            "conflicts_resolved": 0,
        }

    def _on_step(self) -> bool:
        self.buffer["reward"] += float(self.locals["rewards"][0])
        self.buffer["length"] += 1

        infos = self.locals.get("infos", None)
        if infos and len(infos) > 0:
            info = infos[0]
            self.buffer["min_distance"] = min(self.buffer["min_distance"], float(info.get("min_distance", 1e9)))
            self.buffer["collisions"] = int(info.get("collisions", 0))
            self.buffer["emergency_stops"] = int(info.get("emergency_stops", 0))
            self.buffer["conflicts_resolved"] = int(info.get("conflicts_resolved", 0))

        dones = self.locals["dones"]
        if dones[0]:
            stage = self.cm._current_stage()

            success = (self.buffer["collisions"] == 0) and (self.buffer["reward"] > stage.reward_success_min)

            episode_data = {
                "reward": self.buffer["reward"],
                "length": self.buffer["length"],
                "success": 1.0 if success else 0.0,
                "min_distance": self.buffer["min_distance"],
                "collision": 1 if self.buffer["collisions"] > 0 else 0,
                "safety_violation": 1 if self.buffer["min_distance"] < 5.0 else 0,
                "emergency_stops": self.buffer["emergency_stops"],
                "conflicts_resolved": self.buffer["conflicts_resolved"],
                "stage": self.cm.current_stage_idx,
                "town": stage.town,
                "stage_name": stage.name
            }

            self.vm.record_episode(episode_data)

            self.buffer = {
                "reward": 0.0,
                "length": 0,
                "min_distance": float("inf"),
                "collisions": 0,
                "emergency_stops": 0,
                "conflicts_resolved": 0,
            }

        return True


# ==============================================================================
# TRAINER
# ==============================================================================

class StableMultiAgentTrainer:
    """
    Stable PPO trainer with town-aware curriculum driving the env config.
    Use train_with_curriculum() for your 1M run.
    """

    def __init__(self, config: dict, total_timesteps: int = 1_000_000):
        self.config = config

        # curriculum + validation
        self.curriculum_manager = CurriculumManager(total_timesteps=total_timesteps)
        self.validation_metrics = ValidationMetrics()

        # SB3 objects
        self.model = None
        self.training_env = None
        self.eval_env = None

        # trackers
        self.episode_rewards = []
        self.episode_steps = []
        self.collision_counts = []
        self.success_flags = []

    # ---------------------------
    # Environment factory
    # ---------------------------
    def create_training_env(self):
        """Creates env using *current* config (which curriculum updates)."""

        def make_env():
            return Monitor(
                StableMultiAgentCarlaEnv(
                    num_vehicles=self.config["num_vehicles"],
                    town=self.config["town"],
                    port=self.config.get("carla_port", 2000),
                    max_episode_steps=self.config.get("max_episode_steps", 1000),
                    pedestrian_count=self.config.get("pedestrian_count", 0),
                    weather=self.config.get("weather", "ClearNoon"),
                    timeout_threshold=self.config.get("timeout_threshold", 300.0)
                )
            )

        return make_env

    # ---------------------------
    # Setup
    # ---------------------------
    # def setup_training(self):
    #     env_fn = self.create_training_env()
    #     self.training_env = VecMonitor(DummyVecEnv([env_fn]))
    #
    #     policy_type = "MlpPolicy" if len(self.training_env.observation_space.shape) == 1 else "MultiInputPolicy"
    #
    #     self.model = PPO(
    #         policy_type,
    #         self.training_env,
    #         learning_rate=self.config.get("learning_rate", 3e-4),
    #         n_steps=self.config.get("n_steps", 512),
    #         batch_size=self.config.get("batch_size", 64),
    #         n_epochs=self.config.get("n_epochs", 10),
    #         gamma=self.config.get("gamma", 0.99),
    #         gae_lambda=self.config.get("gae_lambda", 0.95),
    #         clip_range=self.config.get("clip_range", 0.2),
    #         ent_coef=0.01,
    #         vf_coef=0.5,
    #         max_grad_norm=0.5,
    #         verbose=1,
    #         tensorboard_log="./experiments/tensorboard_logs/",
    #         device=self.config.get("device", "cuda"),
    #     )
    # In stable_ppo_trainer.py

    def setup_training(self):
        """
        Setup training with optimized parameters.
        CRITICAL FIX: Checks if model exists to preserve 'intelligence' across stages.
        """
        # 1. Create the new environment (the "new world")
        env_fn = self.create_training_env()
        self.training_env = VecMonitor(DummyVecEnv([env_fn]))

        # 2. CHECK: Do we already have a brain?
        if self.model is not None:
            print(f"🔄 TRANSFER LEARNING: Keeping existing brain, updating environment.")
            print(f"   Current Agent Steps: {self.model.num_timesteps}")
            self.model.set_env(self.training_env)
            return  # <--- EXIT HERE so we don't overwrite self.model

        # 3. If no brain exists, create one (only happens once at the very start)
        print("✨ INITIALIZING: Creating new PPO Agent from scratch.")

        # Determine policy based on observation space
        if len(self.training_env.observation_space.shape) == 1:
            policy_type = "MlpPolicy"
        else:
            policy_type = "MultiInputPolicy"

        self.model = PPO(
            policy_type,
            self.training_env,
            learning_rate=self.config.get("learning_rate", 3e-4),
            n_steps=self.config.get("n_steps", 512),
            batch_size=self.config.get("batch_size", 64),
            n_epochs=self.config.get("n_epochs", 10),
            gamma=self.config.get("gamma", 0.99),
            gae_lambda=self.config.get("gae_lambda", 0.95),
            clip_range=self.config.get("clip_range", 0.2),
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./experiments/tensorboard_logs/",
            device=self.config.get("device", "cuda"),
        )

    def setup_evaluation(self):
        eval_env_fn = self.create_training_env()
        self.eval_env = eval_env_fn()

    # ---------------------------
    # Curriculum training loop
    # ---------------------------
    # def train_with_curriculum(self, total_timesteps: int = 1_000_000):
    #     """
    #     Main entry for your run.
    #     Curriculum controls stages & towns.
    #     """
    #
    #     if not self.eval_env:
    #         self.setup_evaluation()
    #
    #     remaining_timesteps = total_timesteps
    #
    #     print("🎓 Starting Town-Aware Curriculum PPO")
    #     print("=" * 70)
    #
    #     # Train in blocks so callback can stop and outer loop reloads env
    #     stage_block_steps = 25_000
    #
    #     while remaining_timesteps > 0 and self.curriculum_manager.current_stage_idx < len(self.curriculum_manager.stages):
    #         cfg = self.curriculum_manager.get_current_config()
    #         self.config.update(cfg)
    #
    #         stage = self.curriculum_manager._current_stage()
    #
    #         print(f"\n📚 Stage: {stage.name}")
    #         print(f"   Town: {stage.town}")
    #         print(f"   Vehicles: {stage.num_vehicles}")
    #         print(f"   Pedestrians: {stage.pedestrian_count}")
    #         print(f"   Difficulty: {stage.level.value}/4")
    #         print(f"   StageBudget: {self.curriculum_manager.per_stage_budget[stage.town]}")
    #
    #         # Reset env/model for this stage (important when town changes)
    #         self.setup_training()
    #
    #         steps_to_train = min(stage_block_steps, remaining_timesteps)
    #         remaining_timesteps -= steps_to_train
    #
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #
    #         curriculum_cb = CurriculumCallback(self.curriculum_manager, self)
    #         validation_cb = ValidationCallback(self.validation_metrics, self.curriculum_manager)
    #
    #         checkpoint_cb = CheckpointCallback(
    #             save_freq=25_000,
    #             save_path=f"./experiments/models/checkpoints_{stage.town}_{timestamp}/",
    #             name_prefix="waymo_driver",
    #             verbose=1,
    #         )
    #
    #         eval_cb = EvalCallback(
    #             self.eval_env,
    #             best_model_save_path=f"./experiments/models/best_model_{stage.town}_{timestamp}/",
    #             log_path=f"./experiments/eval_logs/eval_{stage.town}_{timestamp}/",
    #             eval_freq=50_000,
    #             n_eval_episodes=5,
    #             deterministic=True,
    #             render=False,
    #             verbose=1,
    #         )
    #
    #         cb_list = CallbackList([curriculum_cb, validation_cb, checkpoint_cb, eval_cb])
    #
    #         self.model.learn(
    #             total_timesteps=steps_to_train,
    #             callback=cb_list,
    #             tb_log_name=f"waymo_curriculum_{stage.town}_{timestamp}",
    #             reset_num_timesteps=False,
    #             progress_bar=True,
    #         )
    #
    #     print("\n🎉 Curriculum completed or timesteps exhausted.")
    #
    #     # Save final model
    #     final_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     final_path = f"./experiments/models/final_model_{final_stamp}/"
    #     os.makedirs(final_path, exist_ok=True)
    #     self.model.save(f"{final_path}/final_model")
    #
    #     # Validation outputs
    #     print("\nGenerating validation reports...")
    #     self.validation_metrics.plot_collision_avoidance_validation()
    #     self.validation_metrics.plot_deadlock_freedom_validation()
    #     self.validation_metrics.plot_training_progress()
    #     self.validation_metrics.plot_waymo_safety_metrics()
    #     self.validation_metrics.generate_validation_report()
    #     self.validation_metrics.export_to_csv()
    #
    #     return self.model
    # ---------------------------
    # Curriculum training loop (CORRECTED)
    # ---------------------------
# ---------------------------
    # Curriculum training loop (FIXED)
    # ---------------------------
    def train_with_curriculum(self, total_timesteps: int = 1_000_000):
        """
        Main entry for your run.
        Curriculum controls stages & towns.
        """
        if not self.eval_env:
            self.setup_evaluation()

        print("🎓 Starting Town-Aware Curriculum PPO")
        print("=" * 70)

        # We loop until the MODEL itself has trained for 1M steps
        # This is safer than maintaining a local 'remaining_timesteps' counter
        stage_block_steps = 25_000

        # FIX: Allow entry if model is None (first run) OR if steps < total
        while (self.model is None or self.model.num_timesteps < total_timesteps) and \
              self.curriculum_manager.current_stage_idx < len(self.curriculum_manager.stages):
            
            # 1. Get configuration for the CURRENT stage
            cfg = self.curriculum_manager.get_current_config()
            self.config.update(cfg)
            stage = self.curriculum_manager._current_stage()

            # 2. Setup Environment (Swaps town if needed, keeps brain, CREATES brain if None)
            self.setup_training()

            # Now self.model is guaranteed to exist
            current_steps = self.model.num_timesteps
            
            print(f"\n📚 Stage: {stage.name} | Town: {stage.town}")
            print(f"   Difficulty: {stage.level.value}/4 | Vehicles: {stage.num_vehicles}")
            print(f"   Global Progress: {current_steps} / {total_timesteps}")

            # 3. Calculate how many steps we have left in the GLOBAL budget
            steps_left_global = total_timesteps - current_steps
            
            # Stop if we reached the global limit
            if steps_left_global <= 0:
                print("🏁 Global timestep budget reached.")
                break

            # Train in small blocks (25k) OR until global budget runs out
            steps_to_train = min(stage_block_steps, steps_left_global)

            # 4. Define Callbacks
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            curriculum_cb = CurriculumCallback(self.curriculum_manager, self)
            validation_cb = ValidationCallback(self.validation_metrics, self.curriculum_manager)
            
            checkpoint_cb = CheckpointCallback(
                save_freq=25_000,
                save_path=f"./experiments/models/checkpoints_{stage.town}_{timestamp}/",
                name_prefix="waymo_driver",
                verbose=1,
            )

            cb_list = CallbackList([curriculum_cb, validation_cb, checkpoint_cb])

            # 5. Train
            # reset_num_timesteps=False ensures the Global Counter keeps going up
            self.model.learn(
                total_timesteps=steps_to_train,
                callback=cb_list,
                tb_log_name=f"waymo_curriculum_{stage.town}_{timestamp}",
                reset_num_timesteps=False, 
                progress_bar=True,
            )
            
            # Loop repeats. 
            # If CurriculumCallback stopped learn() early (stage complete), 
            # we loop back, get new config, and start next stage immediately.

        print("\n🎉 Curriculum completed or timesteps exhausted.")

        # Save final model if it exists
        if self.model:
            final_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = f"./experiments/models/final_model_{final_stamp}/"
            os.makedirs(final_path, exist_ok=True)
            self.model.save(f"{final_path}/final_model")

        # Validation outputs
        print("\nGenerating validation reports...")
        self.validation_metrics.plot_collision_avoidance_validation()
        self.validation_metrics.plot_deadlock_freedom_validation()
        self.validation_metrics.plot_training_progress()
        self.validation_metrics.plot_waymo_safety_metrics()
        self.validation_metrics.generate_validation_report()
        self.validation_metrics.export_to_csv()

        return self.model

    # ---------------------------
    # Non-curriculum single-run fallback
    # ---------------------------
    def train_agents(self, total_timesteps=1_000_000):
        """If you ever want single-town training without curriculum."""
        if not self.eval_env:
            self.setup_evaluation()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validation_cb = ValidationCallback(self.validation_metrics, self.curriculum_manager)

        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=f"./experiments/models/checkpoints_{timestamp}/",
            name_prefix="stable_model",
            verbose=1
        )

        cb_list = CallbackList([validation_cb, checkpoint_cb])

        print(f"🚀 Starting Training (no curriculum): {total_timesteps} steps")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=cb_list,
            tb_log_name=f"waymo_training_{timestamp}",
            progress_bar=True
        )

        final_model_path = f"./experiments/models/final_model_{timestamp}/"
        os.makedirs(final_model_path, exist_ok=True)
        self.model.save(f"{final_model_path}/final_model")

        return self.model
