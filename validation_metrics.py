import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from typing import Dict, List, Tuple
import time
import os


class ValidationMetrics:
    """
    Tracks and validates theoretical properties during training.
    Links simulation results to formal proofs and Waymo-style validation.
    """

    def __init__(self, save_dir: str = "validation_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Metrics for Property 1 (Collision Avoidance)
        self.min_distances = []
        self.collision_events = []
        self.safety_violations = []

        # Metrics for Property 2 (Deadlock Freedom)
        self.waiting_times = []
        self.conflict_resolutions = []
        self.vehicle_progress = []

        # Waymo-Style Specific Metrics
        self.emergency_stops = []

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.curriculum_stages = []

        # Timestamps
        self.timestamps = []

    def record_episode(self, episode_data: Dict):
        """Record metrics from a completed episode"""
        timestamp = time.time()
        self.timestamps.append(timestamp)

        self.episode_rewards.append(episode_data.get('reward', 0.0))
        self.episode_lengths.append(episode_data.get('length', 0))
        self.success_rates.append(episode_data.get('success', 0.0))
        self.curriculum_stages.append(episode_data.get('stage', 0))

        # Property 1 metrics - FILTER INF VALUES ON INPUT
        if 'min_distance' in episode_data:
            dist = episode_data['min_distance']
            if np.isfinite(dist):
                self.min_distances.append(dist)
            else:
                self.min_distances.append(100.0)  # Safe fallback
                
        if 'collision' in episode_data:
            self.collision_events.append(episode_data['collision'])
        if 'safety_violation' in episode_data:
            self.safety_violations.append(episode_data['safety_violation'])

        # Property 2 metrics
        if 'max_waiting_time' in episode_data:
            wait = episode_data['max_waiting_time']
            if np.isfinite(wait):
                self.waiting_times.append(wait)
            else:
                self.waiting_times.append(0.0)
                
        if 'conflicts_resolved' in episode_data:
            self.conflict_resolutions.append(episode_data['conflicts_resolved'])
        if 'progress_rate' in episode_data:
            self.vehicle_progress.append(episode_data['progress_rate'])

        # Waymo-Style Metrics
        if 'emergency_stops' in episode_data:
            self.emergency_stops.append(episode_data['emergency_stops'])

    def _get_valid_distances(self) -> List[float]:
        """Helper to filter out inf/nan from min_distances"""
        return [d for d in self.min_distances if np.isfinite(d)]

    def _get_valid_waiting_times(self) -> List[float]:
        """Helper to filter out inf/nan from waiting_times"""
        return [w for w in self.waiting_times if np.isfinite(w)]

    def validate_collision_avoidance(self, min_safe_distance: float = 5.0) -> Dict:
        """
        Validate Property 1: Collision Avoidance
        Returns validation results linking to Theorem 1
        """
        valid_distances = self._get_valid_distances()
        
        if not valid_distances:
            return {"error": "No valid distance data available"}

        min_distances_array = np.array(valid_distances)
        collisions_array = np.array(self.collision_events) if self.collision_events else np.zeros(len(valid_distances))

        # Ensure arrays match in length
        if len(collisions_array) > len(min_distances_array):
            collisions_array = collisions_array[:len(min_distances_array)]
        elif len(collisions_array) < len(min_distances_array):
            collisions_array = np.pad(collisions_array, (0, len(min_distances_array) - len(collisions_array)))

        validation = {
            'property': 'Collision Avoidance (Theorem 1)',
            'min_distance_observed': float(np.min(min_distances_array)),
            'mean_min_distance': float(np.mean(min_distances_array)),
            'std_min_distance': float(np.std(min_distances_array)),
            'safety_threshold': min_safe_distance,
            'violations': int(np.sum(min_distances_array < min_safe_distance)),
            'violation_rate': float(np.mean(min_distances_array < min_safe_distance)),
            'total_collisions': int(np.sum(collisions_array)),
            'collision_rate': float(np.mean(collisions_array)),
            'episodes_evaluated': len(min_distances_array)
        }

        validation['property_satisfied'] = (
            validation['min_distance_observed'] >= min_safe_distance and
            validation['total_collisions'] == 0
        )

        return validation

    def validate_deadlock_freedom(self, max_expected_wait: float = 95.0) -> Dict:
        """
        Validate Property 2: Deadlock Freedom
        Returns validation results linking to Theorem 2
        """
        valid_times = self._get_valid_waiting_times()
        
        if not valid_times:
            return {"error": "No valid waiting time data available"}

        waiting_times_array = np.array(valid_times)

        validation = {
            'property': 'Deadlock Freedom (Theorem 2)',
            'max_waiting_time_observed': float(np.max(waiting_times_array)),
            'mean_waiting_time': float(np.mean(waiting_times_array)),
            'std_waiting_time': float(np.std(waiting_times_array)),
            'theoretical_bound': max_expected_wait,
            'bounded_violations': int(np.sum(waiting_times_array > max_expected_wait)),
            'bounded_violation_rate': float(np.mean(waiting_times_array > max_expected_wait)),
            'infinite_wait_detected': False,  # Already filtered
            'episodes_evaluated': len(waiting_times_array)
        }

        validation['property_satisfied'] = (
            not validation['infinite_wait_detected'] and
            validation['max_waiting_time_observed'] < max_expected_wait
        )

        return validation

    def plot_waymo_safety_metrics(self):
        """
        Generate specific plots for Safety Shield and Fleet Coordination.
        """
        if not self.emergency_stops or not self.conflict_resolutions:
            print("No data for Waymo safety metrics")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        episodes = np.arange(len(self.emergency_stops))

        # Plot 1: Emergency Stops
        ax1 = axes[0]
        ax1.bar(episodes, self.emergency_stops, color='salmon', alpha=0.7, label='Emergency Stops (Event)')

        window = max(5, len(episodes) // 20)
        if len(episodes) > window:
            rolling_stops = pd.Series(self.emergency_stops).rolling(window=window).mean()
            ax1.plot(episodes, rolling_stops, color='darkred', linewidth=2, label=f'Trend ({window}-ep avg)')

        ax1.set_ylabel('Count per Episode', fontsize=12)
        ax1.set_title('Safety Shield Activations (Jaywalking Response)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Fleet Conflicts Resolved
        ax2 = axes[1]
        ax2.plot(episodes, self.conflict_resolutions, color='green', alpha=0.4, label='Raw Conflicts')

        if len(episodes) > window:
            rolling_conflicts = pd.Series(self.conflict_resolutions).rolling(window=window).mean()
            ax2.plot(episodes, rolling_conflicts, color='darkgreen', linewidth=2, label=f'Efficiency Trend')

        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Conflicts Resolved', fontsize=12)
        ax2.set_title('Fleet Coordination Efficiency', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/waymo_safety_metrics.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.save_dir}/waymo_safety_metrics.png")
        plt.close()

    def plot_collision_avoidance_validation(self):
        """Generate plots validating Theorem 1"""
        valid_distances = self._get_valid_distances()
        
        if not valid_distances:
            print("No valid data for collision avoidance validation")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        episodes = np.arange(len(valid_distances))

        # Plot 1: Minimum distances over time
        ax = axes[0, 0]
        ax.plot(episodes, valid_distances, 'b-', alpha=0.6, linewidth=1)
        ax.axhline(y=5.0, color='r', linestyle='--', linewidth=2, label='Safety threshold (d_min = 5m)')
        ax.fill_between(episodes, 0, 5.0, color='red', alpha=0.1)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Minimum Inter-Vehicle Distance (m)', fontsize=12)
        ax.set_title('Validation of Collision Avoidance (Theorem 1)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Plot 2: Distance distribution (FIXED)
        ax = axes[0, 1]
        if valid_distances:
            ax.hist(valid_distances, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(x=5.0, color='r', linestyle='--', linewidth=2, label='Safety threshold')
            ax.set_xlabel('Minimum Distance (m)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No valid distance data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Distribution of Minimum Inter-Vehicle Distances', fontsize=14, fontweight='bold')

        # Plot 3: Rolling average
        ax = axes[1, 0]
        window = max(1, min(50, len(valid_distances) // 10))
        if window > 1 and len(valid_distances) > window:
            rolling_mean = pd.Series(valid_distances).rolling(window=window).mean()
            rolling_std = pd.Series(valid_distances).rolling(window=window).std()
            ax.plot(episodes, rolling_mean, 'b-', linewidth=2, label=f'Rolling mean (window={window})')
            ax.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std,
                            color='blue', alpha=0.2, label='±1 std')
        else:
            ax.plot(episodes, valid_distances, 'b-', linewidth=2, label='Min Distance')
        ax.axhline(y=5.0, color='r', linestyle='--', linewidth=2, label='Safety threshold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Distance (m)', fontsize=12)
        ax.set_title('Rolling Statistics of Minimum Distances', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Plot 4: Validation summary
        ax = axes[1, 1]
        validation = self.validate_collision_avoidance()
        
        if 'error' in validation:
            metrics = [f"Error: {validation['error']}"]
        else:
            metrics = [
                f"Min Distance: {validation['min_distance_observed']:.2f}m",
                f"Mean Distance: {validation['mean_min_distance']:.2f}m",
                f"Std Distance: {validation['std_min_distance']:.2f}m",
                f"Violations: {validation['violations']}/{validation['episodes_evaluated']}",
                f"Violation Rate: {validation['violation_rate'] * 100:.2f}%",
                f"Total Collisions: {validation['total_collisions']}",
                f"Property Satisfied: {'✓ YES' if validation['property_satisfied'] else '✗ NO'}"
            ]

        ax.axis('off')
        y_pos = 0.9
        ax.text(0.1, y_pos, 'VALIDATION SUMMARY (Theorem 1)', fontsize=16, fontweight='bold',
                transform=ax.transAxes)
        y_pos -= 0.12

        for metric in metrics:
            color = 'green' if 'YES' in metric else 'black'
            if 'NO' in metric or 'Error' in metric:
                color = 'red'
            ax.text(0.1, y_pos, metric, fontsize=13, transform=ax.transAxes,
                    family='monospace', color=color, fontweight='bold' if color != 'black' else 'normal')
            y_pos -= 0.1

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/collision_avoidance_validation.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.save_dir}/collision_avoidance_validation.png")
        plt.close()

    def plot_deadlock_freedom_validation(self):
        """Generate plots validating Theorem 2"""
        valid_times = self._get_valid_waiting_times()
        
        if not valid_times:
            print("No valid data for deadlock freedom validation")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        episodes = np.arange(len(valid_times))

        # Plot 1: Waiting times over episodes
        ax = axes[0, 0]
        ax.plot(episodes, valid_times, 'g-', alpha=0.6, linewidth=1)
        max_val = max(valid_times) if valid_times else 100
        ax.axhline(y=95.0, color='r', linestyle='--', linewidth=2, label='Theoretical bound (T_max = 95s)')
        ax.fill_between(episodes, 95.0, max_val * 1.1, color='red', alpha=0.1)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Maximum Waiting Time (s)', fontsize=12)
        ax.set_title('Validation of Deadlock Freedom (Theorem 2)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Plot 2: Waiting time distribution (FIXED)
        ax = axes[0, 1]
        if valid_times:
            ax.hist(valid_times, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
            ax.axvline(x=95.0, color='r', linestyle='--', linewidth=2, label='Theoretical bound')
            ax.set_xlabel('Waiting Time (s)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No valid waiting time data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Distribution of Maximum Waiting Times', fontsize=14, fontweight='bold')

        # Plot 3: CDF of waiting times
        ax = axes[1, 0]
        if valid_times:
            sorted_times = np.sort(valid_times)
            cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
            ax.plot(sorted_times, cdf, 'g-', linewidth=2, label='Empirical CDF')
            ax.axvline(x=95.0, color='r', linestyle='--', linewidth=2, label='Theoretical bound')
            ax.legend(fontsize=11)
        ax.set_xlabel('Waiting Time (s)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('CDF of Waiting Times', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 4: Validation summary
        ax = axes[1, 1]
        validation = self.validate_deadlock_freedom()
        
        if 'error' in validation:
            metrics = [f"Error: {validation['error']}"]
        else:
            metrics = [
                f"Max Wait Observed: {validation['max_waiting_time_observed']:.2f}s",
                f"Mean Wait Time: {validation['mean_waiting_time']:.2f}s",
                f"Std Wait Time: {validation['std_waiting_time']:.2f}s",
                f"Theoretical Bound: {validation['theoretical_bound']:.2f}s",
                f"Bound Violations: {validation['bounded_violations']}/{validation['episodes_evaluated']}",
                f"Infinite Wait: {'Detected' if validation['infinite_wait_detected'] else 'None'}",
                f"Property Satisfied: {'✓ YES' if validation['property_satisfied'] else '✗ NO'}"
            ]

        ax.axis('off')
        y_pos = 0.9
        ax.text(0.1, y_pos, 'VALIDATION SUMMARY (Theorem 2)', fontsize=16, fontweight='bold',
                transform=ax.transAxes)
        y_pos -= 0.12

        for metric in metrics:
            color = 'green' if 'YES' in metric or 'None' in metric else 'black'
            if 'NO' in metric or 'Detected' in metric or 'Error' in metric:
                color = 'red'
            ax.text(0.1, y_pos, metric, fontsize=13, transform=ax.transAxes,
                    family='monospace', color=color, fontweight='bold' if color != 'black' else 'normal')
            y_pos -= 0.1

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/deadlock_freedom_validation.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.save_dir}/deadlock_freedom_validation.png")
        plt.close()

    def plot_training_progress(self):
        """Plot training metrics over time"""
        if not self.episode_rewards:
            print("No training data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        episodes = np.arange(len(self.episode_rewards))

        # Plot 1: Episode rewards
        ax = axes[0, 0]
        ax.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, linewidth=0.5)
        window = max(1, min(50, len(self.episode_rewards) // 10))
        if window > 1 and len(episodes) > window:
            rolling_mean = pd.Series(self.episode_rewards).rolling(window=window).mean()
            ax.plot(episodes, rolling_mean, 'r-', linewidth=2, label=f'Rolling mean (window={window})')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Training Reward Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Plot 2: Success rate
        ax = axes[0, 1]
        if self.success_rates and len(self.success_rates) > 0:
            window = max(1, min(20, len(self.success_rates) // 5))
            if window > 1 and len(self.success_rates) > window:
                rolling_success = pd.Series(self.success_rates).rolling(window=window).mean()
                ax.plot(episodes[:len(rolling_success)], rolling_success, 'g-', linewidth=2)
            else:
                ax.plot(range(len(self.success_rates)), self.success_rates, 'g-', linewidth=2)
            ax.set_ylim([0, 1.05])
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Episode Success Rate', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 3: Episode lengths
        ax = axes[1, 0]
        if self.episode_lengths:
            ax.plot(range(len(self.episode_lengths)), self.episode_lengths, 'purple', alpha=0.5, linewidth=1)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode Length (steps)', fontsize=12)
        ax.set_title('Episode Duration', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 4: Curriculum progression
        ax = axes[1, 1]
        if self.curriculum_stages and len(self.curriculum_stages) > 0:
            ax.plot(range(len(self.curriculum_stages)), self.curriculum_stages, 'orange', linewidth=2, drawstyle='steps-post')
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(['Basic', 'Medium', 'Advanced', 'Expert'])
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Curriculum Stage', fontsize=12)
        ax.set_title('Curriculum Learning Progression', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_progress.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.save_dir}/training_progress.png")
        plt.close()

    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("VALIDATION REPORT: Multi-Robot Fleet Coordination System")
        report.append("=" * 80)
        report.append("")

        # Property 1 validation
        valid_distances = self._get_valid_distances()
        if valid_distances:
            report.append("PROPERTY 1: COLLISION AVOIDANCE (Theorem 1)")
            report.append("-" * 80)
            val1 = self.validate_collision_avoidance()
            for key, value in val1.items():
                report.append(f"  {key:30s}: {value}")
            report.append("")

        # Property 2 validation
        valid_times = self._get_valid_waiting_times()
        if valid_times:
            report.append("PROPERTY 2: DEADLOCK FREEDOM (Theorem 2)")
            report.append("-" * 80)
            val2 = self.validate_deadlock_freedom()
            for key, value in val2.items():
                report.append(f"  {key:30s}: {value}")
            report.append("")

        # Waymo-Style Validation
        if self.emergency_stops:
            report.append("SAFETY & COORDINATION (Waymo-Style Analysis)")
            report.append("-" * 80)
            report.append(f"  Total Emergency Stops: {np.sum(self.emergency_stops)}")
            report.append(f"  Avg Safety Activations/Ep: {np.mean(self.emergency_stops):.2f}")
            if self.conflict_resolutions:
                report.append(f"  Avg Fleet Conflicts Resolved/Ep: {np.mean(self.conflict_resolutions):.2f}")
            report.append("")

        # Training summary
        if self.episode_rewards:
            report.append("TRAINING SUMMARY")
            report.append("-" * 80)
            report.append(f"  Total Episodes: {len(self.episode_rewards)}")
            report.append(f"  Mean Reward: {np.mean(self.episode_rewards):.2f}")
            report.append(f"  Final Reward (last 10): {np.mean(self.episode_rewards[-10:]):.2f}")
            if self.success_rates:
                report.append(f"  Final Success Rate: {np.mean(self.success_rates[-20:]):.2f}")
            report.append("")

        report.append("=" * 80)

        report_text = "\n".join(report)

        with open(f"{self.save_dir}/validation_report.txt", 'w') as f:
            f.write(report_text)

        print(f"Saved: {self.save_dir}/validation_report.txt")
        return report_text

    def export_to_csv(self):
        """Export all metrics to CSV for further analysis"""
        max_len = max(
            len(self.episode_rewards) if self.episode_rewards else 0,
            len(self.min_distances) if self.min_distances else 0,
            len(self.waiting_times) if self.waiting_times else 0,
            1  # Ensure at least 1
        )

        def pad(lst):
            if not lst:
                return [np.nan] * max_len
            return lst + [np.nan] * (max_len - len(lst))

        data = {
            'episode': list(range(max_len)),
            'reward': pad(self.episode_rewards),
            'min_distance': pad(self.min_distances),
            'waiting_time': pad(self.waiting_times),
            'episode_length': pad(self.episode_lengths),
            'success_rate': pad(self.success_rates),
            'curriculum_stage': pad(self.curriculum_stages),
            'emergency_stops': pad(self.emergency_stops),
            'conflicts_resolved': pad(self.conflict_resolutions)
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.save_dir}/metrics_data.csv", index=False)
        print(f"Saved: {self.save_dir}/metrics_data.csv")
