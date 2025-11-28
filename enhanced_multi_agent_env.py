# # import gymnasium as gym
# # from gymnasium import spaces
# # import carla
# # import numpy as np
# # import random
# # import math
# # import time
# # from collections import defaultdict
# # from typing import Tuple, Optional, List, Dict
# #
# # # ==============================================================================
# # # INTEGRATION: Import the "Brain" components
# # # ==============================================================================
# # try:
# #     from utils.waypoint_manager.waypoint_system import WaymoWaypointManager
# #     try:
# #         from utils.fleet_coordinator.fleet_manager import FleetCoordinator
# #     except ImportError:
# #         from utils.fleet_coordinator.fleet_coordinator import FleetCoordinator
# # except ImportError:
# #     print("CRITICAL WARNING: Could not import Waypoint/Fleet managers.")
# #
# #     class WaymoWaypointManager:
# #         def __init__(self, *args, **kwargs): pass
# #         def assign_route(self, *args, **kwargs): return True
# #         def update_route_progress(self, *args, **kwargs):
# #             return {'progress': 0.0, 'dist_delta': 0.0, 'dist_to_next_wp': 0.0}
# #         def get_route_status(self, *args, **kwargs):
# #             return {'destination_reached': False}
# #
# #     class FleetCoordinator:
# #         def __init__(self, *args, **kwargs): pass
# #         def start_coordination(self): pass
# #         def stop_coordination(self): pass
# #         def get_pending_commands(self): return []
# #         def update_vehicle_state(self, *args, **kwargs): pass
# #         def get_coordination_statistics(self):
# #             return {'recent_conflicts': 0, 'coordination_efficiency': 0.0}
# #         def get_all_states(self): return {}
# #
# #
# # # ==============================================================================
# # # CONFIGURATION CONSTANTS (TRAINABLE SAFE V1)
# # # ==============================================================================
# #
# # # Reward Scaling
# # REWARD_ALPHA = 10         # Route progress scaling factor
# # PENALTY_BETA = 0.02         # Smoothness penalty scaling (smaller early)
# # P_COLLISION_EGO = -1.0      # TEMP LOW for early training (raise later)
# # P_COLLISION_STATIC = -0.2
# # P_SPACING_REWARD = 0.02     # small alive spacing bonus
# # P_TAILGATING_PENALTY = -0.2
# # P_STUCK_CUMULATIVE = -0.5   # penalty once vehicle stuck > 10s (NO termination)
# #
# # # Safety Thresholds
# # SHIELD_DISTANCE_BUFFER = 2.0
# # CRITICAL_PROXIMITY_THRESHOLD = 4.0
# # SAFE_FOLLOWING_DIST = 10.0
# #
# # # Physics
# # MAX_SPEED_MPS = 10.0
# # MAX_DECELERATION = 8.0
# # REACTION_TIME = 0.5
# #
# #
# # class StableMultiAgentCarlaEnv(gym.Env):
# #     """
# #     Stable Multi-Agent CARLA Env (v1 trainable).
# #     Fixes spawn instability, reward explosions, early stuck terminations,
# #     and safety shield bugs.
# #     """
# #
# #     metadata = {"render_modes": []}
# #
# #     # Expose thresholds as class attrs
# #     CRITICAL_PROXIMITY_THRESHOLD = CRITICAL_PROXIMITY_THRESHOLD
# #     SAFE_FOLLOWING_DIST = SAFE_FOLLOWING_DIST
# #     SHIELD_DISTANCE_BUFFER = SHIELD_DISTANCE_BUFFER
# #
# #     def __init__(
# #         self,
# #         num_vehicles: int = 5,
# #         town: str = "Town03",
# #         host: str = "localhost",
# #         port: int = 2000,
# #         tm_port: int = 8000,
# #         max_episode_steps: int = 1000,
# #         weather: str = "ClearNoon",
# #         pedestrian_count: int = 0,
# #         timeout_threshold: float = 300.0
# #     ):
# #         super().__init__()
# #
# #         self.num_vehicles = num_vehicles
# #         self.town = town
# #         self.host = host
# #         self.port = port
# #         self.tm_port = tm_port
# #         self.max_episode_steps = max_episode_steps
# #         self.weather = weather
# #         self.pedestrian_count = pedestrian_count
# #         self.timeout_threshold = timeout_threshold
# #
# #         # Action space (Throttle, Steer, Brake) * N
# #         self.action_space = spaces.Box(
# #             low=np.array([0.0, -1.0, 0.0] * num_vehicles, dtype=np.float32),
# #             high=np.array([1.0, 1.0, 1.0] * num_vehicles, dtype=np.float32),
# #             dtype=np.float32,
# #         )
# #
# #         # Observation space: 10 features per vehicle
# #         obs_dim = 10 * num_vehicles
# #         self.observation_space = spaces.Box(
# #             low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
# #         )
# #
# #         # CARLA handles
# #         self.client = None
# #         self.world = None
# #         self.tm = None
# #
# #         # Actors & sensors
# #         self.vehicles: List[carla.Vehicle] = []
# #         self.pedestrians: List[carla.Walker] = []
# #         self.sensors: List[carla.Sensor] = []
# #         self.spawn_points: List[carla.Transform] = []
# #
# #         # Managers
# #         self.waypoint_manager: Optional[WaymoWaypointManager] = None
# #         self.fleet_coordinator: Optional[FleetCoordinator] = None
# #
# #         # Tracking
# #         self.collision_histories = defaultdict(list)
# #         self._stuck_counter: Dict[int, float] = {}
# #         self._prev_actions: np.ndarray = np.zeros((num_vehicles, 3), dtype=np.float32)
# #         self._vehicle_neighbor_cache = {}
# #
# #         # Metrics
# #         self.episode_step = 0
# #         self.episode_start_time = 0
# #         self.emergency_stops = 0
# #         self.total_ego_collisions = 0
# #         self.shield_activations = 0
# #         self.current_min_distance = float('inf')
# #
# #     # ==============================================================================
# #     # RESET / STEP
# #     # ==============================================================================
# #
# #     def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
# #         super().reset(seed=seed)
# #         if seed is not None:
# #             random.seed(seed)
# #             np.random.seed(seed)
# #
# #         self._cleanup()
# #         self._setup_carla()
# #
# #         # init managers
# #         self.waypoint_manager = WaymoWaypointManager(self.world, max_vehicles=self.num_vehicles)
# #         self.fleet_coordinator = FleetCoordinator(self.waypoint_manager, max_vehicles=self.num_vehicles)
# #         self.fleet_coordinator.start_coordination()
# #
# #         # clear trackers
# #         self.collision_histories.clear()
# #         self._stuck_counter.clear()
# #         self._prev_actions = np.zeros((self.num_vehicles, 3), dtype=np.float32)
# #         self._vehicle_neighbor_cache = {}
# #
# #         # safer spawn (no teleport)
# #         spawn_attempts = 0
# #         min_required = min(self.num_vehicles, 3)
# #
# #         while len(self.vehicles) < min_required and spawn_attempts < 5:
# #             self._spawn_vehicles_simple()
# #             for _ in range(10):
# #                 self.world.tick()
# #
# #             self.vehicles = [v for v in self.vehicles if v.is_alive]
# #
# #             if len(self.vehicles) < min_required:
# #                 print(f"Warning: Low vehicle count ({len(self.vehicles)}). Retrying spawn...")
# #                 self._cleanup_vehicles_only()
# #                 spawn_attempts += 1
# #
# #         if self.pedestrian_count > 0:
# #             self._spawn_pedestrians()
# #
# #         for v in self.vehicles:
# #             self.waypoint_manager.assign_route(v.id)
# #
# #         # reset metrics
# #         self.episode_step = 0
# #         self.episode_start_time = time.time()
# #         self.emergency_stops = 0
# #         self.total_ego_collisions = 0
# #         self.shield_activations = 0
# #         self.current_min_distance = float('inf')
# #
# #         # build neighbor cache now
# #         self._update_neighbor_cache()
# #         self._update_global_min_distance()
# #
# #         return self._get_observations(), {}
# #
# #     def step(self, action: np.ndarray):
# #         self.episode_step += 1
# #
# #         # prune dead actors
# #         self.vehicles = [v for v in self.vehicles if v.is_alive]
# #         if not self.vehicles:
# #             return self._get_observations(), 0.0, True, False, {"error": "All vehicles destroyed"}
# #
# #         action = action.reshape(self.num_vehicles, 3)
# #         current_actions = action.copy()
# #
# #         # emergency jaywalk event
# #         if self.pedestrian_count > 0 and random.random() < 0.02:
# #             self._trigger_emergency_jaywalk()
# #
# #         # fleet commands
# #         pending_cmds = self.fleet_coordinator.get_pending_commands()
# #         yielding_vehicles = set()
# #         for cmd_type, vid in pending_cmds:
# #             if cmd_type in ("REASSIGN_ROUTE", "OPTIMIZE_ROUTE"):
# #                 self.waypoint_manager.assign_route(vid)
# #             elif cmd_type == "APPLY_YIELD":
# #                 yielding_vehicles.add(vid)
# #
# #         # neighbors + min dist
# #         self._update_neighbor_cache()
# #         self._update_global_min_distance()
# #
# #         # stuck penalties (no termination)
# #         stuck_penalties = self._handle_stuck_vehicles()
# #
# #         # apply controls
# #         for idx, vehicle in enumerate(self.vehicles):
# #             if idx >= len(action):
# #                 break
# #             vid = vehicle.id
# #
# #             throttle, steer, brake = action[idx]
# #
# #             # telemetry update
# #             t = vehicle.get_transform()
# #             v = vehicle.get_velocity()
# #             route_info = self.waypoint_manager.update_route_progress(vid, t.location)
# #
# #             self.fleet_coordinator.update_vehicle_state(
# #                 vehicle_id=vid,
# #                 location=t.location,
# #                 velocity=v,
# #                 heading=t.rotation.yaw,
# #                 route_progress=route_info.get('progress', 0.0)
# #             )
# #
# #             steer += np.random.normal(0, 0.02)
# #
# #             # targeted shield
# #             shield_active, safe_control = self._check_targeted_shield(vehicle)
# #
# #             if shield_active:
# #                 final_control = safe_control
# #                 self.shield_activations += 1
# #             else:
# #                 final_control = carla.VehicleControl(
# #                     throttle=float(np.clip(throttle, 0.0, 1.0)),
# #                     steer=float(np.clip(steer, -1.0, 1.0)),
# #                     brake=float(np.clip(brake, 0.0, 1.0))
# #                 )
# #
# #             # yield/ped override (below shield)
# #             if not shield_active:
# #                 if vid in yielding_vehicles:
# #                     final_control.throttle = 0.0
# #                     final_control.brake = 0.5
# #                 elif self._check_pedestrian_hazard(vehicle):
# #                     final_control.throttle = 0.0
# #                     final_control.brake = 1.0
# #                     self.emergency_stops += 1
# #
# #             try:
# #                 vehicle.apply_control(final_control)
# #             except:
# #                 pass
# #
# #         self._prev_actions = current_actions
# #
# #         # tick sim
# #         try:
# #             self.world.tick()
# #         except Exception as e:
# #             print(f"Tick error: {e}")
# #             return self._get_observations(), 0.0, True, True, {}
# #
# #         # reward + info
# #         obs = self._get_observations()
# #         reward = self._calculate_rewards(current_actions, stuck_penalties)
# #         fleet_stats = self.fleet_coordinator.get_coordination_statistics()
# #
# #         # termination logic (only real safety)
# #         terminated = False
# #         truncated = False
# #
# #         if self.total_ego_collisions > 0:
# #             terminated = True
# #             print(f"DEBUG: Terminated - Ego Collision ({self.total_ego_collisions})")
# #
# #         if self.current_min_distance < self.CRITICAL_PROXIMITY_THRESHOLD and self.current_min_distance > 0.1:
# #             terminated = True
# #             print(f"DEBUG: Terminated - Critical Proximity ({self.current_min_distance:.2f}m)")
# #
# #         if time.time() - self.episode_start_time > self.timeout_threshold:
# #             truncated = True
# #         if self.episode_step >= self.max_episode_steps:
# #             truncated = True
# #
# #         if len(self.vehicles) < max(1, self.num_vehicles // 3):
# #             terminated = True
# #
# #         info = {
# #             "vehicles": len(self.vehicles),
# #             "episode_step": self.episode_step,
# #             "min_distance": self.current_min_distance,
# #             "collisions": self.total_ego_collisions,
# #             "emergency_stops": self.emergency_stops,
# #             "shield_activations": self.shield_activations,
# #             "conflicts_resolved": fleet_stats.get('recent_conflicts', 0),
# #             "coordination_efficiency": fleet_stats.get('coordination_efficiency', 0.0)
# #         }
# #
# #         return obs, reward, terminated, truncated, info
# #
# #     # ==============================================================================
# #     # SAFETY / NEIGHBORS
# #     # ==============================================================================
# #
# #     def _update_neighbor_cache(self):
# #         """Nearest neighbor per vehicle. Falls back to raw CARLA if fleet states empty."""
# #         self._vehicle_neighbor_cache = {}
# #
# #         all_states = {}
# #         try:
# #             all_states = self.fleet_coordinator.get_all_states()
# #         except:
# #             all_states = {}
# #
# #         # fallback: build from CARLA directly
# #         if not all_states:
# #             alive = [v for v in self.vehicles if v.is_alive]
# #             for ego in alive:
# #                 min_dist = 100.0
# #                 best_neighbor = None
# #                 for other in alive:
# #                     if other.id == ego.id:
# #                         continue
# #                     d = ego.get_location().distance(other.get_location())
# #                     if d < min_dist:
# #                         min_dist = d
# #                         best_neighbor = other
# #                 self._vehicle_neighbor_cache[ego.id] = {
# #                     "nnd": min_dist,
# #                     "neighbor_actor": best_neighbor,
# #                     "neighbor_state": None
# #                 }
# #             return
# #
# #         # main path: use coordinator states
# #         for ego_v in self.vehicles:
# #             ego_id = ego_v.id
# #             if ego_id not in all_states:
# #                 continue
# #
# #             ego_state = all_states[ego_id]
# #             ego_loc = ego_state.location
# #             if not hasattr(ego_loc, "x"):
# #                 continue
# #
# #             min_dist = 100.0
# #             best_neighbor = None
# #
# #             for other_id, other_state in all_states.items():
# #                 if other_id == ego_id:
# #                     continue
# #                 other_loc = other_state.location
# #                 if not hasattr(other_loc, "x"):
# #                     continue
# #
# #                 dist = math.sqrt(
# #                     (ego_loc.x - other_loc.x) ** 2 +
# #                     (ego_loc.y - other_loc.y) ** 2 +
# #                     (ego_loc.z - other_loc.z) ** 2
# #                 )
# #
# #                 if dist < min_dist:
# #                     min_dist = dist
# #                     best_neighbor = other_state
# #
# #             self._vehicle_neighbor_cache[ego_id] = {
# #                 "nnd": min_dist,
# #                 "neighbor_state": best_neighbor,
# #                 "neighbor_actor": None
# #             }
# #
# #     def _update_global_min_distance(self):
# #         if not self._vehicle_neighbor_cache:
# #             self.current_min_distance = 100.0
# #             return
# #         try:
# #             self.current_min_distance = min(d["nnd"] for d in self._vehicle_neighbor_cache.values())
# #         except:
# #             self.current_min_distance = 100.0
# #
# #     def _check_targeted_shield(self, vehicle: carla.Vehicle) -> Tuple[bool, carla.VehicleControl]:
# #         """Targeted shield for this vehicle only."""
# #         vid = vehicle.id
# #         neighbor_data = self._vehicle_neighbor_cache.get(vid)
# #
# #         safe_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
# #
# #         if not neighbor_data:
# #             return False, safe_control
# #
# #         nnd = neighbor_data["nnd"]
# #
# #         ego_vel = vehicle.get_velocity()
# #         ego_speed = math.hypot(ego_vel.x, ego_vel.y)
# #
# #         # assume neighbor speed ~ ego speed if unknown
# #         rel_speed = ego_speed
# #
# #         braking_dist = (rel_speed ** 2) / (2 * MAX_DECELERATION) if rel_speed > 0 else 0.0
# #         d_safe = (REACTION_TIME * ego_speed) + braking_dist + self.SHIELD_DISTANCE_BUFFER
# #
# #         if nnd < d_safe:
# #             return True, safe_control
# #
# #         return False, safe_control
# #
# #     def _check_pedestrian_hazard(self, vehicle) -> bool:
# #         """Cone-based pedestrian hazard check."""
# #         v_loc = vehicle.get_location()
# #         v_fwd = vehicle.get_transform().get_forward_vector()
# #
# #         for ped in self.pedestrians:
# #             try:
# #                 actor = ped[0] if isinstance(ped, tuple) else ped
# #                 if not actor.is_alive:
# #                     continue
# #                 p_loc = actor.get_location()
# #             except:
# #                 continue
# #
# #             dist = v_loc.distance(p_loc)
# #             if dist < 12.0:
# #                 to_ped = p_loc - v_loc
# #                 length = to_ped.length()
# #                 if length > 0:
# #                     to_ped = to_ped / length
# #                     dot = v_fwd.dot(to_ped)
# #                     if dot > 0.5:
# #                         return True
# #         return False
# #
# #     # ==============================================================================
# #     # REWARD / OBS
# #     # ==============================================================================
# #
# #     # def _calculate_rewards(self, current_actions: np.ndarray, stuck_penalties: Dict[int, float]) -> float:
# #     #     total_reward = 0.0
# #     #
# #     #     # smoothness penalty
# #     #     if self._prev_actions.shape == current_actions.shape:
# #     #         delta = current_actions - self._prev_actions
# #     #         smoothness = -PENALTY_BETA * np.sum(delta[:, 0] ** 2 + delta[:, 1] ** 2)
# #     #         total_reward += smoothness
# #     #
# #     #     for v in self.vehicles:
# #     #         if not v.is_alive:
# #     #             continue
# #     #         vid = v.id
# #     #
# #     #         # speed reward
# #     #         vel = v.get_velocity()
# #     #         speed = math.hypot(vel.x, vel.y)
# #     #         r_speed = 0.1 *min(speed / MAX_SPEED_MPS, 1.0)
# #     #
# #     #         # route reward (CLIPPED!)
# #     #         route_info = self.waypoint_manager.update_route_progress(vid, v.get_location())
# #     #         dist_delta = route_info.get("dist_delta", 0.0)
# #     #         dist_delta = float(np.clip(dist_delta, -5.0, 5.0))
# #     #         r_route = dist_delta * REWARD_ALPHA
# #     #
# #     #         # spacing reward
# #     #         r_spacing = 0.0
# #     #         ndata = self._vehicle_neighbor_cache.get(vid)
# #     #         if ndata:
# #     #             nnd = ndata["nnd"]
# #     #             if nnd > SAFE_FOLLOWING_DIST:
# #     #                 r_spacing = P_SPACING_REWARD
# #     #             elif nnd < CRITICAL_PROXIMITY_THRESHOLD + 2.0:
# #     #                 r_spacing = P_TAILGATING_PENALTY
# #     #
# #     #         # collision penalty
# #     #         r_col = 0.0
# #     #         if self.collision_histories.get(vid):
# #     #             r_col = P_COLLISION_STATIC
# #     #             for event in self.collision_histories[vid]:
# #     #                 other = event.other_actor
# #     #                 if other.type_id.startswith("vehicle") or other.type_id.startswith("walker"):
# #     #                     r_col = P_COLLISION_EGO
# #     #                     break
# #     #
# #     #         # stuck penalty
# #     #         r_stuck = stuck_penalties.get(vid, 0.0)
# #     #
# #     #         total_reward += (r_speed + r_route + r_spacing + r_col + r_stuck)
# #     #
# #     #     return total_reward
# #     def _calculate_rewards(self, current_actions: np.ndarray, stuck_penalties: Dict[int, float]) -> float:
# #         total_reward = 0.0
# #
# #         # smoothness penalty
# #         if self._prev_actions.shape == current_actions.shape:
# #             delta = current_actions - self._prev_actions
# #             smoothness = -PENALTY_BETA * np.sum(delta[:, 0] ** 2 + delta[:, 1] ** 2)
# #             total_reward += smoothness
# #
# #         for v in self.vehicles:
# #             if not v.is_alive:
# #                 continue
# #             vid = v.id
# #
# #             # speed reward
# #             vel = v.get_velocity()
# #             speed = math.hypot(vel.x, vel.y)
# #             r_speed = min(speed / MAX_SPEED_MPS, 1.0)
# #
# #             # route reward (CLIPPED!)
# #             route_info = self.waypoint_manager.update_route_progress(vid, v.get_location())
# #             dist_delta = route_info.get("dist_delta", 0.0)
# #             dist_delta = float(np.clip(dist_delta, -5.0, 5.0))
# #             r_route = dist_delta * REWARD_ALPHA
# #
# #             # spacing reward
# #             r_spacing = 0.0
# #             ndata = self._vehicle_neighbor_cache.get(vid)
# #             if ndata:
# #                 nnd = ndata["nnd"]
# #                 if nnd > SAFE_FOLLOWING_DIST:
# #                     r_spacing = P_SPACING_REWARD
# #                 elif nnd < CRITICAL_PROXIMITY_THRESHOLD + 2.0:
# #                     r_spacing = P_TAILGATING_PENALTY
# #
# #             # collision penalty
# #             r_col = 0.0
# #             if self.collision_histories.get(vid):
# #                 r_col = P_COLLISION_STATIC
# #                 for event in self.collision_histories[vid]:
# #                     other = event.other_actor
# #                     if other.type_id.startswith("vehicle") or other.type_id.startswith("walker"):
# #                         r_col = P_COLLISION_EGO
# #                         break
# #
# #             # stuck penalty
# #             r_stuck = stuck_penalties.get(vid, 0.0)
# #
# #             total_reward += (r_speed + r_route + r_spacing + r_col + r_stuck)
# #
# #         return total_reward
# #
# #     def _get_observations(self):
# #         obs = []
# #         for v in self.vehicles:
# #             if v.is_alive:
# #                 try:
# #                     t = v.get_transform()
# #                     vel = v.get_velocity()
# #                     speed = math.hypot(vel.x, vel.y)
# #
# #                     # base 7
# #                     obs.extend([
# #                         t.location.x / 1000.0,
# #                         t.location.y / 1000.0,
# #                         t.rotation.yaw / 180.0,
# #                         vel.x / MAX_SPEED_MPS,
# #                         vel.y / MAX_SPEED_MPS,
# #                         speed / MAX_SPEED_MPS,
# #                     ])
# #
# #                     route_info = self.waypoint_manager.update_route_progress(v.id, t.location)
# #                     obs.append(route_info.get("dist_to_next_wp", 0.0) / 100.0)
# #
# #                     # neighbor 3
# #                     ndata = self._vehicle_neighbor_cache.get(v.id)
# #                     if ndata:
# #                         nnd = ndata["nnd"]
# #                         obs.extend([nnd / 100.0, 0.0, 0.0])  # relspeed/angle optional early
# #                     else:
# #                         obs.extend([1.0, 0.0, 0.0])
# #
# #                 except:
# #                     obs.extend([0.0] * 10)
# #             else:
# #                 obs.extend([0.0] * 10)
# #
# #         target_len = self.num_vehicles * 10
# #         if len(obs) < target_len:
# #             obs.extend([0.0] * (target_len - len(obs)))
# #
# #         return np.array(obs[:target_len], dtype=np.float32)
# #
# #     # ==============================================================================
# #     # SPAWN / STUCK / CLEANUP
# #     # ==============================================================================
# #
# #     def _spawn_vehicles_simple(self):
# #         """Simple stable spawn: one spawn point per vehicle."""
# #         bp_lib = self.world.get_blueprint_library().filter("vehicle.*")
# #         safe_bp_lib = [x for x in bp_lib if int(x.get_attribute("number_of_wheels")) == 4]
# #         collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
# #
# #         spawn_points = self.world.get_map().get_spawn_points()
# #         if not spawn_points:
# #             return
# #
# #         random.shuffle(spawn_points)
# #
# #         for i in range(self.num_vehicles):
# #             tf = spawn_points[i % len(spawn_points)]
# #             tf.location.z += 0.5
# #
# #             bp = random.choice(safe_bp_lib)
# #             if bp.has_attribute("color"):
# #                 bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
# #
# #             vehicle = self.world.try_spawn_actor(bp, tf)
# #             if vehicle:
# #                 vehicle.set_autopilot(False)
# #                 self.vehicles.append(vehicle)
# #
# #                 sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
# #                 sensor.listen(lambda event: self._on_collision(event))
# #                 self.sensors.append(sensor)
# #
# #                 self.world.tick()
# #
# #     def _handle_stuck_vehicles(self):
# #         penalties = {}
# #         delta_time = 0.05
# #
# #         for v in self.vehicles:
# #             if not v.is_alive:
# #                 continue
# #
# #             vid = v.id
# #             if vid not in self._stuck_counter:
# #                 self._stuck_counter[vid] = 0.0
# #
# #             speed = math.hypot(v.get_velocity().x, v.get_velocity().y)
# #             if speed < 0.2:
# #                 self._stuck_counter[vid] += delta_time
# #                 if self._stuck_counter[vid] > 10.0:
# #                     penalties[vid] = P_STUCK_CUMULATIVE
# #             else:
# #                 self._stuck_counter[vid] = 0.0
# #
# #         return penalties
# #
# #     def _setup_carla(self):
# #         self.client = carla.Client(self.host, self.port)
# #         self.client.set_timeout(300.0)
# #
# #         try:
# #             self.world = self.client.get_world()
# #             if self.town not in self.world.get_map().name:
# #                 self.world = self.client.load_world(self.town)
# #         except:
# #             self.world = self.client.load_world(self.town)
# #
# #         self.tm = self.client.get_trafficmanager(self.tm_port)
# #         self.tm.set_synchronous_mode(True)
# #         self.tm.set_random_device_seed(int(time.time()))
# #         self.tm.global_percentage_speed_difference(-30.0)
# #
# #         settings = self.world.get_settings()
# #         settings.synchronous_mode = True
# #         settings.fixed_delta_seconds = 0.05
# #         self.world.apply_settings(settings)
# #
# #         self.spawn_points = self.world.get_map().get_spawn_points()
# #
# #     def _on_collision(self, event):
# #         actor_id = event.actor.id
# #         self.collision_histories[actor_id].append(event)
# #         other = event.other_actor
# #         if other.type_id.startswith("vehicle") or other.type_id.startswith("walker"):
# #             self.total_ego_collisions += 1
# #
# #     def _cleanup_vehicles_only(self):
# #         if self.client:
# #             for sensor in self.sensors:
# #                 if sensor.is_alive:
# #                     sensor.destroy()
# #             self.sensors = []
# #             batch = [carla.command.DestroyActor(v) for v in self.vehicles]
# #             self.client.apply_batch(batch)
# #
# #         self.vehicles = []
# #
# #     def _spawn_pedestrians(self):
# #         if self.pedestrian_count <= 0:
# #             return
# #         walker_spawns = [pt for pt in self.spawn_points if pt.location.z <= 1.0]
# #         bp_lib = self.world.get_blueprint_library().filter("walker.pedestrian.*")
# #
# #         for _ in range(self.pedestrian_count):
# #             try:
# #                 sp = random.choice(walker_spawns)
# #                 bp = random.choice(bp_lib)
# #                 walker = self.world.try_spawn_actor(bp, sp)
# #                 if walker:
# #                     self.pedestrians.append(walker)
# #             except:
# #                 pass
# #
# #     def _trigger_emergency_jaywalk(self):
# #         if not self.vehicles:
# #             return
# #         try:
# #             target_vehicle = random.choice(self.vehicles)
# #             loc = target_vehicle.get_location()
# #             fwd = target_vehicle.get_transform().get_forward_vector()
# #             right = target_vehicle.get_transform().get_right_vector()
# #             spawn_loc = loc + (fwd * 15.0) + (right * 4.0)
# #             spawn_loc.z += 1.0
# #
# #             bp = random.choice(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
# #             walker = self.world.try_spawn_actor(bp, carla.Transform(spawn_loc))
# #
# #             if walker:
# #                 direction = -right
# #                 length = math.sqrt(direction.x ** 2 + direction.y ** 2 + direction.z ** 2)
# #                 if length > 0:
# #                     direction.x /= length
# #                     direction.y /= length
# #                     direction.z /= length
# #                 control = carla.WalkerControl(direction, 3.0, False)
# #                 walker.apply_control(control)
# #                 self.pedestrians.append(walker)
# #         except:
# #             pass
# #
# #     def _cleanup(self):
# #         if self.fleet_coordinator:
# #             self.fleet_coordinator.stop_coordination()
# #
# #         if self.client:
# #             for s in self.sensors:
# #                 if s.is_alive:
# #                     s.destroy()
# #             self.sensors = []
# #
# #             batch = [carla.command.DestroyActor(v) for v in self.vehicles]
# #             if self.pedestrians:
# #                 batch += [carla.command.DestroyActor(p) for p in self.pedestrians if p.is_alive]
# #             self.client.apply_batch(batch)
# #
# #         self.vehicles = []
# #         self.pedestrians = []
# #         self.waypoint_manager = None
# #         self.fleet_coordinator = None
# #         self._stuck_counter = {}
# #
# #     def close(self):
# #         self._cleanup()
# #         if self.world:
# #             try:
# #                 settings = self.world.get_settings()
# #                 settings.synchronous_mode = False
# #                 self.world.apply_settings(settings)
# #             except:
# #                 pass
#
#
#
# import gymnasium as gym
# from gymnasium import spaces
# import carla
# import numpy as np
# import random
# import math
# import time
# from collections import defaultdict
# from typing import Tuple, Optional, List, Dict
#
# # ==============================================================================
# # INTEGRATION: Import the "Brain" components
# # ==============================================================================
# try:
#     from utils.waypoint_manager.waypoint_system import WaymoWaypointManager
#     try:
#         from utils.fleet_coordinator.fleet_manager import FleetCoordinator
#     except ImportError:
#         from utils.fleet_coordinator.fleet_coordinator import FleetCoordinator
# except ImportError:
#     print("CRITICAL WARNING: Could not import Waypoint/Fleet managers.")
#
#     class WaymoWaypointManager:
#         def __init__(self, *args, **kwargs): pass
#         def assign_route(self, *args, **kwargs): return True
#         def update_route_progress(self, *args, **kwargs):
#             return {'progress': 0.0, 'dist_delta': 0.0, 'dist_to_next_wp': 0.0}
#         def get_route_status(self, *args, **kwargs):
#             return {'destination_reached': False}
#
#     class FleetCoordinator:
#         def __init__(self, *args, **kwargs): pass
#         def start_coordination(self): pass
#         def stop_coordination(self): pass
#         def get_pending_commands(self): return []
#         def update_vehicle_state(self, *args, **kwargs): pass
#         def get_coordination_statistics(self):
#             return {'recent_conflicts': 0, 'coordination_efficiency': 0.0}
#         def get_all_states(self): return {}
#
#
# # ==============================================================================
# # CONFIGURATION CONSTANTS (TRAINABLE SAFE V1)
# # ==============================================================================
#
# # Reward Scaling / Shaping
# REWARD_ALPHA = 10.0              # local route progress scaling
# PENALTY_BETA = 0.002             # smoothness penalty scaling
# ALIVE_REWARD = 0.05              # small positive "alive" reward each step
# SPEED_REWARD_SCALE = 0.05        # positive for moving, bounded
#
# # Collision penalties
# P_COLLISION_EGO = -1.0
# P_COLLISION_STATIC = -0.2
#
# # Spacing balance
# P_SPACING_REWARD = 0.10
# P_TAILGATING_PENALTY = -0.05
#
# # Stuck
# P_STUCK_CUMULATIVE = -0.5        # penalty once vehicle stuck > threshold
# STUCK_TIME_THRESHOLD = 20.0      # FIX: delay stuck penalty for early exploration
#
# # Safety Thresholds
# SHIELD_DISTANCE_BUFFER = 2.0
# CRITICAL_PROXIMITY_THRESHOLD = 4.0
# SAFE_FOLLOWING_DIST = 10.0
#
# # Physics
# MAX_SPEED_MPS = 10.0
# MAX_DECELERATION = 8.0
# REACTION_TIME = 0.5
#
# # Spawn safety
# MIN_SPAWN_SEPARATION = 20.0      # meters
#
# # Reward clipping per vehicle (FIX: stabilize critic)
# PER_VEHICLE_REWARD_CLIP = 2.0    # clip to [-2, 2]
#
#
# class StableMultiAgentCarlaEnv(gym.Env):
#     """
#     Stable Multi-Agent CARLA Env (trainable).
#     Includes robust shaping, safer shield, stable spawns, and reward clipping.
#     """
#
#     metadata = {"render_modes": []}
#
#     # Expose thresholds as class attrs
#     CRITICAL_PROXIMITY_THRESHOLD = CRITICAL_PROXIMITY_THRESHOLD
#     SAFE_FOLLOWING_DIST = SAFE_FOLLOWING_DIST
#     SHIELD_DISTANCE_BUFFER = SHIELD_DISTANCE_BUFFER
#
#     def __init__(
#         self,
#         num_vehicles: int = 5,
#         town: str = "Town03",
#         host: str = "localhost",
#         port: int = 2000,
#         tm_port: int = 8000,
#         max_episode_steps: int = 1000,
#         weather: str = "ClearNoon",
#         pedestrian_count: int = 0,
#         timeout_threshold: float = 300.0
#     ):
#         super().__init__()
#
#         self.num_vehicles = num_vehicles
#         self.town = town
#         self.host = host
#         self.port = port
#         self.tm_port = tm_port
#         self.max_episode_steps = max_episode_steps
#         self.weather = weather
#         self.pedestrian_count = pedestrian_count
#         self.timeout_threshold = timeout_threshold
#
#         # Action space (Throttle, Steer, Brake) * N
#         self.action_space = spaces.Box(
#             low=np.array([0.0, -1.0, 0.0] * num_vehicles, dtype=np.float32),
#             high=np.array([1.0, 1.0, 1.0] * num_vehicles, dtype=np.float32),
#             dtype=np.float32,
#         )
#
#         # Observation space: 10 features per vehicle
#         obs_dim = 10 * num_vehicles
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
#         )
#
#         # CARLA handles
#         self.client = None
#         self.world = None
#         self.tm = None
#
#         # Actors & sensors
#         self.vehicles: List[carla.Vehicle] = []
#         self.pedestrians: List[carla.Walker] = []
#         self.sensors: List[carla.Sensor] = []
#         self.spawn_points: List[carla.Transform] = []
#
#         # Managers
#         self.waypoint_manager: Optional[WaymoWaypointManager] = None
#         self.fleet_coordinator: Optional[FleetCoordinator] = None
#
#         # Tracking
#         self.collision_histories = defaultdict(list)
#         self._stuck_counter: Dict[int, float] = {}
#         self._prev_actions: np.ndarray = np.zeros((num_vehicles, 3), dtype=np.float32)
#         self._vehicle_neighbor_cache = {}
#
#         # Local progress tracking
#         self._prev_dist_to_wp: Dict[int, float] = {}
#
#         # Metrics
#         self.episode_step = 0
#         self.episode_start_time = 0
#         self.emergency_stops = 0
#         self.total_ego_collisions = 0
#         self.shield_activations = 0
#         self.current_min_distance = float('inf')
#
#     # ==============================================================================
#     # RESET / STEP
#     # ==============================================================================
#
#     def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
#         super().reset(seed=seed)
#         if seed is not None:
#             random.seed(seed)
#             np.random.seed(seed)
#
#         self._cleanup()
#         self._setup_carla()
#
#         # init managers
#         self.waypoint_manager = WaymoWaypointManager(self.world, max_vehicles=self.num_vehicles)
#         self.fleet_coordinator = FleetCoordinator(self.waypoint_manager, max_vehicles=self.num_vehicles)
#         self.fleet_coordinator.start_coordination()
#
#         # clear trackers
#         self.collision_histories.clear()
#         self._stuck_counter.clear()
#         self._prev_actions = np.zeros((self.num_vehicles, 3), dtype=np.float32)
#         self._vehicle_neighbor_cache = {}
#         self._prev_dist_to_wp = {}
#
#         # safer spawn (distance-filtered)
#         spawn_attempts = 0
#         min_required = min(self.num_vehicles, 3)
#
#         while len(self.vehicles) < min_required and spawn_attempts < 5:
#             self._spawn_vehicles_simple()
#             for _ in range(10):
#                 self.world.tick()
#
#             self.vehicles = [v for v in self.vehicles if v.is_alive]
#
#             if len(self.vehicles) < min_required:
#                 print(f"Warning: Low vehicle count ({len(self.vehicles)}). Retrying spawn...")
#                 self._cleanup_vehicles_only()
#                 spawn_attempts += 1
#
#         if self.pedestrian_count > 0:
#             self._spawn_pedestrians()
#
#         for v in self.vehicles:
#             self.waypoint_manager.assign_route(v.id)
#
#         # init local progress distances
#         for v in self.vehicles:
#             try:
#                 t = v.get_transform()
#                 route_info = self.waypoint_manager.update_route_progress(v.id, t.location)
#                 self._prev_dist_to_wp[v.id] = float(route_info.get("dist_to_next_wp", 0.0))
#             except:
#                 self._prev_dist_to_wp[v.id] = 0.0
#
#         # reset metrics
#         self.episode_step = 0
#         self.episode_start_time = time.time()
#         self.emergency_stops = 0
#         self.total_ego_collisions = 0
#         self.shield_activations = 0
#         self.current_min_distance = float('inf')
#
#         # build neighbor cache now
#         self._update_neighbor_cache()
#         self._update_global_min_distance()
#
#         return self._get_observations(), {}
#
#     def step(self, action: np.ndarray):
#         self.episode_step += 1
#
#         # prune dead actors
#         self.vehicles = [v for v in self.vehicles if v.is_alive]
#         if not self.vehicles:
#             return self._get_observations(), 0.0, True, False, {"error": "All vehicles destroyed"}
#
#         action = action.reshape(self.num_vehicles, 3)
#         current_actions = action.copy()
#
#         # emergency jaywalk event
#         if self.pedestrian_count > 0 and random.random() < 0.02:
#             self._trigger_emergency_jaywalk()
#
#         # fleet commands
#         pending_cmds = self.fleet_coordinator.get_pending_commands()
#         yielding_vehicles = set()
#         for cmd_type, vid in pending_cmds:
#             if cmd_type in ("REASSIGN_ROUTE", "OPTIMIZE_ROUTE"):
#                 self.waypoint_manager.assign_route(vid)
#             elif cmd_type == "APPLY_YIELD":
#                 yielding_vehicles.add(vid)
#
#         # neighbors + min dist
#         self._update_neighbor_cache()
#         self._update_global_min_distance()
#
#         # stuck penalties (no termination)
#         stuck_penalties = self._handle_stuck_vehicles()
#
#         # apply controls
#         for idx, vehicle in enumerate(self.vehicles):
#             if idx >= len(action):
#                 break
#             vid = vehicle.id
#
#             throttle, steer, brake = action[idx]
#
#             # telemetry update
#             t = vehicle.get_transform()
#             v = vehicle.get_velocity()
#             route_info = self.waypoint_manager.update_route_progress(vid, t.location)
#
#             self.fleet_coordinator.update_vehicle_state(
#                 vehicle_id=vid,
#                 location=t.location,
#                 velocity=v,
#                 heading=t.rotation.yaw,
#                 route_progress=route_info.get('progress', 0.0)
#             )
#
#             steer += np.random.normal(0, 0.02)
#
#             # targeted shield (less aggressive)
#             shield_active, safe_control = self._check_targeted_shield(vehicle)
#
#             if shield_active:
#                 final_control = safe_control
#                 self.shield_activations += 1
#             else:
#                 final_control = carla.VehicleControl(
#                     throttle=float(np.clip(throttle, 0.0, 1.0)),
#                     steer=float(np.clip(steer, -1.0, 1.0)),
#                     brake=float(np.clip(brake, 0.0, 1.0))
#                 )
#
#             # yield/ped override (below shield)
#             if not shield_active:
#                 if vid in yielding_vehicles:
#                     final_control.throttle = 0.0
#                     final_control.brake = 0.5
#                 elif self._check_pedestrian_hazard(vehicle):
#                     final_control.throttle = 0.0
#                     final_control.brake = 1.0
#                     self.emergency_stops += 1
#
#             try:
#                 vehicle.apply_control(final_control)
#             except:
#                 pass
#
#         self._prev_actions = current_actions
#
#         # tick sim
#         try:
#             self.world.tick()
#         except Exception as e:
#             print(f"Tick error: {e}")
#             return self._get_observations(), 0.0, True, True, {}
#
#         # reward + info
#         obs = self._get_observations()
#         reward = self._calculate_rewards(current_actions, stuck_penalties)
#         fleet_stats = self.fleet_coordinator.get_coordination_statistics()
#
#         # termination logic (only real safety)
#         terminated = False
#         truncated = False
#
#         if self.total_ego_collisions > 0:
#             terminated = True
#             print(f"DEBUG: Terminated - Ego Collision ({self.total_ego_collisions})")
#
#         if self.current_min_distance < self.CRITICAL_PROXIMITY_THRESHOLD and self.current_min_distance > 0.1:
#             terminated = True
#             print(f"DEBUG: Terminated - Critical Proximity ({self.current_min_distance:.2f}m)")
#
#         if time.time() - self.episode_start_time > self.timeout_threshold:
#             truncated = True
#         if self.episode_step >= self.max_episode_steps:
#             truncated = True
#
#         if len(self.vehicles) < max(1, self.num_vehicles // 3):
#             terminated = True
#
#         info = {
#             "vehicles": len(self.vehicles),
#             "episode_step": self.episode_step,
#             "min_distance": self.current_min_distance,
#             "collisions": self.total_ego_collisions,
#             "emergency_stops": self.emergency_stops,
#             "shield_activations": self.shield_activations,
#             "conflicts_resolved": fleet_stats.get('recent_conflicts', 0),
#             "coordination_efficiency": fleet_stats.get('coordination_efficiency', 0.0)
#         }
#
#         return obs, reward, terminated, truncated, info
#
#     # ==============================================================================
#     # SAFETY / NEIGHBORS
#     # ==============================================================================
#
#     def _update_neighbor_cache(self):
#         """Nearest neighbor per vehicle. Falls back to raw CARLA if fleet states empty."""
#         self._vehicle_neighbor_cache = {}
#
#         all_states = {}
#         try:
#             all_states = self.fleet_coordinator.get_all_states()
#         except:
#             all_states = {}
#
#         # fallback: build from CARLA directly
#         if not all_states:
#             alive = [v for v in self.vehicles if v.is_alive]
#             for ego in alive:
#                 min_dist = 100.0
#                 best_neighbor = None
#                 for other in alive:
#                     if other.id == ego.id:
#                         continue
#                     d = ego.get_location().distance(other.get_location())
#                     if d < min_dist:
#                         min_dist = d
#                         best_neighbor = other
#                 self._vehicle_neighbor_cache[ego.id] = {
#                     "nnd": min_dist,
#                     "neighbor_actor": best_neighbor,
#                     "neighbor_state": None
#                 }
#             return
#
#         # main path: use coordinator states
#         for ego_v in self.vehicles:
#             ego_id = ego_v.id
#             if ego_id not in all_states:
#                 continue
#
#             ego_state = all_states[ego_id]
#             ego_loc = ego_state.location
#             if not hasattr(ego_loc, "x"):
#                 continue
#
#             min_dist = 100.0
#             best_neighbor = None
#
#             for other_id, other_state in all_states.items():
#                 if other_id == ego_id:
#                     continue
#                 other_loc = other_state.location
#                 if not hasattr(other_loc, "x"):
#                     continue
#
#                 dist = math.sqrt(
#                     (ego_loc.x - other_loc.x) ** 2 +
#                     (ego_loc.y - other_loc.y) ** 2 +
#                     (ego_loc.z - other_loc.z) ** 2
#                 )
#
#                 if dist < min_dist:
#                     min_dist = dist
#                     best_neighbor = other_state
#
#             self._vehicle_neighbor_cache[ego_id] = {
#                 "nnd": min_dist,
#                 "neighbor_state": best_neighbor,
#                 "neighbor_actor": None
#             }
#
#     def _update_global_min_distance(self):
#         if not self._vehicle_neighbor_cache:
#             self.current_min_distance = 100.0
#             return
#         try:
#             self.current_min_distance = min(d["nnd"] for d in self._vehicle_neighbor_cache.values())
#         except:
#             self.current_min_distance = 100.0
#
#     def _check_targeted_shield(self, vehicle: carla.Vehicle) -> Tuple[bool, carla.VehicleControl]:
#         """Targeted shield for this vehicle only (less aggressive)."""
#         vid = vehicle.id
#         neighbor_data = self._vehicle_neighbor_cache.get(vid)
#
#         safe_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
#
#         if not neighbor_data:
#             return False, safe_control
#
#         nnd = neighbor_data["nnd"]
#
#         ego_vel = vehicle.get_velocity()
#         ego_speed = math.hypot(ego_vel.x, ego_vel.y)
#
#         neighbor_speed = 0.0
#         try:
#             if neighbor_data.get("neighbor_actor") is not None:
#                 nvel = neighbor_data["neighbor_actor"].get_velocity()
#                 neighbor_speed = math.hypot(nvel.x, nvel.y)
#             elif neighbor_data.get("neighbor_state") is not None:
#                 nvel = neighbor_data["neighbor_state"].velocity
#                 neighbor_speed = math.hypot(nvel.x, nvel.y)
#         except:
#             neighbor_speed = 0.0
#
#         rel_speed = max(ego_speed - neighbor_speed, 0.0)
#
#         d_safe = CRITICAL_PROXIMITY_THRESHOLD + self.SHIELD_DISTANCE_BUFFER + (REACTION_TIME * rel_speed)
#
#         if nnd < d_safe:
#             return True, safe_control
#
#         return False, safe_control
#
#     def _check_pedestrian_hazard(self, vehicle) -> bool:
#         """Cone-based pedestrian hazard check."""
#         v_loc = vehicle.get_location()
#         v_fwd = vehicle.get_transform().get_forward_vector()
#
#         for ped in self.pedestrians:
#             try:
#                 actor = ped[0] if isinstance(ped, tuple) else ped
#                 if not actor.is_alive:
#                     continue
#                 p_loc = actor.get_location()
#             except:
#                 continue
#
#             dist = v_loc.distance(p_loc)
#             if dist < 12.0:
#                 to_ped = p_loc - v_loc
#                 length = to_ped.length()
#                 if length > 0:
#                     to_ped = to_ped / length
#                     dot = v_fwd.dot(to_ped)
#                     if dot > 0.5:
#                         return True
#         return False
#
#     # ==============================================================================
#     # REWARD / OBS
#     # ==============================================================================
#
#     def _calculate_rewards(self, current_actions: np.ndarray, stuck_penalties: Dict[int, float]) -> float:
#         total_reward = 0.0
#
#         # smoothness penalty (global)
#         if self._prev_actions.shape == current_actions.shape:
#             delta = current_actions - self._prev_actions
#             smoothness = -PENALTY_BETA * np.sum(delta[:, 0] ** 2 + delta[:, 1] ** 2)
#             total_reward += smoothness
#
#         for v in self.vehicles:
#             if not v.is_alive:
#                 continue
#             vid = v.id
#
#             r_alive = ALIVE_REWARD
#
#             vel = v.get_velocity()
#             speed = math.hypot(vel.x, vel.y)
#             r_speed = SPEED_REWARD_SCALE * min(speed / MAX_SPEED_MPS, 1.0)
#
#             # local route progress reward (FIX: never penalize noisy progress)
#             r_route = 0.0
#             try:
#                 route_info = self.waypoint_manager.update_route_progress(vid, v.get_location())
#                 cur_dist = float(route_info.get("dist_to_next_wp", 0.0))
#                 prev_dist = float(self._prev_dist_to_wp.get(vid, cur_dist))
#
#                 local_delta = prev_dist - cur_dist
#                 local_delta = max(local_delta, 0.0)                 # FIX: no negative progress penalty
#                 local_delta = float(np.clip(local_delta, 0.0, 5.0))
#
#                 r_route = local_delta * REWARD_ALPHA
#
#                 self._prev_dist_to_wp[vid] = cur_dist
#             except:
#                 r_route = 0.0
#
#             r_spacing = 0.0
#             ndata = self._vehicle_neighbor_cache.get(vid)
#             if ndata:
#                 nnd = ndata["nnd"]
#                 if nnd > SAFE_FOLLOWING_DIST:
#                     r_spacing = P_SPACING_REWARD
#                 elif nnd < CRITICAL_PROXIMITY_THRESHOLD + 2.0:
#                     r_spacing = P_TAILGATING_PENALTY
#
#             r_col = 0.0
#             if self.collision_histories.get(vid):
#                 r_col = P_COLLISION_STATIC
#                 for event in self.collision_histories[vid]:
#                     other = event.other_actor
#                     if other.type_id.startswith("vehicle") or other.type_id.startswith("walker"):
#                         r_col = P_COLLISION_EGO
#                         break
#
#             r_stuck = stuck_penalties.get(vid, 0.0)
#
#             per_vehicle_reward = r_alive + r_speed + r_route + r_spacing + r_col + r_stuck
#             per_vehicle_reward = float(np.clip(per_vehicle_reward,
#                                                -PER_VEHICLE_REWARD_CLIP,
#                                                PER_VEHICLE_REWARD_CLIP))  # FIX: stabilize critic
#
#             total_reward += per_vehicle_reward
#
#         return float(total_reward)
#
#     def _get_observations(self):
#         obs = []
#         for v in self.vehicles:
#             if v.is_alive:
#                 try:
#                     t = v.get_transform()
#                     vel = v.get_velocity()
#                     speed = math.hypot(vel.x, vel.y)
#
#                     obs.extend([
#                         t.location.x / 1000.0,
#                         t.location.y / 1000.0,
#                         t.rotation.yaw / 180.0,
#                         vel.x / MAX_SPEED_MPS,
#                         vel.y / MAX_SPEED_MPS,
#                         speed / MAX_SPEED_MPS,
#                     ])
#
#                     route_info = self.waypoint_manager.update_route_progress(v.id, t.location)
#                     obs.append(route_info.get("dist_to_next_wp", 0.0) / 100.0)
#
#                     ndata = self._vehicle_neighbor_cache.get(v.id)
#                     if ndata:
#                         nnd = ndata["nnd"]
#                         obs.extend([nnd / 100.0, 0.0, 0.0])
#                     else:
#                         obs.extend([1.0, 0.0, 0.0])
#
#                 except:
#                     obs.extend([0.0] * 10)
#             else:
#                 obs.extend([0.0] * 10)
#
#         target_len = self.num_vehicles * 10
#         if len(obs) < target_len:
#             obs.extend([0.0] * (target_len - len(obs)))
#
#         return np.array(obs[:target_len], dtype=np.float32)
#
#     # ==============================================================================
#     # SPAWN / STUCK / CLEANUP
#     # ==============================================================================
#
#     def _spawn_vehicles_simple(self):
#         """Simple stable spawn: distance-filtered spawn points."""
#         bp_lib = self.world.get_blueprint_library().filter("vehicle.*")
#         safe_bp_lib = [x for x in bp_lib if int(x.get_attribute("number_of_wheels")) == 4]
#         collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
#
#         spawn_points = self.world.get_map().get_spawn_points()
#         if not spawn_points:
#             return
#
#         random.shuffle(spawn_points)
#
#         spawned_locs = [v.get_location() for v in self.vehicles if v.is_alive]
#         count = len(self.vehicles)
#
#         for tf in spawn_points:
#             if count >= self.num_vehicles:
#                 break
#
#             ok = True
#             for loc in spawned_locs:
#                 if tf.location.distance(loc) < MIN_SPAWN_SEPARATION:
#                     ok = False
#                     break
#             if not ok:
#                 continue
#
#             tf.location.z += 0.5
#             bp = random.choice(safe_bp_lib)
#             if bp.has_attribute("color"):
#                 bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
#
#             vehicle = self.world.try_spawn_actor(bp, tf)
#             if vehicle:
#                 vehicle.set_autopilot(False)
#                 self.vehicles.append(vehicle)
#                 spawned_locs.append(vehicle.get_location())
#                 count += 1
#
#                 sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
#                 sensor.listen(lambda event: self._on_collision(event))
#                 self.sensors.append(sensor)
#
#                 self.world.tick()
#
#     def _handle_stuck_vehicles(self):
#         penalties = {}
#         delta_time = 0.05
#
#         for v in self.vehicles:
#             if not v.is_alive:
#                 continue
#
#             vid = v.id
#             if vid not in self._stuck_counter:
#                 self._stuck_counter[vid] = 0.0
#
#             speed = math.hypot(v.get_velocity().x, v.get_velocity().y)
#             if speed < 0.2:
#                 self._stuck_counter[vid] += delta_time
#                 if self._stuck_counter[vid] > STUCK_TIME_THRESHOLD:
#                     penalties[vid] = P_STUCK_CUMULATIVE
#             else:
#                 self._stuck_counter[vid] = 0.0
#
#         return penalties
#
#     def _setup_carla(self):
#         self.client = carla.Client(self.host, self.port)
#         self.client.set_timeout(300.0)
#
#         try:
#             self.world = self.client.get_world()
#             if self.town not in self.world.get_map().name:
#                 self.world = self.client.load_world(self.town)
#         except:
#             self.world = self.client.load_world(self.town)
#
#         self.tm = self.client.get_trafficmanager(self.tm_port)
#         self.tm.set_synchronous_mode(True)
#         self.tm.set_random_device_seed(int(time.time()))
#         self.tm.global_percentage_speed_difference(-30.0)
#
#         settings = self.world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = 0.05
#         self.world.apply_settings(settings)
#
#         self.spawn_points = self.world.get_map().get_spawn_points()
#
#     def _on_collision(self, event):
#         actor_id = event.actor.id
#         self.collision_histories[actor_id].append(event)
#         other = event.other_actor
#         if other.type_id.startswith("vehicle") or other.type_id.startswith("walker"):
#             self.total_ego_collisions += 1
#
#     def _cleanup_vehicles_only(self):
#         if self.client:
#             for sensor in self.sensors:
#                 if sensor.is_alive:
#                     sensor.destroy()
#             self.sensors = []
#             batch = [carla.command.DestroyActor(v) for v in self.vehicles]
#             self.client.apply_batch(batch)
#
#         self.vehicles = []
#
#     def _spawn_pedestrians(self):
#         if self.pedestrian_count <= 0:
#             return
#         walker_spawns = [pt for pt in self.spawn_points if pt.location.z <= 1.0]
#         bp_lib = self.world.get_blueprint_library().filter("walker.pedestrian.*")
#
#         for _ in range(self.pedestrian_count):
#             try:
#                 sp = random.choice(walker_spawns)
#                 bp = random.choice(bp_lib)
#                 walker = self.world.try_spawn_actor(bp, sp)
#                 if walker:
#                     self.pedestrians.append(walker)
#             except:
#                 pass
#
#     def _trigger_emergency_jaywalk(self):
#         if not self.vehicles:
#             return
#         try:
#             target_vehicle = random.choice(self.vehicles)
#             loc = target_vehicle.get_location()
#             fwd = target_vehicle.get_transform().get_forward_vector()
#             right = target_vehicle.get_transform().get_right_vector()
#             spawn_loc = loc + (fwd * 15.0) + (right * 4.0)
#             spawn_loc.z += 1.0
#
#             bp = random.choice(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
#             walker = self.world.try_spawn_actor(bp, carla.Transform(spawn_loc))
#
#             if walker:
#                 direction = -right
#                 length = math.sqrt(direction.x ** 2 + direction.y ** 2 + direction.z ** 2)
#                 if length > 0:
#                     direction.x /= length
#                     direction.y /= length
#                     direction.z /= length
#                 control = carla.WalkerControl(direction, 3.0, False)
#                 walker.apply_control(control)
#                 self.pedestrians.append(walker)
#         except:
#             pass
#
#     def _cleanup(self):
#         if self.fleet_coordinator:
#             self.fleet_coordinator.stop_coordination()
#
#         if self.client:
#             for s in self.sensors:
#                 if s.is_alive:
#                     s.destroy()
#             self.sensors = []
#
#             batch = [carla.command.DestroyActor(v) for v in self.vehicles]
#             if self.pedestrians:
#                 batch += [carla.command.DestroyActor(p) for p in self.pedestrians if p.is_alive]
#             self.client.apply_batch(batch)
#
#         self.vehicles = []
#         self.pedestrians = []
#         self.waypoint_manager = None
#         self.fleet_coordinator = None
#         self._stuck_counter = {}
#         self._prev_dist_to_wp = {}
#
#     def close(self):
#         self._cleanup()
#         if self.world:
#             try:
#                 settings = self.world.get_settings()
#                 settings.synchronous_mode = False
#                 self.world.apply_settings(settings)
#             except:
#                 pass



import gymnasium as gym
from gymnasium import spaces
import carla
import numpy as np
import random
import math
import time
from collections import defaultdict
from typing import Tuple, Optional, List, Dict

# ==============================================================================
# INTEGRATION: Import the "Brain" components
# ==============================================================================
try:
    from utils.waypoint_manager.waypoint_system import WaymoWaypointManager
    try:
        from utils.fleet_coordinator.fleet_manager import FleetCoordinator
    except ImportError:
        from utils.fleet_coordinator.fleet_coordinator import FleetCoordinator
except ImportError:
    print("CRITICAL WARNING: Could not import Waypoint/Fleet managers.")

    class WaymoWaypointManager:
        def __init__(self, *args, **kwargs): pass
        def assign_route(self, *args, **kwargs): return True
        def update_route_progress(self, *args, **kwargs):
            return {'progress': 0.0, 'dist_delta': 0.0, 'dist_to_next_wp': 0.0}
        def get_route_status(self, *args, **kwargs):
            return {'destination_reached': False}

    class FleetCoordinator:
        def __init__(self, *args, **kwargs): pass
        def start_coordination(self): pass
        def stop_coordination(self): pass
        def get_pending_commands(self): return []
        def update_vehicle_state(self, *args, **kwargs): pass
        def get_coordination_statistics(self):
            return {'recent_conflicts': 0, 'coordination_efficiency': 0.0}
        def get_all_states(self): return {}


# ==============================================================================
# CONFIGURATION CONSTANTS (TRAINABLE SAFE V1)
# ==============================================================================

# Reward Scaling / Shaping
REWARD_ALPHA = 10.0              # local route progress scaling
PENALTY_BETA = 0.002             # smoothness penalty scaling
ALIVE_REWARD = 0.05              # small positive "alive" reward each step
SPEED_REWARD_SCALE = 0.05        # positive for moving, bounded

# Collision penalties
P_COLLISION_EGO = -1000.0
P_COLLISION_STATIC = -50

# Spacing balance
P_SPACING_REWARD = 0.10
P_TAILGATING_PENALTY = -0.05

# Stuck
P_STUCK_CUMULATIVE = -0.5        # penalty once vehicle stuck > threshold
STUCK_TIME_THRESHOLD = 20.0      # FIX: delay stuck penalty for early exploration

# Safety Thresholds
SHIELD_DISTANCE_BUFFER = 2.0
CRITICAL_PROXIMITY_THRESHOLD = 4.0
SAFE_FOLLOWING_DIST = 10.0

# Physics
MAX_SPEED_MPS = 10.0
MAX_DECELERATION = 8.0
REACTION_TIME = 0.5
THEORETICAL_SAFE_DISTANCE = 5.0  # From your proofs
MAX_PROXIMITY_PENALTY = -200.0   # Cap for the proactive penalty
REWARD_COORDINATION_BONUS = 20.0 # Reward for resolving conflicts
# Spawn safety
MIN_SPAWN_SEPARATION = 20.0      # meters

# Reward clipping per vehicle (FIX: stabilize critic)
PER_VEHICLE_REWARD_CLIP = 2.0    # clip to [-2, 2]


class StableMultiAgentCarlaEnv(gym.Env):
    """
    Stable Multi-Agent CARLA Env (trainable).
    Includes robust shaping, safer shield, stable spawns, and reward clipping.
    """

    metadata = {"render_modes": []}

    # Expose thresholds as class attrs
    CRITICAL_PROXIMITY_THRESHOLD = CRITICAL_PROXIMITY_THRESHOLD
    SAFE_FOLLOWING_DIST = SAFE_FOLLOWING_DIST
    SHIELD_DISTANCE_BUFFER = SHIELD_DISTANCE_BUFFER

    def __init__(
        self,
        num_vehicles: int = 5,
        town: str = "Town03",
        host: str = "localhost",
        port: int = 2000,
        tm_port: int = 8000,
        max_episode_steps: int = 1000,
        weather: str = "ClearNoon",
        pedestrian_count: int = 0,
        timeout_threshold: float = 300.0
    ):
        super().__init__()

        self.num_vehicles = num_vehicles
        self.town = town
        self.host = host
        self.port = port
        self.tm_port = tm_port
        self.max_episode_steps = max_episode_steps
        self.weather = weather
        self.pedestrian_count = pedestrian_count
        self.timeout_threshold = timeout_threshold

        # --- FIX START: CONSTANT OBSERVATION/ACTION SPACES ---
        # Maximum capacity for hardest stage (Expert)
        self.MAX_VEHICLES_CAPACITY = 30
        self.VEHICLE_FEATURE_SIZE = 10  # 10 floats per vehicle

        # Action space (Throttle, Steer, Brake) * MAX_VEHICLES
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0] * self.MAX_VEHICLES_CAPACITY, dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0] * self.MAX_VEHICLES_CAPACITY, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space fixed to MAX capacity
        obs_dim = self.VEHICLE_FEATURE_SIZE * self.MAX_VEHICLES_CAPACITY
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # --- FIX END ---

        # CARLA handles
        self.client = None
        self.world = None
        self.tm = None

        # Actors & sensors
        self.vehicles: List[carla.Vehicle] = []
        self.pedestrians: List[carla.Walker] = []
        self.sensors: List[carla.Sensor] = []
        self.spawn_points: List[carla.Transform] = []

        # Managers
        self.waypoint_manager: Optional[WaymoWaypointManager] = None
        self.fleet_coordinator: Optional[FleetCoordinator] = None

        # Tracking
        self.collision_histories = defaultdict(list)
        self._stuck_counter: Dict[int, float] = {}
        self._prev_actions: np.ndarray = np.zeros((self.MAX_VEHICLES_CAPACITY, 3), dtype=np.float32)
        self._vehicle_neighbor_cache = {}

        # Local progress tracking
        self._prev_dist_to_wp: Dict[int, float] = {}

        # Metrics
        self.episode_step = 0
        self.episode_start_time = 0
        self.emergency_stops = 0
        self.total_ego_collisions = 0
        self.shield_activations = 0
        self.current_min_distance = float('inf')

    # ==============================================================================
    # RESET / STEP
    # ==============================================================================

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._cleanup()
        self._setup_carla()

        # init managers
        self.waypoint_manager = WaymoWaypointManager(self.world, max_vehicles=self.num_vehicles)
        self.fleet_coordinator = FleetCoordinator(self.waypoint_manager, max_vehicles=self.num_vehicles)
        self.fleet_coordinator.start_coordination()

        # clear trackers
        self.collision_histories.clear()
        self._stuck_counter.clear()
        self._prev_actions = np.zeros((self.MAX_VEHICLES_CAPACITY, 3), dtype=np.float32)
        self._vehicle_neighbor_cache = {}
        self._prev_dist_to_wp = {}

        # safer spawn (distance-filtered)
        spawn_attempts = 0
        min_required = min(self.num_vehicles, 3)

        while len(self.vehicles) < min_required and spawn_attempts < 5:
            self._spawn_vehicles_simple()
            for _ in range(10):
                self.world.tick()

            self.vehicles = [v for v in self.vehicles if v.is_alive]

            if len(self.vehicles) < min_required:
                print(f"Warning: Low vehicle count ({len(self.vehicles)}). Retrying spawn...")
                self._cleanup_vehicles_only()
                spawn_attempts += 1

        if self.pedestrian_count > 0:
            self._spawn_pedestrians()

        for v in self.vehicles:
            self.waypoint_manager.assign_route(v.id)

        # init local progress distances
        for v in self.vehicles:
            try:
                t = v.get_transform()
                route_info = self.waypoint_manager.update_route_progress(v.id, t.location)
                self._prev_dist_to_wp[v.id] = float(route_info.get("dist_to_next_wp", 0.0))
            except:
                self._prev_dist_to_wp[v.id] = 0.0

        # reset metrics
        self.episode_step = 0
        self.episode_start_time = time.time()
        self.emergency_stops = 0
        self.total_ego_collisions = 0
        self.shield_activations = 0
        self.current_min_distance = float('inf')

        # build neighbor cache now
        self._update_neighbor_cache()
        self._update_global_min_distance()

        return self._get_observations(), {}

    def step(self, action: np.ndarray):
        self.episode_step += 1

        # prune dead actors
        self.vehicles = [v for v in self.vehicles if v.is_alive]
        if not self.vehicles:
            return self._get_observations(), 0.0, True, False, {"error": "All vehicles destroyed"}

        # --- FIX START: CONSTANT ACTION HANDLING ---
        # Model outputs actions for MAX vehicles
        all_actions = action.reshape(self.MAX_VEHICLES_CAPACITY, 3)

        # Only use actions for currently alive vehicles
        valid_actions = all_actions[:len(self.vehicles)]

        # For smoothness/reward stability, zero out unused slots
        current_actions = all_actions.copy()
        if len(self.vehicles) < self.MAX_VEHICLES_CAPACITY:
            current_actions[len(self.vehicles):, :] = 0.0
        # --- FIX END ---

        # emergency jaywalk event
        if self.pedestrian_count > 0 and random.random() < 0.02:
            self._trigger_emergency_jaywalk()

        # fleet commands
        pending_cmds = self.fleet_coordinator.get_pending_commands()
        yielding_vehicles = set()
        for cmd_type, vid in pending_cmds:
            if cmd_type in ("REASSIGN_ROUTE", "OPTIMIZE_ROUTE"):
                self.waypoint_manager.assign_route(vid)
            elif cmd_type == "APPLY_YIELD":
                yielding_vehicles.add(vid)

        # neighbors + min dist
        self._update_neighbor_cache()
        self._update_global_min_distance()

        # stuck penalties (no termination)
        stuck_penalties = self._handle_stuck_vehicles()

        # apply controls
        for idx, vehicle in enumerate(self.vehicles):
            if idx >= len(valid_actions):
                break
            vid = vehicle.id

            throttle, steer, brake = valid_actions[idx]

            # telemetry update
            t = vehicle.get_transform()
            v = vehicle.get_velocity()
            route_info = self.waypoint_manager.update_route_progress(vid, t.location)

            self.fleet_coordinator.update_vehicle_state(
                vehicle_id=vid,
                location=t.location,
                velocity=v,
                heading=t.rotation.yaw,
                route_progress=route_info.get('progress', 0.0)
            )

            steer += np.random.normal(0, 0.02)

            # targeted shield (less aggressive)
            shield_active, safe_control = self._check_targeted_shield(vehicle)

            if shield_active:
                final_control = safe_control
                self.shield_activations += 1
            else:
                final_control = carla.VehicleControl(
                    throttle=float(np.clip(throttle, 0.0, 1.0)),
                    steer=float(np.clip(steer, -1.0, 1.0)),
                    brake=float(np.clip(brake, 0.0, 1.0))
                )

            # yield/ped override (below shield)
            if not shield_active:
                if vid in yielding_vehicles:
                    final_control.throttle = 0.0
                    final_control.brake = 0.5
                elif self._check_pedestrian_hazard(vehicle):
                    final_control.throttle = 0.0
                    final_control.brake = 1.0
                    self.emergency_stops += 1

            try:
                vehicle.apply_control(final_control)
            except:
                pass

        self._prev_actions = current_actions

        # tick sim
        try:
            self.world.tick()
        except Exception as e:
            print(f"Tick error: {e}")
            return self._get_observations(), 0.0, True, True, {}

        # reward + info
        obs = self._get_observations()
        reward = self._calculate_rewards(current_actions, stuck_penalties)
        fleet_stats = self.fleet_coordinator.get_coordination_statistics()
        reward += fleet_stats.get('recent_conflicts', 0) * REWARD_COORDINATION_BONUS

        # termination logic (only real safety)
        terminated = False
        truncated = False

        if self.total_ego_collisions > 0:
            terminated = True
            print(f"DEBUG: Terminated - Ego Collision ({self.total_ego_collisions})")

        if self.current_min_distance < self.CRITICAL_PROXIMITY_THRESHOLD and self.current_min_distance > 0.1:
            terminated = True
            print(f"DEBUG: Terminated - Critical Proximity ({self.current_min_distance:.2f}m)")

        if time.time() - self.episode_start_time > self.timeout_threshold:
            truncated = True
        if self.episode_step >= self.max_episode_steps:
            truncated = True

        if len(self.vehicles) < max(1, self.num_vehicles // 3):
            terminated = True

        info = {
            "vehicles": len(self.vehicles),
            "episode_step": self.episode_step,
            "min_distance": self.current_min_distance,
            "collisions": self.total_ego_collisions,
            "emergency_stops": self.emergency_stops,
            "shield_activations": self.shield_activations,
            "conflicts_resolved": fleet_stats.get('recent_conflicts', 0),
            "coordination_efficiency": fleet_stats.get('coordination_efficiency', 0.0)
        }

        return obs, reward, terminated, truncated, info

    # ==============================================================================
    # SAFETY / NEIGHBORS
    # ==============================================================================

    def _update_neighbor_cache(self):
        """Nearest neighbor per vehicle. Falls back to raw CARLA if fleet states empty."""
        self._vehicle_neighbor_cache = {}

        all_states = {}
        try:
            all_states = self.fleet_coordinator.get_all_states()
        except:
            all_states = {}

        # fallback: build from CARLA directly
        if not all_states:
            alive = [v for v in self.vehicles if v.is_alive]
            for ego in alive:
                min_dist = 100.0
                best_neighbor = None
                for other in alive:
                    if other.id == ego.id:
                        continue
                    d = ego.get_location().distance(other.get_location())
                    if d < min_dist:
                        min_dist = d
                        best_neighbor = other
                self._vehicle_neighbor_cache[ego.id] = {
                    "nnd": min_dist,
                    "neighbor_actor": best_neighbor,
                    "neighbor_state": None
                }
            return

        # main path: use coordinator states
        for ego_v in self.vehicles:
            ego_id = ego_v.id
            if ego_id not in all_states:
                continue

            ego_state = all_states[ego_id]
            ego_loc = ego_state.location
            if not hasattr(ego_loc, "x"):
                continue

            min_dist = 100.0
            best_neighbor = None

            for other_id, other_state in all_states.items():
                if other_id == ego_id:
                    continue
                other_loc = other_state.location
                if not hasattr(other_loc, "x"):
                    continue

                dist = math.sqrt(
                    (ego_loc.x - other_loc.x) ** 2 +
                    (ego_loc.y - other_loc.y) ** 2 +
                    (ego_loc.z - other_loc.z) ** 2
                )

                if dist < min_dist:
                    min_dist = dist
                    best_neighbor = other_state

            self._vehicle_neighbor_cache[ego_id] = {
                "nnd": min_dist,
                "neighbor_state": best_neighbor,
                "neighbor_actor": None
            }

    def _update_global_min_distance(self):
        if not self._vehicle_neighbor_cache:
            self.current_min_distance = 100.0
            return
        try:
            self.current_min_distance = min(d["nnd"] for d in self._vehicle_neighbor_cache.values())
        except:
            self.current_min_distance = 100.0

    def _check_targeted_shield(self, vehicle: carla.Vehicle) -> Tuple[bool, carla.VehicleControl]:
        """Targeted shield for this vehicle only (less aggressive)."""
        vid = vehicle.id
        neighbor_data = self._vehicle_neighbor_cache.get(vid)

        safe_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        if not neighbor_data:
            return False, safe_control

        nnd = neighbor_data["nnd"]

        ego_vel = vehicle.get_velocity()
        ego_speed = math.hypot(ego_vel.x, ego_vel.y)

        neighbor_speed = 0.0
        try:
            if neighbor_data.get("neighbor_actor") is not None:
                nvel = neighbor_data["neighbor_actor"].get_velocity()
                neighbor_speed = math.hypot(nvel.x, nvel.y)
            elif neighbor_data.get("neighbor_state") is not None:
                nvel = neighbor_data["neighbor_state"].velocity
                neighbor_speed = math.hypot(nvel.x, nvel.y)
        except:
            neighbor_speed = 0.0

        rel_speed = max(ego_speed - neighbor_speed, 0.0)

        d_safe = CRITICAL_PROXIMITY_THRESHOLD + self.SHIELD_DISTANCE_BUFFER + (REACTION_TIME * rel_speed)

        if nnd < d_safe:
            return True, safe_control

        return False, safe_control

    def _check_pedestrian_hazard(self, vehicle) -> bool:
        """Cone-based pedestrian hazard check."""
        v_loc = vehicle.get_location()
        v_fwd = vehicle.get_transform().get_forward_vector()

        for ped in self.pedestrians:
            try:
                actor = ped[0] if isinstance(ped, tuple) else ped
                if not actor.is_alive:
                    continue
                p_loc = actor.get_location()
            except:
                continue

            dist = v_loc.distance(p_loc)
            if dist < 12.0:
                to_ped = p_loc - v_loc
                length = to_ped.length()
                if length > 0:
                    to_ped = to_ped / length
                    dot = v_fwd.dot(to_ped)
                    if dot > 0.5:
                        return True
        return False

    # ==============================================================================
    # REWARD / OBS
    # ==============================================================================

    def _calculate_rewards(self, current_actions: np.ndarray, stuck_penalties: Dict[int, float]) -> float:
        total_reward = 0.0

        # smoothness penalty (global)
        if self._prev_actions.shape == current_actions.shape:
            delta = current_actions - self._prev_actions
            smoothness = -PENALTY_BETA * np.sum(delta[:, 0] ** 2 + delta[:, 1] ** 2)
            total_reward += smoothness

        for v in self.vehicles:
            if not v.is_alive:
                continue
            vid = v.id

            r_alive = ALIVE_REWARD

            vel = v.get_velocity()
            speed = math.hypot(vel.x, vel.y)
            r_speed = SPEED_REWARD_SCALE * min(speed / MAX_SPEED_MPS, 1.0)

            # local route progress reward (FIX: never penalize noisy progress)
            r_route = 0.0
            try:
                route_info = self.waypoint_manager.update_route_progress(vid, v.get_location())
                cur_dist = float(route_info.get("dist_to_next_wp", 0.0))
                prev_dist = float(self._prev_dist_to_wp.get(vid, cur_dist))

                local_delta = prev_dist - cur_dist
                local_delta = max(local_delta, 0.0)                 # FIX: no negative progress penalty
                local_delta = float(np.clip(local_delta, 0.0, 5.0))

                r_route = local_delta * REWARD_ALPHA

                self._prev_dist_to_wp[vid] = cur_dist
            except:
                r_route = 0.0

            r_spacing = 0.0
            ndata = self._vehicle_neighbor_cache.get(vid)
            if ndata:
                nnd = ndata["nnd"]
                if nnd < THEORETICAL_SAFE_DISTANCE:
                    # Exponential penalty as distance -> 0
                    # Formula: - (Target / Actual) * Scaling
                    # Example: at 2.5m (half safe dist), penalty is -40.0
                    proximity_penalty = -(THEORETICAL_SAFE_DISTANCE / (nnd + 1e-6)) * 20.0
                    r_spacing = float(np.clip(proximity_penalty, MAX_PROXIMITY_PENALTY, -0.5))
                elif nnd > SAFE_FOLLOWING_DIST:
                    r_spacing = P_SPACING_REWARD
                # if nnd > SAFE_FOLLOWING_DIST:
                #     r_spacing = P_SPACING_REWARD
                # elif nnd < CRITICAL_PROXIMITY_THRESHOLD + 2.0:
                #     r_spacing = P_TAILGATING_PENALTY

            r_col = 0.0
            if self.collision_histories.get(vid):
                r_col = P_COLLISION_STATIC
                for event in self.collision_histories[vid]:
                    other = event.other_actor
                    if other.type_id.startswith("vehicle") or other.type_id.startswith("walker"):
                        r_col = P_COLLISION_EGO
                        break

            r_stuck = stuck_penalties.get(vid, 0.0)

            per_vehicle_reward = r_alive + r_speed + r_route + r_spacing + r_col + r_stuck
            per_vehicle_reward = float(np.clip(per_vehicle_reward,
                                               -PER_VEHICLE_REWARD_CLIP,
                                               PER_VEHICLE_REWARD_CLIP))  # FIX: stabilize critic

            total_reward += per_vehicle_reward

        return float(total_reward)

    def _get_observations(self):
        obs = []

        # 1. Loop through actual vehicles
        for v in self.vehicles:
            if v.is_alive:
                try:
                    t = v.get_transform()
                    vel = v.get_velocity()
                    speed = math.hypot(vel.x, vel.y)

                    # 10 features per vehicle
                    vehicle_features = [
                        t.location.x / 1000.0,
                        t.location.y / 1000.0,
                        t.rotation.yaw / 180.0,
                        vel.x / MAX_SPEED_MPS,
                        vel.y / MAX_SPEED_MPS,
                        speed / MAX_SPEED_MPS,
                    ]

                    route_info = self.waypoint_manager.update_route_progress(v.id, t.location)
                    vehicle_features.append(route_info.get("dist_to_next_wp", 0.0) / 100.0)

                    ndata = self._vehicle_neighbor_cache.get(v.id)
                    if ndata:
                        nnd = ndata["nnd"]
                        vehicle_features.extend([nnd / 100.0, 0.0, 0.0])
                    else:
                        vehicle_features.extend([1.0, 0.0, 0.0])

                    obs.extend(vehicle_features)

                except:
                    obs.extend([0.0] * self.VEHICLE_FEATURE_SIZE)
            else:
                obs.extend([0.0] * self.VEHICLE_FEATURE_SIZE)

        # --- FIX START: PADDING ---
        current_features_count = len(obs)
        target_len = self.MAX_VEHICLES_CAPACITY * self.VEHICLE_FEATURE_SIZE

        if current_features_count < target_len:
            padding_needed = target_len - current_features_count
            obs.extend([0.0] * padding_needed)

        final_obs = obs[:target_len]
        return np.array(final_obs, dtype=np.float32)
        # --- FIX END ---

    # ==============================================================================
    # SPAWN / STUCK / CLEANUP
    # ==============================================================================

    def _spawn_vehicles_simple(self):
        """Simple stable spawn: distance-filtered spawn points."""
        bp_lib = self.world.get_blueprint_library().filter("vehicle.*")
        safe_bp_lib = [x for x in bp_lib if int(x.get_attribute("number_of_wheels")) == 4]
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return

        random.shuffle(spawn_points)

        spawned_locs = [v.get_location() for v in self.vehicles if v.is_alive]
        count = len(self.vehicles)

        for tf in spawn_points:
            if count >= self.num_vehicles:
                break

            ok = True
            for loc in spawned_locs:
                if tf.location.distance(loc) < MIN_SPAWN_SEPARATION:
                    ok = False
                    break
            if not ok:
                continue

            tf.location.z += 0.5
            bp = random.choice(safe_bp_lib)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

            vehicle = self.world.try_spawn_actor(bp, tf)
            if vehicle:
                vehicle.set_autopilot(False)
                self.vehicles.append(vehicle)
                spawned_locs.append(vehicle.get_location())
                count += 1

                sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
                sensor.listen(lambda event: self._on_collision(event))
                self.sensors.append(sensor)

                self.world.tick()

    def _handle_stuck_vehicles(self):
        penalties = {}
        delta_time = 0.05

        for v in self.vehicles:
            if not v.is_alive:
                continue

            vid = v.id
            if vid not in self._stuck_counter:
                self._stuck_counter[vid] = 0.0

            speed = math.hypot(v.get_velocity().x, v.get_velocity().y)
            if speed < 0.2:
                self._stuck_counter[vid] += delta_time
                if self._stuck_counter[vid] > STUCK_TIME_THRESHOLD:
                    penalties[vid] = P_STUCK_CUMULATIVE
            else:
                self._stuck_counter[vid] = 0.0

        return penalties

    def _setup_carla(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(300.0)

        try:
            self.world = self.client.get_world()
            if self.town not in self.world.get_map().name:
                self.world = self.client.load_world(self.town)
        except:
            self.world = self.client.load_world(self.town)

        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(int(time.time()))
        self.tm.global_percentage_speed_difference(-30.0)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.spawn_points = self.world.get_map().get_spawn_points()

    def _on_collision(self, event):
        actor_id = event.actor.id
        self.collision_histories[actor_id].append(event)
        other = event.other_actor
        if other.type_id.startswith("vehicle") or other.type_id.startswith("walker"):
            self.total_ego_collisions += 1

    def _cleanup_vehicles_only(self):
        if self.client:
            for sensor in self.sensors:
                if sensor.is_alive:
                    sensor.destroy()
            self.sensors = []
            batch = [carla.command.DestroyActor(v) for v in self.vehicles]
            self.client.apply_batch(batch)

        self.vehicles = []

    def _spawn_pedestrians(self):
        if self.pedestrian_count <= 0:
            return
        walker_spawns = [pt for pt in self.spawn_points if pt.location.z <= 1.0]
        bp_lib = self.world.get_blueprint_library().filter("walker.pedestrian.*")

        for _ in range(self.pedestrian_count):
            try:
                sp = random.choice(walker_spawns)
                bp = random.choice(bp_lib)
                walker = self.world.try_spawn_actor(bp, sp)
                if walker:
                    self.pedestrians.append(walker)
            except:
                pass

    def _trigger_emergency_jaywalk(self):
        if not self.vehicles:
            return
        try:
            target_vehicle = random.choice(self.vehicles)
            loc = target_vehicle.get_location()
            fwd = target_vehicle.get_transform().get_forward_vector()
            right = target_vehicle.get_transform().get_right_vector()
            spawn_loc = loc + (fwd * 15.0) + (right * 4.0)
            spawn_loc.z += 1.0

            bp = random.choice(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
            walker = self.world.try_spawn_actor(bp, carla.Transform(spawn_loc))

            if walker:
                direction = -right
                length = math.sqrt(direction.x ** 2 + direction.y ** 2 + direction.z ** 2)
                if length > 0:
                    direction.x /= length
                    direction.y /= length
                    direction.z /= length
                control = carla.WalkerControl(direction, 3.0, False)
                walker.apply_control(control)
                self.pedestrians.append(walker)
        except:
            pass

    def _cleanup(self):
        if self.fleet_coordinator:
            self.fleet_coordinator.stop_coordination()

        if self.client:
            for s in self.sensors:
                if s.is_alive:
                    s.destroy()
            self.sensors = []

            batch = [carla.command.DestroyActor(v) for v in self.vehicles]
            if self.pedestrians:
                batch += [carla.command.DestroyActor(p) for p in self.pedestrians if p.is_alive]
            self.client.apply_batch(batch)

        self.vehicles = []
        self.pedestrians = []
        self.waypoint_manager = None
        self.fleet_coordinator = None
        self._stuck_counter = {}
        self._prev_dist_to_wp = {}

    def close(self):
        self._cleanup()
        if self.world:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except:
                pass
