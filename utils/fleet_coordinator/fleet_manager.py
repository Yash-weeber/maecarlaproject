import numpy as np
import time
from collections import defaultdict, deque
import threading
from threading import Lock, Event
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Deque, Set

# --- Mock CARLA Classes (Prevents crash if CARLA isn't ready yet) ---
class MockLocation:
    """Mock for carla.Location"""
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    def distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    def __repr__(self):
        return f"MockLocation({self.x:.2f}, {self.y:.2f})"
    # Fix: Added vector math support for environment calculations
    def __sub__(self, other):
        return MockLocation(self.x - other.x, self.y - other.y, self.z - other.z)
    def __truediv__(self, scalar):
        return MockLocation(self.x / scalar, self.y / scalar, self.z / scalar)
    def length(self):
        return self.distance(MockLocation(0,0,0))

class MockVector3D:
    """Mock for carla.Vector3D"""
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    # Fix: Added dot product for environment calculations
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

# Dynamic type assignment
CARLA_LOCATION = 'carla.Location'
CARLA_VECTOR3D = 'carla.Vector3D'
try:
    import carla
    CARLA_LOCATION = carla.Location
    CARLA_VECTOR3D = carla.Vector3D
except ImportError:
    CARLA_LOCATION = MockLocation
    CARLA_VECTOR3D = MockVector3D
# ------------------------------------------------------------

# Configuration Constants
UPDATE_INTERVAL = 0.5
PREDICTION_HORIZON = 10.0
CONFLICT_DETECTION_RADIUS = 30.0
CRITICAL_CONFLICT_DISTANCE = 5.0
GRID_CELL_SIZE = 50
RE_ROUTE_PROGRESS_THRESHOLD = 0.3
RE_ROUTE_MIN_DISTANCE = 100.0


@dataclass
class VehicleState:
    """Data class for vehicle state information"""
    vehicle_id: int
    location: CARLA_LOCATION
    velocity: CARLA_VECTOR3D
    heading: float
    route_progress: float
    destination_id: Optional[int] = None
    last_update: float = 0.0


@dataclass
class Conflict:
    """Data class for conflict detection"""
    conflict_type: str
    vehicles: List[int]
    severity: float
    location: CARLA_LOCATION
    estimated_time: float
    resolution_priority: float


class FleetCoordinator:
    """Advanced fleet coordination system for conflict resolution and optimization"""

    def __init__(self, waypoint_manager, max_vehicles=20):
        self.waypoint_manager = waypoint_manager
        self.max_vehicles = max_vehicles

        # --- THREAD SAFETY ---
        self.lock = Lock()
        self.is_ready = Event()
        
        # Queue for suggested actions to be executed by the main thread
        self.pending_commands: Deque[Tuple[str, int]] = deque() 

        # Vehicle tracking
        self.vehicle_states: Dict[int, VehicleState] = {}
        self.vehicle_priorities: Dict[int, float] = {}

        # Conflict management
        self.active_conflicts: List[Conflict] = []
        # FIX: Increased history size to prevent dropping events too early
        self.conflict_history: deque = deque(maxlen=1000) 
        # FIX: Track active pairs to prevent duplicate logging (The "Spike" Fix)
        self.active_conflict_pairs: Set[Tuple[int, int]] = set()
        
        self.conflict_resolver = ConflictResolver() 

        # Performance optimization
        self.route_optimizer = RouteOptimizer() 
        self.traffic_predictor = TrafficPredictor() 

        # Threading
        self.coordination_thread = None
        self.running = False

    def start_coordination(self):
        """Start the fleet coordination system"""
        if self.running:
            return

        self.running = True
        self.coordination_thread = threading.Thread(target=self._coordination_loop, name="FleetCoordThread")
        self.coordination_thread.daemon = True
        self.coordination_thread.start()

    def stop_coordination(self):
        """Stop the fleet coordination system"""
        self.running = False
        self.is_ready.set()
        if self.coordination_thread:
            self.coordination_thread.join(timeout=2.0)

    def _update_vehicle_states(self):
        """Placeholder for coordination loop compliance."""
        pass

    def _coordination_loop(self):
        """Main coordination loop running in separate thread"""
        
        print("Fleet Coordinator waiting for vehicle states...")
        if not self.is_ready.wait(timeout=10.0):
             print("Warning: Fleet Coordinator timed out waiting for vehicle states.")
             self.running = False
             return
        print("Fleet Coordinator started.")

        while self.running:
            try:
                start_time = time.time()

                self._update_vehicle_states()

                with self.lock:
                    conflicts = self._detect_conflicts()

                    for conflict in conflicts:
                        self._resolve_conflict(conflict)
                    
                    self._optimize_routes()

                self._update_traffic_predictions()

                elapsed = time.time() - start_time
                sleep_time = max(0, UPDATE_INTERVAL - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Fleet coordination error: {e}")
                time.sleep(UPDATE_INTERVAL)

    def get_pending_commands(self) -> List[Tuple[str, int]]:
        """Retrieves and clears the queue of pending commands (Call this from Main Thread)."""
        with self.lock:
            commands = list(self.pending_commands)
            self.pending_commands.clear()
            return commands

    def update_vehicle_state(self, vehicle_id: int, location, velocity, heading, route_progress):
        """Update state for a specific vehicle (Thread Safe)"""
        with self.lock:
            # Note: We must ensure the correct Location/Vector3D type is used if CARLA is imported
            # This handles type consistency between mock and real objects
            LocType = type(self.vehicle_states[vehicle_id].location) if vehicle_id in self.vehicle_states else CARLA_LOCATION
            VelType = type(self.vehicle_states[vehicle_id].velocity) if vehicle_id in self.vehicle_states else CARLA_VECTOR3D
            
            self.vehicle_states[vehicle_id] = VehicleState(
                vehicle_id=vehicle_id,
                location=location,
                velocity=velocity,
                heading=heading,
                route_progress=route_progress,
                last_update=time.time()
            )
            self._update_vehicle_priority(vehicle_id)
        
        self.is_ready.set()
    
    def get_all_states(self) -> Dict[int, VehicleState]:
        """
        Allow environment to access states for neighbor calculation.
        Required for the targeted shield logic.
        """
        with self.lock:
            return self.vehicle_states.copy()

    def _update_vehicle_priority(self, vehicle_id: int):
        """Update vehicle priority for conflict resolution"""
        if vehicle_id not in self.vehicle_states:
            return

        state = self.vehicle_states[vehicle_id]
        priority = 0.5

        speed = np.sqrt(state.velocity.x ** 2 + state.velocity.y ** 2 + state.velocity.z ** 2)
        priority += state.route_progress * 0.2
        priority += min(speed / 50.0, 0.2)

        recent_conflicts = sum(1 for c in self.conflict_history
                               if vehicle_id in c.vehicles and
                               time.time() - c.estimated_time < 300.0)
        priority -= recent_conflicts * 0.1

        self.vehicle_priorities[vehicle_id] = np.clip(priority, 0.1, 1.0)

    def _detect_conflicts(self) -> List[Conflict]:
        """Detect potential conflicts between vehicles (Fixes P4 Metrics)"""
        conflicts = []
        vehicle_ids = list(self.vehicle_states.keys())
        
        # Track which pairs are conflicting IN THIS STEP
        current_step_conflict_pairs = set()

        for i in range(len(vehicle_ids)):
            for j in range(i + 1, len(vehicle_ids)):
                v1_id, v2_id = vehicle_ids[i], vehicle_ids[j]
                
                # Unique pair ID (sorted tuple ensures (1,2) is same as (2,1))
                pair_id = tuple(sorted((v1_id, v2_id)))

                # 1. Spatial Check
                spatial_conflict = self._check_spatial_conflict(v1_id, v2_id)
                if spatial_conflict:
                    conflicts.append(spatial_conflict)
                    current_step_conflict_pairs.add(pair_id)
                    
                    # Only log to history if this is a NEW conflict (prevents metric spikes)
                    if pair_id not in self.active_conflict_pairs:
                        self.conflict_history.append(spatial_conflict)

                # 2. Route Check
                route_conflict = self._check_route_conflict(v1_id, v2_id)
                if route_conflict:
                    conflicts.append(route_conflict)
                    if pair_id not in current_step_conflict_pairs:
                        current_step_conflict_pairs.add(pair_id)
                        if pair_id not in self.active_conflict_pairs:
                             self.conflict_history.append(route_conflict)

        # Update active pairs so we don't double count next tick
        self.active_conflict_pairs = current_step_conflict_pairs
        
        self.active_conflicts = conflicts
        return conflicts

    def _check_spatial_conflict(self, v1_id: int, v2_id: int) -> Optional[Conflict]:
        """Check for spatial conflicts between two vehicles"""
        state1 = self.vehicle_states.get(v1_id)
        state2 = self.vehicle_states.get(v2_id)

        if not state1 or not state2:
            return None

        current_distance = state1.location.distance(state2.location)

        if current_distance > CONFLICT_DETECTION_RADIUS:
            return None

        future_pos1 = self._predict_future_position(state1, PREDICTION_HORIZON)
        future_pos2 = self._predict_future_position(state2, PREDICTION_HORIZON)

        future_distance = future_pos1.distance(future_pos2)

        if future_distance < CRITICAL_CONFLICT_DISTANCE:
            severity = 1.0 - (future_distance / CRITICAL_CONFLICT_DISTANCE)
            
            LocationType = type(state1.location)
            conflict_location = LocationType(
                x=(future_pos1.x + future_pos2.x) / 2,
                y=(future_pos1.y + future_pos2.y) / 2,
                z=(future_pos1.z + future_pos2.z) / 2
            )

            return Conflict(
                conflict_type='spatial',
                vehicles=[v1_id, v2_id],
                severity=severity,
                location=conflict_location,
                estimated_time=time.time() + PREDICTION_HORIZON,
                resolution_priority=severity
            )

        return None

    def _predict_future_position(self, state: VehicleState, time_horizon: float):
        """Predict future position based on current velocity"""
        future_x = state.location.x + state.velocity.x * time_horizon
        future_y = state.location.y + state.velocity.y * time_horizon
        future_z = state.location.z + state.velocity.z * time_horizon

        return type(state.location)(x=future_x, y=future_y, z=future_z)

    def _check_route_conflict(self, v1_id: int, v2_id: int) -> Optional[Conflict]:
        """Check for route/destination conflicts"""
        v1_route = self.waypoint_manager.active_routes.get(v1_id)
        v2_route = self.waypoint_manager.active_routes.get(v2_id)

        if not v1_route or not v2_route:
            return None

        if v1_route.get('destination_id') == v2_route.get('destination_id') and v1_route.get('destination_id') is not None:
            v1_progress = v1_route.get('progress', 0.0)
            v2_progress = v2_route.get('progress', 0.0)

            severity = abs(v1_progress - v2_progress)

            if severity < 0.3:
                destination = v1_route.get('destination', {}).get('location')
                if destination is None: return None

                return Conflict(
                    conflict_type='destination',
                    vehicles=[v1_id, v2_id],
                    severity=severity,
                    location=destination,
                    estimated_time=time.time(),
                    resolution_priority=1.0 - severity
                )

        return None

    def _resolve_conflict(self, conflict: Conflict):
        """Resolve a detected conflict"""
        if conflict.conflict_type == 'spatial':
            self._resolve_spatial_conflict(conflict)
        elif conflict.conflict_type == 'destination':
            self._resolve_destination_conflict(conflict)

    #     # Note: We add to history in _detect_conflicts to handle deduplication properly.
    #
    # def _resolve_spatial_conflict(self, conflict: Conflict):
    #     """Resolve spatial conflict between vehicles"""
    #     vehicles = conflict.vehicles
    #     if len(vehicles) != 2: return
    #
    #     v1_id, v2_id = vehicles
    #     v1_priority = self.vehicle_priorities.get(v1_id, 0.5)
    #     v2_priority = self.vehicle_priorities.get(v2_id, 0.5)
    #
    #     if v1_priority < v2_priority:
    #         yielding_vehicle = v1_id
    #         priority_vehicle = v2_id
    #     elif v2_priority < v1_priority:
    #         yielding_vehicle = v2_id
    #         priority_vehicle = v1_id
    #     else:
    #         # Tie-break using route progress
    #         v1_state = self.vehicle_states.get(v1_id)
    #         v2_state = self.vehicle_states.get(v2_id)
    #         if v1_state and v2_state:
    #             yielding_vehicle = v1_id if v1_state.route_progress < v2_state.route_progress else v2_id
    #         else:
    #             return
    #
    #     self._apply_yielding_behavior(yielding_vehicle, priority_vehicle, conflict)
    #
    #     # Queue command for main thread
    #     self.pending_commands.append(("APPLY_YIELD", yielding_vehicle))

    def _resolve_spatial_conflict(self, conflict: Conflict):
        """Resolve spatial conflict between vehicles"""
        vehicles = conflict.vehicles
        if len(vehicles) != 2: return

        v1_id, v2_id = vehicles
        v1_priority = self.vehicle_priorities.get(v1_id, 0.5)
        v2_priority = self.vehicle_priorities.get(v2_id, 0.5)

        if v1_priority < v2_priority:
            yielding_vehicle = v1_id
            priority_vehicle = v2_id
        elif v2_priority < v1_priority:
            yielding_vehicle = v2_id
            priority_vehicle = v1_id
        else:
            # Tie-break using route progress
            v1_state = self.vehicle_states.get(v1_id)
            v2_state = self.vehicle_states.get(v2_id)
            if v1_state and v2_state:
                if v1_state.route_progress < v2_state.route_progress:
                    yielding_vehicle = v1_id
                    priority_vehicle = v2_id  # <--- WAS MISSING
                else:
                    yielding_vehicle = v2_id
                    priority_vehicle = v1_id  # <--- WAS MISSING
            else:
                return

        self._apply_yielding_behavior(yielding_vehicle, priority_vehicle, conflict)

        # Queue command for main thread
        self.pending_commands.append(("APPLY_YIELD", yielding_vehicle))
    def _resolve_destination_conflict(self, conflict: Conflict):
        """Resolve destination conflict by reassigning route"""
        vehicles = conflict.vehicles
        if len(vehicles) != 2: return

        v1_id, v2_id = vehicles
        v1_route = self.waypoint_manager.active_routes.get(v1_id)
        v2_route = self.waypoint_manager.active_routes.get(v2_id)

        if not v1_route or not v2_route: return

        if v1_route.get('progress', 0) < v2_route.get('progress', 0):
            reassign_vehicle = v1_id
        else:
            reassign_vehicle = v2_id

        self.pending_commands.append(("REASSIGN_ROUTE", reassign_vehicle))
        # print(f"Queued route reassignment for vehicle {reassign_vehicle}")

    def _apply_yielding_behavior(self, yielding_vehicle: int, priority_vehicle: int, conflict: Conflict):
        """Apply yielding behavior to resolve spatial conflict"""
        # Logging disabled for performance
        pass

    def _optimize_routes(self):
        """Optimize routes globally for efficiency"""
        if len(self.vehicle_states) < 2: return

        route_usage = defaultdict(int)

        for vehicle_id, route_info in self.waypoint_manager.active_routes.items():
            if 'waypoints' in route_info:
                for waypoint in route_info['waypoints']:
                    loc = getattr(waypoint.transform, 'location', MockLocation())
                    grid_x = int(getattr(loc, 'x', 0) // GRID_CELL_SIZE)
                    grid_y = int(getattr(loc, 'y', 0) // GRID_CELL_SIZE)
                    route_usage[(grid_x, grid_y)] += 1

        congested_areas = {pos: count for pos, count in route_usage.items() if count > 2}

        if congested_areas:
            self._suggest_alternative_routes(congested_areas)

    def _suggest_alternative_routes(self, congested_areas: Dict):
        """Suggest alternative routes for vehicles in congested areas"""
        for vehicle_id, route_info in self.waypoint_manager.active_routes.items():
            if vehicle_id not in self.vehicle_states: continue

            in_congestion = False
            if 'waypoints' in route_info:
                for waypoint in route_info['waypoints']:
                    loc = getattr(waypoint.transform, 'location', MockLocation())
                    grid_x = int(getattr(loc, 'x', 0) // GRID_CELL_SIZE)
                    grid_y = int(getattr(loc, 'y', 0) // GRID_CELL_SIZE)
                    if (grid_x, grid_y) in congested_areas:
                        in_congestion = True
                        break

            if in_congestion and route_info.get('progress', 0) < RE_ROUTE_PROGRESS_THRESHOLD:
                current_location = self.vehicle_states[vehicle_id].location
                destination = route_info.get('destination', {}).get('location')

                try:
                    distance_check = current_location.distance(destination)
                except (AttributeError, TypeError):
                    distance_check = 0.0

                if distance_check > RE_ROUTE_MIN_DISTANCE:
                    self.pending_commands.append(("OPTIMIZE_ROUTE", vehicle_id))

    def _update_traffic_predictions(self):
        """Update traffic predictions for better coordination"""
        pass

    def get_coordination_statistics(self) -> Dict:
        """Get fleet coordination performance statistics"""
        with self.lock:
            current_time = time.time()
            
            # Count recent conflict EVENTS
            recent_conflicts = [c for c in self.conflict_history
                                if current_time - c.estimated_time < 300]

            stats = {
                'active_vehicles': len(self.vehicle_states),
                'active_conflicts': len(self.active_conflicts),
                'recent_conflicts': len(recent_conflicts),
                'conflict_rate': len(recent_conflicts) / max(len(self.vehicle_states), 1),
                'average_vehicle_priority': np.mean(
                    list(self.vehicle_priorities.values())) if self.vehicle_priorities else 0.0,
                'coordination_efficiency': self._calculate_coordination_efficiency()
            }
        return stats

    def _calculate_coordination_efficiency(self) -> float:
        """Calculate overall coordination efficiency"""
        if not self.vehicle_states: return 0.0

        conflict_penalty = len(self.active_conflicts) * 0.1
        utilization_bonus = len(self.vehicle_states) / self.max_vehicles

        efficiency = max(0.0, utilization_bonus - conflict_penalty)
        return min(efficiency, 1.0)


# Supporting classes for conflict resolution and route optimization

class ConflictResolver:
    """Helper class for conflict resolution algorithms"""
    def __init__(self):
        self.resolution_strategies = {
            'spatial': self._resolve_spatial_strategy,
            'destination': self._resolve_destination_strategy
        }
    def resolve(self, conflict: Conflict):
        strategy = self.resolution_strategies.get(conflict.conflict_type)
        if strategy: return strategy(conflict)
        return None
    def _resolve_spatial_strategy(self, conflict: Conflict): pass
    def _resolve_destination_strategy(self, conflict: Conflict): pass


class RouteOptimizer:
    """Helper class for route optimization"""
    def __init__(self):
        self.optimization_history = deque(maxlen=50)
    def optimize_route(self, current_route, traffic_data): pass


class TrafficPredictor:
    """Helper class for traffic prediction"""
    def __init__(self):
        self.traffic_history = deque(maxlen=100)
    def predict_traffic(self, location, time_horizon): pass
