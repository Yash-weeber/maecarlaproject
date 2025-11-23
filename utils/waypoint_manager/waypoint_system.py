import carla
import numpy as np
import random
from collections import defaultdict, deque
import threading
import time

# --- Conditionally import GlobalRoutePlanner ---
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    print("Warning: CARLA GlobalRoutePlanner not available. Using simplified routing.")
    GlobalRoutePlanner = None


class WaymoWaypointManager:
    """Advanced waypoint management system mimicking Waymo operations"""

    def __init__(self, world, max_vehicles=20):
        self.world = world
        self.max_vehicles = max_vehicles
        self.map = world.get_map()

        # Waypoint and route management
        self.active_routes = {}
        self.destination_pool = []
        self.occupied_destinations = set()
        self.route_history = defaultdict(deque)

        # Performance tracking
        self.trip_statistics = defaultdict(dict)
        
        # Conditional instantiation based on whether the import succeeded
        if GlobalRoutePlanner is not None:
            self.global_route_optimizer = GlobalRoutePlanner(self.map, 2.0)
        else:
            # If import failed, set it to None. The routing function will use the fallback.
            print("INFO: Initializing without GlobalRoutePlanner (using simplified routing fallback).")
            self.global_route_optimizer = None
        
        # Threading for real-time updates
        self.update_thread = None
        self.running = False

        self._initialize_destination_pool()

    def _initialize_destination_pool(self):
        """Create diverse destination points across the map"""
        spawn_points = self.map.get_spawn_points()

        # Classify destinations by area type for realistic distribution
        self.destination_pool = []

        for i, spawn_point in enumerate(spawn_points):
            # Analyze location to determine area type
            area_type = self._classify_area(spawn_point.location)
            priority = self._calculate_priority(spawn_point, area_type)

            destination = {
                'id': i,
                'location': spawn_point.location,
                'transform': spawn_point,
                'area_type': area_type,
                'priority': priority,
                'usage_count': 0,
                'last_used': 0,
                'accessibility_score': self._calculate_accessibility(spawn_point)
            }

            self.destination_pool.append(destination)

        # Sort by priority for efficient selection
        self.destination_pool.sort(key=lambda x: x['priority'], reverse=True)

    def _classify_area(self, location):
        """Classify area type based on location characteristics"""
        # Simplified classification - can be enhanced with semantic segmentation
        x, y = location.x, location.y

        # Central areas (high activity)
        if abs(x) < 50 and abs(y) < 50:
            return 'downtown'
        elif abs(x) < 100 and abs(y) < 100:
            return 'commercial'
        elif abs(x) < 200 and abs(y) < 200:
            return 'residential'
        else:
            return 'suburban'

    def _calculate_priority(self, spawn_point, area_type):
        """Calculate destination priority based on area type and accessibility"""
        area_weights = {
            'downtown': 1.0,
            'commercial': 0.8,
            'residential': 0.6,
            'suburban': 0.4
        }

        base_priority = area_weights.get(area_type, 0.5)

        # Add randomness for variety
        return base_priority + random.uniform(-0.1, 0.1)

    def _calculate_accessibility(self, spawn_point):
        """Calculate how accessible a destination is"""
        # Count nearby waypoints and road connections
        waypoint = self.map.get_waypoint(spawn_point.location)
        if not waypoint:
            return 0.0

        # Check for multiple road connections
        connections = 0
        try:
            # Check left and right lanes
            if waypoint.get_left_lane():
                connections += 1
            if waypoint.get_right_lane():
                connections += 1

            # Check junction connectivity
            if waypoint.is_junction:
                connections += 2

        except:
            pass

        return min(connections / 5.0, 1.0)  # Normalize to [0, 1]

    def assign_route(self, vehicle_id, preferred_area=None):
        """Assign a route ensuring no destination conflicts"""
        # Get available destinations (not currently occupied)
        available_destinations = [
            dest for dest in self.destination_pool
            if dest['id'] not in self.occupied_destinations
        ]

        if not available_destinations:
            # Fallback: allow reuse if pool exhausted
            available_destinations = self.destination_pool

        # Filter by preferred area if specified
        if preferred_area:
            area_destinations = [
                dest for dest in available_destinations
                if dest['area_type'] == preferred_area
            ]
            if area_destinations:
                available_destinations = area_destinations

        # Avoid recently visited destinations
        recent_destinations = set(self.route_history[vehicle_id])
        available_destinations = [
            dest for dest in available_destinations
            if dest['id'] not in recent_destinations
        ]

        if not available_destinations:
            # If no unvisited destinations, use all available
            available_destinations = [
                dest for dest in self.destination_pool
                if dest['id'] not in self.occupied_destinations
            ]

        # Select destination based on priority and usage
        destination = self._select_optimal_destination(vehicle_id, available_destinations)

        if not destination:
            return False

        # Calculate route to destination
        current_location = self._get_vehicle_location(vehicle_id)
        if not current_location:
            return False

        route = self._calculate_optimal_route(current_location, destination['location'])

        if not route:
            return False
            
        # Calculate initial distance to destination for progress tracking
        initial_dist = current_location.distance(destination['location'])

        # Create route assignment
        route_assignment = {
            'destination_id': destination['id'],
            'destination': destination,
            'waypoints': route,
            'current_waypoint_index': 0,
            'start_time': time.time(),
            'estimated_duration': len(route) * 0.5,  # Rough estimate
            'status': 'active',
            'progress': 0.0,
            'last_dist_to_dest': initial_dist # For dist_delta calculation
        }

        # Update tracking data
        self.active_routes[vehicle_id] = route_assignment
        self.occupied_destinations.add(destination['id'])
        self.route_history[vehicle_id].append(destination['id'])

        # Limit history size
        if len(self.route_history[vehicle_id]) > 10:
            self.route_history[vehicle_id].popleft()

        # Update destination usage
        destination['usage_count'] += 1
        destination['last_used'] = time.time()

        return True

    def _select_optimal_destination(self, vehicle_id, available_destinations):
        """Select optimal destination using weighted selection"""
        if not available_destinations:
            return None

        # Calculate weights based on priority, usage, and time since last use
        current_time = time.time()
        weights = []

        for dest in available_destinations:
            # Base weight from priority
            weight = dest['priority']

            # Reduce weight for frequently used destinations
            usage_penalty = min(dest['usage_count'] * 0.1, 0.5)
            weight -= usage_penalty

            # Increase weight for destinations unused for a while
            time_since_use = current_time - dest['last_used']
            time_bonus = min(time_since_use / 3600.0, 0.3)  # Max 0.3 bonus after 1 hour
            weight += time_bonus

            # Accessibility bonus
            weight += dest['accessibility_score'] * 0.2

            weights.append(max(weight, 0.1))  # Minimum weight

        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(available_destinations)

        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0

        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return available_destinations[i]

        return available_destinations[-1]  # Fallback

    def update_route_progress(self, vehicle_id, current_position):
        """Update vehicle progress, handle transitions, and calculate metrics"""
        if vehicle_id not in self.active_routes:
            return {'status': 'no_route', 'progress': 0.0, 'dist_delta': 0.0, 'dist_to_next_wp': 0.0}

        route = self.active_routes[vehicle_id]
        waypoints = route['waypoints']
        current_index = route['current_waypoint_index']

        if current_index >= len(waypoints):
            return {'status': 'completed', 'progress': 1.0, 'dist_delta': 0.0, 'dist_to_next_wp': 0.0}

        # Check if reached current waypoint
        current_waypoint = waypoints[current_index]
        distance_to_waypoint = current_position.distance(current_waypoint.transform.location)
        
        # Metrics for observation/reward
        dist_to_next_wp = distance_to_waypoint

        if distance_to_waypoint < 8.0:  # 8 meter threshold
            route['current_waypoint_index'] += 1

            # Check if route completed
            if route['current_waypoint_index'] >= len(waypoints):
                return self._handle_route_completion(vehicle_id)

        # Update progress
        progress = route['current_waypoint_index'] / len(waypoints)
        route['progress'] = progress
        
        # Calculate distance delta (how much closer to destination did we get?)
        dest_loc = route['destination']['location']
        current_dist_to_dest = current_position.distance(dest_loc)
        last_dist = route.get('last_dist_to_dest', current_dist_to_dest)
        
        dist_delta = last_dist - current_dist_to_dest
        route['last_dist_to_dest'] = current_dist_to_dest

        # Get next waypoint info
        next_waypoint = None
        if route['current_waypoint_index'] < len(waypoints):
            next_waypoint = waypoints[route['current_waypoint_index']]

        return {
            'status': 'in_progress',
            'progress': progress,
            'next_waypoint': next_waypoint,
            'distance_to_waypoint': distance_to_waypoint,
            'waypoints_remaining': len(waypoints) - route['current_waypoint_index'],
            'dist_delta': dist_delta,      
            'dist_to_next_wp': dist_to_next_wp 
        }
        
    def get_route_status(self, vehicle_id):
        """Returns route completion status for termination checks."""
        if vehicle_id not in self.active_routes:
            return {'destination_reached': False}
        
        route = self.active_routes[vehicle_id]
        # Check internal status flag set by _handle_route_completion
        is_reached = route.get('status') == 'destination_reached'
        return {'destination_reached': is_reached}

    def _handle_route_completion(self, vehicle_id):
        """Handle vehicle reaching destination"""
        route = self.active_routes[vehicle_id]
        destination_id = route['destination_id']

        # Calculate trip statistics
        trip_duration = time.time() - route['start_time']
        self.trip_statistics[vehicle_id][destination_id] = {
            'duration': trip_duration,
            'completion_time': time.time(),
            'estimated_duration': route['estimated_duration'],
            'efficiency': route['estimated_duration'] / trip_duration if trip_duration > 0 else 1.0
        }

        # Free up destination
        if destination_id in self.occupied_destinations:
            self.occupied_destinations.remove(destination_id)

        # Remove active route logic deferred to re-assignment, but mark as done
        route['status'] = 'destination_reached'
        
        # Simulate passenger pickup/dropoff time
        wait_time = random.uniform(3.0, 8.0)

        return {
            'status': 'destination_reached',
            'progress': 1.0,
            'trip_duration': trip_duration,
            'wait_time': wait_time,
            'dist_delta': 0.0,
            'dist_to_next_wp': 0.0,
            'destination_reached': True
        }

    def _calculate_optimal_route(self, start_location, end_location):
        """Calculate optimal route between two locations"""
        start_waypoint = self.map.get_waypoint(start_location)
        end_waypoint = self.map.get_waypoint(end_location)

        if not start_waypoint or not end_waypoint:
            return None

        # Use CARLA's GlobalRoutePlanner if available (self.global_route_optimizer will be None if the import failed)
        if self.global_route_optimizer:
            try:
                route = self.global_route_optimizer.trace_route(
                    start_location, end_location
                )
                # Extract waypoints from the route
                waypoints = [waypoint for waypoint, _ in route]
                return waypoints if len(waypoints) > 1 else None
            except:
                # Fall back to simple routing if GlobalRoutePlanner fails
                pass

        # Simple routing fallback
        route = []
        current_waypoint = start_waypoint
        max_waypoints = 100

        for _ in range(max_waypoints):
            if current_waypoint.transform.location.distance(end_location) < 15.0:
                route.append(end_waypoint)
                break

            next_waypoints = current_waypoint.next(7.0)
            if not next_waypoints:
                break

            best_waypoint = min(
                next_waypoints,
                key=lambda wp: wp.transform.location.distance(end_location)
            )

            route.append(best_waypoint)
            current_waypoint = best_waypoint

        return route if len(route) > 1 else None

    def _get_vehicle_location(self, vehicle_id):
        """Get current vehicle location - to be implemented by environment"""
        # HACK: The environment should pass location to assign_route, 
        # but if not, we try to find it in the world using the ID directly.
        actor = self.world.get_actor(vehicle_id)
        if actor:
            return actor.get_location()
        return None

    def get_fleet_statistics(self):
        """Get comprehensive fleet performance statistics"""
        stats = {
            'active_routes': len(self.active_routes),
            'occupied_destinations': len(self.occupied_destinations),
            'total_destinations': len(self.destination_pool),
            'utilization_rate': len(self.occupied_destinations) / len(self.destination_pool),
            'average_trip_efficiency': 0.0,
            'destination_distribution': defaultdict(int)
        }

        # Calculate average efficiency
        total_efficiency = 0.0
        trip_count = 0

        for vehicle_trips in self.trip_statistics.values():
            for trip_data in vehicle_trips.values():
                total_efficiency += trip_data['efficiency']
                trip_count += 1

        if trip_count > 0:
            stats['average_trip_efficiency'] = total_efficiency / trip_count

        # Count destination distribution
        for dest in self.destination_pool:
            stats['destination_distribution'][dest['area_type']] += dest['usage_count']

        return stats
