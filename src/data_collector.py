"""
Data Collector Module for SUMO Traffic Simulation
Extracts accident-risk features using TraCI
"""

import traci
import pandas as pd
import numpy as np
from collections import defaultdict
import yaml
import os
from datetime import datetime


class DataCollector:
    """Collects traffic data from SUMO simulation for accident risk prediction"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize data collector with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data = []
        self.vehicle_history = defaultdict(list)
        self.pedestrian_data = []
        self.intersection_data = []
        
        # Risk-related counters
        self.lane_change_counter = defaultdict(int)
        self.last_lane = {}
        
    def get_vehicle_features(self, veh_id, step):
        """Extract comprehensive features for a single vehicle"""
        try:
            # Basic kinematic features
            position = traci.vehicle.getPosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            acceleration = traci.vehicle.getAcceleration(veh_id)
            angle = traci.vehicle.getAngle(veh_id)
            
            # Lane and road information
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_position = traci.vehicle.getLanePosition(veh_id)
            lane_index = traci.vehicle.getLaneIndex(veh_id)
            road_id = traci.vehicle.getRoadID(veh_id)
            
            # Detect lane changes
            if veh_id in self.last_lane:
                if self.last_lane[veh_id] != lane_id:
                    self.lane_change_counter[veh_id] += 1
            self.last_lane[veh_id] = lane_id
            
            # Vehicle type and dimensions
            veh_type = traci.vehicle.getTypeID(veh_id)
            length = traci.vehicle.getLength(veh_id)
            width = traci.vehicle.getWidth(veh_id)
            
            # Leader vehicle (for TTC calculation)
            leader_info = traci.vehicle.getLeader(veh_id, dist=100)
            if leader_info:
                leader_id, distance = leader_info
                leader_speed = traci.vehicle.getSpeed(leader_id)
                # Time to Collision (TTC)
                if speed > leader_speed and speed > 0:
                    ttc = distance / (speed - leader_speed)
                else:
                    ttc = float('inf')
            else:
                distance = 100
                ttc = float('inf')
            
            # Traffic light information
            tls_id = traci.vehicle.getNextTLS(veh_id)
            if tls_id:
                tls_distance = tls_id[0][2]
                tls_state = tls_id[0][3]
            else:
                tls_distance = -1
                tls_state = 'none'
            
            # Waiting time (indicator of congestion)
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            accumulated_waiting = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            
            # Get nearby pedestrians (simulated - SUMO pedestrian support)
            pedestrian_proximity = self._get_pedestrian_proximity(position)
            
            # Calculate local traffic density
            traffic_density = self._calculate_traffic_density(position, road_id)
            
            # Historical features (if available)
            speed_variance = self._calculate_speed_variance(veh_id, speed)
            
            features = {
                'step': step,
                'timestamp': step * self.config['sumo']['step_length'],
                'vehicle_id': veh_id,
                'x': position[0],
                'y': position[1],
                'speed': speed,
                'acceleration': acceleration,
                'angle': angle,
                'lane_id': lane_id,
                'lane_position': lane_position,
                'lane_index': lane_index,
                'road_id': road_id,
                'vehicle_type': veh_type,
                'length': length,
                'width': width,
                'leader_distance': distance,
                'ttc': min(ttc, 999),  # Cap at 999 for numerical stability
                'tls_distance': tls_distance,
                'tls_state': tls_state,
                'waiting_time': waiting_time,
                'accumulated_waiting': accumulated_waiting,
                'lane_changes': self.lane_change_counter[veh_id],
                'pedestrian_proximity': pedestrian_proximity,
                'traffic_density': traffic_density,
                'speed_variance': speed_variance
            }
            
            return features
            
        except traci.exceptions.TraCIException as e:
            print(f"Error getting features for vehicle {veh_id}: {e}")
            return None
    
    def _get_pedestrian_proximity(self, position, radius=20):
        """Calculate proximity to pedestrians (simulated)"""
        # In a real scenario, this would query pedestrian positions
        # For now, we simulate pedestrian presence near intersections
        x, y = position
        
        # Check if near intersection (simplified)
        intersections = [(0, 0), (0, 200), (0, -200), (200, 0), (-200, 0), (200, 200)]
        
        min_dist = float('inf')
        for ix, iy in intersections:
            dist = np.sqrt((x - ix)**2 + (y - iy)**2)
            if dist < min_dist:
                min_dist = dist
        
        # Simulate pedestrian presence (higher near intersections)
        if min_dist < 30:
            # Random pedestrians near intersection
            return np.random.randint(0, 5)
        else:
            return 0
    
    def _calculate_traffic_density(self, position, road_id, radius=50):
        """Calculate local traffic density"""
        try:
            # Get all vehicles on the same road
            vehicles_on_road = traci.edge.getLastStepVehicleIDs(road_id)
            
            # Count vehicles within radius
            x, y = position
            nearby_count = 0
            
            for veh in vehicles_on_road:
                veh_pos = traci.vehicle.getPosition(veh)
                dist = np.sqrt((x - veh_pos[0])**2 + (y - veh_pos[1])**2)
                if dist < radius:
                    nearby_count += 1
            
            # Normalize by area (vehicles per 100m²)
            area = np.pi * radius**2 / 10000  # Convert to 100m²
            density = nearby_count / area if area > 0 else 0
            
            return density
            
        except:
            return 0
    
    def _calculate_speed_variance(self, veh_id, current_speed, window=10):
        """Calculate speed variance over recent history"""
        self.vehicle_history[veh_id].append(current_speed)
        
        # Keep only recent history
        if len(self.vehicle_history[veh_id]) > window:
            self.vehicle_history[veh_id] = self.vehicle_history[veh_id][-window:]
        
        if len(self.vehicle_history[veh_id]) > 1:
            return np.var(self.vehicle_history[veh_id])
        else:
            return 0
    
    def collect_step_data(self, step):
        """Collect data for all vehicles at current simulation step"""
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            features = self.get_vehicle_features(veh_id, step)
            if features:
                self.data.append(features)
        
        # Collect intersection-level aggregated data
        self._collect_intersection_data(step)
    
    def _collect_intersection_data(self, step):
        """Collect aggregated data at intersections"""
        intersections = ['center', 'north', 'south', 'east', 'west', 'northeast']
        
        for junction_id in intersections:
            try:
                # Get junction position
                junction_pos = traci.junction.getPosition(junction_id)
                
                # Count vehicles near junction
                all_vehicles = traci.vehicle.getIDList()
                vehicles_near = 0
                total_speed = 0
                total_waiting = 0
                
                for veh_id in all_vehicles:
                    veh_pos = traci.vehicle.getPosition(veh_id)
                    dist = np.sqrt((junction_pos[0] - veh_pos[0])**2 + 
                                 (junction_pos[1] - veh_pos[1])**2)
                    
                    if dist < 50:  # Within 50m of junction
                        vehicles_near += 1
                        total_speed += traci.vehicle.getSpeed(veh_id)
                        total_waiting += traci.vehicle.getWaitingTime(veh_id)
                
                avg_speed = total_speed / vehicles_near if vehicles_near > 0 else 0
                avg_waiting = total_waiting / vehicles_near if vehicles_near > 0 else 0
                
                self.intersection_data.append({
                    'step': step,
                    'timestamp': step * self.config['sumo']['step_length'],
                    'junction_id': junction_id,
                    'vehicles_count': vehicles_near,
                    'avg_speed': avg_speed,
                    'avg_waiting_time': avg_waiting
                })
                
            except:
                pass
    
    def save_data(self, output_dir='data'):
        """Save collected data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save vehicle data
        if self.data:
            df = pd.DataFrame(self.data)
            vehicle_file = os.path.join(output_dir, f'vehicle_data_{timestamp}.csv')
            df.to_csv(vehicle_file, index=False)
            print(f"Saved vehicle data: {vehicle_file} ({len(df)} records)")
        
        # Save intersection data
        if self.intersection_data:
            df_int = pd.DataFrame(self.intersection_data)
            intersection_file = os.path.join(output_dir, f'intersection_data_{timestamp}.csv')
            df_int.to_csv(intersection_file, index=False)
            print(f"Saved intersection data: {intersection_file} ({len(df_int)} records)")
        
        return vehicle_file if self.data else None
    
    def get_dataframe(self):
        """Return collected data as pandas DataFrame"""
        return pd.DataFrame(self.data)


if __name__ == "__main__":
    print("Data Collector Module - Use run_simulation.py to collect data")
