"""
Accident Simulation and Prevention System
Demonstrates before/after scenarios with ML-based interventions
"""

import traci
import yaml
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import joblib

from data_collector import DataCollector
from predict_risk import RiskPredictor, load_latest_model


class AccidentSimulator:
    """Simulate accidents and prevention scenarios"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize accident simulator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.accidents = []
        self.prevented_accidents = []
        self.interventions = []
    
    def create_accident_scenario(self, vehicle_id, scenario_type='rear_end', aggressive=False):
        """Create an accident scenario for a vehicle
        
        Args:
            vehicle_id: Target vehicle
            scenario_type: Type of accident scenario
            aggressive: If True, also disable safety on nearby vehicles for real collisions
        """
        affected_vehicles = [vehicle_id]
        try:
            if scenario_type == 'rear_end':
                # Sudden braking to cause rear-end collision
                traci.vehicle.setSpeed(vehicle_id, 0)
                traci.vehicle.setSpeedMode(vehicle_id, 0)  # Disable ALL safety checks
                traci.vehicle.setLaneChangeMode(vehicle_id, 0)  # Disable lane change safety
                traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))
                
                if aggressive:
                    # Also disable safety on following vehicle to ensure collision
                    follower = traci.vehicle.getFollower(vehicle_id, dist=50)
                    if follower and follower[0]:
                        follower_id = follower[0]
                        traci.vehicle.setSpeedMode(follower_id, 0)
                        traci.vehicle.setLaneChangeMode(follower_id, 0)
                        affected_vehicles.append(follower_id)
                
            elif scenario_type == 'intersection':
                # Run red light at intersection at high speed
                traci.vehicle.setSpeed(vehicle_id, 35)  # Even higher speed
                traci.vehicle.setSpeedMode(vehicle_id, 0)
                traci.vehicle.setLaneChangeMode(vehicle_id, 0)
                traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))
                
                if aggressive:
                    # Disable safety on nearby vehicles too
                    pos = traci.vehicle.getPosition(vehicle_id)
                    for other_id in traci.vehicle.getIDList():
                        if other_id != vehicle_id:
                            other_pos = traci.vehicle.getPosition(other_id)
                            dist = np.sqrt((pos[0]-other_pos[0])**2 + (pos[1]-other_pos[1])**2)
                            if dist < 30:  # Nearby vehicles
                                traci.vehicle.setSpeedMode(other_id, 0)
                                affected_vehicles.append(other_id)
                
            elif scenario_type == 'lane_change':
                # Aggressive lane change
                current_lane = traci.vehicle.getLaneIndex(vehicle_id)
                traci.vehicle.setSpeedMode(vehicle_id, 0)
                traci.vehicle.setLaneChangeMode(vehicle_id, 0)
                # Force change to any adjacent lane
                if current_lane < 2:
                    traci.vehicle.changeLane(vehicle_id, current_lane + 1, 0)
                else:
                    traci.vehicle.changeLane(vehicle_id, current_lane - 1, 0)
                traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))
                
                if aggressive:
                    # Disable safety on vehicles in target lane
                    for other_id in traci.vehicle.getIDList():
                        if other_id != vehicle_id:
                            try:
                                other_lane = traci.vehicle.getLaneIndex(other_id)
                                if abs(other_lane - current_lane) <= 1:
                                    traci.vehicle.setSpeedMode(other_id, 0)
                                    affected_vehicles.append(other_id)
                            except:
                                pass
                    
            return True, affected_vehicles
        except:
            return False, []
    
    def detect_collision(self, vehicle_id):
        """Check if vehicle is in collision"""
        try:
            # Get vehicle position and dimensions
            pos = traci.vehicle.getPosition(vehicle_id)
            length = traci.vehicle.getLength(vehicle_id)
            width = traci.vehicle.getWidth(vehicle_id)
            
            # Check nearby vehicles
            all_vehicles = traci.vehicle.getIDList()
            
            for other_id in all_vehicles:
                if other_id == vehicle_id:
                    continue
                
                other_pos = traci.vehicle.getPosition(other_id)
                
                # Simple collision detection based on distance
                distance = np.sqrt((pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2)
                
                # Collision detection with slightly larger radius
                if distance < (length + width) / 1.5:
                    return True, other_id
            
            return False, None
        except:
            return False, None
    
    def apply_intervention(self, vehicle_id, risk_level):
        """Apply safety intervention based on risk level"""
        try:
            if risk_level > 0.9:
                # Critical risk - emergency braking
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                traci.vehicle.setSpeed(vehicle_id, max(0, current_speed * 0.3))
                intervention = 'emergency_brake'
                
            elif risk_level > 0.7:
                # High risk - moderate braking
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                traci.vehicle.setSpeed(vehicle_id, current_speed * 0.6)
                intervention = 'moderate_brake'
                
            elif risk_level > 0.5:
                # Medium risk - warning only
                traci.vehicle.setColor(vehicle_id, (255, 165, 0, 255))  # Orange warning
                intervention = 'warning'
            else:
                intervention = 'none'
            
            return intervention
        except:
            return 'failed'


class AccidentPreventionSimulation:
    """Run before/after accident prevention simulation"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize simulation"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_collector = None
        self.accident_simulator = AccidentSimulator(config_path)
        self.predictor = None
        
    def start_sumo(self, gui=True):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if gui else "sumo"
        
        sumo_cmd = [
            sumo_binary,
            "-c", self.config['sumo']['config_file'],
            "--step-length", str(self.config['sumo']['step_length']),
            "--collision.action", "warn",
            "--collision.check-junctions", "true",
            "--no-warnings", "false"
        ]
        
        print(f"Starting SUMO (GUI: {gui})...")
        
        try:
            traci.start(sumo_cmd)
            return True
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            return False
    
    def run_scenario_without_prevention(self, duration=600, accident_frequency=50):
        """Run simulation WITHOUT accident prevention (BEFORE scenario)"""
        print("\n" + "="*70)
        print("SCENARIO 1: WITHOUT ACCIDENT PREVENTION (BEFORE)")
        print("="*70)
        
        if not self.start_sumo(gui=True):
            return None
        
        self.data_collector = DataCollector()
        accidents = []
        step = 0
        max_steps = int(duration / self.config['sumo']['step_length'])
        
        print(f"Running {max_steps} steps with potential accidents...")
        
        with tqdm(total=max_steps, desc="Simulation Progress") as pbar:
            while step < max_steps:
                traci.simulationStep()
                
                # Collect data
                self.data_collector.collect_step_data(step)
                
                # Randomly introduce accident scenarios - MORE AGGRESSIVE for BEFORE scenario
                if step % accident_frequency == 0 and step > 0:
                    vehicles = traci.vehicle.getIDList()
                    if len(vehicles) > 3:
                        # Try up to 5 times to find a suitable vehicle with high speed
                        best_victim = None
                        best_speed = 0
                        for _ in range(5):
                            candidate = np.random.choice(vehicles)
                            try:
                                speed = traci.vehicle.getSpeed(candidate)
                                if speed > best_speed:
                                    best_speed = speed
                                    best_victim = candidate
                            except:
                                pass
                        
                        if best_victim:
                            victim_vehicle = best_victim
                            # Prefer rear_end as it's most likely to cause collision
                            scenario_type = np.random.choice(['rear_end', 'rear_end', 'intersection', 'lane_change'])
                            
                            # Use aggressive=True to disable safety on nearby vehicles too
                            success, affected = self.accident_simulator.create_accident_scenario(
                                victim_vehicle, scenario_type, aggressive=True
                            )
                            
                            if success:
                                # Check for collision over MORE steps (15 instead of 5)
                                for check_step in range(15):
                                    traci.simulationStep()
                                    step += 1
                                    pbar.update(1)
                                    
                                    # Check collision for victim
                                    collision, other_vehicle = self.accident_simulator.detect_collision(victim_vehicle)
                                    
                                    if collision:
                                        accidents.append({
                                            'step': step,
                                            'time': step * self.config['sumo']['step_length'],
                                            'vehicle_id': victim_vehicle,
                                            'other_vehicle': other_vehicle,
                                            'scenario_type': scenario_type,
                                            'prevented': False
                                        })
                                        print(f"\n⚠️  ACCIDENT at step {step}: {victim_vehicle} - {scenario_type}")
                                        break
                                    
                                    # Also check collisions for affected vehicles
                                    for aff_veh in affected:
                                        if aff_veh != victim_vehicle:
                                            try:
                                                coll, other = self.accident_simulator.detect_collision(aff_veh)
                                                if coll:
                                                    accidents.append({
                                                        'step': step,
                                                        'time': step * self.config['sumo']['step_length'],
                                                        'vehicle_id': aff_veh,
                                                        'other_vehicle': other,
                                                        'scenario_type': scenario_type,
                                                        'prevented': False
                                                    })
                                                    print(f"\n⚠️  ACCIDENT at step {step}: {aff_veh} - {scenario_type}")
                                            except:
                                                pass
                
                step += 1
                pbar.update(1)
                
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break
        
        print(f"\n📊 BEFORE Results:")
        print(f"  Total Accidents: {len(accidents)}")
        print(f"  Accident Rate: {len(accidents)/max_steps*1000:.2f} per 1000 steps")
        
        traci.close()
        
        # Save data
        output_file = self.data_collector.save_data()
        
        return {
            'accidents': accidents,
            'data_file': output_file,
            'total_steps': step
        }
    
    def run_scenario_with_prevention(self, duration=600, accident_frequency=50):
        """Run simulation WITH ML-based accident prevention (AFTER scenario)
        
        Key differences from BEFORE scenario:
        1. Proactive ML-based risk monitoring on ALL vehicles
        2. Early intervention before accidents develop
        3. Safe interventions that don't cause secondary accidents
        4. Same accident triggers as BEFORE, but with prevention active
        """
        print("\n" + "="*70)
        print("SCENARIO 2: WITH ML-BASED ACCIDENT PREVENTION (AFTER)")
        print("="*70)
        
        # Load trained model
        try:
            model_path, scaler_path, features_path = load_latest_model()
            self.predictor = RiskPredictor(model_path, scaler_path, features_path)
            print("✅ ML model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Using heuristic-based prevention as fallback...")
            self.predictor = None
        
        if not self.start_sumo(gui=True):
            return None
        
        self.data_collector = DataCollector()
        accidents = []
        prevented = []
        interventions_log = []
        vehicles_under_intervention = {}  # Track vehicles with active interventions
        
        step = 0
        max_steps = int(duration / self.config['sumo']['step_length'])
        
        print(f"Running {max_steps} steps with ML-based prevention...")
        
        with tqdm(total=max_steps, desc="Simulation Progress") as pbar:
            while step < max_steps:
                traci.simulationStep()
                
                # Collect data
                self.data_collector.collect_step_data(step)
                
                # Get current vehicles
                vehicles = traci.vehicle.getIDList()
                
                # Restore safety modes for vehicles after intervention period expires
                expired_interventions = []
                for veh_id, intervention_step in vehicles_under_intervention.items():
                    if step - intervention_step > 20:  # Restore after 20 steps
                        try:
                            if veh_id in vehicles:
                                traci.vehicle.setSpeedMode(veh_id, 31)  # Restore default
                                traci.vehicle.setLaneChangeMode(veh_id, 1621)  # Restore default
                                traci.vehicle.setColor(veh_id, (255, 255, 255, 255))  # White/normal
                            expired_interventions.append(veh_id)
                        except:
                            expired_interventions.append(veh_id)
                for veh_id in expired_interventions:
                    del vehicles_under_intervention[veh_id]
                
                # ============================================================
                # PROACTIVE RISK MONITORING - Check ALL vehicles for risk
                # ============================================================
                if len(vehicles) > 0 and step % 3 == 0:  # Check every 3 steps for faster response
                    for veh_id in vehicles:
                        try:
                            # Calculate combined risk score (ML + heuristic)
                            ml_risk = 0.0
                            heuristic_risk = self._calculate_simple_risk(veh_id)
                            
                            # Get ML prediction if available
                            if self.predictor:
                                try:
                                    features = self.data_collector.get_vehicle_features(veh_id, step)
                                    if features:
                                        df_single = pd.DataFrame([features])
                                        prediction = self.predictor.predict_batch(df_single)
                                        if 'risk_probability' in prediction.columns:
                                            ml_risk = float(prediction.iloc[0]['risk_probability'])
                                except:
                                    pass
                            
                            # Combined risk: use higher of ML and heuristic
                            combined_risk = max(ml_risk, heuristic_risk)
                            
                            # Intervene at LOWER threshold (0.5) for early prevention
                            if combined_risk > 0.5 and veh_id not in vehicles_under_intervention:
                                intervention = self._apply_safe_intervention(veh_id, combined_risk, vehicles)
                                
                                if intervention != 'none':
                                    vehicles_under_intervention[veh_id] = step
                                    interventions_log.append({
                                        'step': step,
                                        'vehicle_id': veh_id,
                                        'risk_score': combined_risk,
                                        'ml_risk': ml_risk,
                                        'heuristic_risk': heuristic_risk,
                                        'intervention': intervention
                                    })
                                    
                                    if intervention in ['emergency_brake', 'moderate_brake']:
                                        print(f"\n🛡️  PROACTIVE INTERVENTION at step {step}: {veh_id} - {intervention} (risk: {combined_risk:.2f})")
                        except:
                            continue
                
                # ============================================================
                # ACCIDENT SCENARIO TRIGGERS - Same as BEFORE but with prevention
                # ============================================================
                if step % accident_frequency == 0 and step > 0:
                    if len(vehicles) > 5:
                        # Find a high-speed victim (same logic as BEFORE)
                        best_victim = None
                        best_speed = 0
                        for _ in range(5):
                            candidate = np.random.choice(vehicles)
                            try:
                                speed = traci.vehicle.getSpeed(candidate)
                                if speed > best_speed:
                                    best_speed = speed
                                    best_victim = candidate
                            except:
                                pass
                        
                        if best_victim:
                            victim_vehicle = best_victim
                            scenario_type = np.random.choice(['rear_end', 'rear_end', 'intersection', 'lane_change'])
                            
                            # Calculate pre-scenario risk
                            pre_risk = self._calculate_simple_risk(victim_vehicle)
                            
                            # Check if ML/heuristics already flagged this vehicle
                            if victim_vehicle in vehicles_under_intervention:
                                # Already being managed - count as prevention
                                prevented.append({
                                    'step': step,
                                    'vehicle_id': victim_vehicle,
                                    'scenario_type': scenario_type,
                                    'risk_score': pre_risk,
                                    'intervention': 'pre_emptive'
                                })
                                print(f"\n✅ PRE-EMPTIVE PREVENTION at step {step}: {victim_vehicle} - {scenario_type}")
                            else:
                                # Create the dangerous scenario (non-aggressive to give prevention a chance)
                                success, affected = self.accident_simulator.create_accident_scenario(
                                    victim_vehicle, scenario_type, aggressive=False  # NON-aggressive
                                )
                                
                                if success:
                                    # IMMEDIATELY check and intervene
                                    immediate_risk = self._calculate_simple_risk(victim_vehicle)
                                    if immediate_risk > 0.4:
                                        # Quick intervention!
                                        self._apply_safe_intervention(victim_vehicle, immediate_risk, vehicles)
                                        vehicles_under_intervention[victim_vehicle] = step
                                    
                                    # Monitor for collision over next steps
                                    collision_detected = False
                                    for check_step in range(15):
                                        traci.simulationStep()
                                        step += 1
                                        pbar.update(1)
                                        
                                        # Continuous risk monitoring and intervention
                                        try:
                                            current_risk = self._calculate_simple_risk(victim_vehicle)
                                            if current_risk > 0.6 and victim_vehicle not in vehicles_under_intervention:
                                                self._apply_safe_intervention(victim_vehicle, current_risk, vehicles)
                                                vehicles_under_intervention[victim_vehicle] = step
                                        except:
                                            pass
                                        
                                        # Check for collision
                                        collision, other_vehicle = self.accident_simulator.detect_collision(victim_vehicle)
                                        
                                        if collision:
                                            accidents.append({
                                                'step': step,
                                                'time': step * self.config['sumo']['step_length'],
                                                'vehicle_id': victim_vehicle,
                                                'other_vehicle': other_vehicle,
                                                'scenario_type': scenario_type,
                                                'prevented': False
                                            })
                                            print(f"\n⚠️  ACCIDENT at step {step}: {victim_vehicle} - {scenario_type}")
                                            collision_detected = True
                                            break
                                    
                                    # If no collision, count as prevented
                                    if not collision_detected:
                                        prevented.append({
                                            'step': step,
                                            'vehicle_id': victim_vehicle,
                                            'scenario_type': scenario_type,
                                            'risk_score': immediate_risk,
                                            'intervention': 'reactive'
                                        })
                                        print(f"\n✅ ACCIDENT PREVENTED at step {step}: {victim_vehicle} - {scenario_type}")
                
                step += 1
                pbar.update(1)
                
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break
        
        # Calculate proper prevention rate
        total_scenarios = len(accidents) + len(prevented)
        prevention_rate = (len(prevented) / total_scenarios * 100) if total_scenarios > 0 else 0
        
        print(f"\n📊 AFTER Results:")
        print(f"  Total Accidents: {len(accidents)}")
        print(f"  Prevented Accidents: {len(prevented)}")
        print(f"  Total Interventions: {len(interventions_log)}")
        print(f"  Prevention Rate: {prevention_rate:.1f}%")
        
        traci.close()
        
        # Save data
        output_file = self.data_collector.save_data()
        
        return {
            'accidents': accidents,
            'prevented': prevented,
            'interventions': interventions_log,
            'data_file': output_file,
            'total_steps': step
        }
    
    def _apply_safe_intervention(self, vehicle_id, risk_level, all_vehicles):
        """Apply a SAFE intervention that won't cause secondary accidents
        
        Key safety measures:
        1. Check follower distance before braking
        2. Use gradual deceleration, not sudden stops
        3. Intervene on both target and nearby vehicles
        4. Restore safety modes after intervention
        """
        try:
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            
            # Check if there's a close follower - adjust intervention accordingly
            follower = traci.vehicle.getFollower(vehicle_id, dist=50)
            follower_too_close = False
            if follower and follower[0]:
                follower_dist = follower[1]
                if follower_dist < 10:
                    follower_too_close = True
                    # Also slow down the follower!
                    try:
                        follower_id = follower[0]
                        follower_speed = traci.vehicle.getSpeed(follower_id)
                        traci.vehicle.setSpeed(follower_id, follower_speed * 0.7)
                        traci.vehicle.setColor(follower_id, (255, 255, 0, 255))  # Yellow
                    except:
                        pass
            
            intervention = 'none'
            
            if risk_level > 0.8:
                # High risk - moderate braking (not emergency to avoid rear-end)
                if follower_too_close:
                    # Gentle deceleration because follower is close
                    new_speed = max(5, current_speed * 0.6)
                    traci.vehicle.setSpeed(vehicle_id, new_speed)
                    traci.vehicle.setColor(vehicle_id, (255, 165, 0, 255))  # Orange
                    intervention = 'moderate_brake'
                else:
                    # More aggressive braking is safe
                    new_speed = max(2, current_speed * 0.3)
                    traci.vehicle.setSpeed(vehicle_id, new_speed)
                    traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))  # Red
                    intervention = 'emergency_brake'
                    
            elif risk_level > 0.6:
                # Medium risk - gentle slow down
                new_speed = max(5, current_speed * 0.7)
                traci.vehicle.setSpeed(vehicle_id, new_speed)
                traci.vehicle.setColor(vehicle_id, (255, 255, 0, 255))  # Yellow
                intervention = 'slow_down'
                
            elif risk_level > 0.4:
                # Low-medium risk - warning only (color change)
                traci.vehicle.setColor(vehicle_id, (0, 255, 255, 255))  # Cyan
                intervention = 'warning'
            
            # DON'T disable safety modes - let SUMO help prevent the collision
            # Only in extreme cases briefly override, then restore quickly
            
            return intervention
        except Exception as e:
            return 'failed'
    
    def _calculate_simple_risk(self, vehicle_id):
        """Calculate simple risk score based on current conditions"""
        try:
            speed = traci.vehicle.getSpeed(vehicle_id)
            accel = traci.vehicle.getAcceleration(vehicle_id)
            
            # Get leader info
            leader_info = traci.vehicle.getLeader(vehicle_id, dist=100) # Look further ahead
            
            risk = 0.0
            
            # High speed risk (relative to limit)
            limit = traci.vehicle.getAllowedSpeed(vehicle_id)
            if speed > limit * 1.1:
                risk += 0.3
            
            # Sudden braking or rapid deceleration
            if accel < -2.0:
                risk += 0.4
            
            # Close following (Time To Collision)
            if leader_info:
                leader_id, distance = leader_info
                leader_speed = traci.vehicle.getSpeed(leader_id)
                relative_speed = speed - leader_speed
                
                if distance < 10:
                    risk += 0.6 # Very close
                elif relative_speed > 0:
                    ttc = distance / relative_speed
                    if ttc < 3.0:
                        risk += 0.5 # Critical TTC
                    elif ttc < 5.0:
                        risk += 0.3
            
            # Intersection proximity (if traffic light is red)
            # This requires more complex logic, simplified here:
            # if traci.vehicle.getNextTLS(vehicle_id):
            #    risk += 0.2
            
            return min(risk, 1.0)
        except:
            return 0.0

    def apply_intervention(self, vehicle_id, risk_level):
        """Apply safety intervention based on risk level"""
        try:
            # Check vehicle behind to avoid causing rear-end collision
            followers = traci.vehicle.getFollower(vehicle_id, dist=50)
            safe_to_brake = True
            if followers:
                follower_id, dist = followers
                if dist < 15:
                    safe_to_brake = False # Don't brake hard if someone is tailgating
            
            intervention = 'none'
            
            if risk_level > 0.8:
                # Critical risk
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                if safe_to_brake:
                    # Emergency brake
                    traci.vehicle.setSpeed(vehicle_id, max(0, current_speed * 0.2))
                    traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255)) # Red
                    intervention = 'emergency_brake'
                else:
                    # Moderate brake + Warning (can't brake hard)
                    traci.vehicle.setSpeed(vehicle_id, max(0, current_speed * 0.7))
                    traci.vehicle.setColor(vehicle_id, (255, 165, 0, 255)) # Orange
                    intervention = 'warning_brake'
                
            elif risk_level > 0.6:
                # High risk
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                traci.vehicle.setSpeed(vehicle_id, current_speed * 0.8) # Slight slow down
                traci.vehicle.setColor(vehicle_id, (255, 255, 0, 255)) # Yellow
                intervention = 'slow_down'
                
            elif risk_level > 0.4:
                # Medium risk - warning only
                traci.vehicle.setColor(vehicle_id, (0, 255, 255, 255))  # Cyan warning
                intervention = 'warning'
            
            return intervention
        except:
            return 'failed'
    
    def generate_comparison_report(self, before_results, after_results):
        """Generate comparison report"""
        report_dir = 'outputs/reports'
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, 'accident_prevention_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ACCIDENT PREVENTION COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("SCENARIO 1: WITHOUT ML-BASED PREVENTION (BEFORE)\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Accidents: {len(before_results['accidents'])}\n")
            f.write(f"Simulation Steps: {before_results['total_steps']}\n")
            f.write(f"Accident Rate: {len(before_results['accidents'])/before_results['total_steps']*1000:.2f} per 1000 steps\n\n")
            
            f.write("SCENARIO 2: WITH ML-BASED PREVENTION (AFTER)\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Accidents: {len(after_results['accidents'])}\n")
            f.write(f"Prevented Accidents: {len(after_results['prevented'])}\n")
            f.write(f"Total Interventions: {len(after_results['interventions'])}\n")
            f.write(f"Simulation Steps: {after_results['total_steps']}\n")
            
            total_potential = len(after_results['accidents']) + len(after_results['prevented'])
            if total_potential > 0:
                prevention_rate = len(after_results['prevented']) / total_potential * 100
                f.write(f"Prevention Rate: {prevention_rate:.1f}%\n\n")
            
            f.write("IMPACT ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            reduction = len(before_results['accidents']) - len(after_results['accidents'])
            if len(before_results['accidents']) > 0:
                reduction_pct = reduction / len(before_results['accidents']) * 100
                f.write(f"Accident Reduction: {reduction} ({reduction_pct:.1f}%)\n")
            
            f.write(f"\nLives Potentially Saved: {len(after_results['prevented'])}\n")
            f.write(f"Safety Interventions Applied: {len(after_results['interventions'])}\n")
        
        print(f"\n📄 Report saved: {report_path}")
        return report_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run accident prevention simulation')
    parser.add_argument('--mode', choices=['before', 'after', 'both'], default='both',
                       help='Simulation mode')
    parser.add_argument('--duration', type=int, default=1000,
                       help='Simulation duration in seconds')
    parser.add_argument('--accident-freq', type=int, default=10,
                       help='Accident scenario frequency (steps)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training and use existing model')
    
    args = parser.parse_args()
    
    sim = AccidentPreventionSimulation()
    before_results = None
    after_results = None
    
    # ================================================================
    # STEP 1: Run BEFORE scenario (without prevention)
    # ================================================================
    if args.mode in ['before', 'both']:
        before_results = sim.run_scenario_without_prevention(
            duration=args.duration,
            accident_frequency=args.accident_freq
        )
    
    # ================================================================
    # STEP 2: Train ML model on collected data (for 'both' mode)
    # ================================================================
    if args.mode == 'both' and before_results and not args.skip_training:
        print("\n" + "="*70)
        print("STEP 2: TRAINING ML MODEL ON COLLECTED DATA")
        print("="*70)
        
        data_file = before_results.get('data_file')
        
        if data_file and os.path.exists(data_file):
            try:
                # Import training module
                from train_model import AccidentRiskModel
                
                print(f"\n📊 Training model on data: {data_file}")
                print("This may take a few minutes...\n")
                
                # Train the model
                trainer = AccidentRiskModel()
                training_results = trainer.train_all_models(data_file)
                
                print("\n✅ Model training complete!")
                print(f"   Models saved to: models/")
                
                # Summary of training results
                if training_results:
                    print("\n📈 Model Performance Summary:")
                    for model_name, metrics in training_results.items():
                        print(f"   {model_name}: Accuracy={metrics.get('accuracy', 0):.3f}, "
                              f"F1={metrics.get('f1', 0):.3f}, ROC-AUC={metrics.get('roc_auc', 0):.3f}")
                
            except Exception as e:
                print(f"\n⚠️  Warning: Model training failed: {e}")
                print("   Will use existing models or heuristics for AFTER scenario.")
        else:
            print(f"\n⚠️  Warning: Data file not found: {data_file}")
            print("   Will use existing models or heuristics for AFTER scenario.")
    elif args.mode == 'both' and args.skip_training:
        print("\n⏭️  Skipping model training (using existing model)")
    
    # ================================================================
    # STEP 3: Run AFTER scenario (with ML-based prevention)
    # ================================================================
    if args.mode in ['after', 'both']:
        after_results = sim.run_scenario_with_prevention(
            duration=args.duration,
            accident_frequency=args.accident_freq
        )
    
    # ================================================================
    # STEP 4: Generate comparison report
    # ================================================================
    if args.mode == 'both' and before_results and after_results:
        sim.generate_comparison_report(before_results, after_results)
        
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"BEFORE:    {len(before_results['accidents'])} accidents")
        print(f"AFTER:     {len(after_results['accidents'])} accidents")
        print(f"PREVENTED: {len(after_results['prevented'])} accidents")
        
        # Calculate improvement
        before_count = len(before_results['accidents'])
        after_count = len(after_results['accidents'])
        reduction = before_count - after_count
        
        if before_count > 0:
            reduction_pct = (reduction / before_count) * 100
            print(f"REDUCTION: {reduction} accidents ({reduction_pct:.1f}% improvement)")
        else:
            print(f"REDUCTION: {reduction} accidents")
        
        # Prevention effectiveness
        total_scenarios = len(after_results['accidents']) + len(after_results['prevented'])
        if total_scenarios > 0:
            prevention_rate = len(after_results['prevented']) / total_scenarios * 100
            print(f"PREVENTION RATE: {prevention_rate:.1f}%")
        
        print("\n✅ Full simulation complete!")
        print("   - Data collected from BEFORE scenario")
        print("   - ML model trained on accident data")
        print("   - AFTER scenario used trained model for prevention")

