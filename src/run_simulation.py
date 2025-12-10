"""
Main Simulation Runner
Orchestrates SUMO simulation and data collection
"""

import traci
import sumolib
import yaml
import os
import sys
from datetime import datetime
from tqdm import tqdm

from data_collector import DataCollector


class SimulationRunner:
    """Run SUMO simulation and collect data"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize simulation runner"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sumo_config = self.config['sumo']
        self.data_collector = None
    
    def start_sumo(self, gui=False):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if gui else "sumo"
        
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_config['config_file'],
            "--step-length", str(self.sumo_config['step_length']),
            "--no-warnings", "true",
            "--duration-log.disable", "true",
            "--no-step-log", "true"
        ]
        
        print(f"Starting SUMO with config: {self.sumo_config['config_file']}")
        print(f"GUI mode: {gui}")
        
        try:
            traci.start(sumo_cmd)
            print("SUMO started successfully!")
            return True
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            print("\nPlease ensure:")
            print("1. SUMO is installed and in your PATH")
            print("2. Network file exists (run generate_network.bat first)")
            print("3. All SUMO configuration files are present")
            return False
    
    def run_simulation(self, duration=None, gui=False, scenario_id=0):
        """Run a single simulation scenario"""
        if duration is None:
            duration = self.sumo_config['simulation_duration']
        
        print(f"\n{'='*70}")
        print(f"Running Simulation Scenario {scenario_id}")
        print(f"{'='*70}")
        print(f"Duration: {duration} seconds")
        
        # Start SUMO
        if not self.start_sumo(gui):
            return None
        
        # Initialize data collector
        self.data_collector = DataCollector()
        
        # Run simulation
        step = 0
        max_steps = int(duration / self.sumo_config['step_length'])
        
        print(f"Simulating {max_steps} steps...")
        
        with tqdm(total=max_steps, desc="Simulation Progress") as pbar:
            while step < max_steps:
                traci.simulationStep()
                
                # Collect data at each step
                self.data_collector.collect_step_data(step)
                
                step += 1
                pbar.update(1)
                
                # Check if simulation ended early
                if traci.simulation.getMinExpectedNumber() <= 0:
                    print("\nAll vehicles have left the simulation")
                    break
        
        print(f"\nSimulation complete! Collected data for {len(self.data_collector.data)} records")
        
        # Close SUMO
        traci.close()
        
        # Save data
        output_file = self.data_collector.save_data()
        
        return output_file
    
    def run_multiple_scenarios(self, num_scenarios=None, gui=False):
        """Run multiple simulation scenarios"""
        if num_scenarios is None:
            num_scenarios = self.sumo_config['num_scenarios']
        
        print(f"\n{'='*70}")
        print(f"RUNNING {num_scenarios} SIMULATION SCENARIOS")
        print(f"{'='*70}\n")
        
        data_files = []
        
        for scenario_id in range(num_scenarios):
            # Vary simulation parameters for diversity
            # You can modify traffic flows, signal timings, etc. here
            
            output_file = self.run_simulation(
                duration=self.sumo_config['simulation_duration'],
                gui=gui,
                scenario_id=scenario_id
            )
            
            if output_file:
                data_files.append(output_file)
            
            print(f"\nCompleted scenario {scenario_id + 1}/{num_scenarios}")
        
        print(f"\n{'='*70}")
        print(f"ALL SCENARIOS COMPLETE")
        print(f"{'='*70}")
        print(f"Generated {len(data_files)} data files:")
        for f in data_files:
            print(f"  - {f}")
        
        return data_files
    
    def quick_test(self, duration=300):
        """Run a quick test simulation"""
        print("\n" + "="*70)
        print("QUICK TEST SIMULATION (5 minutes)")
        print("="*70)
        
        return self.run_simulation(duration=duration, gui=False, scenario_id=0)


def merge_data_files(data_files, output_file='data/merged_data.csv'):
    """Merge multiple data files into one"""
    import pandas as pd
    
    print(f"\nMerging {len(data_files)} data files...")
    
    dfs = []
    for file in data_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    
    print(f"Merged data saved: {output_file}")
    print(f"Total records: {len(merged_df)}")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SUMO traffic simulation')
    parser.add_argument('--test', action='store_true', help='Run quick test (5 minutes)')
    parser.add_argument('--gui', action='store_true', help='Run with SUMO GUI')
    parser.add_argument('--scenarios', type=int, default=None, help='Number of scenarios to run')
    parser.add_argument('--duration', type=int, default=None, help='Simulation duration in seconds')
    
    args = parser.parse_args()
    
    runner = SimulationRunner()
    
    if args.test:
        # Quick test
        data_file = runner.quick_test()
        print(f"\nTest complete! Data saved to: {data_file}")
        print("\nNext steps:")
        print(f"1. Train models: python src/train_model.py {data_file}")
        print(f"2. Make predictions: python src/predict_risk.py {data_file}")
    
    elif args.scenarios:
        # Multiple scenarios
        data_files = runner.run_multiple_scenarios(num_scenarios=args.scenarios, gui=args.gui)
        
        # Merge data files
        if len(data_files) > 1:
            merged_file = merge_data_files(data_files)
            print(f"\nNext steps:")
            print(f"1. Train models: python src/train_model.py {merged_file}")
            print(f"2. Make predictions: python src/predict_risk.py {merged_file}")
    
    else:
        # Single scenario
        data_file = runner.run_simulation(duration=args.duration, gui=args.gui)
        print(f"\nSimulation complete! Data saved to: {data_file}")
        print("\nNext steps:")
        print(f"1. Train models: python src/train_model.py {data_file}")
        print(f"2. Make predictions: python src/predict_risk.py {data_file}")
