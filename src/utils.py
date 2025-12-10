"""
Utility functions for the accident risk prediction system
"""

import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime


def load_config(config_path='config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'data',
        'models',
        'outputs',
        'outputs/visualizations',
        'outputs/reports',
        'sumo_network'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory ready: {directory}")


def get_data_summary(data_file):
    """Get summary statistics of data file"""
    df = pd.read_csv(data_file)
    
    summary = {
        'file': data_file,
        'total_records': len(df),
        'unique_vehicles': df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else 0,
        'time_range': (df['timestamp'].min(), df['timestamp'].max()) if 'timestamp' in df.columns else (0, 0),
        'columns': list(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    return summary


def print_data_summary(data_file):
    """Print formatted data summary"""
    summary = get_data_summary(data_file)
    
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"File: {summary['file']}")
    print(f"Total Records: {summary['total_records']:,}")
    print(f"Unique Vehicles: {summary['unique_vehicles']:,}")
    print(f"Time Range: {summary['time_range'][0]:.0f}s - {summary['time_range'][1]:.0f}s")
    print(f"Memory Usage: {summary['memory_usage_mb']:.2f} MB")
    print(f"Columns: {len(summary['columns'])}")
    print("="*60 + "\n")


def merge_csv_files(file_pattern, output_file):
    """Merge multiple CSV files matching a pattern"""
    import glob
    
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching: {file_pattern}")
        return None
    
    print(f"Merging {len(files)} files...")
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"  - {file}: {len(df):,} records")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    
    print(f"\nMerged data saved: {output_file}")
    print(f"Total records: {len(merged_df):,}")
    
    return output_file


def clean_old_files(directory, pattern, keep_latest=3):
    """Clean old files, keeping only the latest N files"""
    import glob
    
    files = glob.glob(os.path.join(directory, pattern))
    
    if len(files) <= keep_latest:
        print(f"No cleanup needed in {directory}")
        return
    
    # Sort by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Delete old files
    files_to_delete = files[keep_latest:]
    
    print(f"\nCleaning up old files in {directory}:")
    for file in files_to_delete:
        os.remove(file)
        print(f"  Deleted: {file}")
    
    print(f"Kept {keep_latest} latest files")


def validate_sumo_installation():
    """Check if SUMO is properly installed"""
    import subprocess
    
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        
        if result.returncode == 0:
            print("✓ SUMO is installed")
            print(f"  Version: {result.stdout.strip()}")
            return True
        else:
            print("✗ SUMO installation issue")
            return False
            
    except FileNotFoundError:
        print("✗ SUMO not found in PATH")
        print("\nPlease install SUMO:")
        print("  Windows: https://sumo.dlr.de/docs/Downloads.php")
        print("  Linux: sudo apt-get install sumo sumo-tools sumo-doc")
        print("  Mac: brew install sumo")
        return False
    except Exception as e:
        print(f"✗ Error checking SUMO: {e}")
        return False


def check_system_requirements():
    """Check if all system requirements are met"""
    print("\n" + "="*60)
    print("SYSTEM REQUIREMENTS CHECK")
    print("="*60)
    
    # Check Python version
    import sys
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("  ✗ Python 3.8+ required")
    else:
        print("  ✓ Python version OK")
    
    # Check SUMO
    validate_sumo_installation()
    
    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'tensorflow', 'matplotlib', 'seaborn', 
        'plotly', 'streamlit', 'yaml', 'tqdm'
    ]
    
    print("\nRequired Packages:")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
    
    # Check directories
    print("\nDirectories:")
    ensure_directories()
    
    print("\n" + "="*60)
    
    return len(missing_packages) == 0


def export_results_summary(results_dict, output_file='outputs/reports/results_summary.txt'):
    """Export results summary to text file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ACCIDENT RISK PREDICTION - RESULTS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write results
        for key, value in results_dict.items():
            f.write(f"{key}:\n")
            f.write(f"{value}\n\n")
    
    print(f"Results summary saved: {output_file}")


if __name__ == "__main__":
    print("Utility Functions Module")
    print("\nRunning system requirements check...")
    check_system_requirements()
