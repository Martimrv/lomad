"""Run anomaly detectors and save results."""
import sys
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, 'backend/scripts')

print("=" * 60)
print("ANOMALY DETECTION PIPELINE")
print("=" * 60)

# Step 1: Run trajectory anomaly detector
print("\n[1/2] Running trajectory anomaly detector...")
try:
    from trajectory_anomaly_detector import main as traj_main
    traj_main()
    print("[OK] Trajectory anomalies generated")
except Exception as e:
    print(f"[FAIL] Error in trajectory detector: {e}")

# Step 2: Run advanced anomaly detector
print("\n[2/2] Running advanced anomaly detector...")
try:
    from advanced_anomaly_detector import main as adv_main
    adv_main()
    print("[OK] Advanced anomalies generated")
except Exception as e:
    print(f"[FAIL] Error in advanced detector: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)

