import sys
sys.path.append("src")

import mujoco
import numpy as np
import time
from live_test_pid import model, data, run_pick_and_place_pid

np.random.seed(0)

N_TRIALS = 20
X_RANGE = (0.35, 0.55)
Y_RANGE = (-0.15, 0.15)
TARGET = np.array([0.45, -0.3, 0.02])

results = []

for trial in range(N_TRIALS):
    obj_x = np.random.uniform(*X_RANGE)
    obj_y = np.random.uniform(*Y_RANGE)

    mujoco.mj_resetData(model, data)
    object_jnt_adr = model.body("object").jntadr[0]
    qpos_adr = model.jnt_qposadr[object_jnt_adr]
    data.qpos[qpos_adr:qpos_adr+3] = [obj_x, obj_y, 0.05]
    data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

    t0 = time.time()
    result = run_pick_and_place_pid(TARGET, viewer=None)
    t1 = time.time()

    result["trial"] = trial
    result["completion_time"] = t1 - t0
    results.append(result)

    grip_lift = result.get('grip_offset_after_lift', -1) * 1000
    grip_transport = result.get('grip_offset_after_transport', -1) * 1000
    print(f"Trial {trial:2d}: success={result['success']}  "
          f"grasped={result['grasped']}  "
          f"grip_after_lift={grip_lift:.1f}mm  "
          f"grip_after_transport={grip_transport:.1f}mm  "
          f"final_dist={result['final_dist']*1000:.1f}mm  "
          f"time={result['completion_time']:.2f}s")

n_success = sum(r["success"] for r in results)
n_grasped = sum(r["grasped"] for r in results)
mean_time = np.mean([r["completion_time"] for r in results])
mean_dist = np.mean([r["final_dist"] for r in results])

print(f"\n=== Week 3 PID Evaluation ({N_TRIALS} trials) ===")
print(f"Success rate: {n_success}/{N_TRIALS} ({100*n_success/N_TRIALS:.0f}%)")
print(f"Grasp rate:   {n_grasped}/{N_TRIALS} ({100*n_grasped/N_TRIALS:.0f}%)")
print(f"Mean completion time: {mean_time:.2f}s")
print(f"Mean final position error: {mean_dist*1000:.1f}mm")
