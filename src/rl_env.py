import os
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from kinematics import damped_least_squares_ik
from pid_controller import PIDController

MODEL_PATH = os.path.join("mujoco_menagerie", "franka_emika_panda", "pick_scene_pid.xml")

# --------------------------- Task parameters ---------------------------
PROXIMITY_NEAR = 0.08
PROXIMITY_CLOSE = 0.045
GRASP_PROXIMITY = 0.085
LIFT_HEIGHT_THRESH = 0.045
# Placement/release thresholds. Transport can enter the 10 cm zone,
# but the gripper is not unlocked until the object is aligned closely
# and lowered near the target/table height.
PLACEMENT_ZONE_THRESH = 0.10
PLACEMENT_RELEASE_DIST = 0.035
PLACEMENT_RELEASE_HEIGHT_MIN = 0.035
PLACEMENT_RELEASE_HEIGHT_MAX = 0.065

# The policy must actually settle the object near the target before the
# gripper is unlocked.  Stability is measured in Cartesian space, not by
# joint-speed norm, because the latter can be small while the object is still
# drifting or oscillating.
PLACEMENT_STABLE_DIST = 0.035
PLACEMENT_STABLE_OBJ_DELTA = 0.004       # 4 mm per RL step
PLACEMENT_STABLE_EE_DELTA = 0.008        # 8 mm per RL step
PLACEMENT_STABLE_STEPS = 3

SUCCESS_DIST_THRESH = PLACEMENT_RELEASE_DIST
SUCCESS_HEIGHT_MIN = 0.035
SUCCESS_HEIGHT_MAX = 0.060

REACH_SCALE = 12.0
LIFT_SCALE = 35.0
TRANSPORT_SCALE = 35.0
PLACEMENT_DESCENT_SCALE = 40.0
TOUCH_BONUS = 2.0
GRASP_BONUS = 12.0
TARGET_BONUS = 5.0
SUCCESS_BONUS = 100.0
DROP_PENALTY = -8.0

HOLD_STEPS = 4
DROP_CONFIRM_STEPS = 6

# Hierarchical controller:
# PPO (high level) -> desired joint position -> PID (low level) -> torque -> MuJoCo.
# PPO changes the desired joint target once per RL step. The PID runs at every
# MuJoCo physics substep, so it is the actual low-level controller.
CONTROL_SUBSTEPS = 20
MAX_JOINT_DELTA = 0.020
MAX_EPISODE_STEPS = 300

# After a valid placement/release, automatically return the arm smoothly to
# the equilibrium/home configuration. The return is still controlled by PID.
RETURN_HOME_MAX_JOINT_DELTA = 0.015
RETURN_HOME_POS_TOL = 0.020
RETURN_HOME_VEL_TOL = 0.08
RETURN_HOME_SETTLE_STEPS = 5
RETURN_HOME_SCALE = 2.0
RETURN_HOME_BONUS = 20.0
PLACEMENT_BONUS = 100.0

# PID gains are the same gains used by the working standalone PID pipeline.
KP = [400, 400, 400, 400, 180, 100, 40]
KI = [10, 10, 10, 10, 5, 5, 2]
KD = [60, 60, 60, 60, 25, 18, 8]

PLACEMENT_CURRICULUM_PROB = 0.30
GRASP_CURRICULUM_PROB = 0.35


class PandaPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"MuJoCo model not found: {MODEL_PATH}. "
                "Run this program from the project root."
            )

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.site_id = self.model.site("attachment_site").id
        self.object_body_id = self.model.body("object").id
        self.gripper_actuator = 7
        self.n_arm_joints = 7

        if self.model.nu < 8:
            raise RuntimeError(
                f"Expected at least 8 actuators (7 arm + gripper), found {self.model.nu}."
            )
        if self.model.nq < 14:
            raise RuntimeError(
                f"Expected a 7-DOF arm plus free object joint, found nq={self.model.nq}."
            )

        self.target_pos = np.array([0.45, -0.30, 0.02], dtype=np.float64)
        self.home_qpos = np.array(
            [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79], dtype=np.float64
        )

        self.max_delta = MAX_JOINT_DELTA
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.current_step = 0

        # 7 q + 7 dq + EE(3) + object(3) + object->target(3)
        # + gripper(1) + EE->object(3) + EE-object distance(1)
        # + object-target distance(1) = 29 values.
        joint_low = self.model.jnt_range[:7, 0].astype(np.float32)
        joint_high = self.model.jnt_range[:7, 1].astype(np.float32)
        low = np.concatenate([
            joint_low,
            -10.0 * np.ones(7),
            -2.0 * np.ones(3),
            -2.0 * np.ones(3),
            -2.0 * np.ones(3),
            [0.0],
            -2.0 * np.ones(3),
            [0.0],
            [0.0],
        ]).astype(np.float32)
        high = np.concatenate([
            joint_high,
            10.0 * np.ones(7),
            2.0 * np.ones(3),
            2.0 * np.ones(3),
            2.0 * np.ones(3),
            [0.045],
            2.0 * np.ones(3),
            [3.0],
            [3.0],
        ]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 7 continuous joint-target deltas + 1 continuous gripper command.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None

        # High-level target generated by PPO.
        self.desired_qpos = self.home_qpos.copy()

        # Low-level joint controller. It runs at the MuJoCo physics timestep.
        self.pid = PIDController(KP, KI, KD, n_joints=7)

        # Torque limits of the seven arm actuators.
        self.torque_low = self.model.actuator_forcerange[:7, 0].copy()
        self.torque_high = self.model.actuator_forcerange[:7, 1].copy()

        self.gripper_closed = False
        self.has_touched_once = False
        self.has_grasped_once = False
        self.has_reached_target_once = False
        self.near_bonus = False
        self.close_bonus = False
        self.consec_holding_steps = 0
        self.consec_not_holding_steps = 0
        self.had_sustained_grasp = False
        self.prev_ee_to_obj = None
        self.prev_obj_to_target = None
        self.prev_obj_height = None
        self.prev_obj_pos = None
        self.prev_ee_pos = None
        self.placement_stable_steps = 0

        # Episode state after a valid placement:
        # task -> release -> smooth return home -> terminate.
        self.returning_home = False
        self.return_home_converged_steps = 0
        self.placement_complete = False
        self.return_home_start_dist = None

    # --------------------------- State helpers ---------------------------
    def _object_qpos_address(self):
        body = self.model.body("object")
        jnt = int(body.jntadr[0])
        return int(self.model.jnt_qposadr[jnt])

    def _get_state(self):
        ee = self.data.site_xpos[self.site_id].copy()
        obj = self.data.xpos[self.object_body_id].copy()
        ee_to_obj_vec = obj - ee
        obj_to_target_vec = self.target_pos - obj
        ee_to_obj = float(np.linalg.norm(ee_to_obj_vec))
        obj_to_target = float(np.linalg.norm(obj[:2] - self.target_pos[:2]))
        return ee, obj, ee_to_obj_vec, obj_to_target_vec, ee_to_obj, obj_to_target

    def _get_obs(self):
        ee, obj, ee_to_obj_vec, obj_to_target_vec, ee_to_obj, obj_to_target = self._get_state()
        gripper_q = float(self.data.qpos[7]) if self.model.nq > 7 else 0.0
        obs = np.concatenate([
            self.data.qpos[:7],
            self.data.qvel[:7],
            ee,
            obj,
            obj_to_target_vec,
            [gripper_q],
            ee_to_obj_vec,
            [ee_to_obj],
            [obj_to_target],
        ]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _set_object_pose(self, xyz, quat=(1.0, 0.0, 0.0, 0.0)):
        adr = self._object_qpos_address()
        self.data.qpos[adr:adr + 3] = xyz
        self.data.qpos[adr + 3:adr + 7] = quat

    def _solve_arm_to(self, target):
        qpos_sol, err, n_iters, converged = damped_least_squares_ik(
            self.model, self.data, np.asarray(target, dtype=np.float64)
        )
        if converged and np.all(np.isfinite(qpos_sol)):
            qpos_sol = np.clip(qpos_sol, self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
            self.data.qpos[:7] = qpos_sol
            self.data.qvel[:7] = 0.0
            # This is only a reset/curriculum teleport. During actual RL
            # control, arm torques are produced exclusively by the PID.
            self.desired_qpos = qpos_sol.copy()
            self.data.ctrl[:7] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return True
        return False

    # ------------------------------- Gym API -----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:7] = self.home_qpos
        self.data.qvel[:] = 0.0

        # PPO's initial desired position equals the reset configuration.
        self.desired_qpos = self.home_qpos.copy()
        self.pid.reset()

        # Arm is torque controlled; PID will provide the torque on the first
        # physics step. Gripper remains directly controlled.
        self.data.ctrl[:7] = 0.0
        self.data.ctrl[self.gripper_actuator] = 255.0  # open
        self.gripper_closed = False

        self.current_step = 0
        self.has_touched_once = False
        self.has_grasped_once = False
        self.has_reached_target_once = False
        self.near_bonus = False
        self.close_bonus = False
        self.consec_holding_steps = 0
        self.consec_not_holding_steps = 0
        self.had_sustained_grasp = False
        self.returning_home = False
        self.return_home_converged_steps = 0
        self.placement_complete = False
        self.return_home_start_dist = None

        obj_x = self.np_random.uniform(0.35, 0.55)
        obj_y = self.np_random.uniform(-0.15, 0.15)
        obj_xyz = np.array([obj_x, obj_y, 0.05])
        self._set_object_pose(obj_xyz)
        mujoco.mj_forward(self.model, self.data)

        # Curriculum 1: learn the final transport/release phase directly.
        roll = self.np_random.random()
        if roll < PLACEMENT_CURRICULUM_PROB:
            near_target_xy = self.target_pos[:2] + self.np_random.uniform(-0.055, 0.055, 2)
            hold_xyz = np.array([near_target_xy[0], near_target_xy[1], 0.11])
            if self._solve_arm_to(hold_xyz):
                ee = self.data.site_xpos[self.site_id].copy()
                # Keep the object just below the attachment site, inside the gripper.
                self._set_object_pose(ee - np.array([0.0, 0.0, 0.025]))
                self.gripper_closed = True
                self.data.ctrl[self.gripper_actuator] = 0.0
                mujoco.mj_forward(self.model, self.data)

        # Curriculum 2: learn close -> lift before asking PPO to discover the reach.
        elif roll < PLACEMENT_CURRICULUM_PROB + GRASP_CURRICULUM_PROB:
            grasp_xyz = np.array([obj_x, obj_y, 0.075])
            self._solve_arm_to(grasp_xyz)
            self.gripper_closed = False
            self.data.ctrl[self.gripper_actuator] = 255.0
            mujoco.mj_forward(self.model, self.data)

        ee, obj, _, _, ee_dist, obj_dist = self._get_state()
        self.prev_ee_to_obj = ee_dist
        self.prev_obj_to_target = obj_dist
        self.prev_obj_height = float(obj[2])
        self.prev_obj_pos = obj.copy()
        self.prev_ee_pos = ee.copy()
        self.placement_stable_steps = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        # ================================================================
        # HIGH-LEVEL CONTROLLER: PPO -> desired joint positions
        # ================================================================
        # During the pick-and-place task, PPO produces a persistent joint
        # target. After a valid release, the environment takes over only the
        # return-home phase so that placement/retraction is always smooth.
        if not self.returning_home:
            dq = action[:7] * self.max_delta
            self.desired_qpos = np.clip(
                self.desired_qpos + dq,
                self.model.jnt_range[:7, 0],
                self.model.jnt_range[:7, 1],
            )

            # Before grasp: PPO controls the gripper.
            # During transport: keep it closed.
            # After the target gate: PPO can open it to release.
            if not self.has_grasped_once:
                if action[7] > 0.25:
                    self.gripper_closed = True
                elif action[7] < -0.25:
                    self.gripper_closed = False
            elif not self.has_reached_target_once:
                self.gripper_closed = True
            else:
                if action[7] > 0.25:
                    self.gripper_closed = True
                elif action[7] < -0.25:
                    self.gripper_closed = False

        else:
            # ============================================================
            # POST-PLACEMENT: automatic smooth return to home/equilibrium
            # ============================================================
            # Keep the gripper open. Move the high-level desired target toward
            # home in small increments; PID converts those targets to torque.
            self.gripper_closed = False

            delta_home = self.home_qpos - self.desired_qpos
            delta_home = np.clip(
                delta_home,
                -RETURN_HOME_MAX_JOINT_DELTA,
                RETURN_HOME_MAX_JOINT_DELTA,
            )
            self.desired_qpos = np.clip(
                self.desired_qpos + delta_home,
                self.model.jnt_range[:7, 0],
                self.model.jnt_range[:7, 1],
            )

        # 0 = closed, 255 = open
        self.data.ctrl[self.gripper_actuator] = (
            0.0 if self.gripper_closed else 255.0
        )

        # ================================================================
        # LOW-LEVEL CONTROLLER: desired joint positions -> PID -> torque
        # ================================================================
        dt = self.model.opt.timestep
        for substep in range(CONTROL_SUBSTEPS):
            current_qpos = self.data.qpos[:7].copy()
            current_qvel = self.data.qvel[:7].copy()

            pid_torque = self.pid.compute(
                self.desired_qpos,
                current_qpos,
                current_qvel,
                dt,
            )

            # Match the standalone PID pipeline: compensate MuJoCo bias
            # forces, then enforce each actuator's torque limit.
            torque = pid_torque + self.data.qfrc_bias[:7]
            torque = np.clip(torque, self.torque_low, self.torque_high)
            self.data.ctrl[:7] = torque

            mujoco.mj_step(self.model, self.data)

            if (
                self.render_mode == "human"
                and self.viewer is not None
                and substep % 2 == 0
            ):
                self.viewer.sync()

        reward, terminated, info = self._compute_reward()
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------ Reward -------------------------------
    def _compute_reward(self):
        ee, obj, _, _, ee_dist, obj_dist = self._get_state()
        height = float(obj[2])
        reward = 0.0
        components = {}

        # Before a sustained grasp, teach reaching. Once holding, stop paying
        # for EE motion and pay for object transport instead.
        near = ee_dist < GRASP_PROXIMITY
        lifted = height > LIFT_HEIGHT_THRESH
        holding = bool(self.gripper_closed and near and lifted)

        if not self.has_grasped_once and self.prev_ee_to_obj is not None:
            components["reach"] = REACH_SCALE * (self.prev_ee_to_obj - ee_dist)
            reward += components["reach"]
        else:
            components["reach"] = 0.0
        self.prev_ee_to_obj = ee_dist

        # One-time proximity achievements.
        components["proximity"] = 0.0
        if ee_dist < PROXIMITY_NEAR and not getattr(self, "near_bonus", False):
            components["proximity"] += 0.75
            self.near_bonus = True
        if ee_dist < PROXIMITY_CLOSE and not getattr(self, "close_bonus", False):
            components["proximity"] += 1.25
            self.close_bonus = True
        reward += components["proximity"]

        # Closing near the object gets a one-time initiation bonus.
        components["touch"] = 0.0
        if near and self.gripper_closed and not self.has_touched_once:
            components["touch"] = TOUCH_BONUS
            reward += TOUCH_BONUS
            self.has_touched_once = True

        # Dense lift progress only when the gripper is closed and near object.
        components["lift"] = 0.0
        height_gain = 0.0
        previous_obj_height = self.prev_obj_height
        if near and self.gripper_closed and self.prev_obj_height is not None:
            height_gain = height - self.prev_obj_height
            components["lift"] = LIFT_SCALE * height_gain
            reward += components["lift"]
        self.prev_obj_height = height

        if holding:
            self.consec_holding_steps += 1
            self.consec_not_holding_steps = 0
        else:
            self.consec_not_holding_steps += 1
            self.consec_holding_steps = 0

        sustained = self.consec_holding_steps >= HOLD_STEPS
        components["grasp"] = 0.0

        if sustained and not self.has_grasped_once:
            components["grasp"] = GRASP_BONUS
            reward += GRASP_BONUS
            self.has_grasped_once = True
            self.had_sustained_grasp = True

        # Transport reward starts from the actual object distance at reset and
        # continues through the whole carry. This makes moving toward target
        # immediately valuable once the object is lifted.
        components["transport"] = 0.0
        components["placement_stability"] = 0.0
        if holding:
            if self.prev_obj_to_target is not None:
                components["transport"] = TRANSPORT_SCALE * (self.prev_obj_to_target - obj_dist)
                reward += components["transport"]
            self.prev_obj_to_target = obj_dist

            # Placement shaping: once the object is near the target, reward
            # downward motion while it is still being held. This teaches
            # align -> descend -> release rather than release from above.
            components["placement_descent"] = 0.0
            if obj_dist < PLACEMENT_ZONE_THRESH and previous_obj_height is not None:
                descent = previous_obj_height - height
                components["placement_descent"] = PLACEMENT_DESCENT_SCALE * descent
                reward += components["placement_descent"]

            # ------------------------------------------------------------
            # PLACEMENT STABILITY GATE
            # ------------------------------------------------------------
            # Entering the target zone is not enough to unlock the gripper.
            # The object must be aligned and physically settled for a few RL
            # steps.  We use Cartesian object/EE motion rather than joint
            # velocity so the gate measures the thing that actually matters:
            # whether the placement is stable.
            obj_step_delta = 0.0
            ee_step_delta = 0.0
            if self.prev_obj_pos is not None:
                obj_step_delta = float(np.linalg.norm(obj - self.prev_obj_pos))
            if self.prev_ee_pos is not None:
                ee_step_delta = float(np.linalg.norm(ee - self.prev_ee_pos))

            placement_ready = (
                obj_dist <= PLACEMENT_STABLE_DIST
                and PLACEMENT_RELEASE_HEIGHT_MIN <= height <= PLACEMENT_RELEASE_HEIGHT_MAX
                and obj_step_delta <= PLACEMENT_STABLE_OBJ_DELTA
                and ee_step_delta <= PLACEMENT_STABLE_EE_DELTA
            )

            if placement_ready:
                self.placement_stable_steps += 1
                components["placement_stability"] = 1.0
                reward += components["placement_stability"]
            else:
                self.placement_stable_steps = 0
                components["placement_stability"] = 0.0

            # Unlock release only after the stable-placement window.
            if (
                self.placement_stable_steps >= PLACEMENT_STABLE_STEPS
                and not self.has_reached_target_once
            ):
                reward += TARGET_BONUS
                self.has_reached_target_once = True
        elif self.consec_not_holding_steps >= DROP_CONFIRM_STEPS:
            # Once a real drop/release is confirmed, restart the transport
            # baseline at the object's current position. Do not keep paying
            # stale progress against a distance from before the drop.
            self.prev_obj_to_target = obj_dist
            if self.had_sustained_grasp:
                reward += DROP_PENALTY
                self.had_sustained_grasp = False

        # Save Cartesian positions for the next RL-step stability check.
        self.prev_obj_pos = obj.copy()
        self.prev_ee_pos = ee.copy()

        terminated = False
        success = False

        # ================================================================
        # VALID PLACEMENT: release object, then begin return-home phase
        # ================================================================
        valid_release = (
            self.has_grasped_once
            and self.has_reached_target_once
            and not self.gripper_closed
            and obj_dist < SUCCESS_DIST_THRESH
            and SUCCESS_HEIGHT_MIN <= height <= SUCCESS_HEIGHT_MAX
        )

        if valid_release and not self.returning_home and not self.placement_complete:
            reward += PLACEMENT_BONUS
            self.placement_complete = True
            self.returning_home = True
            self.return_home_converged_steps = 0
            self.return_home_start_dist = float(
                np.max(np.abs(self.data.qpos[:7] - self.home_qpos))
            )
            # Clear integral/derivative memory before the retreat.
            self.pid.reset()

        # ================================================================
        # RETURN-HOME: terminate only after the arm is actually settled
        # ================================================================
        if self.returning_home:
            home_pos_err = float(
                np.max(np.abs(self.home_qpos - self.data.qpos[:7]))
            )
            home_vel_err = float(
                np.max(np.abs(self.data.qvel[:7]))
            )

            # Small dense shaping for completing the retreat.
            if self.return_home_start_dist is not None:
                reward += RETURN_HOME_SCALE * (
                    self.return_home_start_dist - home_pos_err
                )
                self.return_home_start_dist = home_pos_err

            if (
                home_pos_err <= RETURN_HOME_POS_TOL
                and home_vel_err <= RETURN_HOME_VEL_TOL
            ):
                self.return_home_converged_steps += 1
            else:
                self.return_home_converged_steps = 0

            # Require several consecutive settled RL steps to avoid ending
            # while the arm is still oscillating.
            if self.return_home_converged_steps >= RETURN_HOME_SETTLE_STEPS:
                reward += RETURN_HOME_BONUS
                reward += SUCCESS_BONUS
                terminated = True
                success = True
                self.returning_home = False

        # Catastrophic escape.
        if height < -0.10 or np.linalg.norm(obj[:2]) > 2.0:
            reward += -20.0
            terminated = True
            success = False

        info = {
            "success": success,
            "reward_reaching": components["reach"],
            "reward_proximity": components["proximity"],
            "reward_touch": components["touch"],
            "reward_grasp": components["grasp"],
            "reward_lift": components["lift"],
            "reward_transport": components["transport"],
            "reward_placement_descent": components.get("placement_descent", 0.0),
            "reward_placement_stability": components.get("placement_stability", 0.0),
            "distance_ee_to_obj": ee_dist,
            "distance_obj_to_target": obj_dist,
            "obj_height": height,
            "near_object": near,
            "is_holding": holding,
            "sustained_holding": sustained,
            "gripper_closed": self.gripper_closed,
            "desired_qpos": self.desired_qpos.copy(),
            "returning_home": self.returning_home,
            "placement_complete": self.placement_complete,
            "placement_stable_steps": self.placement_stable_steps,
            "placement_ready": bool(
                obj_dist <= PLACEMENT_STABLE_DIST
                and PLACEMENT_RELEASE_HEIGHT_MIN <= height <= PLACEMENT_RELEASE_HEIGHT_MAX
            ),
            "home_position_error": float(
                np.max(np.abs(self.home_qpos - self.data.qpos[:7]))
            ),
            "home_velocity_error": float(
                np.max(np.abs(self.data.qvel[:7]))
            ),
            "current_step": self.current_step,
        }
        return reward, terminated, info

    # ------------------------------- Render ------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        import mujoco.viewer
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
