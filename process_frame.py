import time
import numpy as np
import cv2
from utils import find_angle, draw_text, draw_dotted_line
import mediapipe as mp

# Initialize Mediapipe modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ----------------------------------------------------------------------
# Utility helpers (you can keep these in utils.py if you already have them)
# ----------------------------------------------------------------------

def find_angle(p1, p2, ref):
    """Return angle (in degrees) at 'ref' formed by p1-ref and p2-ref."""
    a = np.array(p1) - np.array(ref)
    b = np.array(p2) - np.array(ref)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def draw_text(frame, text, pos=(30,30), text_color=(255,255,255),
              text_color_bg=(0,0,0), font_scale=0.7, thickness=2):
    """Draw text with background rectangle."""
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x-2, y-h-5), (x+w+2, y+4), text_color_bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_dotted_line(frame, pt1, pt2, line_color=(255,255,0), thickness=2, gap=10):
    """Draw a dotted line between pt1 and pt2."""
    dist = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    for i in range(0, dist, gap):
        x = int(pt1[0] + (pt2[0] - pt1[0]) * i / dist)
        y = int(pt1[1] + (pt2[1] - pt1[1]) * i / dist)
        cv2.circle(frame, (x, y), thickness, line_color, -1)

def smooth(prev, current, alpha=0.6):
    """Exponential moving average for smoothing."""
    if prev is None:
        return current
    return alpha * current + (1 - alpha) * prev

# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------

class ProcessFrame:
    def __init__(self, thresholds=None, flip_frame=False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds or {}

        self.state = {
            'exercise': None,
            'count': 0,
            'dir': 0,
            'incorrect': 0,
            'start_time': time.time(),
            'smoothed_angles': {},
            'frame_idx': 0
        }

        self.COLORS = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'white': (255, 255, 255),
            'yellow': (0, 255, 255),
            'blue': (255, 127, 0)
        }

    def process(self, frame_bgr, pose, exercise):
        if self.state['exercise'] != exercise:
            self.state.update({
                'exercise': exercise,
                'count': 0,
                'dir': 0,
                'incorrect': 0,
                'smoothed_angles': {},
                'start_time': time.time()
            })

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        h, w = frame_bgr.shape[:2]

        if not results.pose_landmarks:
            draw_text(frame_bgr, "NO PERSON DETECTED", (30,80), (255,255,255), (0,0,255))
            self._draw_summary(frame_bgr)
            return frame_bgr, {}

        lm = results.pose_landmarks.landmark
        def coord(i):
            if lm[i].visibility < 0.1:
                return None
            return (int(lm[i].x * w), int(lm[i].y * h))

        joints = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        coords = {k: coord(v) for k, v in joints.items()}

        def npcoord(p): return np.array(p) if p is not None else None
        def angle_triplet(a,b,c,name):
            pa,pb,pc = npcoord(coords[a]), npcoord(coords[b]), npcoord(coords[c])
            if pa is None or pb is None or pc is None:
                return None
            ang = find_angle(pa, pc, pb)
            prev = self.state['smoothed_angles'].get(name)
            sm = smooth(prev, ang)
            self.state['smoothed_angles'][name] = sm
            return sm

        left_elbow = angle_triplet('left_shoulder','left_elbow','left_wrist','left_elbow')
        right_elbow = angle_triplet('right_shoulder','right_elbow','right_wrist','right_elbow')
        left_shldr_hip_knee = angle_triplet('left_shoulder','left_hip','left_knee','left_hip')
        right_shldr_hip_knee = angle_triplet('right_shoulder','right_hip','right_knee','right_hip')
        left_hip_knee_ankle = angle_triplet('left_hip','left_knee','left_ankle','left_knee')
        right_hip_knee_ankle = angle_triplet('right_hip','right_knee','right_ankle','right_knee')

        ex = self.state['exercise'].lower()
        if 'push' in ex:
            self._logic_pushups(frame_bgr, left_elbow, right_elbow,
                                left_shldr_hip_knee, right_shldr_hip_knee, coords)
        elif 'squat' in ex:
            self._logic_squats(frame_bgr, left_hip_knee_ankle, right_hip_knee_ankle,
                               left_shldr_hip_knee, right_shldr_hip_knee, coords)
        elif 'sit' in ex:
            self._logic_situps(frame_bgr, left_shldr_hip_knee, right_shldr_hip_knee,
                               left_hip_knee_ankle, right_hip_knee_ankle, coords)
        elif 'bicep' in ex:
            frame_bgr = self._logic_bicep(frame_bgr, left_elbow, right_elbow, coords)
        else:
            draw_text(frame_bgr, "UNKNOWN EXERCISE", (30,80), (255,255,255), (0,0,255))

        self._draw_summary(frame_bgr)
        return frame_bgr, {
            'count': self.state['count'],
            'incorrect': self.state['incorrect'],
            'exercise': self.state['exercise'],
            'time_elapsed': int(time.time() - self.state['start_time'])
        }


    # -------------------
    # Exercise logic implementations
    # -------------------

    def _logic_pushups(self, frame, left_elbow, right_elbow, left_body_angle, right_body_angle, coords):
        """
        Push-up logic:
         - main angle: elbow angle (shoulder-elbow-wrist)
         - body alignment: shoulder-hip-knee angle (close to 180 -> straight)
         - require both arms symmetric-ish and body aligned for a rep to be counted correct
        """
        # progress: 0 (top) -> 100 (bottom). elbow angle roughly: ~160 (extended) down to ~80 (bent)
        def per_from_angle(a):
            if a is None: return None
            return np.interp(a, (85, 165), (100, 0))  # extended -> 0%? we want top to be 0, bottom to be 100

        left_per = per_from_angle(left_elbow)
        right_per = per_from_angle(right_elbow)
        valid_arms = (left_per is not None) and (right_per is not None)
        avg_per = (left_per + right_per) / 2 if valid_arms else None

        # body alignment check: we expect shoulder-hip-knee angles near 180 (straight)
        body_ok = True
        for angle in (left_body_angle, right_body_angle):
            if angle is None or angle < 150:
                body_ok = False

        # count transitions: top (avg_per < 10) -> down (avg_per > 90) -> top again
        if avg_per is not None:
            # show angle numbers near elbows
            if coords.get('left_elbow'):
                draw_text(frame, f"{int(left_elbow or 0)}", pos=(coords['left_elbow'][0]+10, coords['left_elbow'][1]), text_color=(255,255,255), text_color_bg=(0,0,0), font_scale=0.5)
            if coords.get('right_elbow'):
                draw_text(frame, f"{int(right_elbow or 0)}", pos=(coords['right_elbow'][0]+10, coords['right_elbow'][1]), text_color=(255,255,255), text_color_bg=(0,0,0), font_scale=0.5)

            # progress bar
            w = frame.shape[1]
            bar = int(np.interp(avg_per, (0, 100), (0, w)))
            cv2.rectangle(frame, (10, 10), (bar, 30), (0, 255, 0), -1)

            # decide transitions
            if avg_per > 85 and self.state['dir'] == 0:
                self.state['dir'] = 1  # went down half
            if avg_per < 15 and self.state['dir'] == 1:
                # completed rep bottom->top
                # count as correct only if body aligned and arm difference small
                arm_diff = abs(left_per - right_per)
                if body_ok and arm_diff < 20:
                    self.state['count'] += 1
                else:
                    self.state['incorrect'] += 1
                self.state['dir'] = 0

            # form feedback
            if not body_ok:
                draw_text(frame, "Keep body straight!", pos=(30, 60), text_color=(255,255,255), text_color_bg=(0,0,255), font_scale=0.6)
            elif left_per is not None and right_per is not None and abs(left_per - right_per) > 20:
                draw_text(frame, "Uneven arms - keep them symmetric!", pos=(30, 60), text_color=(255,255,255), text_color_bg=(0,0,255), font_scale=0.6)

    def _logic_squats(self, frame, left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle, coords):
        """
        Squats logic:
         - main angle: knee angle (hip-knee-ankle) — when knee angle decreases toward ~90 it is deep
         - hip hinge: shoulder-hip-knee should move properly
         - count rep on bottom->top cycle similarly
        """
        def per_from_angle(a):
            if a is None: return None
            # knee angle: standing ~170-180 -> bottom ~90
            return np.interp(a, (170, 90), (0, 100))  # standing -> 0, bottom -> 100

        left_per = per_from_angle(left_knee_angle)
        right_per = per_from_angle(right_knee_angle)
        valid = left_per is not None and right_per is not None
        avg_per = (left_per + right_per) / 2 if valid else None

        # check hip hinge (we want hips to push back; hip angle roughly >= 100 indicates hinge)
        hip_ok = True
        for a in (left_hip_angle, right_hip_angle):
            if a is None or a < 80:
                hip_ok = False

        if avg_per is not None:
            # draw progress bar
            w = frame.shape[1]
            bar = int(np.interp(avg_per, (0, 100), (0, w)))
            cv2.rectangle(frame, (10, 10), (bar, 30), (0, 255, 0), -1)

            # transitions
            if avg_per > 85 and self.state['dir'] == 0:
                self.state['dir'] = 1
            if avg_per < 15 and self.state['dir'] == 1:
                # bottom->top completed
                # valid rep only if hip hinge OK and legs evenly loaded
                leg_diff = abs(left_per - right_per)
                if hip_ok and leg_diff < 20:
                    self.state['count'] += 1
                else:
                    self.state['incorrect'] += 1
                self.state['dir'] = 0

            # Feedback
            if not hip_ok:
                draw_text(frame, "Hinge at hips (push hips back)!", pos=(30, 60), text_color=(255,255,255), text_color_bg=(0,0,255), font_scale=0.6)
            if left_per is not None and right_per is not None and abs(left_per - right_per) > 20:
                draw_text(frame, "Shift weight evenly!", pos=(30, 90), text_color=(255,255,255), text_color_bg=(0,0,255), font_scale=0.6)

    def _logic_situps(self, frame, left_torso_angle, right_torso_angle, left_leg_angle, right_leg_angle, coords):
        """
        Sit-ups:
         - torso angle (shoulder-hip-knee) moving from ~180 (lying) to ~70-90 (sitting up)
         - legs should be anchored: leg angle (hip-knee-ankle) ~40-60
        """
        def per_from_angle(a):
            if a is None: return None
            # torso: lying ~180 -> sit ~70. Map to 0..100 for progress
            return np.interp(a, (180, 70), (0, 100))

        left_per = per_from_angle(left_torso_angle)
        right_per = per_from_angle(right_torso_angle)
        avg_per = None
        if left_per is not None and right_per is not None:
            avg_per = (left_per + right_per) / 2

        # leg anchor check: expect ~40-60 deg for knee (or hip-knee-ankle)
        legs_ok = True
        for a in (left_leg_angle, right_leg_angle):
            if a is None or not (35 <= a <= 65):
                legs_ok = False

        if avg_per is not None:
            w = frame.shape[1]
            bar = int(np.interp(avg_per, (0, 100), (0, w)))
            cv2.rectangle(frame, (10, 10), (bar, 30), (0, 255, 0), -1)

            if avg_per > 85 and self.state['dir'] == 0:
                self.state['dir'] = 1
            if avg_per < 15 and self.state['dir'] == 1:
                if legs_ok:
                    self.state['count'] += 1
                else:
                    self.state['incorrect'] += 1
                self.state['dir'] = 0

            if not legs_ok:
                draw_text(frame, "Keep legs anchored (~45°)", pos=(30, 60), text_color=(255,255,255), text_color_bg=(0,0,255), font_scale=0.6)

    def _logic_bicep(self, frame, left_elbow, right_elbow, coords):
        """
        Bicep curl detection logic (class method).
        - Supports single-arm and dual-arm curls.
        - Updates self.state['count'] and self.state['incorrect'] for unified summary.
        - Performs a basic form check: shoulder vertical movement between down->up should be small.
        """

        # Draw angle text if available
        if coords.get('left_elbow') and left_elbow is not None:
            draw_text(frame, f"{int(left_elbow)}°",
                      pos=(coords['left_elbow'][0]-20, coords['left_elbow'][1]-15),
                      text_color=(0,255,255), text_color_bg=(0,0,0), font_scale=0.5)
        if coords.get('right_elbow') and right_elbow is not None:
            draw_text(frame, f"{int(right_elbow)}°",
                      pos=(coords['right_elbow'][0]+20, coords['right_elbow'][1]-15),
                      text_color=(0,255,255), text_color_bg=(0,0,0), font_scale=0.5)

        # Draw dotted connection lines for both arms (safe calls)
        for side in ["left", "right"]:
            s = coords.get(f"{side}_shoulder")
            e = coords.get(f"{side}_elbow")
            w = coords.get(f"{side}_wrist")
            if s and e:
                draw_dotted_line(frame, s, e, line_color=(255,255,0))
            if e and w:
                draw_dotted_line(frame, e, w, line_color=(255,255,0))

        # thresholds for motion
        up_threshold = 50    # angle < this => curled (top)
        down_threshold = 160 # angle > this => extended (bottom)

        # init per-arm tracking in self.state if missing
        for arm in ("left", "right"):
            dir_key = f"{arm}_dir"            # "down" or "up"
            prev_sh_key = f"{arm}_sh_down_y"  # recorded shoulder y when we detect "down"
            rep_key = f"{arm}_correct"        # per-arm correct count
            if dir_key not in self.state:
                self.state[dir_key] = "down"
            if prev_sh_key not in self.state:
                self.state[prev_sh_key] = None
            if rep_key not in self.state:
                self.state[rep_key] = 0

        # helper to register a completed rep for an arm
        def register_rep(arm, correct=True):
            rep_key = f"{arm}_correct"
            if correct:
                self.state[rep_key] += 1
                # unified counter that your UI already shows
                self.state['count'] += 1
            else:
                # count as incorrect rep
                self.state['incorrect'] += 1

        # FORM CHECK SETTINGS
        # maximum allowed shoulder vertical movement (in pixels) during rep (tweak if needed)
        MAX_SH_MOVE_PX = 25

        # LEFT ARM logic
        if left_elbow is not None:
            # went to down/extended -> record shoulder y
            if left_elbow > down_threshold:
                self.state["left_dir"] = "down"
                if coords.get('left_shoulder'):
                    self.state["left_sh_down_y"] = coords['left_shoulder'][1]
            # completed up (curl)
            if left_elbow < up_threshold and self.state["left_dir"] == "down":
                # basic form check: did shoulder move too much?
                sh_y_down = self.state.get("left_sh_down_y")
                sh_y_now = coords['left_shoulder'][1] if coords.get('left_shoulder') else None
                if sh_y_down is not None and sh_y_now is not None:
                    sh_move = abs(sh_y_now - sh_y_down)
                else:
                    sh_move = 0  # if we don't have coords, be permissive

                if sh_move <= MAX_SH_MOVE_PX:
                    register_rep("left", correct=True)
                else:
                    register_rep("left", correct=False)

                self.state["left_dir"] = "up"
                # reset stored down position
                self.state["left_sh_down_y"] = None

        # RIGHT ARM logic
        if right_elbow is not None:
            if right_elbow > down_threshold:
                self.state["right_dir"] = "down"
                if coords.get('right_shoulder'):
                    self.state["right_sh_down_y"] = coords['right_shoulder'][1]
            if right_elbow < up_threshold and self.state["right_dir"] == "down":
                sh_y_down = self.state.get("right_sh_down_y")
                sh_y_now = coords['right_shoulder'][1] if coords.get('right_shoulder') else None
                if sh_y_down is not None and sh_y_now is not None:
                    sh_move = abs(sh_y_now - sh_y_down)
                else:
                    sh_move = 0

                if sh_move <= MAX_SH_MOVE_PX:
                    register_rep("right", correct=True)
                else:
                    register_rep("right", correct=False)

                self.state["right_dir"] = "up"
                self.state["right_sh_down_y"] = None

        # If user is curling only one arm, we count only that arm. If both curl, both are counted independently.

        # Overlay summary: show both-arm counts and unified correct/incorrect
        cv2.rectangle(frame, (20, 20), (380, 150), (0, 0, 0), -1)
        left_count = self.state.get("left_correct", 0)
        right_count = self.state.get("right_correct", 0)
        unified_correct = int(self.state.get('count', 0))
        unified_incorrect = int(self.state.get('incorrect', 0))

        cv2.putText(frame, f"L reps: {left_count}", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"R reps: {right_count}", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Correct(total): {unified_correct}", (210, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
        cv2.putText(frame, f"Incorrect(total): {unified_incorrect}", (210, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,200), 2)

        # Optional feedback if arms are very uneven
        if (left_elbow is not None) and (right_elbow is not None):
            if abs(left_elbow - right_elbow) > 40:
                draw_text(frame, "Uneven curl angles — try to curl both similarly", pos=(30, 185),
                          text_color=(255,255,255), text_color_bg=(0,0,255), font_scale=0.55)

        return frame




    # UI summary: correct / incorrect / elapsed time
    def _draw_summary(self, frame):
        h, w = frame.shape[:2]
        draw_text(frame, f"Exercise: {self.state['exercise']}", (int(w*0.02), int(h*0.88)), (255,255,255), (0,0,0), 0.6)
        draw_text(frame, f"Correct: {int(self.state['count'])}", (int(w*0.65), int(h*0.02)), (255,255,255), (0,150,0), 0.7)
        draw_text(frame, f"Incorrect: {int(self.state['incorrect'])}", (int(w*0.65), int(h*0.06)), (255,255,255), (180,0,0), 0.7)
        draw_text(frame, f"Time: {int(time.time() - self.state['start_time'])}s", (int(w*0.02), int(h*0.92)), (255,255,255), (0,0,0), 0.6)