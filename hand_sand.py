import cv2
import numpy as np
import mediapipe as mp
import time

# =========================
# Tunables (you can tweak)
# =========================
CAM_INDEX = 0
OUTPUT_WIDTH = 860          # resize camera feed to this width (height auto)
N_PARTICLES = 64000         # raise/lower for performance/denser look
BOX_MARGIN = 150             # pixel margin from frame edges for the “glass box”
PARTICLE_SIZE_PX = 1        # 1 = tiny dots (like your screenshots)
FORCE_STRENGTH = -1000.0     # repulsion intensity
FORCE_RADIUS = 200.0        # effective influence radius (pixels)
DAMPING = 0.96              # 0.85–0.96 (higher = smoother trails)
BROWNIAN = 0.10             # random jitter (0–0.3)
GRAVITY = 0.0              # subtle downward pull (0–0.5)
MAX_SPEED = 8.0             # clamp velocity
BOX_BOUNCE = 2          # energy retained on wall bounce
PARTICLE_ALPHA = 1       # opacity of particle overlay (0–1)
DRAW_BOX = False             # show faint bounding box
MIRROR = True               # selfie-style mirror

# ===========
# Hand model
# ===========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
_is_hand_closed = False # Global state for fist gesture

def is_fist(hand_landmarks, width, height):
    """Return true if the hand is in a fist gesture."""
    # Tip of the middle finger
    m_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    # MCP (knuckle) of the middle finger
    m_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    # Wrist
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Calculate distances
    tip_to_mcp_dist = np.sqrt((m_tip.x - m_mcp.x)**2 + (m_tip.y - m_mcp.y)**2)
    mcp_to_wrist_dist = np.sqrt((m_mcp.x - wrist.x)**2 + (m_mcp.y - wrist.y)**2)

    # A simple heuristic: if the tip is closer to the MCP than the MCP is to the wrist,
    # it's likely a fist. Adjust the ratio for sensitivity.
    return tip_to_mcp_dist < mcp_to_wrist_dist * 0.6

def get_hand_center_bounded(frame_bgr):
    """Return (x,y) of smoothed hand center in image coords, or None if not found."""
    global _smoothed_hx, _smoothed_hy, _is_hand_closed
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        _is_hand_closed = False
        return None

    hand_landmarks = res.multi_hand_landmarks[0]
    _is_hand_closed = is_fist(hand_landmarks, w, h)

    # Use average of palm-area landmarks for a stable center
    palm_idx = [0,1,5,9,13,17]  # wrist + MCPs
    xs = [hand_landmarks.landmark[i].x for i in palm_idx]
    ys = [hand_landmarks.landmark[i].y for i in palm_idx]
    hx = int(np.clip(np.mean(xs) * w, 0, w-1))
    hy = int(np.clip(np.mean(ys) * h, 0, h-1))

    # Exponential smoothing to avoid jitter
    alpha = 0.35
    if _smoothed_hx is None:
        _smoothed_hx, _smoothed_hy = hx, hy
    else:
        _smoothed_hx = int((1-alpha)*_smoothed_hx + alpha*hx)
        _smoothed_hy = int((1-alpha)*_smoothed_hy + alpha*hy)
    return _smoothed_hx, _smoothed_hy

_smoothed_hx, _smoothed_hy = None, None

# ===========
# Camera
# ===========
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit("Could not open webcam.")

# Get a first frame to size everything
ret, frame = cap.read()
if not ret:
    raise SystemExit("Webcam gave no frames.")
if MIRROR:
    frame = cv2.flip(frame, 1)

# Resize target
h0, w0 = frame.shape[:2]
scale = OUTPUT_WIDTH / float(w0)
OUT_W = OUTPUT_WIDTH
OUT_H = int(h0 * scale)

# Precompute box bounds
LEFT = BOX_MARGIN
RIGHT = OUT_W - BOX_MARGIN
TOP = BOX_MARGIN // 2
BOTTOM = OUT_H - BOX_MARGIN * 2

# =================
# Particle buffers
# =================
rng = np.random.default_rng(7)
px = rng.uniform(LEFT, RIGHT, N_PARTICLES).astype(np.float32)
py = rng.uniform(TOP, BOTTOM, N_PARTICLES).astype(np.float32)
vx = np.zeros(N_PARTICLES, np.float32)
vy = np.zeros(N_PARTICLES, np.float32)

# For Gaussian-like force falloff
SIG2 = (FORCE_RADIUS * FORCE_RADIUS)

# Add a nice “filled box” look by biasing positions to a tighter cube then expand
px = (px - (LEFT+RIGHT)/2.0) * 0.92 + (LEFT+RIGHT)/2.0
py = (py - (TOP+BOTTOM)/2.0) * 0.92 + (TOP+BOTTOM)/2.0

# =================
# Helper rendering
# =================
def render_particles_overlay(width, height, px, py):
    """
    Fast-ish particle drawing: set pixels at particle coords on a single-channel
    image then convert to BGR and blur slightly to get that soft speckle look.
    """
    img = np.zeros((height, width), dtype=np.uint8)

    xi = px.astype(np.int32)
    yi = py.astype(np.int32)

    # Clip to bounds
    np.clip(xi, 0, width-1, out=xi)
    np.clip(yi, 0, height-1, out=yi)

    # Draw as points; for size > 1, draw a tiny square
    if PARTICLE_SIZE_PX <= 1:
        img[yi, xi] = 255
    else:
        r = int(PARTICLE_SIZE_PX // 2)
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                x2 = np.clip(xi + dx, 0, width-1)
                y2 = np.clip(yi + dy, 0, height-1)
                img[y2, x2] = 255

    # Soft glow for a denser, airy feel
    img = cv2.GaussianBlur(img, (0,0), 0.6)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ==============
# Main loop
# ==============
prev = time.time()
fps_hist = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        # Resize to target output size
        frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)

        # Hand center
        hand = get_hand_center_bounded(frame)

        # Physics step (vectorized)
        if not _is_hand_closed:
            if hand is not None:
                hx, hy = hand

                # Repulsion only inside the box (cheaper & cleaner)
                # Find particles within a bounding square around the hand to reduce math
                # First a quick coarse mask, then compute forces
                dx = px - hx
                dy = py - hy

                # Coarse neighborhood (square)
                mask_coarse = (np.abs(dx) < FORCE_RADIUS*1.5) & (np.abs(dy) < FORCE_RADIUS*1.5)

                if np.any(mask_coarse):
                    dxc = dx[mask_coarse]
                    dyc = dy[mask_coarse]
                    d2 = dxc*dxc + dyc*dyc + 1e-3

                    # Smooth Gaussian-ish falloff = exp(-r^2 / (2*sigma^2))
                    falloff = np.exp(-d2 / (2.0*SIG2)).astype(np.float32)

                    # Force magnitude ~ (strength / (r^2 + eps)) * falloff
                    mag = (FORCE_STRENGTH / d2) * falloff

                    # Normalize direction and apply
                    invr = 1.0 / np.sqrt(d2)
                    fx = dxc * invr * mag
                    fy = dyc * invr * mag

                    vx[mask_coarse] += fx.astype(np.float32)
                    vy[mask_coarse] += fy.astype(np.float32)

            # Gravity + Brownian jitter
            vy += GRAVITY
            vx += (rng.standard_normal(N_PARTICLES).astype(np.float32) * BROWNIAN)
            vy += (rng.standard_normal(N_PARTICLES).astype(np.float32) * BROWNIAN)

            # Speed clamp
            speed2 = vx*vx + vy*vy
            mask_fast = speed2 > (MAX_SPEED*MAX_SPEED)
            if np.any(mask_fast):
                s = np.sqrt(speed2[mask_fast])
                vx[mask_fast] *= (MAX_SPEED / s)
                vy[mask_fast] *= (MAX_SPEED / s)

            # Integrate
            px += vx
            py += vy

            # Damping
            vx *= DAMPING
            vy *= DAMPING

            # Wall collisions (inside the “glass box”)
            # Left
            hit = px < LEFT
            if np.any(hit):
                px[hit] = LEFT + (LEFT - px[hit])
                vx[hit] *= -BOX_BOUNCE
            # Right
            hit = px > RIGHT
            if np.any(hit):
                px[hit] = RIGHT - (px[hit] - RIGHT)
                vx[hit] *= -BOX_BOUNCE
            # Top
            hit = py < TOP
            if np.any(hit):
                py[hit] = TOP + (TOP - py[hit])
                vy[hit] *= -BOX_BOUNCE
            # Bottom
            hit = py > BOTTOM
            if np.any(hit):
                py[hit] = BOTTOM - (py[hit] - BOTTOM)
                vy[hit] *= -BOX_BOUNCE

        # Render particle overlay and composite with camera frame
        overlay = render_particles_overlay(OUT_W, OUT_H, px, py)

        # Slight desaturate/darken background so white particles pop (screenshot vibe)
        bg = cv2.convertScaleAbs(frame, alpha=0.9, beta=0)

        # Additive-ish blend: first alpha mix, then add a bit more highlight
        out = cv2.addWeighted(bg, 1.0, overlay, PARTICLE_ALPHA, 0)
        out = cv2.add(out, (overlay // 8))  # tiny extra sparkle

        # Draw translucent box for that “contained volume” look
        if DRAW_BOX:
            box_color = (230, 230, 230)
            cv2.rectangle(out, (LEFT, TOP), (RIGHT, BOTTOM), box_color, 1, cv2.LINE_AA)

        # FPS (optional small overlay)
        # now = time.time()
        # fps = 1.0 / max(1e-6, (now - prev))
        # prev = now
        # fps_hist.append(fps)
        # if len(fps_hist) > 30:
        #     fps_hist.pop(0)
        # cv2.putText(out, f"{np.mean(fps_hist):.1f} FPS | particles: {N_PARTICLES}",
        #             (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)

        cv2.imshow("Hand-controlled Sand Box", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
