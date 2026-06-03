import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# config parte grafica
BEV_SIZE = 400  
BEV_SCALE = 110  
COLOR_RADAR = (30, 30, 30)

EDGES_SOLID  = [(0,2),(2,6),(6,4),(4,0), (0,1),(2,3),(4,5),(6,7)]
EDGES_DASHED = [(1,3),(3,7),(7,5),(5,1)]

# dimensioni medie
SEMANTIC_HEIGHTS = {
    'person': 1.75, 'chair': 1.0, 'bottle': 0.28,
    'cup': 0.09, 'wine glass': 0.22, 'bowl': 0.08,
    'apple': 0.15, 'orange': 0.12, 'potted plant': 0.25,
    'vase': 0.18, 'cell phone': 0.012
}

SEMANTIC_GIRTH = {
    'bottle': 0.08, 'cup': 0.09, 'wine glass': 0.08, 
    'bowl': 0.16, 'apple': 0.15, 'orange': 0.12, 
    'potted plant': 0.16, 'vase': 0.12, 'cell phone': 0.15
}
DEFAULT_GIRTH = 0.08

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=6):
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    if dist < 1: return
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = tuple((np.array(pt1) + (np.array(pt2) - np.array(pt1)) * (i / dashes)).astype(int))
        end   = tuple((np.array(pt1) + (np.array(pt2) - np.array(pt1)) * ((i + 0.5) / dashes)).astype(int))
        cv2.line(img, start, end, color, thickness)

def get_camera_params(w, h):
    fx = fy = w * 1.25  
    return np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float32)

def get_obb_axes_and_verts(pos, ext, ang):
    ca, sa = np.cos(ang), np.sin(ang)
    axes = np.array([[ca, sa], [-sa, ca]]) 
    cx, cz = pos[0], pos[2]
    hw, hd = ext[0], ext[2] 
    corners = []
    for sx in [-1, 1]:
        for sz in [-1, 1]:
            c = np.array([cx, cz]) + sx * hw * axes[0] + sz * hd * axes[1]
            corners.append(c)
    return np.array(corners), axes

def project_obb(corners, axis):
    dots = corners @ axis
    return dots.min(), dots.max()

def check_collision_sat(obj1, obj2):
    s1 = obj1['kf'].x.flatten()
    s2 = obj2['kf'].x.flatten()
    v1, ax1 = get_obb_axes_and_verts(s1[0:3], s1[3:6], s1[6])
    v2, ax2 = get_obb_axes_and_verts(s2[0:3], s2[3:6], s2[6])
    
    for axis in [ax1[0], ax1[1], ax2[0], ax2[1]]:
        min1, max1 = project_obb(v1, axis)
        min2, max2 = project_obb(v2, axis)
        if max1 < min2 or max2 < min1:
            return False 
    return True

def create_kalman():
    kf = KalmanFilter(dim_x=7, dim_z=7)
    kf.F = np.eye(7)
    kf.H = np.eye(7)
    kf.P = np.eye(7) * 5.0
    kf.R = np.diag([0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.1])
    kf.Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    return kf

class ObjectronTracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}
        self.max_age = 20

    def update(self, detections):
        for fid in list(self.objects.keys()):
            self.objects[fid]['last_seen'] += 1
            if self.objects[fid]['last_seen'] > self.max_age:
                del self.objects[fid]

        if not detections:
            return self.objects

        obj_ids = list(self.objects.keys())
        if not obj_ids:
            for det in detections:
                self._add_new(det)
            return self.objects

        prev_states  = np.array([self.objects[i]['kf'].x[:3, 0] for i in obj_ids])
        curr_centers = np.array([d['center'] for d in detections])
        dist_matrix  = np.linalg.norm(prev_states[:, np.newaxis] - curr_centers, axis=2)
        rows, cols   = linear_sum_assignment(dist_matrix)

        used_cols = set()
        for r, c in zip(rows, cols):
            if dist_matrix[r, c] < 1.0:  
                oid = obj_ids[r]
                z = np.array([*detections[c]['center'], *detections[c]['extents'], detections[c]['angle']])
                self.objects[oid]['kf'].predict()
                self.objects[oid]['kf'].update(z.reshape((7, 1)))
                self.objects[oid]['last_seen'] = 0
                self.objects[oid]['label'] = detections[c]['label']
                used_cols.add(c)

        for i, det in enumerate(detections):
            if i not in used_cols:
                self._add_new(det)

        return self.objects

    def _add_new(self, det):
        kf = create_kalman()
        kf.x = np.array([*det['center'], *det['extents'], det['angle']]).reshape((7, 1))
        self.objects[self.next_id] = {'kf': kf, 'label': det['label'], 'last_seen': 0}
        self.next_id += 1

# pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO("yolov8n-seg.pt")
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if device == "cuda" else -1
)
tracker = ObjectronTracker()

input_mode = input("1: Webcam | 2: Video | 3: Immagine\nScelta: ").strip()

if input_mode == "3":
    img_path = input("Path immagine: ").strip()
    cap = None
else:
    src = 0 if input_mode == "1" else input("Path video: ").strip()
    cap = cv2.VideoCapture(src)

cv2.namedWindow("OBJECTRON 3D - SCENE", cv2.WINDOW_NORMAL)
cv2.namedWindow("MAPPA DALL'ALTO (RADAR)", cv2.WINDOW_NORMAL)

while True:
    if input_mode == "3":
        frame = cv2.imread(img_path)
        if frame is None: break
    else:
        ret, frame = cap.read()
        if not ret or frame is None: break

    h, w = frame.shape[:2]
    K = get_camera_params(w, h)
    overlay_base = frame.copy()

    radar = np.full((BEV_SIZE, BEV_SIZE, 3), COLOR_RADAR, dtype=np.uint8)
    cv2.line(radar, (BEV_SIZE//2, BEV_SIZE), (BEV_SIZE//2, 0), (50, 50, 50), 1)

    with torch.no_grad():
        depth_out = depth_pipe(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        depth_raw = cv2.resize(np.array(depth_out["depth"]).astype(float), (w, h))

    p5, p95 = np.percentile(depth_raw, 5), np.percentile(depth_raw, 95)
    depth_metric = (depth_raw - p5) / (p95 - p5 + 1e-5) * 1.5 + 0.35 

    results = yolo(frame, conf=0.35, verbose=False)
    raw_detections = []
    
    if results[0].masks is not None:
        for i, mask_data in enumerate(results[0].masks.data.cpu().numpy()):
            label = yolo.names[int(results[0].boxes.cls[i])]
            
            if label not in SEMANTIC_HEIGHTS: continue
            
            m = cv2.resize(mask_data, (w, h)) > 0.5
            coords = np.argwhere(m)
            if len(coords) < 100: continue
            
            v_coords, u_coords = coords[:, 0], coords[:, 1]
            u_cen, v_cen = float(np.mean(u_coords)), float(np.mean(v_coords))
            
            z_med = float(np.median(depth_metric[m]))
            if z_med < 0.2: continue 
            
            center_x = (u_cen - K[0, 2]) * z_med / K[0, 0]
            center_y = (v_cen - K[1, 2]) * z_med / K[1, 1]
            center_z = z_med

            # Adattamento dinamico
            u_min, u_max = float(np.min(u_coords)), float(np.max(u_coords))
            v_min, v_max = float(np.min(v_coords)), float(np.max(v_coords))
            
            measured_width = float(((u_max - u_min) * z_med) / K[0, 0])
            measured_height = float(((v_max - v_min) * z_med) / K[1, 1])

            real_h = max(SEMANTIC_HEIGHTS[label], measured_height)
            real_girth = max(SEMANTIC_GIRTH.get(label, DEFAULT_GIRTH), measured_width)
            ext_x = measured_width / 2.0
            
            # Yaw
            angle = 0.0 

            raw_detections.append({
                'center':  [center_x, center_y, center_z],
                'extents': [ext_x, real_h / 2.0, real_girth / 2.0], 
                'angle': angle,
                'label': label
            })

    tracked_objs = tracker.update(raw_detections)
    colliding_ids = {
        oid for oid in tracked_objs
        for oid2 in tracked_objs
        if oid != oid2 and check_collision_sat(tracked_objs[oid], tracked_objs[oid2])
    }

    for oid, obj in tracked_objs.items():
        state = obj['kf'].x.flatten()
        pos, ext, ang = state[0:3], state[3:6], state[6]
        
        color = (0, 0, 255) if oid in colliding_ids else (0, 255, 0)

        cos_a, sin_a = np.cos(ang), np.sin(ang)
        R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        v3d = []
        for iz in [-1, 1]:
            for ix in [-1, 1]:
                for iy in [-1, 1]:
                    v_rel = np.array([ix * ext[0], iy * ext[1], iz * ext[2]])
                    v3d.append(pos + R @ v_rel)

        pts = np.array(v3d)          
        proj = (K @ pts.T).T         
        if not np.all(proj[:, 2] > 0.02): continue
        
        v2d = (proj[:, :2] / proj[:, 2:3]).astype(int)
        v2d[:, 0] = np.clip(v2d[:, 0], -2000, w + 2000)
        v2d[:, 1] = np.clip(v2d[:, 1], -2000, h + 2000)

        pts_base = np.array([v2d[0], v2d[4], v2d[6], v2d[2]], np.int32)
        cv2.fillPoly(overlay_base, [pts_base], color)
        frame = cv2.addWeighted(overlay_base, 0.2, frame, 0.8, 0)

        for s, e_idx in EDGES_SOLID:
            cv2.line(frame, tuple(v2d[s]), tuple(v2d[e_idx]), color, 2) 
        for s, e_idx in EDGES_DASHED:
            draw_dashed_line(frame, v2d[s], v2d[e_idx], color, 1) 

        label_y = max(int(v2d[:, 1].min()) - 10, 15)
        label_x = max(int(v2d[:, 0].min()), 5)
        info = f"ID:{oid} [{obj['label']}] Z:{pos[2]:.2f}m"
        if oid in colliding_ids: info += " [COLLISION]"
        
        cv2.putText(frame, info, (label_x + 1, label_y + 1), 0, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info, (label_x, label_y), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        rx = int(BEV_SIZE // 2 + pos[0] * BEV_SCALE)
        rz = int(BEV_SIZE - pos[2] * BEV_SCALE)
        if 0 < rx < BEV_SIZE and 0 < rz < BEV_SIZE:
            rect_size = (max(2, int(ext[0] * 2 * BEV_SCALE)), max(2, int(ext[2] * 2 * BEV_SCALE)))
            rect_pts  = cv2.boxPoints(((rx, rz), rect_size, 0))
            cv2.drawContours(radar, [np.int32(rect_pts)], 0, color, -1)
            cv2.putText(radar, f"{oid}", (rx + 5, rz), 0, 0.35, (255, 255, 255), 1)

    for d in range(1, 6):
        cv2.circle(radar, (BEV_SIZE//2, BEV_SIZE), d * BEV_SCALE, (60, 60, 60), 1)
        cv2.putText(radar, f"{d}m", (BEV_SIZE//2 + 5, BEV_SIZE - d * BEV_SCALE), 0, 0.3, (120, 120, 120), 1)

    cv2.imshow("OBJECTRON 3D - SCENE", frame)
    cv2.imshow("MAPPA DALL'ALTO (RADAR)", radar)

    key = cv2.waitKey(0 if input_mode == "3" else 1) & 0xFF
    if key == ord('q') or input_mode == "3": break

if cap: cap.release()
cv2.destroyAllWindows()