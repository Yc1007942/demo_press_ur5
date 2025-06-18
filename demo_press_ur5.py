
MODEL_FILES = [                           
    "fold4.pt",                          
]
DEFAULT_PT_IMG = 380                      
DEVICE        = "cuda"                    
CAM_INDEX     = 0                        

# ([x y z rx ry rz], depth_mm, speed_m_s, step_mm)
POSES = [
    ([  0.481,  0.136, -0.111,  2.889,  1.232,  0.001 ], 6.0, 0.001, 0.01),
    ([  0.481,  0.136, -0.111,  2.889,  1.232,  0.001 ], 5.0, 0.0001, 0.01),
]
HOLD_SEC    = 2.0
RETURN_POSE = [0.563, 0.232, 0.083, -1.664, -0.857, -1.768]
ROBOT_IP, PORT = "10.10.10.1", 50002
TCP_OFFSET  = (0, 0, 0.26, 0, 0, 0)
import sys, signal, time, glob, os
from pathlib import Path
import cv2, numpy as np
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.set_visible_devices([], "GPU")          
import torch, torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b4
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import setting, find_marker, A_utility
PT_DEVICE = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
class ModelWrapper:
    _pt_pre = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],
                    [0.229,0.224,0.225])
    ])
    def __init__(self, fp:str):
        self.fp = fp
        ext = fp.split(".")[-1].lower()
        if ext in ("keras","h5"):                     
            self.kind = "tf"
            self.m    = tf.keras.models.load_model(fp, compile=False)
            self.inp  = self.m.input_shape[1]
            print(f"TF model ({self.inp}²) loaded from '{fp}'")
        elif ext == "pt":                             
            self.kind = "pt"
            chk = torch.load(fp, map_location="cpu")
            sd  = chk["model"] if isinstance(chk,dict) and "model" in chk else chk
            net = efficientnet_b4(weights=None)
            in_dim = net.classifier[1].in_features
            net.classifier[1] = torch.nn.Sequential(
                torch.nn.Dropout(0.3), torch.nn.Linear(in_dim,10))
            net.load_state_dict(sd, strict=False)
            net.to(PT_DEVICE).eval()
            self.m   = net
            self.inp = DEFAULT_PT_IMG
            print(f" PT model ({self.inp}²) loaded from '{fp}'")
        else:
            raise ValueError(f"Unsupported extension: {fp}")
    @torch.no_grad()
    def predict(self, rgb:np.ndarray)->np.ndarray:
        if self.kind == "tf":
            img = cv2.resize(rgb,(self.inp,self.inp), cv2.INTER_AREA)
            x   = tf.keras.applications.efficientnet_v2.preprocess_input(img.astype(np.float32))
            p   = self.m(x[None,...], training=False).numpy()[0]
            return p
        else:
            img = cv2.resize(rgb,(self.inp,self.inp), cv2.INTER_AREA)
            t   = self._pt_pre(img).unsqueeze(0).to(PT_DEVICE)
            logits = self.m(t)
            return F.softmax(logits,1).cpu().numpy()[0]
WRAPPERS = [ModelWrapper(f) for f in MODEL_FILES]
print(f" Ensemble size: {len(WRAPPERS)}")
cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
if not cam.isOpened(): sys.exit("Cannot open camera.")
cam.set(3,800); cam.set(4,600)
setting.init()
print("Camera initialised.")
rtde_c = RTDEControlInterface(ROBOT_IP, PORT); rtde_c.setTcp(TCP_OFFSET)
rtde_r = RTDEReceiveInterface(ROBOT_IP)
print("✔  UR5 RTDE connection established.")
def clean_exit(*_):
    print("\n[EXIT] Stopping robot & closing camera …")
    rtde_c.stopScript(); cam.release(); cv2.destroyAllWindows(); sys.exit(0)
signal.signal(signal.SIGINT, clean_exit)
def predict_and_show():
    ok, frame_bgr = cam.read()
    if not ok: return
    t0   = time.time()
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preds= [wr.predict(rgb) for wr in WRAPPERS]          
    preds= np.stack(preds)                               
    ens  = preds.mean(0)                                 
    y,dy = 25,22
    for i,(wr,p) in enumerate(zip(WRAPPERS,preds)):
        cls,conf = p.argmax(), p.max()*100
        tag      = Path(wr.fp).name[:10]
        txt      = f"[{i+1}:{tag}] {cls+1}  {conf:5.1f}%"
        cv2.putText(frame_bgr, txt,(10,y+dy*i),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2,cv2.LINE_AA)

    e_cls,e_conf = ens.argmax(), ens.max()*100
    cv2.putText(frame_bgr, f"[ENS] {e_cls+1}  {e_conf:5.1f}%",
                (10,y+dy*len(WRAPPERS)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(0,0,255),2,cv2.LINE_AA)

    fps = 1./(time.time()-t0)
    cv2.putText(frame_bgr, f"{fps:4.1f} FPS",(10,18),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1,cv2.LINE_AA)

    cv2.imshow("GelSight live (ensemble)", frame_bgr)
    if cv2.waitKey(1)&0xFF==ord('q'):
        raise KeyboardInterrupt

def rotvec_to_R(rx, ry, rz):
    v=np.asarray([rx,ry,rz]); th=np.linalg.norm(v)
    if th<1e-9: return np.eye(3)
    k=v/th; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K

# ─────── Main press loop ───────
print(f"[DEMO] {len(POSES)} press positions queued.")
for idx,(pose,depth,speed,step) in enumerate(POSES,1):
    print(f"\n► Pose {idx}: depth {depth} mm, speed {speed} m/s, step {step} mm")
    rtde_c.moveL(pose,0.25,0.2); time.sleep(0.05)

    tcp=list(rtde_r.getActualTCPPose())
    R  = rotvec_to_R(*tcp[3:])
    step_vec = -(step/1000)*R[:,0]
    steps=int(depth/step+0.5)

    for _ in range(steps):
        predict_and_show()
        tcp[:3]=(np.array(tcp[:3])+step_vec).tolist()
        rtde_c.moveL(tcp,speed,speed)

    t0=time.time()
    while time.time()-t0 < HOLD_SEC:
        predict_and_show()

    rtde_c.moveL(RETURN_POSE,0.25,0.2)
print("\nDemo finished – robot parked.")
clean_exit()
