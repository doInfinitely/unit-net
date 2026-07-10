"""CACHED FOR POSTERITY (v1, 2026-07-10): the original five-stage voxel
explorer with the big lofted receptive-field arcs — superseded by
resnet_voxels.py (whole net, direct clearance curves, pan) but the high
arcs were cute. Writes resnet_voxels_v1.html.

Generate resnet_voxels_v1.html: the conv layers of the unit-net ResNet18
with their IMPLICIT NEURONS made explicit — each stage a C x H x W lattice
of voxels (cubes with gaps), colored by real activations from one forward
pass of an Imagenette image; hover/click a voxel to draw its true
receptive-field connections as outward-bowing Bezier curves (no clipping
through intervening neurons). Weight sharing is visible: same-channel
voxels carry the same incoming pattern on shifted windows.
"""
import json
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from resnet_transplant import transplant_resnet

DEV = "cpu"
N_CH = 8          # channels displayed per conv stage

m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
transplant_resnet(m)

tf = T.Compose([T.Resize(180), T.CenterCrop(160), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
import glob
img_path = sorted(glob.glob(
    "imagenette2-160/val/n03028079/*.JPEG"))[3]     # a church
from PIL import Image
pil = Image.open(img_path).convert("RGB")
x = tf(pil).unsqueeze(0)

acts = {}
with torch.no_grad():
    a = m.conv1(x)
    a = m.bn1(a); a = m.relu(a)
    p = m.maxpool(a)
    acts["pool"] = p
    b0 = m.layer1[0]
    c1 = b0.relu(b0.bn1(b0.conv1(p)))
    acts["l1c1"] = c1
    c2 = b0.bn2(b0.conv2(c1))
    acts["l1c2"] = torch.relu(c2 + p)
    b2 = m.layer2[0]
    acts["l2c1"] = b2.relu(b2.bn1(b2.conv1(acts["l1c2"])))

def topch(t, y0, y1):
    sub = t[0, :, y0:y1, y0:y1]
    return sub.mean(dim=(1, 2)).argsort(descending=True)[:N_CH].tolist()

CH = {"pool": topch(acts["pool"], 18, 22),
      "l1c1": topch(acts["l1c1"], 18, 22),
      "l1c2": topch(acts["l1c2"], 18, 22),
      "l2c1": topch(acts["l2c1"], 9, 11)}

raw = T.Compose([T.Resize(180), T.CenterCrop(160), T.ToTensor()])(pil)
crop_rgb = raw[:, 72:88, 72:88].numpy()

stages = []
voxels = []
def add_stage(name, note, n_ch, hw, org):
    stages.append({"name": name, "note": note, "C": n_ch, "S": hw,
                   "org": org})

add_stage("input", "16×16 crop, RGB", 3, 16, 72)
for y in range(16):
    for xx in range(16):
        voxels.append({"st": 0, "ch": 0, "y": y, "x": xx,
                       "rgb": [round(float(crop_rgb[c, y, xx]), 3)
                               for c in range(3)]})
def norm(t):
    t = t.clamp(min=0)
    return (t / (t.max() + 1e-9)).numpy()

for si, (key, hw, org, note) in enumerate(
        [("pool", 4, 18, "conv1 7×7 s2 + maxpool (RF approx)"),
         ("l1c1", 4, 18, "layer1.0.conv1 3×3 (exact RF)"),
         ("l1c2", 4, 18, "layer1.0.conv2 3×3 + skip (exact RF)"),
         ("l2c1", 2, 9, "layer2.0.conv1 3×3 s2 (exact RF)")], start=1):
    add_stage(key, note, N_CH, hw, org)
    A = norm(acts[key][0])
    for ci, ch in enumerate(CH[key]):
        for y in range(hw):
            for xx in range(hw):
                voxels.append({"st": si, "ch": ci, "y": y, "x": xx,
                               "v": round(float(A[ch, org + y, org + xx]),
                                          3)})

def vidx():
    d = {}
    for i, v in enumerate(voxels):
        d[(v["st"], v["ch"], v["y"], v["x"])] = i
    return d
VI = vidx()
incoming = {}

W1 = m.conv1.weight.data.numpy()
for ci, ch in enumerate(CH["pool"]):
    Wc = W1[ch]
    for y in range(4):
        for xx in range(4):
            dst = VI[(1, ci, y, xx)]
            lst = []
            base_y, base_x = (18 + y) * 4 - 72 - 3, (18 + xx) * 4 - 72 - 3
            for dy in range(7):
                for dx in range(7):
                    sy, sx = base_y + dy, base_x + dx
                    if not (0 <= sy < 16 and 0 <= sx < 16):
                        continue
                    w = float(Wc[:, dy, dx].sum())
                    if abs(w) < 5e-3:
                        continue
                    lst.append([VI[(0, 0, sy, sx)], round(w, 4)])
            incoming[dst] = lst

def conv_edges(si_dst, key_dst, si_src, key_src, conv, stride, hw_d, hw_s):
    W = conv.weight.data.numpy()
    for ci, ch in enumerate(CH[key_dst]):
        for y in range(hw_d):
            for xx in range(hw_d):
                dst = VI[(si_dst, ci, y, xx)]
                lst = incoming.get(dst, [])
                for cj, chs in enumerate(CH[key_src]):
                    for dy in range(3):
                        for dx in range(3):
                            sy = stride * y + dy - 1
                            sx = stride * xx + dx - 1
                            if not (0 <= sy < hw_s and 0 <= sx < hw_s):
                                continue
                            w = float(W[ch, chs, dy, dx])
                            if abs(w) < 5e-3:
                                continue
                            lst.append([VI[(si_src, cj, sy, sx)],
                                        round(w, 4)])
                incoming[dst] = lst

b0, b2 = m.layer1[0], m.layer2[0]
conv_edges(2, "l1c1", 1, "pool", b0.conv1, 1, 4, 4)
conv_edges(3, "l1c2", 2, "l1c1", b0.conv2, 1, 4, 4)
conv_edges(4, "l2c1", 3, "l1c2", b2.conv1, 2, 2, 4)

data = {"stages": stages, "voxels": voxels,
        "incoming": {str(k): v for k, v in incoming.items()}}

HTML = """<title>Unit-ResNet: explicit neurons (v1)</title>
<style>
body{margin:0;background:#0e1013;color:#edeef0;overflow:hidden;
font:13px ui-monospace,Menlo,monospace}
#hdr{position:fixed;top:0;left:0;right:0;padding:9px 16px;display:flex;
gap:18px;z-index:2;color:#858b96;flex-wrap:wrap}
.sw{display:inline-block;width:18px;height:3px;border-radius:2px;
vertical-align:middle;margin-right:5px}
canvas{display:block;cursor:grab}
</style>
<div id="hdr"><b style="color:#edeef0">unit-ResNet18 — every neuron (v1,
the cute lofted arcs)</b>
<span><span class="sw" style="background:#4d8fe0"></span>excitatory path</span>
<span><span class="sw" style="background:#e05c5c"></span>inhibitory path</span>
<span>voxel brightness = activation on a real church image &middot;
hover a voxel for its receptive-field cone &middot; click to pin</span></div>
<canvas id="cv"></canvas>
<script>
const D=__DATA__;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let rx=-.35,ry=.7,zoom=1,drag=false,moved=false,px=0,py=0,spin=true;
let mouse=[0,0],hover=null,pin=null;
cv.onmousedown=e=>{drag=true;moved=false;spin=false;px=e.clientX;
  py=e.clientY;};
onmouseup=e=>{if(drag&&!moved){pin=hover;}drag=false;};
onmousemove=e=>{if(drag){
  if(Math.abs(e.clientX-px)+Math.abs(e.clientY-py)>3)moved=true;
  ry+=(e.clientX-px)*.006;rx+=(e.clientY-py)*.006;
  rx=Math.max(-1.5,Math.min(1.5,rx));px=e.clientX;py=e.clientY;return;}
  mouse=[e.clientX,e.clientY];};
cv.addEventListener('wheel',e=>{e.preventDefault();
  zoom=Math.max(.4,Math.min(6,zoom*Math.exp(-e.deltaY*.001)));},
  {passive:false});

const SX=[-3.4,-1.2,0.6,2.2,3.8];
const pos3=[];
D.voxels.forEach(v=>{
  const st=D.stages[v.st];
  const cgap=0.16, sp=1.9/st.S;
  const X=SX[v.st]+(v.ch-(st.C-1)/2)*cgap;
  const Y=(v.y-(st.S-1)/2)*sp;
  const Z=(v.x-(st.S-1)/2)*sp;
  pos3.push([X,Y,Z]);
});
function project(p,W,H){
  const cy=Math.cos(ry),sy=Math.sin(ry),cx=Math.cos(rx),sx=Math.sin(rx);
  let x=p[0]*cy+p[2]*sy, z=-p[0]*sy+p[2]*cy, y=p[1];
  let y2=y*cx-z*sx, z2=y*sx+z*cx;
  const f=5.2, s=Math.min(W,H)*.24*zoom*f/(f+z2);
  return [W/2+x*s, H/2+y2*s, z2, s/(Math.min(W,H)*.24*zoom)];
}
function bez(a,b,W,H){
  const lift=0.55+0.25*Math.abs(a[0]-b[0]);
  const c1=[(2*a[0]+b[0])/3, a[1]-lift, (2*a[2]+b[2])/3];
  const c2=[(a[0]+2*b[0])/3, b[1]-lift, (a[2]+2*b[2])/3];
  const pts=[];
  for(let t=0;t<=1.001;t+=1/14){
    const u=1-t;
    const p=[0,1,2].map(i=>u*u*u*a[i]+3*u*u*t*c1[i]+3*u*t*t*c2[i]
      +t*t*t*b[i]);
    pts.push(project(p,W,H));
  }
  return pts;
}
function draw(){
  requestAnimationFrame(draw);
  if(spin)ry+=.0018;
  const W=innerWidth,H=innerHeight;cv.width=W;cv.height=H;
  ctx.clearRect(0,0,W,H);
  const P=D.voxels.map((_,i)=>project(pos3[i],W,H));
  hover=null;let hd=9;
  P.forEach((p,i)=>{const d=Math.hypot(p[0]-mouse[0],p[1]-mouse[1]);
    if(d<hd){hd=d;hover=i;}});
  const focus=hover!==null?hover:pin;
  const order=P.map((p,i)=>[p[2],i]).sort((a,b)=>b[0]-a[0]);
  for(const [,i] of order){
    const v=D.voxels[i],p=P[i];
    const sz=Math.max(1.5,5.2*p[3])*(D.stages[v.st].S>8?0.62:1);
    let col;
    if(v.rgb){col=`rgb(${v.rgb[0]*255|0},${v.rgb[1]*255|0},
      ${v.rgb[2]*255|0})`;}
    else{const t=v.v;
      col=`rgb(${25+60*t|0},${35+140*t|0},${55+200*t|0})`;}
    ctx.fillStyle=col;
    const s=sz*2;
    ctx.fillRect(p[0]-s/2,p[1]-s/2,s,s);
    if(i===focus){ctx.strokeStyle='#eda100';ctx.lineWidth=2;
      ctx.strokeRect(p[0]-s/2-2,p[1]-s/2-2,s+4,s+4);}
  }
  if(focus!==null){
    const inc=D.incoming[String(focus)];
    if(inc){
      let mx=0;inc.forEach(([,w])=>mx=Math.max(mx,Math.abs(w)));
      for(const [src,w] of inc){
        const pts=bez(pos3[src],pos3[focus],W,H);
        const m2=Math.abs(w)/mx;
        ctx.strokeStyle=w>=0?`rgba(77,143,224,${.15+.8*m2})`
          :`rgba(224,92,92,${.15+.8*m2})`;
        ctx.lineWidth=.5+2.6*m2;
        ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);
        for(const q of pts.slice(1))ctx.lineTo(q[0],q[1]);
        ctx.stroke();
        const sp2=P[src];
        ctx.strokeStyle='#eda100';ctx.lineWidth=1;
        ctx.strokeRect(sp2[0]-4,sp2[1]-4,8,8);
      }
    }
  }
  ctx.fillStyle='rgba(160,165,175,.8)';ctx.font='10.5px ui-monospace';
  ctx.textAlign='center';
  D.stages.forEach((st,s)=>{
    const p=project([SX[s],1.55,0],W,H);
    ctx.fillText(st.name,p[0],p[1]);
    ctx.fillStyle='rgba(130,135,145,.55)';
    ctx.fillText(st.note,p[0],p[1]+13);
    ctx.fillStyle='rgba(160,165,175,.8)';
  });
}
requestAnimationFrame(draw);
</script>"""

html = HTML.replace("__DATA__", json.dumps(data))
open("resnet_voxels_v1.html", "w").write(html)
import os
n_edges = sum(len(v) for v in incoming.values())
print("wrote resnet_voxels_v1.html:",
      round(os.path.getsize("resnet_voxels_v1.html") / 1e6, 2), "MB,",
      len(voxels), "voxels,", n_edges, "receptive-field edges")
