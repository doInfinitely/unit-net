"""resnet_voxels.html v2: the WHOLE unit-net ResNet18 with implicit conv
neurons explicit. One aligned center column followed through every stride
(16x16 -> 4x4 -> ... -> 1x1) so every receptive field is exact; all 18
conv stages + class row; activations from a real image; direct
low-clearance curves; orbit + PAN (right-drag / shift-drag) + zoom.
"""
import glob
import json
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
from resnet_transplant import transplant_resnet, IMNET_IDX

N_CH = 8
CLASSNAMES = ["tench", "springer", "cassette", "chainsaw", "church",
              "horn", "garbage", "gaspump", "golf", "parachute"]

m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
transplant_resnet(m)

tf = T.Compose([T.Resize(180), T.CenterCrop(160), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
pil = Image.open(sorted(glob.glob(
    "imagenette2-160/val/n03028079/*.JPEG"))[3]).convert("RGB")
x = tf(pil).unsqueeze(0)

# ---- forward, capturing every stage ----
caps = []      # (name, tensor, spatial_crop_origin, crop_size, stride_in,
#                 conv_module or None, src_stage_index)
with torch.no_grad():
    a = m.relu(m.bn1(m.conv1(x)))
    p = m.maxpool(a)
    caps.append(("pool", p, 18, 4, None, None))
    prev = p
    blocks = [(m.layer1, 18, 4, 1), (m.layer2, 9, 2, 2),
              (m.layer3, 5, 1, 2), (m.layer4, 2, 1, 2)]
    for layer, org, hw, first_stride in blocks:
        for bi, block in enumerate(layer):
            s1 = first_stride if bi == 0 else 1
            c1 = block.relu(block.bn1(block.conv1(prev)))
            caps.append((f"{layer[0].conv1.out_channels}c1b{bi}", c1,
                         org, hw, block.conv1, s1))
            c2 = block.bn2(block.conv2(c1))
            idn = prev if block.downsample is None else block.downsample(prev)
            out = torch.relu(c2 + idn)
            caps.append((f"{layer[0].conv1.out_channels}c2b{bi}", out,
                         org, hw, block.conv2, 1))
            prev = out

# nicer stage names
names = ["input", "conv1·pool"]
for ln, layer in (("L1", m.layer1), ("L2", m.layer2), ("L3", m.layer3),
                  ("L4", m.layer4)):
    for bi in range(len(layer)):
        names += [f"{ln}.{bi}.c1", f"{ln}.{bi}.c2+skip"]
names.append("classes")

# displayed channels per stage: liveliest in crop
CH = []
for name, t, org, hw, conv, stride in caps:
    sub = t[0, :, org:org + hw, org:org + hw]
    CH.append(sub.mean(dim=(1, 2)).argsort(descending=True)[:N_CH].tolist())

raw = T.Compose([T.Resize(180), T.CenterCrop(160), T.ToTensor()])(pil)
crop_rgb = raw[:, 72:88, 72:88].numpy()

stages = [{"name": names[0], "C": 3, "S": 16}]
voxels = []
for y in range(16):
    for xx in range(16):
        voxels.append({"st": 0, "ch": 0, "y": y, "x": xx,
                       "rgb": [round(float(crop_rgb[c, y, xx]), 3)
                               for c in range(3)]})
for si, (name, t, org, hw, conv, stride) in enumerate(caps, start=1):
    stages.append({"name": names[si], "C": N_CH, "S": hw})
    A = t[0].clamp(min=0)
    A = (A / (A.max() + 1e-9)).numpy()
    for ci, ch in enumerate(CH[si - 1]):
        for y in range(hw):
            for xx in range(hw):
                voxels.append({"st": si, "ch": ci, "y": y, "x": xx,
                               "v": round(float(A[ch, org + y, org + xx]),
                                          3)})
stages.append({"name": names[-1], "C": 10, "S": 1})
for c in range(10):
    voxels.append({"st": len(caps) + 1, "ch": c, "y": 0, "x": 0, "v": 0.5,
                   "lab": CLASSNAMES[c]})

VI = {(v["st"], v["ch"], v["y"], v["x"]): i for i, v in enumerate(voxels)}
incoming = {}

# conv1+pool stage <- input (effective RF, approx)
W1 = m.conv1.weight.data.numpy()
for ci, ch in enumerate(CH[0]):
    for y in range(4):
        for xx in range(4):
            dst = VI[(1, ci, y, xx)]
            lst = []
            by, bx = (18 + y) * 4 - 72 - 3, (18 + xx) * 4 - 72 - 3
            for dy in range(7):
                for dx in range(7):
                    sy, sx = by + dy, bx + dx
                    if 0 <= sy < 16 and 0 <= sx < 16:
                        w = float(W1[ch][:, dy, dx].sum())
                        if abs(w) >= 5e-3:
                            lst.append([VI[(0, 0, sy, sx)], round(w, 4)])
            incoming[dst] = lst

# exact 3x3 RFs between consecutive stages
for si in range(2, len(caps) + 1):
    name, t, org_d, hw_d, conv, stride = caps[si - 1]
    _, _, org_s, hw_s, _, _ = caps[si - 2]
    W = conv.weight.data.numpy()
    ch_d, ch_s = CH[si - 1], CH[si - 2]
    for ci, ch in enumerate(ch_d):
        for y in range(hw_d):
            for xx in range(hw_d):
                dst = VI[(si, ci, y, xx)]
                lst = incoming.get(dst, [])
                ay, ax = org_d + y, org_d + xx
                for cj, chs in enumerate(ch_s):
                    for dy in range(3):
                        for dx in range(3):
                            sy = stride * ay + dy - 1 - org_s
                            sx = stride * ax + dx - 1 - org_s
                            if 0 <= sy < hw_s and 0 <= sx < hw_s:
                                w = float(W[ch, chs, dy, dx])
                                if abs(w) >= 5e-3:
                                    lst.append([VI[(si - 1, cj, sy, sx)],
                                                round(w, 4)])
                incoming[dst] = lst

# classes <- last stage (fc rows over displayed channels)
Wfc = m.fc.weight.data.numpy()[IMNET_IDX]
last = len(caps)
for c in range(10):
    dst = VI[(last + 1, c, 0, 0)]
    lst = []
    for cj, chs in enumerate(CH[-1]):
        w = float(Wfc[c, chs])
        if abs(w) >= 1e-3:
            lst.append([VI[(last, cj, 0, 0)], round(w, 4)])
    incoming[dst] = lst

data = {"stages": stages, "voxels": voxels,
        "incoming": {str(k): v for k, v in incoming.items()}}

HTML = """<title>Unit-ResNet: every neuron, whole net</title>
<style>
body{margin:0;background:#0e1013;color:#edeef0;overflow:hidden;
font:13px ui-monospace,Menlo,monospace}
#hdr{position:fixed;top:0;left:0;right:0;padding:9px 16px;display:flex;
gap:16px;z-index:2;color:#858b96;flex-wrap:wrap}
.sw{display:inline-block;width:18px;height:3px;border-radius:2px;
vertical-align:middle;margin-right:5px}
canvas{display:block;cursor:grab}
</style>
<div id="hdr"><b style="color:#edeef0">unit-ResNet18 — all 18 conv stages
</b>
<span><span class="sw" style="background:#4d8fe0"></span>excitatory</span>
<span><span class="sw" style="background:#e05c5c"></span>inhibitory</span>
<span>left-drag orbit &middot; right- or shift-drag PAN &middot; wheel zoom
&middot; hover/click a voxel for its exact receptive-field cone &middot;
brightness = activation on a church image</span></div>
<canvas id="cv"></canvas>
<script>
const D=__DATA__;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let rx=-.3,ry=.5,zoom=.85,panX=0,panY=0;
let drag=0,moved=false,px=0,py=0,spin=true,mouse=[0,0],hover=null,pin=null;
cv.oncontextmenu=e=>e.preventDefault();
cv.onmousedown=e=>{drag=(e.button===2||e.shiftKey)?2:1;moved=false;
  spin=false;px=e.clientX;py=e.clientY;};
onmouseup=e=>{if(drag===1&&!moved){pin=hover;}drag=0;};
onmousemove=e=>{
  if(drag===1){if(Math.abs(e.clientX-px)+Math.abs(e.clientY-py)>3)moved=true;
    ry+=(e.clientX-px)*.006;rx+=(e.clientY-py)*.006;
    rx=Math.max(-1.5,Math.min(1.5,rx));px=e.clientX;py=e.clientY;return;}
  if(drag===2){panX+=e.clientX-px;panY+=e.clientY-py;
    px=e.clientX;py=e.clientY;moved=true;return;}
  mouse=[e.clientX,e.clientY];};
cv.addEventListener('wheel',e=>{e.preventDefault();
  const f=Math.exp(-e.deltaY*.001);
  panX=e.clientX-(e.clientX-panX)*f;panY=e.clientY-(e.clientY-panY)*f;
  zoom=Math.max(.15,Math.min(10,zoom*f));},{passive:false});

// stage anchors: within-block tight, between-block roomy
const SX=[];let acc=-7.2;
for(let s=0;s<D.stages.length;s++){
  SX.push(acc);
  const cur=D.stages[s].name,nxt=(D.stages[s+1]||{}).name||"";
  const blockBreak=nxt.startsWith("L")&&cur.slice(0,4)!==nxt.slice(0,4)
    ||nxt==="classes"||s===0;
  acc+=blockBreak?1.15:0.72;
}
const pos3=D.voxels.map(v=>{
  const st=D.stages[v.st];
  const cg=st.S===1?0.11:0.055, sp=st.S>1?1.7/st.S:0;
  return [SX[v.st]+(v.ch-(st.C-1)/2)*cg,
          (v.y-(st.S-1)/2)*sp,(v.x-(st.S-1)/2)*sp];
});
function project(p,W,H){
  const cy=Math.cos(ry),sy=Math.sin(ry),cx=Math.cos(rx),sx=Math.sin(rx);
  let x=p[0]*cy+p[2]*sy, z=-p[0]*sy+p[2]*cy, y=p[1];
  let y2=y*cx-z*sx, z2=y*sx+z*cx;
  const f=7, s=Math.min(W,H)*.30*zoom*f/(f+z2);
  return [W/2+x*s+panX-0*panX+ (panX), H/2+y2*s+panY, z2, s]
    .map((v,i)=>i===0?W/2+x*s+panX:(i===1?H/2+y2*s+panY:v));
}
function bez(a,b,W,H){
  // direct path with a small clearance bow: lift just enough to clear a
  // slab, perpendicular-ish (up in Y), scaled by crossing distance
  const dx=Math.abs(a[0]-b[0]);
  const lift=Math.min(.28,.06+.09*dx);
  const c1=[a[0]*.67+b[0]*.33,(a[1]*.67+b[1]*.33)-lift,a[2]*.67+b[2]*.33];
  const c2=[a[0]*.33+b[0]*.67,(a[1]*.33+b[1]*.67)-lift,a[2]*.33+b[2]*.67];
  const pts=[];
  for(let t=0;t<=1.001;t+=1/12){
    const u=1-t;
    pts.push(project([0,1,2].map(i=>
      u*u*u*a[i]+3*u*u*t*c1[i]+3*u*t*t*c2[i]+t*t*t*b[i]),W,H));
  }
  return pts;
}
function draw(){
  requestAnimationFrame(draw);
  if(spin)ry+=.0014;
  const W=innerWidth,H=innerHeight;cv.width=W;cv.height=H;
  ctx.clearRect(0,0,W,H);
  const P=pos3.map(p=>project(p,W,H));
  hover=null;let hd=9;
  P.forEach((p,i)=>{const d=Math.hypot(p[0]-mouse[0],p[1]-mouse[1]);
    if(d<hd){hd=d;hover=i;}});
  const focus=hover!==null?hover:pin;
  const order=P.map((p,i)=>[p[2],i]).sort((a,b)=>b[0]-a[0]);
  for(const [,i] of order){
    const v=D.voxels[i],p=P[i],st=D.stages[v.st];
    const base=st.S>8?2.2:(st.S>2?4.5:6);
    const s=Math.max(1.2,base*p[3]/(Math.min(W,H)*.3));
    let col;
    if(v.rgb)col=`rgb(${v.rgb[0]*255|0},${v.rgb[1]*255|0},${v.rgb[2]*255|0})`;
    else if(v.lab)col='#eda100';
    else{const t=v.v;col=`rgb(${25+60*t|0},${35+140*t|0},${55+200*t|0})`;}
    ctx.fillStyle=col;
    ctx.fillRect(p[0]-s,p[1]-s,2*s,2*s);
    if(i===focus){ctx.strokeStyle='#eda100';ctx.lineWidth=2;
      ctx.strokeRect(p[0]-s-2,p[1]-s-2,2*s+4,2*s+4);}
    if(v.lab){ctx.fillStyle='rgba(230,232,236,.9)';
      ctx.font='9.5px ui-monospace';ctx.textAlign='left';
      ctx.fillText(v.lab,p[0]+s+4,p[1]+3);}
  }
  if(focus!==null&&D.incoming[String(focus)]){
    const inc=D.incoming[String(focus)];
    let mx=0;inc.forEach(([,w])=>mx=Math.max(mx,Math.abs(w)));
    for(const [src,w] of inc){
      const pts=bez(pos3[src],pos3[focus],W,H);
      const m2=Math.abs(w)/mx;
      ctx.strokeStyle=w>=0?`rgba(77,143,224,${.14+.8*m2})`
        :`rgba(224,92,92,${.14+.8*m2})`;
      ctx.lineWidth=.5+2.4*m2;
      ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);
      for(const q of pts.slice(1))ctx.lineTo(q[0],q[1]);
      ctx.stroke();
      const sp2=P[src];
      ctx.strokeStyle='rgba(237,161,0,.8)';ctx.lineWidth=1;
      ctx.strokeRect(sp2[0]-3.5,sp2[1]-3.5,7,7);
    }
  }
  ctx.font='9.5px ui-monospace';ctx.textAlign='center';
  D.stages.forEach((st,s)=>{
    const p=project([SX[s],1.35,0],W,H);
    ctx.fillStyle='rgba(160,165,175,.8)';
    ctx.save();ctx.translate(p[0],p[1]);ctx.rotate(-.9);
    ctx.fillText(st.name,0,0);ctx.restore();});
}
requestAnimationFrame(draw);
</script>"""

html = HTML.replace("__DATA__", json.dumps(data))
open("resnet_voxels.html", "w").write(html)
import os
print("wrote resnet_voxels.html:",
      round(os.path.getsize("resnet_voxels.html") / 1e6, 2), "MB,",
      len(voxels), "voxels,",
      sum(len(v) for v in incoming.values()), "edges,",
      len(stages), "stages")
