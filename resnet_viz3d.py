"""Generate resnet_3d.html: interactive 3D explorer for the unit-net
transplanted ResNet18. Stages = channel planes along depth; edges =
kernel-summed channel->channel net weights (top-k per node); click conv1
channels to see their RGB 7x7 filters, deeper channels for incoming
profiles + kernel pattern; class nodes for their fc rows."""
import json
import numpy as np
import torch
import torchvision
from resnet_transplant import transplant_resnet

IMAGENETTE = {0: "tench", 217: "springer", 482: "cassette", 491: "chainsaw",
              497: "church", 566: "horn", 569: "garbage truck",
              571: "gas pump", 574: "golf ball", 701: "parachute"}
TOPK_EDGE = 2      # edges kept per destination node
TOPK_IN = 16       # incoming entries in click payload

m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
transplant_resnet(m)

stages = [("input", 3, None)]        # (name, channels, conv module)
stages.append(("conv1", 64, m.conv1))
skips = []                            # (src_stage, dst_stage, kind)
si = 1
for ln, layer in (("L1", m.layer1), ("L2", m.layer2),
                  ("L3", m.layer3), ("L4", m.layer4)):
    for bi, block in enumerate(layer):
        pre = si
        stages.append((f"{ln}b{bi}c1", block.conv1.out_channels,
                       block.conv1))
        si += 1
        stages.append((f"{ln}b{bi}c2", block.conv2.out_channels,
                       block.conv2))
        si += 1
        skips.append((pre, si, "conv" if block.downsample is not None
                      else "identity",
                      block.downsample[0] if block.downsample is not None
                      else None))
stages.append(("classes", 10, "fc"))
si += 1

nodes, edges, payloads = [], [], {}
grid = {3: (3, 1), 64: (8, 8), 128: (16, 8), 256: (16, 16), 512: (32, 16),
        10: (10, 1)}
Z = np.linspace(-2.4, 2.4, len(stages))


def add_edges(mat, s_src, s_dst, kind="conv"):
    """mat: (out, in) signed net weights. Keep top-K per out node."""
    out, cin = mat.shape
    idx = np.argsort(-np.abs(mat), axis=1)[:, :TOPK_EDGE]
    for o in range(out):
        for j in idx[o]:
            w = float(mat[o, j])
            if abs(w) < 1e-4:
                continue
            edges.append([s_src, int(j), s_dst, o, round(w, 4), kind])


for s, (name, C, conv) in enumerate(stages):
    gx, gy = grid[C]
    for c in range(C):
        nodes.append({"s": s, "c": c, "x": (c % gx - (gx - 1) / 2) / max(gx, 6) * 2.2,
                      "y": ((c // gx) - (gy - 1) / 2) / max(gy, 6) * 2.2,
                      "z": float(Z[s])})
    if conv is None:
        continue
    if conv == "fc":
        W = m.fc.weight.data.numpy()
        sub = W[list(IMAGENETTE.keys())]           # (10, 512)
        add_edges(sub, s - 1, s, "fc")
        for o, (idx_, nm) in enumerate(IMAGENETTE.items()):
            top = np.argsort(-np.abs(sub[o]))[:TOPK_IN]
            payloads[f"{s}:{o}"] = {
                "title": f"class '{nm}'",
                "in": [[int(t), round(float(sub[o, t]), 4)] for t in top]}
        continue
    W = conv.weight.data.numpy()                   # (out, in, kh, kw)
    net = W.sum(axis=(2, 3))
    add_edges(net, s - 1, s, "conv")
    for o in range(W.shape[0]):
        top = np.argsort(-np.abs(net[o]))[:TOPK_IN]
        pay = {"title": f"{name} ch{o}",
               "in": [[int(t), round(float(net[o, t]), 4)] for t in top],
               "kern": np.round(np.abs(W[o]).mean(0), 4).tolist()}
        if name == "conv1":                        # RGB filter image
            f = W[o]                               # (3,7,7)
            f = (f - f.min()) / (f.max() - f.min() + 1e-9)
            pay["rgb"] = np.round(f.transpose(1, 2, 0), 3).tolist()
        payloads[f"{s}:{o}"] = pay

skip_list = []
for pre, post, kind, ds in skips:
    skip_list.append([pre, post, kind])
    if ds is not None:
        net = ds.weight.data.numpy().sum(axis=(2, 3))
        add_edges(net, pre, post, "skip")

stage_meta = [{"name": n, "C": C} for n, C, _ in stages]
data = {"stages": stage_meta, "nodes": nodes, "edges": edges,
        "skips": skip_list, "payloads": payloads}

HTML = """<title>Unit-ResNet18 in 3D</title>
<style>
body{margin:0;background:#101216;color:#edeef0;overflow:hidden;
font:13px ui-monospace,Menlo,monospace}
#hdr{position:fixed;top:0;left:0;right:0;padding:9px 16px;display:flex;
gap:18px;align-items:center;z-index:2;flex-wrap:wrap;color:#858b96}
.sw{display:inline-block;width:18px;height:3px;border-radius:2px;
vertical-align:middle;margin-right:6px}
canvas#cv{display:block;cursor:grab}
#panel{position:fixed;right:14px;top:52px;background:#1c1f24;
border:1px solid #2e323a;border-radius:6px;padding:13px;display:none;
z-index:3;max-width:250px}
#panel h3{margin:0 0 8px;font-size:13px;color:#edeef0}
#panel canvas{display:block;image-rendering:pixelated;border-radius:3px;
margin-bottom:8px}
#panel .bar{height:10px;margin:1px 0;border-radius:2px}
#panel .row{display:flex;align-items:center;gap:6px;font-size:10.5px;
color:#858b96}
#pclose{position:absolute;top:6px;right:10px;cursor:pointer;color:#858b96}
</style>
<div id="hdr">
 <b style="color:#edeef0">unit-ResNet18</b>
 <span><span class="sw" style="background:#4d8fe0"></span>net excitatory</span>
 <span><span class="sw" style="background:#e05c5c"></span>net inhibitory</span>
 <span><span class="sw" style="background:#9085e9"></span>residual skip</span>
 <span>95.5% of params unit-net-legal &middot; 99.95% agreement with original
 &middot; click conv1 nodes for RGB filters &middot; drag/wheel to orbit</span>
</div>
<div id="panel"><span id="pclose">&times;</span><h3 id="ptitle"></h3>
<div id="pbody"></div></div>
<canvas id="cv"></canvas>
<script>
const D=__DATA__;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let rx=-.28,ry=.85,zoom=1,drag=false,moved=false,px=0,py=0,spin=true;
let mouse=[0,0],hover=null,sel=null;
cv.onmousedown=e=>{drag=true;moved=false;spin=false;px=e.clientX;
  py=e.clientY;};
onmouseup=e=>{if(drag&&!moved){if(hover){sel=hover;showPanel(sel);}
  else{sel=null;panel.style.display='none';}}drag=false;};
onmousemove=e=>{if(drag){
  if(Math.abs(e.clientX-px)+Math.abs(e.clientY-py)>3)moved=true;
  ry+=(e.clientX-px)*.006;rx+=(e.clientY-py)*.006;
  rx=Math.max(-1.5,Math.min(1.5,rx));px=e.clientX;py=e.clientY;return;}
  mouse=[e.clientX,e.clientY];};
cv.addEventListener('wheel',e=>{e.preventDefault();
  zoom=Math.max(.35,Math.min(6,zoom*Math.exp(-e.deltaY*.001)));},
  {passive:false});
function project(p,W,H){
  const cy=Math.cos(ry),sy=Math.sin(ry),cx=Math.cos(rx),sx=Math.sin(rx);
  let x=p[0]*cy+p[2]*sy, z=-p[0]*sy+p[2]*cy, y=p[1];
  let y2=y*cx-z*sx, z2=y*sx+z*cx;
  const f=4.0, s=Math.min(W,H)*.30*zoom*f/(f+z2);
  return [W/2+x*s, H/2+y2*s, z2, s/(Math.min(W,H)*.30*zoom)];
}
const byStage={};D.nodes.forEach(n=>{(byStage[n.s]=byStage[n.s]||[])[n.c]=n;});
let wmax=0;D.edges.forEach(e=>wmax=Math.max(wmax,Math.abs(e[4])));
function draw(){
  requestAnimationFrame(draw);
  if(spin)ry+=.002;
  const W=innerWidth,H=innerHeight;cv.width=W;cv.height=H;
  ctx.clearRect(0,0,W,H);
  const P=new Map();
  D.nodes.forEach(n=>P.set(n.s+':'+n.c,project([n.x,n.y,n.z],W,H)));
  // hover hit-test
  hover=null;let hd=10;
  D.nodes.forEach(n=>{const p=P.get(n.s+':'+n.c);
    const d=Math.hypot(p[0]-mouse[0],p[1]-mouse[1]);
    if(d<hd){hd=d;hover=n.s+':'+n.c;}});
  const focus=hover||sel;
  // skip arcs
  ctx.strokeStyle='rgba(144,133,233,.5)';ctx.lineWidth=1.4;
  D.skips.forEach(([a,b,kind])=>{
    const za=project([0,-1.5,D.stages[a]?byStage[a][0].z:0],W,H);
    const zb=project([0,-1.5,byStage[b][0].z],W,H);
    ctx.beginPath();ctx.moveTo(za[0],za[1]);
    ctx.quadraticCurveTo((za[0]+zb[0])/2,(za[1]+zb[1])/2-40*za[3]*60,
      zb[0],zb[1]);
    ctx.setLineDash(kind==='identity'?[4,4]:[]);ctx.stroke();
    ctx.setLineDash([]);});
  // edges
  for(const [ss,j,sd,o,w,kind] of D.edges){
    const a=P.get(ss+':'+j),b=P.get(sd+':'+o);
    if(!a||!b)continue;
    const m=Math.abs(w)/wmax;
    let al=.04+.7*Math.pow(m,.6);
    if(focus){const t=(sd+':'+o)===focus||(ss+':'+j)===focus;
      al*=t?1:.05;}
    ctx.strokeStyle=kind==='skip'?`rgba(144,133,233,${al})`:
      (w>=0?`rgba(77,143,224,${al})`:`rgba(224,92,92,${al})`);
    ctx.lineWidth=.3+2.2*m;
    ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
  }
  // nodes
  D.nodes.forEach(n=>{const p=P.get(n.s+':'+n.c);
    const isF=focus===(n.s+':'+n.c);
    ctx.fillStyle=n.s===D.stages.length-1?'#eda100':
      (isF?'#eda100':'#c9cbd1');
    ctx.beginPath();ctx.arc(p[0],p[1],(isF?3.6:1.7)*Math.min(p[3]*2.4,2.2),
      0,7);ctx.fill();});
  // stage labels
  ctx.fillStyle='rgba(160,165,175,.75)';ctx.font='10px ui-monospace';
  ctx.textAlign='center';
  D.stages.forEach((st,s)=>{const n0=byStage[s][0];
    const p=project([0,1.55,n0.z],W,H);
    ctx.fillText(st.name,p[0],p[1]);});
}
requestAnimationFrame(draw);
const panel=document.getElementById('panel');
function showPanel(key){
  const pay=D.payloads[key];if(!pay)return;
  document.getElementById('ptitle').textContent=pay.title;
  const body=document.getElementById('pbody');body.innerHTML='';
  if(pay.rgb){
    const c=document.createElement('canvas');c.width=c.height=7*18;
    const g=c.getContext('2d');
    pay.rgb.forEach((row,y)=>row.forEach((px_,x)=>{
      g.fillStyle=`rgb(${px_[0]*255|0},${px_[1]*255|0},${px_[2]*255|0})`;
      g.fillRect(x*18,y*18,18,18);}));
    const cap=document.createElement('div');cap.className='row';
    cap.textContent='RGB 7×7 filter (unit-budget row)';
    body.appendChild(cap);body.appendChild(c);
  } else if(pay.kern){
    const k=pay.kern,s=k.length;
    const c=document.createElement('canvas');c.width=c.height=s*22;
    const g=c.getContext('2d');
    let mx=0;k.forEach(r=>r.forEach(v=>mx=Math.max(mx,v)));
    k.forEach((row,y)=>row.forEach((v,x)=>{
      const t=v/(mx+1e-9);
      g.fillStyle=`rgb(${30+50*t|0},${40+130*t|0},${60+180*t|0})`;
      g.fillRect(x*22,y*22,22,22);}));
    const cap=document.createElement('div');cap.className='row';
    cap.textContent='mean |kernel| pattern';
    body.appendChild(cap);body.appendChild(c);
  }
  const cap2=document.createElement('div');cap2.className='row';
  cap2.textContent='strongest incoming channels (net P−N):';
  body.appendChild(cap2);
  let mx=Math.max(...pay.in.map(x=>Math.abs(x[1])));
  pay.in.forEach(([ch,w])=>{
    const row=document.createElement('div');row.className='row';
    const bar=document.createElement('div');bar.className='bar';
    bar.style.width=(4+90*Math.abs(w)/mx)+'px';
    bar.style.background=w>=0?'#4d8fe0':'#e05c5c';
    const lab=document.createElement('span');
    lab.textContent=`ch${ch} ${w>=0?'+':''}${w}`;
    row.appendChild(bar);row.appendChild(lab);body.appendChild(row);});
  panel.style.display='block';
}
document.getElementById('pclose').onclick=()=>{sel=null;
  panel.style.display='none';};
</script>"""

html = HTML.replace("__DATA__", json.dumps(data))
open("resnet_3d.html", "w").write(html)
import os
print("wrote resnet_3d.html:",
      round(os.path.getsize("resnet_3d.html") / 1e6, 2), "MB,",
      len(nodes), "nodes,", len(edges), "edges")
