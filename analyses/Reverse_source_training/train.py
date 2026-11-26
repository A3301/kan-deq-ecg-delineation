"""Train the two heads exclusively on RDB. This process never reads LUDB."""
import csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import UNetKAN

ROOT=Path(__file__).resolve().parent
TRAIN=ROOT/'training'; MODELS=ROOT/'models'
STEPS=2400; BATCH=32; WIDTH=1024

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def create(head):
    return UNetKAN(where=['head']) if head=='KAN' else UNetKAN(where=[],mlp_head=29)

def taper_probabilities(net, x):
    starts=(0,x.shape[-1]-2048);taper=torch.hann_window(2048,periodic=False,device=x.device)+.001
    pred=torch.zeros(len(x),4,x.shape[-1],device=x.device);weight=torch.zeros(x.shape[-1],device=x.device)
    for k in starts:
        prob=net(x[:,None,k:k+2048].contiguous()).softmax(dim=1);pred[:,:,k:k+2048]+=prob*taper;weight[k:k+2048]+=taper
    return pred/weight

@torch.inference_mode()
def validation(net,x,y,indices):
    net.eval();loss=torch.zeros((),device=x.device);n=0
    for ix in indices.split(16):
        yy=y[ix].long();p=taper_probabilities(net,x[ix]);loss+=F.nll_loss(p.clamp_min(1e-12).log(),yy,ignore_index=-100,reduction='sum');n+=(yy!=-100).sum()
    net.train();return (loss/n).item()

def run(head,seed,x,y,training,validation_rows,meta):
    key=f'{head}_{seed}';weight=MODELS/f'{key}.pt';completion=MODELS/f'{key}.json'
    if completion.is_file():
        ck=json.loads(completion.read_text());assert ck['steps']==STEPS and ck['weight_sha256']==digest(weight) and ck['protocol']==meta['protocol'];return
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);net=create(head).cuda();net.train()
    backbone=hashlib.sha256()
    for n,p in net.state_dict().items():
        if n.startswith(('enc.','dec.')):backbone.update(p.cpu().numpy().tobytes())
    rng=np.random.default_rng(27400+seed);draw=training[rng.integers(0,len(training),size=(STEPS,BATCH))];start=rng.integers(0,x.shape[1]-WIDTH+1,size=(STEPS,BATCH));draw_sha=hashlib.sha256(draw.tobytes()+start.tobytes()).hexdigest();draw=torch.as_tensor(draw,device=x.device);start=torch.as_tensor(start,device=x.device);time_indices=torch.arange(WIDTH,device=x.device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,fused=True);best=float('inf');best_step=None;log=[];beg=time.monotonic()
    for step in range(1,STEPS+1):
        idx=draw[step-1,:,None];win=start[step-1,:,None]+time_indices
        logits=net(x[idx,win].unsqueeze(1));loss=F.cross_entropy(logits,y[idx,win].long(),ignore_index=-100);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%400==0:
            v=validation(net,x,y,validation_rows)
            if not np.isfinite(v):raise ValueError('Nonfinite validation objective')
            if v<best:
                best,best_step=v,step;temp=weight.with_suffix('.part');torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},temp);os.replace(temp,weight)
            log.append(dict(step=step,validation_CE=v,elapsed_seconds=round(time.monotonic()-beg,2)));print(key,step,round(v,5),log[-1]['elapsed_seconds'],flush=True)
    result=dict(head=head,seed=seed,steps=STEPS,parameters=sum(p.numel() for p in net.parameters()),backbone_initial_sha256=backbone.hexdigest(),minibatch_sha256=draw_sha,best_step=best_step,best_validation_CE=best,checkpoints=log,weight_sha256=digest(weight),**meta)
    assert result['parameters']==({'KAN':309140,'GELU':309133}[head]);completion.write_text(json.dumps(result,indent=2),encoding='utf8')

def main():
    MODELS.mkdir(exist_ok=True);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('high');assert torch.cuda.is_available()
    audit=json.loads((TRAIN/'source_qc.json').read_text());assert digest(ROOT/'protocol.txt')==audit['protocol'] and digest(TRAIN/'rdb_split.csv')==audit['roster_hash']
    rows=list(csv.DictReader((TRAIN/'rdb_split.csv').open()));tr=np.array([i for i,r in enumerate(rows) if r['partition']=='train']);va=np.array([i for i,r in enumerate(rows) if r['partition']=='validation']);assert len(tr)+len(va)==2398
    x=torch.from_numpy(np.load(TRAIN/'rdb_x.npy')).cuda();y=torch.from_numpy(np.load(TRAIN/'rdb_y.npy')).cuda();assert x.shape==y.shape==(2398,2500)
    meta=dict(protocol=audit['protocol'],roster=audit['roster_hash'],train_code=digest(Path(__file__)),network_code=digest(ROOT/'model.py'),environment=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(),numpy=np.__version__),batch=BATCH,crop=WIDTH,lr=.001,train_n=len(tr),val_n=len(va))
    for seed in [1,2,3]:
        for head in ['KAN','GELU']:run(head,seed,x,y,tr,torch.as_tensor(va,device=x.device),meta)
        checks=[json.loads((MODELS/f'{head}_{seed}.json').read_text()) for head in ['KAN','GELU']]
        assert all(checks[0][field]==checks[1][field] for field in ('backbone_initial_sha256','minibatch_sha256'))
    print('All six RDB fits completed.',flush=True)

if __name__=='__main__':main()
