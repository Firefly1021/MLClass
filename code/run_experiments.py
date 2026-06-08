import os, math, json, random, csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG = os.path.join(ROOT, 'figures')
DATA = os.path.join(ROOT, 'data')
os.makedirs(FIG, exist_ok=True); os.makedirs(DATA, exist_ok=True)
SEED = 20260608
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_default_dtype(torch.float32)
try: torch.set_num_threads(4)
except Exception: pass

class MLP(nn.Module):
    def __init__(self, in_dim=1, width=32, depth=3, out_dim=1):
        super().__init__(); layers=[]; dim=in_dim
        for _ in range(depth):
            lin=nn.Linear(dim,width); nn.init.xavier_normal_(lin.weight); nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]; dim=width
        out=nn.Linear(dim,out_dim); nn.init.xavier_normal_(out.weight); nn.init.zeros_(out.bias)
        layers.append(out); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class FourierMLP(nn.Module):
    def __init__(self, freqs, width=32, depth=2):
        super().__init__(); self.register_buffer('freqs', torch.tensor(freqs, dtype=torch.float32).reshape(1,-1))
        self.mlp=MLP(in_dim=2*len(freqs), width=width, depth=depth, out_dim=1)
    def forward(self,x):
        z=x@self.freqs
        return self.mlp(torch.cat([torch.sin(z), torch.cos(z)], dim=1))

def rel_l2(y,yt): return torch.sqrt(torch.mean((y-yt)**2)/torch.mean(yt**2)).item()

def train_fit(k, kind, epochs=500, lr=2e-3):
    x=torch.linspace(0,1,120).reshape(-1,1); y=torch.sin(k*x)
    model=MLP() if kind=='mlp' else FourierMLP([1,2,4,8,16,32,k])
    opt=torch.optim.Adam(model.parameters(), lr=lr); hist=[]
    for ep in range(epochs+1):
        opt.zero_grad(); pred=model(x); loss=torch.mean((pred-y)**2); loss.backward(); opt.step()
        if ep%25==0: hist.append((ep, loss.item(), rel_l2(model(x).detach(), y)))
    return model, np.array(hist)

def trial(model,x,k): return x*math.sin(k)+x*(1-x)*model(x)

def train_pinn(k, kind, epochs=350, lr=1e-3):
    x=torch.linspace(0,1,60).reshape(-1,1).requires_grad_(True)
    model=MLP(width=24,depth=2) if kind=='mlp' else FourierMLP([1,2,4,8,16,32,k], width=24, depth=1)
    opt=torch.optim.Adam(model.parameters(), lr=lr); hist=[]
    x_eval=torch.linspace(0,1,300).reshape(-1,1); y_true=torch.sin(k*x_eval)
    for ep in range(epochs+1):
        opt.zero_grad(); u=trial(model,x,k)
        du=torch.autograd.grad(u,x,torch.ones_like(u),create_graph=True)[0]
        d2u=torch.autograd.grad(du,x,torch.ones_like(du),create_graph=True)[0]
        r=-d2u-(k**2)*u; loss=torch.mean(r**2); loss.backward(); opt.step()
        if ep%25==0:
            with torch.no_grad(): hist.append((ep, loss.item(), rel_l2(trial(model,x_eval,k), y_true)))
    return model, np.array(hist)

# Clean previous generated data/figures
for d in [FIG, DATA]:
    for fn in os.listdir(d):
        if fn.endswith(('.csv','.json','.pdf','.png')): os.remove(os.path.join(d,fn))

fit_results={}
for k in [5.0,40.0]:
    for kind in ['mlp','fourier']:
        m,h=train_fit(k,kind); fit_results[(k,kind)]=(m,h)
        np.savetxt(os.path.join(DATA,f'fit_k{k:g}_{kind}.csv'), h, delimiter=',', header='epoch,mse,rel_l2', comments='')
plt.figure(figsize=(6.3,4.2))
for (k,kind),(_,h) in fit_results.items():
    plt.semilogy(h[:,0],h[:,2],label=('MLP' if kind=='mlp' else 'Fourier-MLP')+f', k={k:g}')
plt.xlabel('Epoch'); plt.ylabel('Relative L2 error'); plt.legend(fontsize=8); plt.grid(True,which='both',alpha=.35); plt.tight_layout()
plt.savefig(os.path.join(FIG,'fig_spectral_bias_training.pdf')); plt.savefig(os.path.join(FIG,'fig_spectral_bias_training.png'),dpi=220); plt.close()

x_plot=torch.linspace(0,1,800).reshape(-1,1)
for k in [5.0,40.0]:
    plt.figure(figsize=(6.3,3.8)); plt.plot(x_plot.numpy().ravel(), torch.sin(k*x_plot).numpy().ravel(), label='Exact')
    for kind in ['mlp','fourier']:
        plt.plot(x_plot.numpy().ravel(), fit_results[(k,kind)][0](x_plot).detach().numpy().ravel(), '--', label='MLP' if kind=='mlp' else 'Fourier-MLP')
    plt.xlabel('x'); plt.ylabel('u(x)'); plt.title(f'Function fitting: sin({k:g} x)'); plt.legend(fontsize=8); plt.grid(True,alpha=.35); plt.tight_layout()
    plt.savefig(os.path.join(FIG,f'fig_fit_k{k:g}.pdf')); plt.savefig(os.path.join(FIG,f'fig_fit_k{k:g}.png'),dpi=220); plt.close()

pinn_table=[]; pinn_models={}
for k in [5.0,10.0,20.0,40.0]:
    for kind in ['mlp','fourier']:
        m,h=train_pinn(k,kind); pinn_models[(k,kind)]=m
        np.savetxt(os.path.join(DATA,f'pinn_k{k:g}_{kind}.csv'), h, delimiter=',', header='epoch,residual_mse,rel_l2', comments='')
        pinn_table.append({'k':k,'model':kind,'residual_mse':float(h[-1,1]),'rel_l2':float(h[-1,2])})
with open(os.path.join(DATA,'pinn_results.json'),'w',encoding='utf-8') as f: json.dump(pinn_table,f,indent=2)
with open(os.path.join(DATA,'summary_table.csv'),'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['k','model','final_residual_mse','final_relative_l2_error'])
    for r in pinn_table: w.writerow([r['k'],r['model'],r['residual_mse'],r['rel_l2']])
plt.figure(figsize=(6.3,4.2))
for k in [5.0,20.0,40.0]:
    for kind in ['mlp','fourier']:
        h=np.loadtxt(os.path.join(DATA,f'pinn_k{k:g}_{kind}.csv'),delimiter=',',skiprows=1)
        plt.semilogy(h[:,0],h[:,2],label=('MLP' if kind=='mlp' else 'Fourier-MLP')+f', k={k:g}')
plt.xlabel('Epoch'); plt.ylabel('Relative L2 error'); plt.legend(fontsize=7); plt.grid(True,which='both',alpha=.35); plt.tight_layout()
plt.savefig(os.path.join(FIG,'fig_pinn_training.pdf')); plt.savefig(os.path.join(FIG,'fig_pinn_training.png'),dpi=220); plt.close()
for k in [10.0,40.0]:
    plt.figure(figsize=(6.3,3.8)); plt.plot(x_plot.numpy().ravel(), torch.sin(k*x_plot).numpy().ravel(), label='Exact')
    for kind in ['mlp','fourier']:
        plt.plot(x_plot.numpy().ravel(), trial(pinn_models[(k,kind)],x_plot,k).detach().numpy().ravel(), '--', label='PINN-MLP' if kind=='mlp' else 'PINN-Fourier')
    plt.xlabel('x'); plt.ylabel('u(x)'); plt.title(f'Helmholtz PINN: k={k:g}'); plt.legend(fontsize=8); plt.grid(True,alpha=.35); plt.tight_layout()
    plt.savefig(os.path.join(FIG,f'fig_pinn_solution_k{k:g}.pdf')); plt.savefig(os.path.join(FIG,f'fig_pinn_solution_k{k:g}.png'),dpi=220); plt.close()
print('Done')
for r in pinn_table: print(r)
