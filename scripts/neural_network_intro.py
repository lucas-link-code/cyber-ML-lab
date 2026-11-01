"""Simple NumPy neural network: XOR, one hidden tanh layer, manual backpropagation."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lab_common import ROOT,out_name
import json


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-30,30)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs',type=int,default=3000)
    ap.add_argument('--rate',type=float,default=.4)
    ap.add_argument('--name',default='nn_xor')
    args=ap.parse_args()
    if args.epochs<1 or args.rate<=0:ap.error('epochs and rate must be positive')
    rng=np.random.default_rng(42)
    x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y=np.array([[0.],[1.],[1.],[0.]])
    w1=rng.normal(0,.7,(2,6));b1=np.zeros((1,6))
    w2=rng.normal(0,.7,(6,1));b2=np.zeros((1,1))
    history=[]
    for epoch in range(args.epochs):
        z1=x@w1+b1;h=np.tanh(z1)
        z2=h@w2+b2;p=sigmoid(z2)
        eps=1e-9
        loss=float(np.mean(-y*np.log(p+eps)-(1-y)*np.log(1-p+eps)))
        history.append(loss)
        # For sigmoid + binary cross-entropy: derivative wrt output logits = p - y.
        dz2=(p-y)/len(x);dw2=h.T@dz2;db2=dz2.sum(axis=0,keepdims=True)
        dz1=(dz2@w2.T)*(1-h**2);dw1=x.T@dz1;db1=dz1.sum(axis=0,keepdims=True)
        w2-=args.rate*dw2;b2-=args.rate*db2
        w1-=args.rate*dw1;b1-=args.rate*db1
        if epoch in (0,9,99,499,args.epochs-1):print(f'epoch={epoch+1:5d} loss={loss:.5f}')
    predicted=sigmoid(np.tanh(x@w1+b1)@w2+b2)
    print('XOR INPUT -> predicted P(label=1)')
    for row,prob in zip(x,predicted):print(f'  {row.astype(int).tolist()} -> {prob[0]:.4f}')
    name=out_name(args.name)
    (ROOT/'outputs'/f'{name}_neural_metrics.json').write_text(json.dumps({'epochs':args.epochs,'rate':args.rate,'initial_loss':history[0],'final_loss':history[-1],'predictions':[float(a[0]) for a in predicted]},indent=2))
    fig,ax=plt.subplots(figsize=(6,4));ax.plot(np.arange(len(history))+1,history)
    ax.set(xlabel='Epoch',ylabel='Binary cross-entropy loss',title='Learning via backpropagation (toy XOR)')
    fig.tight_layout();fig.savefig(ROOT/'charts'/f'{name}_neural_network_loss.png',dpi=130);plt.close(fig)
    print(f'Saved: charts/{name}_neural_network_loss.png; outputs/{name}_neural_metrics.json')
    print('This is a learning demo, not malware detection.')

if __name__=='__main__':main()
