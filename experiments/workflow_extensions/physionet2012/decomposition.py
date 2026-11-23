"""Signed log-task contributions for a multiplicative AST, frozen coefficients only."""
import sys
from .study import (OUT,DATA,Context,SEED,references,data,prep,sim,PERIOD,segments,serve,
                    BUDGETS,np,json,MONOMIALS,write)
def run(t):
    frozen=json.loads((OUT/(t+'_frozen.json')).read_text())
    split=json.loads((DATA/'split.json').read_text())
    test=[r for r in data.load(DATA,[t]) if split[r['subject_id']]=='test']
    ctx=Context(t,references(t,t=='BUN_CR'),SEED);prep(ctx,test)
    powers=np.asarray(MONOMIALS[t][ctx.work.target][1])
    median=np.array(frozen['median'])
    out=[]
    for r in test:
        logtruth=np.log(data.replay(r))
        for B in BUDGETS:
            mod={(bool(int(k.split('/')[0])),k.split('/')[1]):np.array(v) for k,v in frozen['models'][f'{B}/period'].items()}
            ps=sim(ctx,r,B,PERIOD,4,True);parts=list(segments(ctx,ps))
            for name in ('joint','root_loss','task'):
                c=serve(ctx,ps,mod,median,'task' if name=='joint' else name,0 if name=='joint' else frozen['strength']['period'],parts)
                terms=(np.log(ctx.work.denormalize(c))-logtruth)*powers
                individual=abs(terms).sum(-1).mean()
                net=abs(terms.sum(-1)).mean()
                out.append(dict(subject=r['subject_id'],B=B,method=name,individual=individual,net=net))
    write(OUT/(t+'_signed.csv'),out)
if __name__=='__main__':run(sys.argv[1])
