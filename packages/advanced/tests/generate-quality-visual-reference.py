"""Independent numerical reference: SciPy distributions and statsmodels REML.
Run manually with Python NumPy/SciPy/statsmodels; no runtime dependency.
"""
from pathlib import Path
import json, warnings
import numpy as np
from scipy import stats, linalg, optimize
import pandas as pd
import statsmodels.api as sm
ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(72491)
sets={"weibull":rng.weibull(2.3,110)*4.5,"gamma":rng.gamma(2.7,1.1,105),"lognormal":rng.lognormal(1,.5,100),"exponential":rng.exponential(2,105),"concentrated":10+rng.normal(0,.025,96)}
refs=[]
for name,x in sets.items():
    lsl=float(np.percentile(x,5)*.8);usl=float(np.percentile(x,95)*1.1)
    fits={}
    for dist in ['normal','weibull','gamma','lognormal','exponential']:
        if dist=='normal':
            mu,sd=stats.norm.fit(x);d=stats.norm(mu,sd);p={'mu':mu,'sigma':sd}
        elif dist=='weibull':
            k,_,scale=stats.weibull_min.fit(x,floc=0);d=stats.weibull_min(k,scale=scale);p={'shape':k,'scale':scale}
        elif dist=='gamma':
            k,_,scale=stats.gamma.fit(x,floc=0);d=stats.gamma(k,scale=scale);p={'shape':k,'scale':scale}
        elif dist=='lognormal':
            s,_,scale=stats.lognorm.fit(x,floc=0);d=stats.lognorm(s,scale=scale);p={'mu':np.log(scale),'sigma':s}
        else:
            _,scale=stats.expon.fit(x,floc=0);d=stats.expon(scale=scale);p={'scale':scale}
        ll=float(d.logpdf(x).sum());q=d.ppf([.00135,.5,.99865]);pl=float(stats.norm.isf(d.cdf(lsl))/3);pu=float(stats.norm.isf(d.sf(usl))/3)
        fits[dist]={'parameters':p,'logLik':ll,'aic':2*len(p)-2*ll,'quantiles':list(q),'Pp':(pl+pu)/2,'Ppk':min(pl,pu),'ppm':float((d.cdf(lsl)+d.sf(usl))*1e6),'isoPp':float((usl-lsl)/(q[2]-q[0])),'isoPpk':float(min((q[1]-lsl)/(q[1]-q[0]),(usl-q[1])/(q[2]-q[1])))}
    refs.append({'name':name,'x':x.tolist(),'lsl':lsl,'usl':usl,'fits':fits})
reml=[]
# Strong effects avoid unstable boundary fixtures; intentionally unequal replicate counts.
for name in ['crossed-interaction','extra-random','fixed-operator','nested']:
    data=[];parts=rng.normal(0,2.0,7);ops=np.array([-1.5,-.1,1.2]);tools=np.array([-.8,.6]);interaction=rng.normal(0,.6,(7,3));npv=rng.normal(0,2,(7,3))
    for p in range(7):
        for op in range(3):
            for tool in range(2 if name=='extra-random' else 1):
                for rep in range(2+(p+op+tool)%3):
                    y=20+(npv[p,op] if name=='nested' else parts[p])+ops[op]+(tools[tool] if name=='extra-random' else 0)+(interaction[p,op] if name=='crossed-interaction' else 0)+rng.normal(0,.28)
                    data.append({'y':float(y),'part':str(p),'operator':str(op),'tool':str(tool),'nested':str(op)+'/'+str(p),'pair':str(p)+'/'+str(op),'group':1})
    df=pd.DataFrame(data);vc={}
    vc['part']='0+C(nested)' if name=='nested' else '0+C(part)'
    if name!='fixed-operator':vc['operator']='0+C(operator)'
    if name=='extra-random':vc['tool']='0+C(tool)'
    if name=='crossed-interaction':vc['pair']='0+C(pair)'
    formula='y~C(operator)' if name=='fixed-operator' else 'y~1'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');fit=sm.MixedLM.from_formula(formula,groups='group',vc_formula=vc,data=df).fit(reml=True,method='lbfgs',maxiter=3000,gtol=1e-10,ftol=1e-12)
    if not fit.converged:raise RuntimeError('Independent REML fixture not converged')
    # Independent dense V = I + sum(lambda_j Z_j Z_j') verification.
    X=np.asarray(fit.model.exog);Y=df.y.to_numpy();n=len(Y);p=X.shape[1];Zs=[np.asarray(m[0]) for m in fit.model.exog_vc.mats];Ks=[z@z.T for z in Zs]
    def dense_profile(log_ratios,detail=False):
        V=np.eye(n)
        for a,K in zip(np.exp(log_ratios),Ks):V+=a*K
        L=linalg.cho_factor(V,lower=True);viX=linalg.cho_solve(L,X);viY=linalg.cho_solve(L,Y);G=X.T@viX;beta=linalg.solve(G,X.T@viY);res=Y-X@beta;variance=float(res@linalg.cho_solve(L,res)/(n-p));obj=2*np.log(np.diag(L[0])).sum()+np.linalg.slogdet(G)[1]+(n-p)*np.log(variance)
        return (variance,np.exp(log_ratios)*variance,beta,float(-.5*(obj+(n-p)*(np.log(2*np.pi)+1)))) if detail else obj
    dense=optimize.minimize(dense_profile,np.log(fit.vcomp/fit.scale),method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-7,'fatol':1e-10})
    dv,dc,db,dll=dense_profile(dense.x,True)
    reml.append({'dense':{'residualVariance':dv,'components':dict(zip(sorted(vc),map(float,dc))),'fixed':db.tolist(),'logLik':dll,'converged':bool(dense.success)},'name':name,'data':data,'components':dict(zip(sorted(vc),map(float,fit.vcomp))),'residualVariance':float(fit.scale),'logLik':float(fit.llf),'fixed':list(map(float,fit.fe_params)),'conditionalFitted':list(map(float,fit.fittedvalues))})
def finite_json(x):
    if isinstance(x,dict):return {k:finite_json(v) for k,v in x.items()}
    if isinstance(x,list):return [finite_json(v) for v in x]
    if isinstance(x,(float,np.floating)) and not np.isfinite(x):return str(x)
    return x
(ROOT/'quality-visual-reference.json').write_text(json.dumps(finite_json({'distributions':refs,'reml':reml,'reference':'SciPy '+__import__('scipy').__version__+'; statsmodels '+sm.__version__}),ensure_ascii=False,indent=2))
print('Generated',len(refs),'distribution cases x 5 fits and',len(reml),'independent REML cases')
