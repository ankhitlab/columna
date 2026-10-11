"""Independent references for the proposed Columna engineering patch: SciPy/statsmodels/NumPy.
Only generates committed fixtures; the app never imports Python.
"""
from pathlib import Path
import json,numpy as np,scipy,statsmodels
from scipy import stats,optimize
from scipy.cluster.hierarchy import linkage,fcluster
from statsmodels.tsa.stattools import acf,pacf_yw
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.inter_rater import fleiss_kappa,cohens_kappa
rng=np.random.default_rng(4206)
F={'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__}}
# Balanced ANOVA components, calculated independently by projections of cell means.
F['grr']=[]
for nested in [False,True]:
 for strong in [False,True]:
  p,o,nr=6,3,3
  A=rng.normal(size=(p,o,nr))*.08
  for j in range(p):
   for k in range(o):A[j,k,:]+=10+j*.7+k*.1+(j*k*.09 if strong else 0)+(rng.normal()*.7 if nested else 0)
  grand=A.mean();cell=A.mean(2);op=A.mean((0,2));part=A.mean((1,2))
  SSE=((A-cell[:,:,None])**2).sum();SSO=p*nr*((op-grand)**2).sum();dfe=p*o*(nr-1);mse=SSE/dfe;mso=SSO/(o-1)
  if nested:
   SSP=nr*((cell-op[None,:])**2).sum();dfp=o*(p-1);msp=SSP/dfp;sspo=0;msPO=0;varP=max(0,(msp-mse)/nr);varO=max(0,(mso-msp)/(p*nr));varPO=0
   expected=[['Operator',SSO,o-1,mso,mso/msp,stats.f.sf(mso/msp,o-1,dfp)],['Part (Operator)',SSP,dfp,msp,msp/mse,stats.f.sf(msp/mse,dfp,dfe)],['Repeatability',SSE,dfe,mse,None,None]]
  else:
   SSP=o*nr*((part-grand)**2).sum();dfp=p-1;msp=SSP/dfp;sspo=nr*((cell-part[:,None]-op[None,:]+grand)**2).sum();dfpo=(p-1)*(o-1);msPO=sspo/dfpo;varP=max(0,(msp-msPO)/(o*nr));varO=max(0,(mso-msPO)/(p*nr));varPO=max(0,(msPO-mse)/nr)
   expected=[['Part',SSP,dfp,msp,msp/msPO,stats.f.sf(msp/msPO,dfp,dfpo)],['Operator',SSO,o-1,mso,mso/msPO,stats.f.sf(mso/msPO,o-1,dfpo)],['Part × Operator',sspo,dfpo,msPO,msPO/mse,stats.f.sf(msPO/mse,dfpo,dfe)],['Repeatability',SSE,dfe,mse,None,None]]
  data={'y':[],'part':[],'operator':[]}
  for j in range(p):
   for k in range(o):
    for h in range(nr):data['y'].append(float(A[j,k,h]));data['part'].append(str(j));data['operator'].append(str(k))
  F['grr'].append({'data':data,'nested':nested,'anova':expected,'sigmaRepeat':np.sqrt(mse),'sigmaReprod':np.sqrt(varO+varPO),'sigmaGageRR':np.sqrt(mse+varO+varPO),'sigmaPart':np.sqrt(varP),'sigmaTotal':np.sqrt(mse+varO+varPO+varP)})
F['series']=[]
for period in [5,6,12]:
 n=96;tt=np.arange(1,n+1);y=30+.2*tt+3*np.sin(2*np.pi*tt/period)+rng.normal(size=n)*.8;lags=10
 r={'period':period,'y':y.tolist(),'acf':acf(y,nlags=lags,fft=False).tolist(),'pacf':pacf_yw(y,nlags=lags,method='mle').tolist(),'ljung':acorr_ljungbox(y,lags=[lags]).iloc[0].tolist(),'decomp':{}}
 for method in ['additive','multiplicative']:
  a=seasonal_decompose(y,model=method,period=period)
  r['decomp'][method]={'trend':a.trend.tolist(),'seasonal':a.seasonal.tolist(),'residual':a.resid.tolist()}
 # 8 point holdout, 4 future: independent linear / quadratic OLS forecasts
 train=y[:-8];t=np.arange(1,len(train)+1);future=np.arange(len(train)+1,n+5)
 r['trend']={}
 for degree in [1,2]:
  X=np.vander(t,degree+1,increasing=True);beta=np.linalg.lstsq(X,train,rcond=None)[0];yp=X@beta;fore=np.vander(future,degree+1,increasing=True)@beta
  r['trend'][str(degree)]={'coef':beta.tolist(),'forecast':fore.tolist(),'fitted':yp.tolist()}
 F['series'].append(r)
# MLE with right censoring; custom log-likelihood optimized independently.
F['reliability']=[]
for distribution in ['weibull','lognormal','exponential']:
 raw=(rng.weibull(2,60)*120 if distribution=='weibull' else np.exp(rng.normal(4,.6,60)) if distribution=='lognormal' else rng.exponential(100,60))
 cut=rng.uniform(40,180,60);times=np.minimum(raw,cut);c=(raw>cut).astype(int);fail=times[c==0];right=times[c==1]
 def objective(theta):
  if distribution=='weibull':a,b=np.exp(theta);dist=stats.weibull_min(a,scale=b)
  elif distribution=='lognormal':a,b=np.exp(theta);dist=stats.lognorm(a,scale=b)
  else:b=np.exp(theta[0]);dist=stats.expon(scale=b)
  return -(dist.logpdf(fail).sum()+dist.logsf(right).sum())
 ini=np.log([1,np.median(times)]) if distribution!='exponential' else [np.log(times.mean())]
 result=optimize.minimize(objective,ini,method='Nelder-Mead',options={'xatol':1e-11,'fatol':1e-10,'maxiter':10000});par=np.exp(result.x)
 if distribution=='weibull':dist=stats.weibull_min(par[0],scale=par[1])
 elif distribution=='lognormal':dist=stats.lognorm(par[0],scale=par[1])
 else:dist=stats.expon(scale=par[0])
 F['reliability'].append({'distribution':distribution,'times':times.tolist(),'censor':c.tolist(),'params':par.tolist(),'logLik':-result.fun,'quantiles':dist.ppf([.01,.05,.1,.5,.9]).tolist(),'R100':dist.sf(100)})
# Independent principal eigensystems / hierarchy (well-separated, no distance ties).
F['multi']=[]
for scale in [False,True]:
 X=rng.normal(size=(36,4));X[:,1]+=.8*X[:,0];X[:,2]+=2*X[:,0];X[:,3]*=8
 Z=(X-X.mean(0))/(X.std(0,ddof=1) if scale else 1);cov=Z.T@Z/(len(X)-1);eig,V=np.linalg.eigh(cov);eig=eig[::-1];V=V[:,::-1]
 trees={m:{'heights':linkage(Z,method=m)[:,2].tolist(),'membership':fcluster(linkage(Z,method=m),3,criterion='maxclust').tolist()} for m in ['single','complete','average','ward']}
 F['multi'].append({'X':X.tolist(),'scale':scale,'eigenvalues':eig.tolist(),'loadings':V.tolist(),'trees':trees})
F['kappa']=[]
for raters in [2,3,5]:
 R=rng.integers(0,3,size=(40,raters));counts=np.array([np.bincount(row,minlength=3) for row in R]);cont=np.zeros((3,3),dtype=int)
 if raters==2:
  for a,b in R:cont[a,b]+=1
 k=cohens_kappa(cont).kappa if raters==2 else fleiss_kappa(counts)
 F['kappa'].append({'ratings':R.tolist(),'kappa':k})
# Three-level quadratic DOE fit / lack-of-fit.
X=[];y=[]
for a in [-1,0,1]:
 for b in [-1,0,1]:
  for rep in range(3):X.append([1,a,b,a*b,a*a,b*b]);y.append(10+2*a-3*b+.5*a*a+a*b+np.sin(rep+2*a+b)*.1)
X=np.array(X);y=np.array(y);beta=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@beta;SSE=res@res;df=len(y)-X.shape[1];SE=np.sqrt(np.diag(np.linalg.inv(X.T@X))*SSE/df)
F['doe']={'a':X[:,1].tolist(),'b':X[:,2].tolist(),'y':y.tolist(),'beta':beta.tolist(),'se':SE.tolist(),'p':(2*stats.t.sf(abs(beta/SE),df)).tolist(),'SSE':SSE,'df':df,'pureError':sum(((g-g.mean())**2).sum() for g in y.reshape(-1,3))}
# Weighted log-rank score/covariance, independently accumulated across risk sets.
F['logrank']=[]
for k in [2,3]:
 t=np.array([2,4,4,6,8,10,13,13,15,18,20,24]*k,dtype=float);g=np.repeat(np.arange(k),12);t+=g*2;c=np.array([0,0,1,0,0,1,0,0,1,0,0,1]*k)
 for method in ['logrank','wilcoxon','tarone-ware']:
  O=np.zeros(k);E=np.zeros(k);V=np.zeros((k,k))
  for tm in np.unique(t[c==0]):
   risk=np.array([np.sum((t>=tm)&(g==j)) for j in range(k)],dtype=float);d=np.array([np.sum((t==tm)&(g==j)&(c==0)) for j in range(k)],dtype=float);N=risk.sum();D=d.sum();weight=1 if method=='logrank' else N if method=='wilcoxon' else np.sqrt(N)
   O+=weight*d;E+=weight*risk/N*D
   if N>1:V+=weight**2*D*(N-D)/(N*(N-1))*(np.diag(risk)-np.outer(risk,risk)/N)
  diff=(O-E)[:-1];stat=diff@np.linalg.solve(V[:-1,:-1],diff)
  F['logrank'].append({'times':t.tolist(),'groups':g.tolist(),'censor':c.tolist(),'method':method,'statistic':stat,'pValue':stats.chi2.sf(stat,k-1),'observed':O.tolist(),'expected':E.tolist(),'nEvents':int(sum(c==0))})
def sanitize(v):
 if isinstance(v,dict):return {k:sanitize(x) for k,x in v.items()}
 if isinstance(v,list):return [sanitize(x) for x in v]
 if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
 if isinstance(v,np.integer):return int(v)
 return v
out=Path(__file__).with_name('engineering-fixtures.json');out.write_text(json.dumps(sanitize(F),indent=2,ensure_ascii=False));print(out,len(out.read_bytes()))
