"""Independent numeric fixtures: SciPy and statsmodels; no browser/runtime dependency."""
from pathlib import Path
import json, math
import numpy as np
import scipy, statsmodels
from scipy import stats as st
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
R={'versions':{'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},'contingency':[],'binomial':[],'twoprop':[],'variance':[],'groups':[],'power':[]}
for tab in [[[12,5],[3,10]],[[1,9],[11,3]],[[0,5],[8,0]],[[50,20,5],[22,41,19],[8,12,33]],[[12000,3200],[14500,9000]],[[10,0,20],[5,1,14]]]:
 for correction in ([False,True] if len(tab)==2 and len(tab[0])==2 else [False]):
  q,p,df,e=st.chi2_contingency(tab,correction=correction);r={'table':tab,'correction':correction,'statistic':q,'p':p,'df':df,'expected':e.tolist()}
  if len(tab)==2 and len(tab[0])==2:r['fisher']={a:st.fisher_exact(tab,alternative=a).pvalue for a in ['two-sided','less','greater']}
  R['contingency'].append(r)
for n,k,p0 in [(10,0,.3),(10,10,.3),(30,12,.5),(1000,45,.05),(1000000,0,.01),(500000,200000,.4),(12,4,.123),(4,2,.5),(1,0,.5),(1,1,.5),(1000,998,.95)]:
 for a in ['two-sided','less','greater']:
  b=st.binomtest(k,n,p0,alternative=a);ci=b.proportion_ci(.95,method='exact')
  z=st.norm.isf(.025 if a=='two-sided' else .05);ph=k/n;den=1+z*z/n;c=(ph+z*z/(2*n))/den;h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
  R['binomial'].append({'n':n,'events':k,'p0':p0,'alternative':a,'p':b.pvalue,'ci':[ci.low,ci.high],'wilson':[0 if a=='less' else c-h,1 if a=='greater' else c+h]})
for a,b in [((20,100),(12,100)),((0,30),(0,40)),((10,10),(30,30)),((4,21),(7,29)),((11,12),(2,10))]:
 ci=confint_proportions_2indep(a[0],a[1],b[0],b[1],method='newcomb',alpha=.05,compare='diff')
 R['twoprop'].append({'a':a,'b':b,'ci':ci,'fisher':st.fisher_exact([[a[0],a[1]-a[0]],[b[0],b[1]-b[0]]]).pvalue})
for n,s,n2,s2 in [(12,2,17,1.5),(3,.03,7,.021),(55,12.3,34,10.1)]:
 for a in ['two-sided','less','greater']:
  for rho in [.7,1,1.8]:
   ratio=(s/s2)**2;q=ratio/rho**2;cdf=st.f.cdf(q,n-1,n2-1);sf=st.f.sf(q,n-1,n2-1);p=cdf if a=='less' else sf if a=='greater' else min(1,2*min(cdf,sf))
   ci=[0,ratio/st.f.ppf(.05,n-1,n2-1)] if a=='less' else [ratio/st.f.isf(.05,n-1,n2-1),None] if a=='greater' else [ratio/st.f.isf(.025,n-1,n2-1),ratio/st.f.ppf(.025,n-1,n2-1)]
   R['variance'].append({'samples':[{'n':n,'sd':s},{'n':n2,'sd':s2}],'ratio0':rho,'alternative':a,'statistic':q,'p':p,'ci':ci})
rng=np.random.default_rng(180)
for j in range(5):
 gs=[rng.normal(loc=i,scale=.8+i*.4,size=15+3*i).tolist() for i in range(2+j%3)]
 R['groups'].append({'data':gs,'bartlett':list(st.bartlett(*gs)),'levene':list(st.levene(*gs,center='mean')),'brown-forsythe':list(st.levene(*gs,center='median'))})
def refpower(o):
 test=o['test'];n=o['n'];d=o['effect'];sigma=o.get('sigma',1);alpha=o.get('alpha',.05);a=o.get('alternative','two-sided');n2=math.ceil(n*o.get('allocation',1));z=st.norm.isf(alpha/(2 if a=='two-sided' else 1))
 if test in ['1-sample t','paired t','2-sample t']:
  df=n+n2-2 if test=='2-sample t' else n-1;delta=d/sigma/(1/n+1/n2)**.5 if test=='2-sample t' else d/sigma*n**.5;c=st.t.isf(alpha/(2 if a=='two-sided' else 1),df)
  return st.nct.sf(c,df,delta) if a=='greater' else st.nct.cdf(-c,df,delta) if a=='less' else st.nct.sf(c,df,delta)+st.nct.cdf(-c,df,delta)
 if test=='1-sample z':shift=d/sigma*n**.5;scale=1
 if test=='1 proportion':p0=o['p0'];se0=(p0*(1-p0)/n)**.5;se1=(d*(1-d)/n)**.5;shift=(d-p0)/se1;scale=se0/se1
 if test=='2 proportions':p1=o['p1'];pbar=(n*p1+n2*d)/(n+n2);se0=(pbar*(1-pbar)*(1/n+1/n2))**.5;se1=(p1*(1-p1)/n+d*(1-d)/n2)**.5;shift=(p1-d)/se1;scale=se0/se1
 if test in ['1-sample z','1 proportion','2 proportions']:
  return st.norm.sf(z*scale-shift) if a=='greater' else st.norm.cdf(-z*scale-shift) if a=='less' else st.norm.sf(z*scale-shift)+st.norm.cdf(-z*scale-shift)
 if test=='one-way anova':k=o['groups'];return st.ncf.sf(st.f.isf(alpha,k-1,k*(n-1)),k-1,k*(n-1),n*d*d/(2*sigma*sigma))
 if test=='chi-square':return st.ncx2.sf(st.chi2.isf(alpha,o['df']),o['df'],n*d*d)
 D=st.chi2(n-1) if test=='1 variance' else st.f(n-1,n2-1)
 return D.sf(D.isf(alpha)/d**2) if a=='greater' else D.cdf(D.ppf(alpha)/d**2) if a=='less' else D.sf(D.isf(alpha/2)/d**2)+D.cdf(D.ppf(alpha/2)/d**2)
for test in ['1-sample z','1-sample t','2-sample t','paired t','1 proportion','2 proportions','one-way anova','1 variance','2 variances','chi-square']:
 for n in [3,12,35,250]:
  for a in (['two-sided'] if test in ['one-way anova','chi-square'] else ['two-sided','less','greater']):
   effect=.32 if test in ['1 proportion','2 proportions'] else 1.4 if 'variance' in test else .5
   o={'test':test,'n':n,'effect':effect,'alpha':.05,'alternative':a,'sigma':1.3,'p0':.5,'p1':.5,'allocation':1.7,'groups':4,'df':3}
   R['power'].append({'options':o,'power':refpower(o)})
path=Path(__file__).with_name('inference-fixtures.json');path.write_text(json.dumps(R,indent=2,ensure_ascii=False));print({k:len(v) for k,v in R.items()})
