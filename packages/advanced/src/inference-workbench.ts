/**
 * Proposed Columna API: categorical inference, binomial proportions, variance inference.
 * Domain validation and interval metadata are part of the statistical library, not the UI.
 * Pearson/Yates chi-square, Fisher 2x2, Clopper–Pearson/Wilson/Newcombe,
 * normal-theory chi-square/F variance inference, Bonett (one sample), Levene/Bartlett.
 */
import {chi2test, chi2gof, levene, bartlett, cleanNumbers, type Alternative} from './tests.js'
import {propTest1, propTest2, varTest1} from './basic.js'
import {normal, chi2, f, lgamma} from './dist.js'
import {descriptiveStats, type DescriptiveStats} from './descriptive.js'

export interface BinomialSample { events: number; trials: number; name?: string }
export interface VarianceSample { n: number; sd: number; name?: string }
type CI = [number, number]
const STD=normal()
const clamp=(x:number)=>Math.max(0,Math.min(1,x))
function conf(c=.95){if(!(c>0&&c<1))throw new RangeError('confidence must be in (0, 1)');return c}
function alt(a:Alternative='two-sided'){if(!['two-sided','less','greater'].includes(a))throw new RangeError('unknown alternative');return a}
function count(n:number,label:string,positive=false){if(!Number.isSafeInteger(n)||n<(positive?1:0))throw new RangeError(label+' must be a '+(positive?'positive':'non-negative')+' safe integer')}
function sum(v:number[]){return v.reduce((a,b)=>a+b,0)}
function total(v:number[]){const n=sum(v);count(n,'total count',true);return n}
function validSample(s:BinomialSample){count(s.trials,'trials',true);count(s.events,'events');if(s.events>s.trials)throw new RangeError('events cannot exceed trials')}
function stats(v:ArrayLike<number|null|undefined>){return descriptiveStats(Array.from(v)) as DescriptiveStats}

/** Exact 2x2 Fisher test via a modal hypergeometric recurrence, avoiding PMF underflow at the observed tail.
 * Both tails use the probability-order definition for two-sided p, not twice the smaller tail.
 * Counts are literal frequencies, not survey weights. The sample odds ratio is NOT a conditional MLE.
 */
export function fisherExact2x2(table:number[][], alternative:Alternative='two-sided'){
  alt(alternative)
  if(table.length!==2||table.some(r=>r.length!==2))throw new RangeError('Fisher requires a 2 x 2 table')
  table.flat().forEach(x=>count(x,'cell'))
  const [[a,b],[c,d]]=table as [[number,number],[number,number]]
  const N=total([a,b,c,d]), K=a+c, m=a+b, low=Math.max(0,m-(N-K)),high=Math.min(K,m)
  const oddsRatio=b*c>0?a*d/(b*c):a*d>0?Infinity:undefined
  if(low===high)return {test:'Fisher exact',pValue:1,oddsRatio,alternative}
  const mode=Math.max(low,Math.min(high,Math.floor((m+1)*(K+1)/(N+2))))
  const logchoose=(n:number,k:number)=>lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)
  const logobs=logchoose(K,a)+logchoose(N-K,m-a)
  const logmode=logchoose(K,mode)+logchoose(N-K,m-mode)
  const logthreshold=logobs-logmode+Math.log1p(1e-7)
  let mass=0, selected=0
  const include=(k:number,weight:number)=>{
    mass+=weight
    if(alternative==='less'?k<=a:alternative==='greater'?k>=a:Math.log(weight)<=logthreshold)selected+=weight
  }
  include(mode,1)
  let w=1
  for(let k=mode;k>low;k--){w*=k*(N-K-m+k)/((K-k+1)*(m-k+1));include(k-1,w);if(w===0)break}
  w=1
  for(let k=mode;k<high;k++){w*=(K-k)*(m-k)/((k+1)*(N-K-m+k+1));include(k+1,w);if(w===0)break}
  return {test:'Fisher exact',pValue:clamp(selected/mass),oddsRatio,alternative}
}

export function contingencyAnalysis(observed:number[][], options:{correction?:boolean; fisher?:boolean; alternative?:Alternative}={}){
  const rows=observed.length,cols=observed[0]?.length??0
  if(rows<2||cols<2||observed.some(r=>r.length!==cols))throw new RangeError('independence needs a rectangular table, at least 2 x 2')
  observed.flat().forEach(x=>count(x,'cell'))
  const n=total(observed.flat()), rowTotals=observed.map(sum),colTotals=Array.from({length:cols},(_,j)=>sum(observed.map(r=>r[j]!)))
  if([...rowTotals,...colTotals].some(x=>x===0))throw new RangeError('zero marginal total: remove empty categories before testing')
  if(options.correction&&(rows!==2||cols!==2))throw new RangeError('Yates correction is available only for 2 x 2')
  if(options.fisher&&(rows!==2||cols!==2))throw new RangeError('Fisher is implemented for 2 x 2 only')
  const test=chi2test(observed,{correction:options.correction}), raw=options.correction?chi2test(observed):test
  const expected=test.expected as number[][]
  const cells=observed.flatMap((row,i)=>row.map((o,j)=>{
    const e=expected[i]![j]!,delta=o-e,adj=Math.max(0,Math.abs(delta)-(options.correction? .5:0))
    return {row:i,column:j,observed:o,expected:e,contribution:adj*adj/e,pearsonResidual:delta/Math.sqrt(e),adjustedResidual:delta/Math.sqrt(e*(1-rowTotals[i]!/n)*(1-colTotals[j]!/n)),rowPercent:100*o/rowTotals[i]!,columnPercent:100*o/colTotals[j]!,totalPercent:100*o/n}
  }))
  const lowExpected=cells.filter(c=>c.expected<5).length,veryLow=cells.filter(c=>c.expected<1).length
  return {...test,n,observed,expected,rowTotals,colTotals,cells,cramersV:Math.sqrt(raw.statistic/(n*Math.min(rows-1,cols-1))),lowExpectedCells:lowExpected,veryLowExpectedCells:veryLow,
    fisher:options.fisher?fisherExact2x2(observed,options.alternative):undefined,
    warnings:[...(lowExpected?[`${lowExpected} expected cells are below 5 (${veryLow} below 1); the chi-square approximation may be unreliable. Use Fisher for a 2 x 2 table.`]:[])]}
}

export function goodnessOfFitAnalysis(observed:number[], expectedInput?:number[],options:{expectedMode?:'probabilities'|'counts';ddof?:number}={}){
  if(observed.length<2)throw new RangeError('goodness of fit needs at least two categories')
  observed.forEach(x=>count(x,'observed count'));const n=total(observed),k=observed.length,ddof=options.ddof??0
  count(ddof,'ddof');if(k-1-ddof<1)throw new RangeError('degrees of freedom must remain >= 1')
  let probabilities:number[]
  if(expectedInput){
    if(expectedInput.length!==k||expectedInput.some(v=>!Number.isFinite(v)||v<=0))throw new RangeError('positive finite expected values for every category are required')
    const norm=options.expectedMode==='counts'?n:1
    if(Math.abs(sum(expectedInput)-norm)>1e-8*Math.max(1,norm))throw new RangeError(options.expectedMode==='counts'?'expected counts must sum to observed total':'expected probabilities must sum to 1')
    probabilities=expectedInput.map(v=>v/norm)
  }else probabilities=observed.map(()=>1/k)
  const r=chi2gof(observed,probabilities,{ddof}),expected=r.expected as number[]
  return {...r,n,observed,expected,probabilities,ddof,cells:observed.map((o,i)=>({observed:o,expected:expected[i]!,contribution:(o-expected[i]!)**2/expected[i]!,pearsonResidual:(o-expected[i]!)/Math.sqrt(expected[i]!)})),lowExpectedCells:expected.filter(e=>e<5).length}
}

/** Wilson score interval; one-sided variants use the matching z quantile. */
export function wilsonInterval(events:number,trials:number,confidence=.95,alternative:Alternative='two-sided'):CI{
  validSample({events,trials});conf(confidence);alt(alternative)
  if(confidence<=.5)throw new RangeError('score intervals require confidence > .5')
  const p=events/trials,z=STD.isf((1-confidence)/(alternative==='two-sided'?2:1)),z2=z*z,den=1+z2/trials
  const center=(p+z2/(2*trials))/den,half=z*Math.sqrt(p*(1-p)/trials+z2/(4*trials*trials))/den
  return [alternative==='less'?0:clamp(center-half),alternative==='greater'?1:clamp(center+half)]
}
/** Newcombe unpaired score-combination interval for p1-p2. Not an inversion of Fisher exact p. */
export function newcombeInterval(a:BinomialSample,b:BinomialSample,confidence=.95,alternative:Alternative='two-sided'):CI{
  validSample(a);validSample(b);conf(confidence);alt(alternative)
  if(confidence<=.5)throw new RangeError('score intervals require confidence > .5')
  // Wilson endpoints use the one-sided normal quantile for a one-sided difference interval.
  const z=STD.ppf(alternative==='two-sided'?.5+confidence/2:confidence)
  const pair=(s:BinomialSample):CI=>{const p=s.events/s.trials,z2=z*z,den=1+z2/s.trials,c=(p+z2/(2*s.trials))/den,h=z*Math.sqrt(p*(1-p)/s.trials+z2/(4*s.trials*s.trials))/den;return [clamp(c-h),clamp(c+h)]}
  const [l1,u1]=pair(a),[l2,u2]=pair(b),p1=a.events/a.trials,p2=b.events/b.trials,d=p1-p2
  return [alternative==='less'?-1:Math.max(-1,d-Math.sqrt((p1-l1)**2+(u2-p2)**2)),alternative==='greater'?1:Math.min(1,d+Math.sqrt((u1-p1)**2+(p2-l2)**2))]
}

export function proportionAnalysis(samples:BinomialSample[],options:{p0?:number;method?:'exact'|'normal'|'fisher';confidence?:number;alternative?:Alternative}={}){
  if(samples.length<1||samples.length>2)throw new RangeError('provide one or two binomial samples')
  samples.forEach(validSample);total(samples.map(s=>s.trials))
  const confidence=conf(options.confidence),alternative=alt(options.alternative),method=options.method??(samples.length===1?'exact':'normal'), warnings:string[]=[]
  // One-sided intervals below 50% are not useful in this workbench and can invert Wilson endpoints.
  if(confidence<=.5)throw new RangeError('proportion intervals require confidence > .5')
  const summary=samples.map((s,i)=>({...s,name:s.name??String(i+1),proportion:s.events/s.trials,wilsonCI:wilsonInterval(s.events,s.trials,confidence),failures:s.trials-s.events}))
  if(samples.length===1){
    if(!['exact','normal'].includes(method))throw new RangeError('one proportion supports exact or normal')
    const s=samples[0]!,p0=options.p0??.5
    if(!(p0>0&&p0<1))throw new RangeError('hypothesized proportion p0 must be in (0,1)')
    const r=propTest1(s.events,s.trials,{p0,confidence,alternative,method:method as 'exact'|'normal'})
    if(method==='normal'&&Math.min(s.trials*p0,s.trials*(1-p0))<5)warnings.push('Small null expected counts: use the exact binomial test instead of the normal approximation.')
    return {...r,ci:method==='normal'?wilsonInterval(s.events,s.trials,confidence,alternative):r.ci,ciMethod:method==='normal'?'Wilson score':'Clopper–Pearson',samples:summary,method,p0,warnings}
  }
  if(!['normal','fisher'].includes(method))throw new RangeError('two proportions support normal or fisher')
  const [a,b]=samples as [BinomialSample,BinomialSample],estimate=a.events/a.trials-b.events/b.trials
  const pooled=(a.events+b.events)/(a.trials+b.trials)
  const degenerate=pooled===0||pooled===1
  const raw=method==='normal'&&!degenerate?propTest2(a.events,a.trials,b.events,b.trials,{confidence,alternative,method:'normal'}):undefined
  const fisher=method==='fisher'?fisherExact2x2([[a.events,a.trials-a.events],[b.events,b.trials-b.events]],alternative):undefined
  if(degenerate)warnings.push('All outcomes are identical. The pooled normal test has zero standard error; p=1 is a limiting convention. The confidence interval still reflects uncertainty.')
  if(method==='normal'&&Math.min(a.trials*pooled,a.trials*(1-pooled),b.trials*pooled,b.trials*(1-pooled))<5)warnings.push('Small expected counts: the pooled z approximation may be inaccurate; consider Fisher exact.')
  if(method==='fisher')warnings.push('Fisher tests conditional association (odds ratio 1). The displayed Newcombe difference interval is approximate and is not an inversion of the Fisher test.')
  return {test:'2 proportions',estimate,statistic:method==='normal'?(degenerate?0:raw!.statistic):undefined,pValue:fisher?.pValue??raw?.pValue??1,ci:newcombeInterval(a,b,confidence,alternative),ciMethod:'Newcombe–Wilson',confidence,alternative,samples:summary,method,fisher,warnings}
}

/** Normal-theory variance test from n and sample s. ratio0 is a ratio of standard deviations, NOT variances. */
export function varianceSummaryAnalysis(samples:VarianceSample[],options:{sigma0?:number;ratio0?:number;confidence?:number;alternative?:Alternative}={}){
  const confidence=conf(options.confidence),alternative=alt(options.alternative)
  if(samples.length<1||samples.length>2)throw new RangeError('provide one or two variance summaries')
  samples.forEach(s=>{count(s.n,'N',true);if(s.n<2||!(s.sd>0&&Number.isFinite(s.sd)))throw new RangeError('variance test needs N >= 2 and finite SD > 0')})
  const alpha=1-confidence,groups=samples.map((s,i)=>({...s,name:s.name??String(i+1),variance:s.sd*s.sd,ci: [Math.sqrt((s.n-1)*s.sd*s.sd/chi2(s.n-1).isf(alpha/2)),Math.sqrt((s.n-1)*s.sd*s.sd/chi2(s.n-1).ppf(alpha/2))] as CI}))
  if(groups.some(s=>!Number.isFinite(s.variance)||s.variance===0))throw new RangeError('squared SD overflows or underflows; rescale the measurements')
  if(samples.length===1){
    const s=groups[0]!,sigma0=options.sigma0??1;if(!(sigma0>0&&Number.isFinite(sigma0)))throw new RangeError('sigma0 must be finite and positive')
    const df=s.n-1,statistic=df*(s.sd/sigma0)**2,d=chi2(df),varCI:CI=alternative==='less'?[0,df*s.variance/d.ppf(alpha)]:alternative==='greater'?[df*s.variance/d.isf(alpha),Infinity]:[df*s.variance/d.isf(alpha/2),df*s.variance/d.ppf(alpha/2)]
    return {test:'1 variance · chi-square',statistic,df,pValue:alternative==='less'?d.cdf(statistic):alternative==='greater'?d.sf(statistic):Math.min(1,2*Math.min(d.cdf(statistic),d.sf(statistic))),estimate:s.variance,sd:s.sd,ci:varCI,sdCI:varCI.map(Math.sqrt) as CI,groups,confidence,alternative,sigma0,method:'chi-square'}
  }
  const [a,b]=groups as [typeof groups[number],typeof groups[number]],ratio0=options.ratio0??1
  if(!(ratio0>0&&Number.isFinite(ratio0)))throw new RangeError('null SD ratio must be positive and finite')
  const ratio=(a.sd/b.sd)**2,statistic=(a.sd/b.sd/ratio0)**2,d=f(a.n-1,b.n-1)
  if(!Number.isFinite(ratio)||ratio===0)throw new RangeError('variance ratio is not numerically representable; rescale measurements')
  const ci:CI=alternative==='less'?[0,ratio/d.ppf(alpha)]:alternative==='greater'?[ratio/d.isf(alpha),Infinity]:[ratio/d.isf(alpha/2),ratio/d.ppf(alpha/2)]
  return {test:'2 variances · F',statistic,df:[a.n-1,b.n-1],pValue:alternative==='less'?d.cdf(statistic):alternative==='greater'?d.sf(statistic):Math.min(1,2*Math.min(d.cdf(statistic),d.sf(statistic))),estimate:ratio,ratio,sdRatio:a.sd/b.sd,ci,sdCI:ci.map(Math.sqrt) as CI,groups,confidence,alternative,ratio0,method:'f'}
}

export function varianceAnalysis(input:Record<string,ArrayLike<number|null|undefined>>,options:{method?:'chi-square'|'bonett'|'f'|'levene'|'brown-forsythe'|'bartlett';sigma0?:number;ratio0?:number;confidence?:number;alternative?:Alternative}={}){
  const entries=Object.entries(input),method=options.method??(entries.length===1?'chi-square':entries.length===2?'f':'brown-forsythe')
  if(!entries.length)throw new RangeError('provide samples')
  const gs=entries.map(([name,v])=>{const s=stats(v);if(s.n<2||!Number.isFinite(s.sd)||!(s.sd>0))throw new RangeError('each selected group needs >= 2 numeric values and positive SD: '+name);return {name,n:s.n,sd:s.sd,nMissing:s.nMissing}})
  const confidence=conf(options.confidence),alternative=alt(options.alternative)
  if(['chi-square','f'].includes(method)){
    if(entries.length!==(method==='chi-square'?1:2))throw new RangeError('wrong number of groups for variance test')
    return {...varianceSummaryAnalysis(gs,options),source:gs}
  }
  if(method==='bonett'){
    if(!(Number.isFinite(options.sigma0??1)&&(options.sigma0??1)>0))throw new RangeError('sigma0 must be finite and positive')
    if(entries.length!==1||gs[0]!.n<6)throw new RangeError('one-sample Bonett needs at least 6 observations')
    if(alternative!=='two-sided')throw new RangeError('Bonett workbench currently exposes the two-sided test only')
    if(STD.isf((1-confidence)/2)>=gs[0]!.n)throw new RangeError('Bonett confidence level too extreme for sample size')
    const r=varTest1(entries[0]![1],{sigma0:options.sigma0??1,confidence,alternative,method:'bonett'})
    if(!Number.isFinite(r.pValue)||r.ci.some(x=>!Number.isFinite(x)||x<=0))throw new RangeError('Bonett approximation failed for these data')
    return {...r,test:'1 variance · Bonett',sdCI:r.ci.map(Math.sqrt) as CI,groups:gs.map(s=>({...s,variance:s.sd*s.sd,ci:r.ci.map(Math.sqrt) as CI})),source:gs,method}
  }
  if(!['levene','brown-forsythe','bartlett'].includes(method)||entries.length<2)throw new RangeError('choose Levene, Brown–Forsythe or Bartlett and at least 2 groups')
  if(alternative!=='two-sided')throw new RangeError('equal-variance omnibus tests are non-directional')
  const arrays=entries.map(([,v])=>Array.from(cleanNumbers(v)))
  const r=method==='bartlett'?bartlett(arrays):levene(arrays,{center:method==='levene'?'mean':'median'})
  if(!Number.isFinite(r.statistic)||!Number.isFinite(r.pValue))throw new RangeError('equal-variance test is degenerate: no variation in absolute deviations')
  return {...r,groups:r.groups.map((g,i)=>({...g,name:gs[i]!.name})),source:gs,method,confidence,alternative}
}

/** Convert row-aligned categorical columns into a frequency table; missing rows are counted, not shifted. */
export function crossTabulate(a:ArrayLike<string|number|null|undefined>,b:ArrayLike<string|number|null|undefined>,weights?:ArrayLike<number|null|undefined>){
  if(a.length!==b.length||(weights&&weights.length!==a.length))throw new RangeError('categorical columns and weights must be row-aligned')
  const rows:string[]=[],columns:string[]=[],ri=new Map<string,number>(),ci=new Map<string,number>(),records:Array<[number,number,number]>=[]
  let omitted=0
  for(let i=0;i<a.length;i++){
    const x=a[i],y=b[i],w=weights?weights[i]:1
    if(x==null||y==null||x===''||y===''||w==null){omitted++;continue}
    if(!['string','number'].includes(typeof x)||!['string','number'].includes(typeof y))throw new RangeError('categories must be strings or numbers')
    count(w,'frequency weight')
    const xs=String(x),ys=String(y)
    if(!ri.has(xs)){ri.set(xs,rows.length);rows.push(xs)}
    if(!ci.has(ys)){ci.set(ys,columns.length);columns.push(ys)}
    records.push([ri.get(xs)!,ci.get(ys)!,w])
  }
  const observed=rows.map(()=>columns.map(()=>0))
  for(const [i,j,w] of records)observed[i]![j]+=w
  return {rows,columns,observed,omitted,sourceRows:a.length,completeRows:records.length}
}

/** Explicit binary event coding. More than two observed levels is not silently dichotomized. */
export function binarySample(v:ArrayLike<string|number|null|undefined>,event:string|number,name='Sample'){
  if(event===undefined||event===null||String(event)==='')throw new RangeError('choose the event value')
  const levels=new Set<string>();let trials=0,events=0,omitted=0
  for(let i=0;i<v.length;i++){
    const x=v[i]
    if(x===null||x===undefined||x===''){omitted++;continue}
    if(typeof x==='number'&&!Number.isFinite(x))throw new RangeError('binary outcomes must be finite')
    const s=String(x);levels.add(s);trials++;if(s===String(event))events++
  }
  if(levels.size>2||levels.size===2&&!levels.has(String(event)))throw new RangeError('binary sample needs at most two outcome levels; event value must be one of the two')
  validSample({events,trials})
  return {events,trials,name,omitted,event:String(event),levels:[...levels]}
}
