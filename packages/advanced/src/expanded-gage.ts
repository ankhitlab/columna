/** Proposed Columna variance-component REML and expanded Gage R&R. MIT.
 * Profiles residual variance, uses the determinant lemma/Woodbury identity on
 * random coefficient space, and validates covariance identifiability after
 * projecting out fixed effects. Independent Gaussian random intercept terms.
 */
import { nelderMead } from './optim.js'
import { normal } from './dist.js'
import { descriptiveStats, type DescriptiveStats } from './descriptive.js'

const sum = (x: number[]) => x.reduce((s, v) => s + v, 0)
function chol(a: Float64Array, n: number, relative = 1e-12): Float64Array {
  const l = new Float64Array(n * n), largest = Math.max(1, ...Array.from({length:n}, (_,i) => Math.abs(a[i*n+i]!)))
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = a[i*n+j]!; for (let k=0;k<j;k++) s -= l[i*n+k]! * l[j*n+k]!
    if (i === j) {if (!(s > relative * largest)) throw new RangeError('Матрица вырождена: эффекты или компоненты дисперсии неидентифицируемы.'); l[i*n+j] = Math.sqrt(s)}
    else l[i*n+j] = s/l[j*n+j]!
  }
  return l
}
function solve(l: Float64Array, b: ArrayLike<number>, n: number): Float64Array {
  const x=Float64Array.from(b)
  for(let i=0;i<n;i++){for(let j=0;j<i;j++) x[i] -= l[i*n+j]!*x[j]!;x[i]/=l[i*n+i]!}
  for(let i=n-1;i>=0;i--){for(let j=i+1;j<n;j++) x[i]-=l[j*n+i]!*x[j]!;x[i]/=l[i*n+i]!}
  return x
}
function logdet(l: Float64Array,n:number){let s=0;for(let i=0;i<n;i++)s+=2*Math.log(l[i*n+i]!);return s}
function dot(a:ArrayLike<number>,b:ArrayLike<number>){let s=0;for(let i=0;i<a.length;i++)s+=a[i]!*b[i]!;return s}
function indexed(labels: ArrayLike<string | number>){const levels:string[]=[], map=new Map<string,number>(), index:number[]=[];for(let i=0;i<labels.length;i++){const v=JSON.stringify([typeof labels[i],labels[i]]);if(!map.has(v)){map.set(v,levels.length);levels.push(String(labels[i]))}index.push(map.get(v)!)}return {levels,index}}
export interface VarianceTerm {name:string; labels:ArrayLike<string|number>; role?:'part'|'gage'}
export interface REMLInput {fixed?:number[][]; fixedNames?:string[]; terms:VarianceTerm[]; maxIter?:number; maxCoefficients?:number; memoryMB?:number; confidence?:number}
/** The supplied fixed matrix includes its intercept; omit it for an intercept-only model. */
export function varianceComponentREML(yInput: number[], options: REMLInput) {
  const n=yInput.length, X=options.fixed??yInput.map(()=>[1]), p=X[0]?.length??0, conf=options.confidence??.95
  if(!(conf>.5&&conf<1)||n<5||!p||n<=p+1||X.length!==n||X.some(r=>r.length!==p||r.some(v=>!Number.isFinite(v)))||yInput.some(v=>!Number.isFinite(v)))throw new RangeError('REML: некорректные данные, фиксированная матрица или недостаточно остаточных DF.')
  if(X.some(row=>row[0]!==1))throw new RangeError('REML: первый столбец фиксированной матрицы должен быть единичной константой.');
  if(!options.terms.length||options.terms.length>16)throw new RangeError('REML: задайте от 1 до 16 случайных компонент.')
  if(options.fixedNames&&options.fixedNames.length!==p)throw new RangeError('Имена фиксированных коэффициентов не совпадают с матрицей.')
  const s=descriptiveStats(yInput) as DescriptiveStats
  if(!(s.sd>0))throw new RangeError('REML: измерения постоянны.')
  const scale=s.sd, y=yInput.map(v=>(v-s.mean)/scale), offset=[0], terms=options.terms.map(t=>{
    if(t.labels.length!==n)throw new RangeError('Длины факторов и измерений различаются.')
    const g=indexed(t.labels);if(g.levels.length<2||g.levels.length===n)throw new RangeError(`${t.name}: нужны минимум 2 уровня и повторные наблюдения; уникальный ID каждой строки смешивается с остатком.`)
    offset.push(offset.at(-1)!+g.levels.length);return {...t,...g}
  }), q=offset.at(-1)!, k=terms.length, b=p+1, df=n-p
  const maxCoefficients=options.maxCoefficients??400, memoryMB=options.memoryMB??256
  if(!Number.isSafeInteger(maxCoefficients)||maxCoefficients<2||!Number.isFinite(memoryMB)||memoryMB<16)throw new RangeError('Некорректный бюджет REML.')
  const bytes=8*(q*q*6+q*b*4+p*p*5+n*(p+k+3))
  if(q>maxCoefficients||bytes>memoryMB*1024*1024)throw new RangeError(`REML: ${q} случайных коэффициентов; оценка памяти ${Math.ceil(bytes/1048576)} МБ. Упростите модель или явно увеличьте бюджет.`)
  const ids=Array.from({length:n},(_,i)=>terms.map((t,j)=>offset[j]!+t.index[i]!)), ZZ=new Float64Array(q*q), ZB=Array.from({length:b},()=>new Float64Array(q)), BB=new Float64Array(b*b)
  for(let i=0;i<n;i++){
    const row=[y[i]!,...X[i]!]
    for(let u=0;u<b;u++){for(let v=0;v<b;v++)BB[u*b+v]+=row[u]!*row[v]!;for(const z of ids[i]!)ZB[u]![z]+=row[u]!}
    for(const a of ids[i]!)for(const z of ids[i]!)ZZ[a*q+z]++
  }
  const XX=new Float64Array(p*p);for(let i=0;i<p;i++)for(let j=0;j<p;j++)XX[i*p+j]=BB[(i+1)*b+j+1]!
  const LX=chol(XX,p), projZ=Array.from({length:q},(_,j)=>solve(LX,Array.from({length:p},(_,u)=>ZB[u+1]![j]!),p))
  // Gram of residualized covariance basis: tr(M K_i M K_j), plus residual I.
  const gram=new Float64Array((k+1)*(k+1));gram[k*(k+1)+k]=df
  for(let i=0;i<q;i++)for(let j=0;j<q;j++){
    let v=ZZ[i*q+j]!;for(let u=0;u<p;u++)v-=ZB[u+1]![i]!*projZ[j]![u]!
    let ti=0,tj=0;while(offset[ti+1]!<=i)ti++;while(offset[tj+1]!<=j)tj++
    gram[ti*(k+1)+tj]+=v*v
    if(i===j){gram[ti*(k+1)+k]+=v;gram[k*(k+1)+ti]+=v}
  }
  const normalized=gram.slice();for(let i=0;i<=k;i++)for(let j=0;j<=k;j++)normalized[i*(k+1)+j]/=Math.sqrt(gram[i*(k+1)+i]!*gram[j*(k+1)+j]!)
  try{chol(normalized,k+1,1e-9)}catch{throw new RangeError('Случайные компоненты неидентифицируемы после учёта фиксированных эффектов. Проверьте смешивание факторов, вложенность, взаимодействия и повторения.')}
  const owners=new Int16Array(q);for(let t=0;t<k;t++)for(let j=offset[t]!;j<offset[t+1]!;j++)owners[j]=t
  let evaluations=0
  const profile=(lambda:number[], details=false)=>{
    evaluations++
    const d=Float64Array.from(owners,t=>Math.sqrt(lambda[t]!)), H=new Float64Array(q*q)
    for(let i=0;i<q;i++)for(let j=0;j<=i;j++){const v=d[i]!*d[j]!*ZZ[i*q+j]!+(i===j?1:0);H[i*q+j]=v;H[j*q+i]=v}
    const L=chol(H,q,1e-15), T=ZB.map(v=>Float64Array.from(v,(a,i)=>a*d[i]!)), solved=T.map(v=>solve(L,v,q)), S=BB.slice()
    for(let i=0;i<b;i++)for(let j=0;j<b;j++)S[i*b+j]-=dot(T[i]!,solved[j]!)
    const G=new Float64Array(p*p), rhs=new Float64Array(p);for(let i=0;i<p;i++){rhs[i]=S[(i+1)*b]!;for(let j=0;j<p;j++)G[i*p+j]=S[(i+1)*b+j+1]!}
    const LG=chol(G,p,1e-14), beta=solve(LG,rhs,p), rss=S[0]!-dot(beta,rhs)
    if(!(rss>1e-14))throw new RangeError('Остаточная вариация исчезла; REML вырождено.')
    const value=logdet(L,q)+logdet(LG,p)+df*Math.log(rss/df)
    if(!details)return {value}
    const sigma2=rss/df*scale*scale, cov=Array.from({length:p},(_,j)=>{const e=new Float64Array(p);e[j]=1;return Array.from(solve(LG,e,p),v=>v*sigma2)})
    const u=Float64Array.from(solved[0]!, (v,i)=>{for(let j=0;j<p;j++)v-=solved[j+1]![i]!*beta[j]!;return d[i]!*v*scale})
    const fixedFitted=X.map(row=>dot(row,beta)*scale+s.mean), fitted=fixedFitted.map((v,i)=>v+sum(ids[i]!.map(j=>u[j]!)))
    return {value,sigma2,beta:Array.from(beta,(v,j)=>v*scale+(j===0?s.mean:0)),cov,fitted,fixedFitted,residuals:yInput.map((v,i)=>v-fitted[i]!),blup:Array.from(u)}
  }
  const maxIter=options.maxIter??1600
  if(!Number.isSafeInteger(maxIter)||maxIter<10||maxIter>20000)throw new RangeError('maxIter: целое от 10 до 20000.')
  const objective=(v:Float64Array)=>{if(Array.from(v).some(x=>x < -25 || x>22))return Infinity;try{return profile(Array.from(v,Math.exp)).value}catch{return Infinity}}
  const starts=[new Array(k).fill(0),new Array(k).fill(2)], fits=starts.map(start=>nelderMead(objective,start,{maxIter,step:1,tol:1e-10})).filter(f=>Number.isFinite(f.value)).sort((a,b)=>a.value-b.value)
  if(!fits.length)throw new RangeError('REML не нашло допустимого решения.')
  let lambda=Array.from(fits[0]!.x,Math.exp), value=fits[0]!.value, iterations=fits[0]!.iterations, converged=fits[0]!.converged
  // Exact boundary refits: no small positive number misrepresented as a zero variance.
  for(let j=0;j<k;j++)if(lambda[j]!<1e-6){const trial=lambda.slice();trial[j]=0;try{if(profile(trial).value<=value+1e-7){lambda=trial;value=profile(trial).value}}catch{/* retain positive fit */}}
  const active=lambda.map((v,i)=>v>0?i:-1).filter(i=>i>=0)
  if(active.length&&active.length<k){const f=nelderMead(v=>{const l=lambda.slice();active.forEach((j,t)=>l[j]=Math.exp(v[t]!));try{return profile(l).value}catch{return Infinity}},active.map(j=>Math.log(lambda[j]!)),{maxIter,step:.2,tol:1e-10});if(f.value<=value+1e-7){active.forEach((j,t)=>lambda[j]=Math.exp(f.x[t]!));value=f.value;iterations+=f.iterations;converged=f.converged}}
  let improvement=0
  for(let j=0;j<k;j++)for(const step of [-.002,.002]){const l=lambda.slice();l[j]=lambda[j]!>0?lambda[j]!*Math.exp(step):step>0?1e-7:0;try{improvement=Math.max(improvement,value-profile(l).value)}catch{/* invalid perturbation */}}
  converged=converged&&improvement<1e-5&&lambda.every(v=>v<Math.exp(21))
  const result=profile(lambda,true), critical=normal().ppf((1+conf)/2)
  const coefficients=result.beta!.map((estimate,j)=>{const se=Math.sqrt(Math.max(0,result.cov![j]![j]!));return {name:options.fixedNames?.[j]??(j?'X'+j:'Константа'),estimate,se,ci:[estimate-critical*se,estimate+critical*se],pValue:se>0?2*normal().sf(Math.abs(estimate/se)):null}})
  return {method:'REML',n,p,df,randomCoefficients:q,memoryBytes:bytes,converged,iterations,evaluations,localImprovement:improvement,residualVariance:result.sigma2!,logLik:-.5*(result.value+df*(Math.log(2*Math.PI)+1+2*Math.log(scale))),coefficients,components:terms.map((t,j)=>({name:t.name,role:t.role??'gage',levels:t.levels.length,variance:lambda[j]!*result.sigma2!,boundary:lambda[j]===0})),fitted:result.fitted!,fixedFitted:result.fixedFitted!,residuals:result.residuals!,blup:terms.flatMap((t,j)=>t.levels.map((level,i)=>({term:t.name,level,value:result.blup![offset[j]!+i]!}))),covariance:result.cov!}
}
export interface ExpandedGageInput {
  y: Array<number|null>; data:Record<string,Array<string|number|null>>; part:string; operator:string
  operatorType?:'random'|'fixed'; structure?:'crossed'|'nested'; additional?:Array<{id:string;type:'random'|'fixed';role?:'part'|'gage'}>
  interactions?:'none'|'part-operator'|'all-pairs'|'full'; tolerance?:number; studyMultiplier?:number; confidence?:number; maxIter?:number; maxCoefficients?:number
  labels?:Record<string,string>
}
export function expandedGageRR(o:ExpandedGageInput){
  const additional=o.additional??[], ids=[o.part,o.operator,...additional.map(f=>f.id)], label=(id:string)=>o.labels?.[id]??id
  if(new Set(ids).size!==ids.length||ids.some(id=>!id||!o.data[id]||o.data[id]!.length!==o.y.length))throw new RangeError('Выберите различные существующие факторы одинаковой длины.')
  if(additional.some(f=>!['random','fixed'].includes(f.type)||f.role!==undefined&&!['part','gage'].includes(f.role)))throw new RangeError('Некорректный тип/роль дополнительного фактора.')
  if(o.operatorType!==undefined&&!['random','fixed'].includes(o.operatorType)||o.structure!==undefined&&!['crossed','nested'].includes(o.structure))throw new RangeError('Некорректный оператор или структура.')
  const mult=o.studyMultiplier??6, conf=o.confidence??.95
  if(!(Number.isFinite(mult)&&mult>0)||o.tolerance!==undefined&&!(Number.isFinite(o.tolerance)&&o.tolerance>0))throw new RangeError('Множитель и допуск должны быть положительными.')
  const keep=o.y.map((v,i)=>v!==null&&ids.every(id=>o.data[id]![i]!==null&&o.data[id]![i]!=='')?i:-1).filter(i=>i>=0), y=keep.map(i=>o.y[i]!) as number[]
  if(y.some(v=>!Number.isFinite(v)))throw new RangeError('Измерения должны быть конечными числами.')
  const data:Record<string,Array<string|number>>={};for(const id of ids)data[id]=keep.map(i=>o.data[id]![i]!) as Array<string|number>
  if(ids.some(id=>new Set(data[id]).size<2))throw new RangeError('Каждый фактор должен иметь минимум два наблюдаемых уровня.')
  const cells=new Map<string,number>();for(let i=0;i<y.length;i++){const key=JSON.stringify(ids.map(id=>data[id]![i]));cells.set(key,(cells.get(key)??0)+1)}
  if(![...cells.values()].some(n=>n>=2))throw new RangeError('Нет повторных измерений при одинаковой комбинации факторов; Repeatability не подтверждена схемой.')
  const nested=o.structure==='nested', opFixed=o.operatorType==='fixed', factors=[{id:o.part,type:'random',role:'part'},{id:o.operator,type:opFixed?'fixed':'random',role:'gage'},...additional.map(f=>({...f,role:f.role??'gage'}))]
  const X=y.map(()=>[1]), names=['Константа'], fixedInfo:Array<{id:string;role:string;levels:string[];indices:number[]}>=[]
  for(const f of factors.filter(f=>f.type==='fixed')){const g=indexed(data[f.id]!);const indices:number[]=[];for(let j=1;j<g.levels.length;j++){indices.push(names.length);names.push(`${label(f.id)}[${g.levels[j]}]`);X.forEach((row,i)=>row.push(g.index[i]===j?1:0))}fixedInfo.push({id:f.id,role:f.role,levels:g.levels,indices})}
  const random=factors.filter(f=>f.type==='random'), terms:VarianceTerm[]=random.map(f=>({name:f.id===o.part&&nested?`${label(o.part)}(${label(o.operator)})`:label(f.id),role:f.role as 'part'|'gage',labels:f.id===o.part&&nested?y.map((_,i)=>JSON.stringify([data[o.operator]![i],data[o.part]![i]])):data[f.id]!}))
  const interaction=o.interactions??(nested?'none':'part-operator')
  if(!['none','part-operator','all-pairs','full'].includes(interaction))throw new RangeError('Неизвестная схема взаимодействий.')
  if(nested&&interaction!=='none')throw new RangeError('Вложенная схема: доступны главные эффекты. Не добавляйте взаимодействие детали с родительским оператором.')
  if(interaction==='part-operator'&&opFixed)throw new RangeError('Для фиксированного оператора выберите главные эффекты либо взаимодействия случайных факторов; случайный Part×фиксированный Operator здесь не параметризован.')
  const pushTerm=(fs:typeof random)=>terms.push({name:fs.map(f=>label(f.id)).join(' × '),role:fs.every(f=>f.role==='part')?'part':'gage',labels:y.map((_,i)=>JSON.stringify(fs.map(f=>data[f.id]![i])))})
  if(interaction==='part-operator')pushTerm(random.filter(f=>f.id===o.part||f.id===o.operator))
  else if(interaction==='all-pairs'||interaction==='full'){for(let mask=1;mask<2**random.length;mask++){const fs=random.filter((_,j)=>mask&(1<<j));if(fs.length>=2&&(interaction==='full'||fs.length===2))pushTerm(fs)}}
  const fit=varianceComponentREML(y,{fixed:X,fixedNames:names,terms,confidence:conf,maxIter:o.maxIter,maxCoefficients:o.maxCoefficients})
  if(!fit.converged)throw new RangeError('REML не достигло подтверждённого локального минимума. Упростите модель или увеличьте число итераций; неподтверждённые компоненты не сохранены.')
  const components=fit.components.map(c=>({...c,kind:'random'})),fixedSpreads=fixedInfo.map(f=>{const estimates=[0,...f.indices.map(j=>fit.coefficients[j]!.estimate)], center=sum(estimates)/estimates.length;return {name:label(f.id),role:f.role,levels:f.levels.length,kind:'fixed-spread',variance:sum(estimates.map(v=>(v-center)**2))/estimates.length,boundary:false}})
  const sources=[...components,...fixedSpreads], repeatability=fit.residualVariance, reproducibility=sum(sources.filter(c=>c.role==='gage').map(c=>c.variance)),partToPart=sum(sources.filter(c=>c.role==='part').map(c=>c.variance)),grr=repeatability+reproducibility,total=grr+partToPart
  const measures=(variance:number)=>({variance,sd:Math.sqrt(variance),study:mult*Math.sqrt(variance),contribution:100*variance/total,studyPercent:100*Math.sqrt(variance/total),tolerancePercent:o.tolerance===undefined?null:100*mult*Math.sqrt(variance)/o.tolerance})
  const summary=[['Total Gage R&R',grr],['Repeatability',repeatability],['Reproducibility',reproducibility],['Part-to-Part',partToPart],['Total',total]].map(([name,v])=>({name:String(name),...measures(Number(v))}))
  const warnings=['REML с независимыми нормально распределёнными случайными эффектами; это не EMS/ANOVA-оценивание Expanded Gage R&R Minitab. Численного тождества всем настройкам Minitab не заявлено.','Компоненты неотрицательны; нулевая оценка на границе не доказывает отсутствия вариации. Wald-интервалы фиксированных коэффициентов условны по оценённой ковариации; Kenward–Roger/Satterthwaite и интервалы компонент не вычислены.','Нет автоматического удаления взаимодействий по p. Выбранная модель сохраняется целиком; несбалансированность и пропуски требуют проверки плана измерений.']
  if(fixedInfo.length)warnings.push('Воспроизводимость включает φ фиксированных факторов: средний квадрат отклонений оценённых уровней при равных весах. Это разброс конкретных фиксированных уровней, не генеральная случайная дисперсия.')
  if(components.some(c=>c.boundary))warnings.push('Есть компоненты на границе 0; будьте осторожны с интерпретацией и переносом выводов на новые уровни факторов.')
  return {kind:'expanded-gage',n:y.length,omitted:o.y.length-y.length,sourceRows:keep.map(i=>i+1),structure:o.structure??'crossed',interaction,fit,summary,sources:sources.map(c=>({...c,...measures(c.variance)})),ndc:Math.max(1,Math.floor(1.41*Math.sqrt(partToPart/grr))),tolerance:o.tolerance,studyMultiplier:mult,replicateCounts:[...cells.entries()].map(([cell,n])=>({cell:JSON.parse(cell),n})),balanced:new Set(cells.values()).size===1&&cells.size===ids.reduce((p,id)=>p*new Set(data[id]).size,1),warnings}
}
