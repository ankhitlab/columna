/** Proposed Columna extension: typed, non-executable worksheet expressions.
 * Expressions are parsed to a closed AST and bind column/constant IDs, never eval/Function.
 * Statistical aggregates, distributions and matrix algebra delegate to existing Columna modules.
 * Date values are UTC Unix milliseconds. This is not an Excel formula interpreter.
 */
import {descriptiveStats, quantile} from './descriptive.js'
import {normal, t, chi2, f, binomial, poisson} from './dist.js'
import {fromRows, transpose, matmul, inverse, type Matrix} from './linalg.js'
export type WorksheetValue = number | string | null
export type WorksheetType = 'number' | 'text' | 'datetime'
export interface WorksheetColumn {id: string; name: string; type: WorksheetType; formula?: {source: string; ast: WorksheetExpression; version: 1}}
export interface WorksheetConstant {id: string; code: string; name: string; type: WorksheetType; value: WorksheetValue}
export interface WorksheetData {columns: WorksheetColumn[]; rows: WorksheetValue[][]; constants?: WorksheetConstant[]}
export type WorksheetExpression = {kind: 'literal'; value: WorksheetValue} | {kind: 'column' | 'constant'; id: string} | {kind:'call'; name:string; args:WorksheetExpression[]} | {kind:'binary'; op:string; left:WorksheetExpression; right:WorksheetExpression} | {kind:'unary'; op:string; arg:WorksheetExpression}
const arities: Record<string,[number,number]> = {
 IF:[3,3], IFERROR:[2,2], COALESCE:[1,99], ISMISSING:[1,1], ROW:[0,0], LAG:[1,2], LEAD:[1,2],
 SUM:[1,1], MEAN:[1,1], MEDIAN:[1,1], SD:[1,1], VAR:[1,1], MIN:[1,1], MAX:[1,1], COUNT:[1,1], NMISSING:[1,1], QUANTILE:[2,2], STANDARDIZE:[1,1], CUMSUM:[1,1], RANK:[1,1],
 ROWSUM:[1,99], ROWMEAN:[1,99], ROWMIN:[1,99], ROWMAX:[1,99],
 ABS:[1,1], SQRT:[1,1], LN:[1,1], LOG10:[1,1], EXP:[1,1], ROUND:[1,2], FLOOR:[1,1], CEIL:[1,1], SIN:[1,1], COS:[1,1], POW:[2,2],
 CONCAT:[1,99], LOWER:[1,1], UPPER:[1,1], TRIM:[1,1], LEN:[1,1], LEFT:[2,2], RIGHT:[2,2], MID:[3,3], REPLACE:[3,3], CONTAINS:[2,2], NUMBER:[1,1], TEXT:[1,1],
 DATE:[1,6], YEAR:[1,1], MONTH:[1,1], DAY:[1,1], HOUR:[1,1], MINUTE:[1,1], SECOND:[1,1], DATEADD:[2,3], DAYS:[2,2], FORMATDATE:[1,2],
 NORMALCDF:[1,3], NORMALPDF:[1,3], NORMALINV:[1,3], TCDF:[2,2], TINV:[2,2], CHI2CDF:[2,2], CHI2INV:[2,2], FCDF:[3,3], FINV:[3,3], BINOMCDF:[3,3], POISSONCDF:[2,2]
}
const binaryOps=['+','-','*','/','^','%','&','==','!=','<','>','<=','>=','AND','OR']
function fail(message:string):never {throw new Error(message)}
function numeric(v:WorksheetValue):number {if(typeof v!=='number'||!Number.isFinite(v))fail('Ожидается конечное число.');return v}
function integer(v:WorksheetValue,min=0):number {const n=numeric(v);if(!Number.isSafeInteger(n)||n<min)fail(`Ожидается целое число ≥ ${min}.`);return n}
function valid(v:WorksheetValue):WorksheetValue {if(typeof v==='number'&&!Number.isFinite(v))fail('Результат NaN/Infinity: проверьте область определения.');return v}
export function worksheetDate(value:WorksheetValue):number|null {
 if(value===null||value==='')return null
 if(typeof value==='number'){if(!Number.isFinite(value)||Math.abs(value)>8640000000000000)fail('Дата вне диапазона.');return value}
 let s=value.trim();const ru=s.match(/^(\d{2})\.(\d{2})\.(\d{4})(.*)$/);if(ru)s=`${ru[3]}-${ru[2]}-${ru[1]}${ru[4]}`
 if(/^\d{2}:\d{2}(:\d{2}(\.\d{1,3})?)?$/.test(s))s='1970-01-01T'+s
 const m=s.match(/^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?(Z|[+-]\d{2}:\d{2})?)?$/)
 if(!m)fail('Дата: YYYY-MM-DD, DD.MM.YYYY или ISO дата/время с необязательным часовым поясом. Без пояса — UTC.')
 const y=+m[1],mo=+m[2],d=+m[3],h=+(m[4]??0),mi=+(m[5]??0),se=+(m[6]??0),ms=+(m[7]??'').padEnd(3,'0')
 const date=new Date(0);date.setUTCFullYear(y,mo-1,d);date.setUTCHours(h,mi,se,ms)
 if(date.getUTCFullYear()!==y||date.getUTCMonth()!==mo-1||date.getUTCDate()!==d||h>23||mi>59||se>59)fail('Несуществующая дата или время.')
 let offset=0;if(m[8]&&m[8]!=='Z'){const z=m[8],hh=+z.slice(1,3),mm=+z.slice(4);if(hh>23||mm>59)fail('Некорректный UTC offset.');offset=(z[0]==='-'?-1:1)*(hh*60+mm)*60000}
 return date.getTime()-offset
}
export function worksheetDateText(value:WorksheetValue,format='datetime'):string {
 if(value===null)return '';const v=worksheetDate(value);if(v===null)return '';const s=new Date(v).toISOString()
 return format==='date'?s.slice(0,10):format==='time'?s.slice(11,23):s
}
interface Token {kind:string; value:string}
export function compileWorksheetExpression(source:string,sheet:WorksheetData):WorksheetExpression {
 if(typeof source!=='string'||source.length>20000)fail('Слишком длинная формула.');let s=source.trim().replace(/^=/,''),p=0,tokens:Token[]=[]
 while(p<s.length){const c=s[p];if(/\s/.test(c)){p++;continue}
  if(c==='"'||c==="'"){const quote=c;let v='',closed=false;p++;while(p<s.length){const k=s[p++];if(k===quote){if(s[p]===quote){v+=quote;p++;}else{closed=true;break}}else if(k==='\\'){const e=s[p++];if(e===undefined)fail('Незакрытая строка.');v+=e==='n'?'\n':e==='t'?'\t':e}else v+=k}if(!closed)fail('Незакрытая строка.');tokens.push({kind:'string',value:v});continue}
  if(c==='['){const end=s.indexOf(']',p+1);if(end<0)fail('Не закрыто имя в квадратных скобках.');tokens.push({kind:'ref',value:s.slice(p+1,end)});p=end+1;continue}
  const num=s.slice(p).match(/^(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/);if(num){tokens.push({kind:'number',value:num[0]});p+=num[0].length;continue}
  const id=s.slice(p).match(/^[\p{L}_][\p{L}\p{N}_]*/u);if(id){tokens.push({kind:'id',value:id[0]});p+=id[0].length;continue}
  const op=s.slice(p).match(/^(<=|>=|==|!=|<>|&&|\|\||[+\-*/^%&=<>!(),;])/);if(!op)fail(`Недопустимый символ «${c}».`);tokens.push({kind:'op',value:op[0]});p+=op[0].length
 }
 tokens.push({kind:'end',value:''});let i=0,depth=0,nodes=0
 const binding=(name:string,byName=false):WorksheetExpression=>{const cols=sheet.columns.filter(c=>c.name===name),ks=(sheet.constants??[]).filter(c=>c.name===name);if(!byName&&/^C\d+$/i.test(name)){const col=sheet.columns[+name.slice(1)-1];if(!col)fail('Нет столбца '+name);return {kind:'column',id:col.id}}if(!byName&&/^K\d+$/i.test(name)){const k=sheet.constants?.find(c=>c.code===name.toUpperCase());if(!k)fail('Нет константы '+name);return {kind:'constant',id:k.id}}if(cols.length+ks.length!==1)fail(`Неоднозначное или неизвестное имя «${name}».`);return cols.length?{kind:'column',id:cols[0].id}:{kind:'constant',id:ks[0].id}}
 const norm=(v:string)=>({'=':'==','<>':'!=','&&':'AND','||':'OR'}[v]??v.toUpperCase())
 const precedence:Record<string,number>={OR:1,AND:2,'==':3,'!=':3,'<':3,'>':3,'<=':3,'>=':3,'&':4,'+':5,'-':5,'*':6,'/':6,'%':6,'^':8}
 function parse(min=0):WorksheetExpression {
  if(++depth>80||++nodes>4000)fail('Формула слишком сложная.');const token=tokens[i++];let left:WorksheetExpression
  if(token.kind==='number')left={kind:'literal',value:valid(Number(token.value))}
  else if(token.kind==='string')left={kind:'literal',value:token.value}
  else if(token.kind==='ref')left=binding(token.value,true)
  else if(['+','-','!','NOT'].includes(token.value.toUpperCase()))left={kind:'unary',op:token.value.toUpperCase(),arg:parse(7)}
  else if(token.value==='('){left=parse();if(tokens[i++].value!==')')fail('Ожидается ).')}
  else if(token.kind==='id'){
   const name=token.value.toUpperCase()
   if(tokens[i].value==='('){i++;const args:WorksheetExpression[]=[];if(tokens[i].value!==')'){do{args.push(parse());if(![',',';'].includes(tokens[i].value))break;i++}while(true)}if(tokens[i++].value!==')')fail('Ожидается ).');const a=arities[name];if(!a||args.length<a[0]||args.length>a[1])fail(`Неизвестная функция или неверное число аргументов: ${name}.`);left={kind:'call',name,args}}
   else if(['NULL','TRUE','FALSE','PI','E'].includes(name))left={kind:'literal',value:({NULL:null,TRUE:1,FALSE:0,PI:Math.PI,E:Math.E} as Record<string,WorksheetValue>)[name]}
   else left=binding(token.value)
  }else fail('Ожидается число, столбец или функция.')
  while(true){const op=norm(tokens[i].value),prec=precedence[op];if(prec===undefined||prec<min)break;i++;left={kind:'binary',op,left,right:parse(op==='^'?prec:prec+1)}}depth--;return left
 }
 const ast=parse();if(tokens[i].kind!=='end')fail('Лишние символы в формуле. Десятичный разделитель — точка.');return ast
}
export function worksheetReferences(ast:WorksheetExpression):{columns:string[]; constants:string[]} {
 const columns=new Set<string>(),constants=new Set<string>();let n=0
 const visit=(a:WorksheetExpression,d=0)=>{if(!a||typeof a!=='object'||++n>4000||d>80)fail('Повреждённое дерево формулы.');switch(a.kind){
  case 'literal':if(a.value!==null&&typeof a.value!=='string'&&typeof a.value!=='number')fail('Недопустимая константа.');valid(a.value);break
  case 'column':case 'constant':if(typeof a.id!=='string')fail('Некорректная ссылка.');(a.kind==='column'?columns:constants).add(a.id);break
  case 'unary':if(!['+','-','NOT','!'].includes(a.op))fail('Некорректный оператор.');visit(a.arg,d+1);break
  case 'binary':if(!binaryOps.includes(a.op))fail('Некорректный оператор.');visit(a.left,d+1);visit(a.right,d+1);break
  case 'call':{const b=arities[a.name];if(!b||!Array.isArray(a.args)||a.args.length<b[0]||a.args.length>b[1])fail('Некорректная функция.');a.args.forEach(x=>visit(x,d+1));break}
  default:fail('Неизвестный элемент формулы.')
 }};visit(ast);return {columns:[...columns],constants:[...constants]}
}
export function worksheetExpressionText(ast:WorksheetExpression,sheet:WorksheetData):string {
 worksheetReferences(ast)
 const show=(a:WorksheetExpression):string=>{switch(a.kind){case 'literal':return a.value===null?'NULL':JSON.stringify(a.value);case 'column':{const j=sheet.columns.findIndex(c=>c.id===a.id);return j<0?'[удалённый столбец]':`C${j+1}`}case 'constant':return sheet.constants?.find(c=>c.id===a.id)?.code??'[удалённая константа]';case 'unary':return `${a.op}(${show(a.arg)})`;case 'binary':return `(${show(a.left)} ${a.op} ${show(a.right)})`;case 'call':return `${a.name}(${a.args.map(show).join(', ')})`}}
 return show(ast)
}
export function worksheetAggregate(values:WorksheetValue[],method:string,p=.5):WorksheetValue {
 const v=values.filter(x=>x!==null);if(method==='count')return v.length;if(method==='missing')return values.length-v.length
 if(method==='first')return v[0]??null;if(!v.length)return null
 v.forEach(numeric);if(method==='quantile')return quantile(v as number[],p)
 const d=descriptiveStats(v as number[]) as Exclude<ReturnType<typeof descriptiveStats>,Array<unknown>>
 const keys:Record<string,string>={sum:'sum',mean:'mean',median:'median',sd:'sd',var:'variance',min:'min',max:'max'}
 if(!keys[method])fail('Неизвестная агрегация.');const out=(d as unknown as Record<string,number>)[keys[method]];return Number.isFinite(out)?out:null
}
export function evaluateWorksheetExpression(ast:WorksheetExpression,sheet:WorksheetData):WorksheetValue[] {
 const refs=worksheetReferences(ast),cols=new Map(sheet.columns.map((c,j)=>[c.id,j])),ks=new Map((sheet.constants??[]).map(k=>[k.id,k.value]));refs.columns.forEach(id=>{if(!cols.has(id))fail('Формула ссылается на удалённый столбец.')} );refs.constants.forEach(id=>{if(!ks.has(id))fail('Формула ссылается на удалённую константу.')})
 const cache=new Map<WorksheetExpression,WorksheetValue>(),vectorCache=new Map<WorksheetExpression,WorksheetValue[]>(),n=sheet.rows.length
 const vector=(a:WorksheetExpression)=>{if(!vectorCache.has(a))vectorCache.set(a,sheet.rows.map((_,i)=>at(a,i)));return vectorCache.get(a)!}
 function hasRowFunction(a:WorksheetExpression):boolean {return a.kind==='call'?(a.name==='ROW'||a.name==='LAG'||a.name==='LEAD'||a.name==='CUMSUM'||a.name==='RANK'||a.name==='STANDARDIZE'||a.args.some(hasRowFunction)):a.kind==='binary'?hasRowFunction(a.left)||hasRowFunction(a.right):a.kind==='unary'?hasRowFunction(a.arg):false}
 function at(a:WorksheetExpression,r:number):WorksheetValue {
  switch(a.kind){case 'literal':return a.value;case 'column':return sheet.rows[r]?.[cols.get(a.id)!]??null;case 'constant':return ks.get(a.id)??null
   case 'unary':{const x=at(a.arg,r);return a.op==='NOT'||a.op==='!'?+!x:x===null?null:valid((a.op==='-'?-1:1)*numeric(x))}
   case 'binary':{
    const x=at(a.left,r);if(a.op==='AND')return x?+!!at(a.right,r):0;if(a.op==='OR')return x?1:+!!at(a.right,r);const y=at(a.right,r);if(x===null||y===null)return null
    if(['==','!=','<','>','<=','>='].includes(a.op)){if(typeof x!==typeof y)fail('Сравнивайте значения одного типа.');return +({'==':x===y,'!=':x!==y,'<':x<y,'>':x>y,'<=':x<=y,'>=':x>=y}[a.op]!)}if(a.op==='&')return String(x)+String(y)
    const u=numeric(x),v=numeric(y);return valid(({'+':()=>u+v,'-':()=>u-v,'*':()=>u*v,'/':()=>u/v,'^':()=>u**v,'%':()=>u%v}[a.op]!)())
   }
   case 'call':{
    const k=a.name,args=a.args,arg=(i:number)=>at(args[i],r)
    if(k==='IF')return at(args[arg(0)?1:2],r)
    if(k==='IFERROR'){try{return arg(0)}catch{return arg(1)}}
    if(k==='COALESCE'){for(const node of args){const x=at(node,r);if(x!==null)return x}return null}
    if(k==='ISMISSING')return +(arg(0)===null)
    if(k==='ROW')return r+1
    if(k==='LAG'||k==='LEAD'){const off=args[1]?integer(arg(1)):1,pos=r+(k==='LAG'?-off:off);return pos<0||pos>=n?null:at(args[0],pos)}
    if(['SUM','MEAN','MEDIAN','SD','VAR','MIN','MAX','COUNT','NMISSING','QUANTILE'].includes(k)){
     if(!cache.has(a)){if(k==='QUANTILE'&&(worksheetReferences(args[1]).columns.length||hasRowFunction(args[1])))fail('Вероятность QUANTILE не должна зависеть от строки.');cache.set(a,worksheetAggregate(vector(args[0]),k==='NMISSING'?'missing':k.toLowerCase(),k==='QUANTILE'?numeric(arg(1)):.5))}return cache.get(a)!
    }
    if(k==='STANDARDIZE'){const x=arg(0);if(x===null)return null;if(!vectorCache.has(a)){const v=vector(args[0]),mean=worksheetAggregate(v,'mean'),sd=worksheetAggregate(v,'sd');if(sd===null||numeric(sd)<=0)fail('Для стандартизации нужен ненулевой SD.');vectorCache.set(a,v.map(z=>z===null?null:(numeric(z)-numeric(mean))/numeric(sd)))}return vectorCache.get(a)![r]}
    if(k==='CUMSUM'||k==='RANK'){
     if(!vectorCache.has(a)){const v=vector(args[0]);if(k==='CUMSUM'){let sum=0;vectorCache.set(a,v.map(x=>x===null?null:valid(sum+=numeric(x))))}else{const sorted=v.map((x,i)=>({x,i})).filter(o=>o.x!==null).sort((a,b)=>numeric(a.x)-numeric(b.x)),out:WorksheetValue[]=Array(n).fill(null);for(let j=0;j<sorted.length;){let end=j+1;while(end<sorted.length&&sorted[end].x===sorted[j].x)end++;for(let u=j;u<end;u++)out[sorted[u].i]=(j+1+end)/2;j=end}vectorCache.set(a,out)}}return vectorCache.get(a)![r]
    }
    const vals=args.map(x=>at(x,r));if(k.startsWith('ROW'))return worksheetAggregate(vals,k.slice(3).toLowerCase())
    if(k==='CONCAT')return vals.map(x=>x??'').join('');if(vals.some(x=>x===null))return null
    const x=vals[0],y=vals[1],z=vals[2],num=(i:number)=>numeric(vals[i]),str=(i:number)=>String(vals[i])
    const math:Record<string,(x:number)=>number>={ABS:Math.abs,SQRT:Math.sqrt,LN:Math.log,LOG10:Math.log10,EXP:Math.exp,FLOOR:Math.floor,CEIL:Math.ceil,SIN:Math.sin,COS:Math.cos}
    if(math[k])return valid(math[k](num(0)))
    switch(k){
     case 'ROUND':{const d=args[1]?num(1):0;if(!Number.isInteger(d)||Math.abs(d)>15)fail('ROUND: число знаков от −15 до 15.');return valid(Math.round(num(0)*10**d)/10**d)}
     case 'POW':return valid(num(0)**num(1));case 'LOWER':return str(0).toLowerCase();case 'UPPER':return str(0).toUpperCase();case 'TRIM':return str(0).trim();case 'LEN':return [...str(0)].length
     case 'LEFT':return [...str(0)].slice(0,integer(y)).join('');case 'RIGHT':{const m=integer(y);return m?[...str(0)].slice(-m).join(''):''}
     case 'MID':return [...str(0)].slice(integer(y,1)-1,integer(y,1)-1+integer(z)).join('');case 'REPLACE':return str(0).split(str(1)).join(str(2));case 'CONTAINS':return +str(0).includes(str(1))
     case 'NUMBER':{const s=str(0).trim().replace(',','.');if(!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(s))fail('NUMBER: некорректное число.');return valid(Number(s))}
     case 'TEXT':return str(0)
     case 'DATE':{if(vals.length===1)return worksheetDate(x);if(vals.length<3)fail('DATE: дата-строка или год, месяц, день[, час, минута, секунда].');const p=(i:number,size=2)=>String(integer(vals[i]??0)).padStart(size,'0');return worksheetDate(`${p(0,4)}-${p(1)}-${p(2)}T${p(3)}:${p(4)}:${p(5)}Z`)}
     case 'YEAR':case 'MONTH':case 'DAY':case 'HOUR':case 'MINUTE':case 'SECOND':{const d=new Date(worksheetDate(x)!);return ({YEAR:d.getUTCFullYear(),MONTH:d.getUTCMonth()+1,DAY:d.getUTCDate(),HOUR:d.getUTCHours(),MINUTE:d.getUTCMinutes(),SECOND:d.getUTCSeconds()})[k]}
     case 'DATEADD':{const unit=String(z??'days').toLowerCase(),scale=({days:86400000,hours:3600000,minutes:60000,seconds:1000,ms:1} as Record<string,number>)[unit];if(!scale)fail('DATEADD: days, hours, minutes, seconds или ms.');return worksheetDate(worksheetDate(x)!+num(1)*scale)}
     case 'DAYS':return (worksheetDate(x)!-worksheetDate(y)!)/86400000;case 'FORMATDATE':return worksheetDateText(x,String(y??'datetime'))
     case 'NORMALCDF':case 'NORMALPDF':case 'NORMALINV':{const d=normal(args.length>1?num(1):0,args.length>2?num(2):1);return valid(k==='NORMALCDF'?d.cdf(num(0)):k==='NORMALPDF'?d.pdf(num(0)):d.ppf(num(0)))}
     case 'TCDF':return valid(t(num(1)).cdf(num(0)));case 'TINV':return valid(t(num(1)).ppf(num(0)));case 'CHI2CDF':return valid(chi2(num(1)).cdf(num(0)));case 'CHI2INV':return valid(chi2(num(1)).ppf(num(0)))
     case 'FCDF':return valid(f(num(1),num(2)).cdf(num(0)));case 'FINV':return valid(f(num(1),num(2)).ppf(num(0)));case 'BINOMCDF':return valid(binomial(integer(y),num(2)).cdf(num(0)));case 'POISSONCDF':return valid(poisson(num(1)).cdf(num(0)))
     default:return fail('Неизвестная функция '+k)
    }
   }
  }
 }
 return sheet.rows.map((_,r)=>{try{return valid(at(ast,r))}catch(e){fail(`Строка ${r+1}: ${(e as Error).message}`)}})
}
export function recalculateWorksheet(sheet:WorksheetData):WorksheetValue[][] {
 const rows=sheet.rows.map(r=>r.slice()),work={...sheet,rows},done=new Set<string>(),active=new Set<string>(),cols=new Map(sheet.columns.map((c,j)=>[c.id,j]))
 function visit(c:WorksheetColumn){if(done.has(c.id))return;if(active.has(c.id))fail(`Циклическая зависимость формул: ${c.name}.`);active.add(c.id)
  if(c.formula){if(c.formula.version!==1)fail('Неизвестная версия формулы.');const refs=worksheetReferences(c.formula.ast);for(const id of refs.columns){const index=cols.get(id);if(index===undefined)fail(`Удалён столбец, нужный формуле «${c.name}».`);visit(sheet.columns[index])}
   const out=evaluateWorksheetExpression(c.formula.ast,work);const j=cols.get(c.id)!;out.forEach((v,r)=>{if(v!==null){if(c.type==='text'){if(typeof v!=='string')fail(`Формула «${c.name}»: результат должен быть текстом (TEXT/FORMATDATE).`)}else{numeric(v);if(c.type==='datetime')worksheetDate(v)}}rows[r][j]=v})
  }active.delete(c.id);done.add(c.id)
 }
 sheet.columns.forEach(visit);return rows
}
export function worksheetMatrix(operation:string,a:number[][],b?:number[][],scalar=1):number[][] {
 const check=(v:number[][])=>{if(!Array.isArray(v)||!v.length||!Array.isArray(v[0])||!v[0].length||v.some(r=>!Array.isArray(r)||r.length!==v[0].length||r.some(x=>typeof x!=='number'||!Number.isFinite(x))))fail('Матрица должна быть непустой, прямоугольной и числовой.');return fromRows(v)}
 const A=check(a),toRows=(m:Matrix)=>Array.from({length:m.rows},(_,i)=>Array.from(m.data.slice(i*m.cols,(i+1)*m.cols)))
 let out:number[][]
 if(operation==='transpose')out=toRows(transpose(A));else if(operation==='inverse')out=toRows(inverse(A));else if(operation==='multiply'){if(!b)fail('Нужна вторая матрица.');const B=check(b);if(A.cols!==B.rows)fail('Число столбцов A должно совпадать с числом строк B.');out=toRows(matmul(A,B))}
 else if(operation==='scale'){numeric(scalar);out=a.map(r=>r.map(x=>x*scalar))}
 else if(operation==='add'||operation==='subtract'){if(!b)fail('Нужна вторая матрица.');const B=check(b);if(A.rows!==B.rows||A.cols!==B.cols)fail('Матрицы должны иметь одинаковые размеры.');out=a.map((r,i)=>r.map((x,j)=>x+(operation==='add'?1:-1)*b![i][j]))}
 else fail('Неизвестная матричная операция.');out.forEach(r=>r.forEach(x=>valid(x)));return out
}
