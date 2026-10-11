/** Plot-ready data and statistics for a visual graph builder. MIT.
 * Rendering/editing remain outside Columna; aggregation, quantiles, fits and CI stay here.
 */
import { descriptiveStats, boxplotStats, histogramBins, meanInterval, type DescriptiveStats, type BoxplotStats } from './descriptive.js'
import { normal } from './dist.js'
import { ols } from './regression.js'
export type GraphCell = number | string | null
export interface VizPoint {x:number;y:number|null;low?:number;high?:number;row?:number;label?:string}
export interface VizChart {
 type:'studio-viz';kind:string;title:string;xLabel:string;yLabel:string
 series:Array<{name:string;points:VizPoint[];line?:boolean;step?:boolean;dashed?:boolean}>
 categories?:string[];yCategories?:string[];boxes?:Array<{name:string;n:number;q1:number;median:number;q3:number;low:number;high:number;outliers:number[]}>
 bins?:Array<{from:number;to:number;count:number;group:string}>;cells?:Array<{x:number;y:number;value:number|null;n:number}>
 note?:string;xDate?:boolean;referenceLines?:Array<{axis:'x'|'y';value:number;label:string}>
}
export interface GraphInput {kind:string;data:Record<string,GraphCell[]>;labels?:Record<string,string>;types?:Record<string,string>;x?:string;y?:string[];value?:string;group?:string;panel?:string;aggregate?:'count'|'mean'|'sum'|'median';bins?:number;confidence?:number;fitLine?:boolean}
const finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v)
const d=(x:number[])=>descriptiveStats(x) as DescriptiveStats
const missing=(x:unknown)=>x===null||x===undefined||x===''
function groups(rows:number[],raw?:GraphCell[]){const m=new Map<string,number[]>();for(const row of rows){if(raw&&missing(raw[row]))continue;const key=raw?String(raw[row]):'Все наблюдения';if(!m.has(key))m.set(key,[]);m.get(key)!.push(row)}return [...m.entries()]}
const chart=(kind:string,title:string,xLabel:string,yLabel:string):VizChart=>({type:'studio-viz',kind,title,xLabel,yLabel,series:[]})
export function graphWorkbench(o:GraphInput){
 const kinds=['histogram','boxplot','scatter','line','bar','interval','heatmap','ecdf','dot','scatter-matrix']
 if(!kinds.includes(o.kind))throw new RangeError('Неизвестный вид графика.')
 const ys=[...new Set(o.y??[])], ids=[o.x,...ys,o.value,o.group,o.panel].filter(Boolean) as string[], cols=[...new Set(ids)]
 if(!cols.length||cols.some(id=>!Object.hasOwn(o.data,id)))throw new RangeError('Выберите переменные графика.')
 const n=o.data[cols[0]!]!.length;if(!n||cols.some(id=>o.data[id]!.length!==n))throw new RangeError('Длины столбцов должны совпадать и быть ненулевыми.')
 const bins=o.bins??12,conf=o.confidence??.95,aggregation=o.aggregate??'count'
 if(!Number.isInteger(bins)||bins<2||bins>100)throw new RangeError('Число интервалов: от 2 до 100.')
 if(!(conf>.5&&conf<1)||!['count','mean','sum','median'].includes(aggregation))throw new RangeError('Некорректный доверительный уровень или агрегат.')
 const label=(id:string)=>o.labels?.[id]??id
 const ensureNumeric=(id:string)=>{if(o.types?.[id]&& !['number','datetime'].includes(o.types[id]!))throw new RangeError(`«${label(id)}»: нужен числовой столбец.`);if(o.data[id]!.some(v=>!missing(v)&&!finite(v)))throw new RangeError(`«${label(id)}»: нечисловые данные.`)}
 if(!['bar','heatmap'].includes(o.kind)&&!ys.length)throw new RangeError('Выберите хотя бы одну переменную Y.')
 if(!['bar','heatmap'].includes(o.kind))ys.forEach(ensureNumeric)
 if(['scatter','scatter-matrix'].includes(o.kind)&&o.kind!=='scatter-matrix'&&!o.x)throw new RangeError('Для рассеяния выберите X.')
 if(['scatter','line'].includes(o.kind)&&o.x)ensureNumeric(o.x)
 if(['bar','heatmap'].includes(o.kind)&&!o.x)throw new RangeError('Выберите категориальную ось X.')
 if(o.kind==='heatmap'&&(!ys.length||ys[0]===o.x))throw new RangeError('Для heatmap нужны различные X и Y категории.')
 if(['bar','heatmap'].includes(o.kind)&&aggregation!=='count'){if(!o.value)throw new RangeError('Для агрегирования выберите числовые значения.');ensureNumeric(o.value)}
 const panels=groups(Array.from({length:n},(_,i)=>i),o.panel?o.data[o.panel]:undefined), charts:VizChart[]=[], notes:string[]=[], tables:Array<{title:string;headers:string[];rows:GraphCell[][]}>=[]
 if(!panels.length)throw new RangeError('Нет непустых панелей.')
 if(panels.length>50)throw new RangeError('Более 50 панелей. Создайте подмножество или выберите другую переменную панелей; данные не обрезаны.')
 if(o.kind==='scatter-matrix'&&(ys.length<2||ys.length>8))throw new RangeError('Матрица рассеяния: от 2 до 8 переменных.')
 let used=new Set<number>()
 for(const [panel,rows]of panels){
  const suffix=o.panel?' · '+label(o.panel)+' = '+panel:'', by=groups(rows,o.group?o.data[o.group]:undefined)
  if(by.length>100)throw new RangeError('Более 100 групп графика; выберите другую группировку.')
  if(['scatter','line','ecdf','scatter-matrix'].includes(o.kind)){
   const pairs=o.kind==='scatter-matrix'?ys.flatMap((y,j)=>ys.slice(0,j).map(x=>[x,y])):ys.map(y=>[o.x??'',y])
   for(const [x,y]of pairs){const c=chart(o.kind==='line'||o.kind==='ecdf'?'line':'scatter',`${o.kind==='ecdf'?'Эмпирическая CDF':label(y!)}${x?' / '+label(x):''}${suffix}`,o.kind==='ecdf'?label(y!):x?label(x):'Исходная строка',o.kind==='ecdf'?'Накопленная доля':label(y!));c.xDate=!!x&&o.types?.[x]==='datetime'
    for(const [group,ii]of by){let points:VizPoint[]=ii.map(i=>({x:x&&o.kind!=='ecdf'?Number(o.data[x]![i]):i+1,y:finite(o.data[y!]![i])?Number(o.data[y!]![i]):null,row:i+1}));if(x&&o.kind!=='ecdf')points=points.filter(p=>finite(o.data[x]![p.row!-1]));points=points.filter(p=>{if(p.y!==null)used.add(p.row!-1);return o.kind==='line'||p.y!==null})
     if(o.kind==='ecdf'){const vv=points.filter(p=>p.y!==null).sort((a,b)=>a.y!-b.y!);points=vv.map((p,j)=>({x:p.y!,y:(j+1)/vv.length,row:p.row}));c.xDate=o.types?.[y!]==='datetime'}
     if(points.some(p=>p.y!==null))c.series.push({name:o.group?group:label(y!),points,line:o.kind==='line'||o.kind==='ecdf',step:o.kind==='ecdf'})
     if(o.fitLine&&c.kind==='scatter'){const good=points.filter(p=>p.y!==null);if(good.length>=3&&d(good.map(p=>p.x)).sd>0&&d(good.map(p=>p.y!)).sd>0){const fit=ols(good.map(p=>p.y!),{X:good.map(p=>p.x)});const sx=d(good.map(p=>p.x));c.series.push({name:'OLS · '+group,points:[sx.min,sx.max].map(x=>({x,y:fit.coefficients[0]!.coef+fit.coefficients[1]!.coef*x})),line:true,dashed:true})}}
    }charts.push(c)
   }
  }else if(['boxplot','histogram','dot','interval'].includes(o.kind)){
   for(const y of ys){const c=chart(o.kind,`${label(y)}${suffix}`,o.kind==='histogram'?label(y):o.group?label(o.group):'Переменная',o.kind==='histogram'?'Частота':label(y));c.boxes=[];c.categories=[];c.bins=[]
    const ss:Array<{group:string;ii:number[];vv:number[]}>=by.map(([group,ii])=>({group,ii:ii.filter(i=>finite(o.data[y]![i])),vv:ii.map(i=>o.data[y]![i]).filter(finite)})).filter(g=>g.vv.length)
    const all=ss.flatMap(g=>g.vv);if(!all.length)throw new RangeError('Нет числовых значений для графика.')
    const template=histogramBins(all,bins)
    for(const [j,g]of ss.entries()){g.ii.forEach(i=>used.add(i));const summary=d(g.vv);c.categories.push(g.group);tables.push({title:label(y)+' · '+g.group+suffix,headers:['N','Среднее','SD','Минимум','Медиана','Максимум'],rows:[[summary.n,summary.mean,finite(summary.sd)?summary.sd:null,summary.min,summary.median,summary.max]]})
     if(o.kind==='boxplot'){const b=boxplotStats(g.vv) as BoxplotStats;c.boxes.push({name:g.group,n:summary.n,q1:b.q1,median:b.median,q3:b.q3,low:b.whiskerLow,high:b.whiskerHigh,outliers:b.outliers})}
     if(o.kind==='dot')c.series.push({name:g.group,points:g.ii.map((i,k)=>({x:j+((k*0.61803398875)%1-.5)*.35,y:Number(o.data[y]![i]),row:i+1}))})
     if(o.kind==='interval'){if(g.vv.length<2||!(summary.sd>0))throw new RangeError(`«${g.group}»: для t-интервала нужны минимум два наблюдения и SD > 0.`);const ci=meanInterval(g.vv,conf);c.series.push({name:g.group,points:[{x:j,y:summary.mean,low:ci[0],high:ci[1]}]})}
     if(o.kind==='histogram'){const counts=template.map(()=>0),bw=template[0]!.to-template[0]!.from;for(const v of g.vv){const k=Math.min(counts.length-1,Math.max(0,Math.floor((v-template[0]!.from)/bw)));counts[k]++}c.bins.push(...template.map((b,i)=>({...b,count:counts[i]!,group:g.group})));if(summary.sd>0){const pdf=normal(summary.mean,summary.sd);c.series.push({name:'Normal · '+g.group,line:true,points:Array.from({length:121},(_,i)=>{const x=template[0]!.from+(template.at(-1)!.to-template[0]!.from)*i/120;return {x,y:pdf.pdf(x)*g.vv.length*bw}})})}}
    }charts.push(c)
   }
  }else{
   const c=chart(o.kind,o.kind==='heatmap'?'Тепловая карта'+suffix:'Агрегаты по категориям'+suffix,label(o.x!),o.kind==='heatmap'?label(ys[0]!):aggregation==='count'?'Число наблюдений':`${aggregation}(${label(o.value!)})`), xcats:string[]=[],ycats:string[]=[], map=new Map<string,{x:number;y:number;group:string;v:number[];n:number}>()
   for(const i of rows){const xv=o.data[o.x!]![i],yv=o.kind==='heatmap'?o.data[ys[0]!]![i]:'',gv=o.group?o.data[o.group]![i]:'Все наблюдения';if(missing(xv)||missing(yv)&&o.kind==='heatmap'||missing(gv)||aggregation!=='count'&&!finite(o.data[o.value!]![i]))continue
    const xn=String(xv),yn=String(yv),gn=String(gv);if(!xcats.includes(xn))xcats.push(xn);if(!ycats.includes(yn))ycats.push(yn);const xi=xcats.indexOf(xn),yi=ycats.indexOf(yn),key=JSON.stringify([xi,yi,gn]);if(!map.has(key))map.set(key,{x:xi,y:yi,group:gn,v:[],n:0});const entry=map.get(key)!;entry.n++;if(aggregation!=='count')entry.v.push(Number(o.data[o.value!]![i]));used.add(i)
   }
   if(o.kind==='heatmap'&&o.group)throw new RangeError('Heatmap: используйте панели вместо цветовой группировки.')
   if(xcats.length*ycats.length>10000)throw new RangeError('Слишком много категориальных ячеек для визуализации; сгруппируйте данные.')
   c.categories=xcats;c.yCategories=ycats;c.cells=[];const tableRows:GraphCell[][]=[]
   for(const e of map.values()){const value=aggregation==='count'?e.n:aggregation==='mean'?d(e.v).mean:aggregation==='median'?d(e.v).median:d(e.v).sum;tableRows.push([xcats[e.x]!,ycats[e.y]!,e.group,e.n,value]);if(o.kind==='heatmap')c.cells.push({x:e.x,y:e.y,value,n:e.n});else{let series=c.series.find(s=>s.name===e.group);if(!series){series={name:e.group,points:[]};c.series.push(series)}series.points.push({x:e.x,y:value,label:xcats[e.x]})}}
   tables.push({title:c.title,headers:['X','Y','Группа','N','Значение'],rows:tableRows});charts.push(c)
  }
 }
 if(!used.size)throw new RangeError('Не осталось полных наблюдений для заданных ролей.')
 if(charts.length>100)throw new RangeError('Получается более 100 графиков; уменьшите число панелей или переменных.')
 for(const c of charts){
  if(['scatter','line','ecdf','interval','dot'].includes(c.kind))tables.push({title:c.title+' · данные графика',headers:['Серия','Исходная строка','X','Y','Нижняя граница','Верхняя граница'],rows:c.series.flatMap(s=>s.points.map(p=>[s.name,p.row??null,p.x,p.y,p.low??null,p.high??null]))})
  if(c.kind==='histogram')tables.push({title:c.title+' · частоты',headers:['Группа','От','До','Частота'],rows:(c.bins??[]).map(b=>[b.group,b.from,b.to,b.count])})
  if(c.kind==='boxplot')tables.push({title:c.title+' · квартили',headers:['Группа','N','Нижний ус','Q1','Медиана','Q3','Верхний ус','Выбросов'],rows:(c.boxes??[]).map(b=>[b.name,b.n,b.low,b.q1,b.median,b.q3,b.high,b.outliers.length])})
 }
 notes.push('Расчёт по всем выбранным данным. Пропуски исключены совместно для ролей каждой серии; в линиях отсутствующее Y оставляет разрыв. Цветовая группировка и панели не изменяют данные листа.')
 if(o.kind==='interval')notes.push(`${conf*100}% t-интервалы среднего, индивидуальные, без поправки за множественные сравнения; не эквивалентны post-hoc.`)
 if(o.kind==='histogram')notes.push('Нормальная кривая — описательный ориентир по среднему и выборочному SD, не проверка нормальности.')
 if(o.kind==='line')notes.push('Линии соединяют наблюдения по исходному порядку строк внутри группы; автоматическая сортировка по X не выполняется.')
 return {kind:o.kind,n:used.size,omitted:n-used.size,charts,tables,notes}
}
