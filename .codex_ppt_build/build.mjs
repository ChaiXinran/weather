import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "D:/_Search/AIforScience/Rewritten/origin/OpenSTL/OpenSTL_雷达降水周进展_20260802.pptx";
const TMP = "D:/_Search/AIforScience/Rewritten/origin/OpenSTL/.codex_ppt_build";
const C = { navy:"#0B2A4A", blue:"#1769D2", cyan:"#45A6E8", pale:"#EAF4FF", pale2:"#F5F9FD", ink:"#132238", gray:"#5F7185", grid:"#D9E5F0", white:"#FFFFFF", green:"#16876C", orange:"#E47A2E", red:"#CF3E4B" };
const FONT = "Microsoft YaHei";
const p = Presentation.create({ slideSize:{ width:1280, height:720 } });

function box(slide, x,y,w,h, fill=C.white, radius="roundRect", line=C.grid, lw=1) {
  return slide.shapes.add({ geometry:radius, position:{left:x,top:y,width:w,height:h}, fill, line:{style:"solid",fill:line,width:lw}, borderRadius:"rounded-xl" });
}
function text(slide, value, x,y,w,h, size=22, color=C.ink, bold=false, align="left") {
  const s=slide.shapes.add({geometry:"textbox",position:{left:x,top:y,width:w,height:h},fill:"none",line:{style:"solid",fill:"none",width:0}});
  s.text=value; s.text.style={fontFamily:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:"middle",wrap:true}; return s;
}
function line(slide,x,y,w,color=C.blue,lw=4){slide.shapes.add({geometry:"rect",position:{left:x,top:y,width:w,height:lw},fill:color,line:{style:"solid",fill:color,width:0}});}
function base(title, kicker, section){
  const s=p.slides.add(); s.background.fill=C.white;
  text(s,kicker.toUpperCase(),72,34,300,28,15,C.blue,true);
  text(s,title,72,66,1120,62,34,C.navy,true);
  line(s,72,136,66,C.blue,5);
  text(s,section,72,680,480,20,12,"#8292A4",false);
  text(s,String(p.slides.items.length).padStart(2,"0"),1160,680,48,20,12,"#8292A4",true,"right");
  return s;
}
function bullet(slide, value, x,y,w, color=C.ink, size=20){
  slide.shapes.add({geometry:"ellipse",position:{left:x,top:y+11,width:8,height:8},fill:C.blue,line:{style:"solid",fill:C.blue,width:0}});
  text(slide,value,x+20,y,w-20,42,size,color,false);
}
function metric(slide,x,y,w,label,value,sub,color=C.blue){
  box(slide,x,y,w,132,C.pale2,"roundRect",C.grid,1);
  text(slide,value,x+18,y+16,w-36,50,32,color,true);
  text(slide,label,x+18,y+65,w-36,28,17,C.ink,true);
  text(slide,sub,x+18,y+94,w-36,23,14,C.gray,false);
}
function notes(slide, refs){slide.speakerNotes.textFrame.setText(`[Sources]\n${refs.join("\n")}`);}

// 1
{
 const s=p.slides.add(); s.background.fill=C.white;
 s.shapes.add({geometry:"rect",position:{left:0,top:0,width:1280,height:720},fill:C.pale2,line:{style:"solid",fill:C.pale2,width:0}});
 s.shapes.add({geometry:"rect",position:{left:0,top:0,width:32,height:720},fill:C.blue,line:{style:"solid",fill:C.blue,width:0}});
 text(s,"OPENSTL · WEEKLY PROGRESS",88,86,500,28,16,C.blue,true);
 text(s,"雷达降水临近预报\n周进展",88,160,690,150,52,C.navy,true);
 line(s,88,338,108,C.blue,6);
 text(s,"从强降水评估基线到显式运动建模诊断",88,372,700,42,23,C.gray,false);
 text(s,"2026.07.29 — 2026.08.02",88,548,400,28,18,C.navy,true);
 text(s,"BTH Radar · SimVP · ConvLSTM · Motion Evolution",88,584,600,26,15,C.gray,false);
 // restrained abstract radar arcs
 for(let i=0;i<4;i++) s.shapes.add({geometry:"arc",position:{left:830+i*48,top:136+i*48,width:310-i*48,height:310-i*48},fill:"none",line:{style:"solid",fill:i%2?C.cyan:C.blue,width:3}});
 text(s,"01",1160,680,48,20,12,"#8292A4",true,"right");
 notes(s,["Internal project records: .research/run_log.md"]);
}

// 2
{
 const s=base("本周完成了从数据协议到运动诊断的闭环","01 · 本周全景","WEEKLY OVERVIEW");
 const xs=[72,300,528,756,984]; const heads=["协议固化","评估升级","损失消融","基线对比","运动诊断"];
 const subs=["4 事件\n932 窗口","降水 + 空间\n对象 + 事件","MSE → Soft CSI","SimVP / ConvLSTM","方向 · 尺度\n门控"];
 line(s,126,315,980,C.grid,4);
 for(let i=0;i<5;i++){
   s.shapes.add({geometry:"ellipse",position:{left:xs[i]+62,top:280,width:34,height:34},fill:i===4?C.blue:C.white,line:{style:"solid",fill:C.blue,width:3}});
   text(s,String(i+1),xs[i]+62,280,34,34,15,i===4?C.white:C.blue,true,"center");
   text(s,heads[i],xs[i],342,158,32,19,C.navy,true,"center");
   text(s,subs[i],xs[i],382,158,60,16,C.gray,false,"center");
 }
 box(s,142,506,996,96,C.pale,"roundRect",C.pale,0);
 text(s,"核心进展",170,526,120,30,18,C.blue,true);
 text(s,"已定位下一阶段瓶颈：不是“有没有运动信号”，而是“何时、何处、以多大幅度移动”。",304,516,790,54,24,C.navy,true);
 notes(s,[".research/run_log.md",".research/project_manifest.yml"]);
}

// 3
{
 const s=base("最低像素误差，并不等于最佳强降水预报","02 · 评价体系","EVALUATION");
 text(s,"统一协议",72,178,210,30,20,C.blue,true);
 bullet(s,"10 帧历史输入 → 20 帧递归预测",72,220,470);
 bullet(s,"0–1 h / 1–2 h 分时段评估",72,268,470);
 bullet(s,"16 / 32 mm·h⁻¹ 两级强降水阈值",72,316,470);
 text(s,"四层指标",620,178,210,30,20,C.blue,true);
 const labels=[["降水","CSI · POD · FAR · Bias"],["空间","FSS · 质心误差"],["对象","IoU · 对象 POD/FAR"],["保持","面积 · 能量 · 强度"]];
 labels.forEach((a,i)=>{box(s,620,222+i*74,500,58,i===0?C.pale:C.pale2,"roundRect",C.grid,1);text(s,a[0],640,231+i*74,90,38,18,C.blue,true);text(s,a[1],744,231+i*74,350,38,18,C.ink,false);});
 box(s,72,480,480,116,C.navy,"roundRect",C.navy,0);
 text(s,"Checkpoint 选择原则",96,500,300,27,18,C.cyan,true);
 text(s,"以强降水 CSI 为主，联合约束 FAR、Bias 与空间误差。",96,530,410,48,22,C.white,true);
 notes(s,["docs/bth_radar_evaluation.md",".research/history/r1_5ep_results.md"]);
}

// 4
{
 const s=base("损失函数改造将强降水综合分数提升 84%","03 · SimVP 损失消融","LOSS ABLATION");
 s.charts.add("bar",{position:{left:70,top:185,width:720,height:390},categories:["MSE","Huber","强像素加权","+ Soft CSI"],series:[{name:"CSI score",values:[0.331,0.381,0.531,0.610],valuesFormatCode:"0.000",fill:C.blue,points:[{idx:0,fill:"#A9C8E8"},{idx:1,fill:"#80B4E6"},{idx:2,fill:C.cyan},{idx:3,fill:C.blue}]}],barOptions:{direction:"column",grouping:"clustered",gapWidth:55},hasLegend:false,xAxis:{textStyle:{fill:C.gray,fontSize:16},line:{style:"solid",fill:C.grid,width:1}},yAxis:{min:0,max:.7,majorUnit:.1,numberFormatCode:"0.0",textStyle:{fill:C.gray,fontSize:14},majorGridlines:{style:"solid",fill:C.grid,width:1}},dataLabels:{showValue:true,position:"outEnd",textStyle:{fill:C.navy,fontSize:16,bold:true}},chartFill:C.white,plotAreaFill:C.white,chartLine:{style:"solid",fill:C.white,width:0}});
 text(s,"主要增益来源",850,202,290,30,20,C.blue,true);
 metric(s,850,250,300,"强像素加权","+39%","0.381 → 0.531",C.green);
 metric(s,850,404,300,"Soft CSI","+15%","0.531 → 0.610",C.blue);
 text(s,"结论：采用分时段 Soft CSI 作为后续训练基线。",850,566,300,48,18,C.navy,true);
 notes(s,[".research/history/r2_ablation_results.md"]);
}

// 5
{
 const s=base("SimVP 达到 0.714，但召回提升伴随空间虚警","04 · SimVP 基线","SIMVP BASELINE");
 metric(s,72,180,250,"综合 CSI","0.714","epoch 3 · 10 epochs",C.blue);
 metric(s,342,180,250,"较 5-epoch","+5.24%","主要来自第二小时",C.green);
 metric(s,612,180,250,"1–2 h CSI@16","0.108","覆盖明显恢复",C.blue);
 metric(s,882,180,250,"1–2 h CSI@32","0.034","仍处于低位",C.orange);
 text(s,"收益",72,354,150,30,20,C.green,true);
 bullet(s,"第二小时覆盖、POD 与强度恢复",72,392,470);
 bullet(s,"Intensity ratio 1–2 h 恢复至 1.019",72,442,470);
 text(s,"代价",650,354,150,30,20,C.red,true);
 bullet(s,"第一小时 Bias@16/32：1.33 / 1.41",650,392,500);
 bullet(s,"第二小时 FAR@16/32：0.789 / 0.905",650,442,500);
 box(s,72,530,1060,72,C.pale,"roundRect",C.pale,0);
 text(s,"定位",96,549,80,30,17,C.blue,true); text(s,"当前候选中召回最完整，但尚未解决长时空间重叠与虚警。",180,541,900,44,22,C.navy,true);
 notes(s,[".research/history/r3_selected_epoch03_analysis.md"]);
}

// 6
{
 const s=base("ConvLSTM 改善局地重叠，仍未解决长时位移","05 · 循环基线","CONVLSTM");
 const cats=["0–1h CSI16","0–1h CSI32","1–2h CSI16","1–2h CSI32"];
 s.charts.add("bar",{position:{left:60,top:190,width:720,height:380},categories:cats,series:[{name:"SimVP R3",values:[.3122,.2258,.1084,.0339],valuesFormatCode:"0.000",fill:"#9ABFE2"},{name:"ConvLSTM 0.788",values:[.3347,.2350,.1098,.0544],valuesFormatCode:"0.000",fill:C.blue}],barOptions:{direction:"column",grouping:"clustered",gapWidth:65},hasLegend:true,legend:{position:"bottom",textStyle:{fill:C.gray,fontSize:14}},xAxis:{textStyle:{fill:C.gray,fontSize:14}},yAxis:{min:0,max:.4,majorUnit:.1,numberFormatCode:"0.0",majorGridlines:{style:"solid",fill:C.grid,width:1},textStyle:{fill:C.gray,fontSize:13}},dataLabels:{showValue:true,position:"outEnd",textStyle:{fill:C.navy,fontSize:13,bold:true}},chartFill:C.white,plotAreaFill:C.white,chartLine:{style:"solid",fill:C.white,width:0}});
 text(s,"改善",850,194,180,28,20,C.green,true);
 bullet(s,"FSS 与匹配对象 IoU",850,232,330,C.ink,18);
 bullet(s,"第一小时强降水 CSI",850,276,330,C.ink,18);
 text(s,"未解决",850,346,180,28,20,C.red,true);
 bullet(s,"质心误差仍为 84–88 km",850,384,330,C.ink,18);
 bullet(s,"第二小时 Bias@32 达 2.88",850,428,330,C.ink,18);
 bullet(s,"事件鲁棒性弱于 SimVP",850,472,330,C.ink,18);
 box(s,842,540,330,66,C.pale,"roundRect",C.pale,0); text(s,"保留为运动分支骨干，不直接替代 SimVP。",864,550,286,44,18,C.navy,true);
 notes(s,[".research/history/convlstm_r2d_ft_checkpoint_analysis.md"]);
}

// 7
{
 const s=base("显式运动改善定位，却让强降水结构快速衰减","06 · R4-b 运动分支","MOTION EVOLUTION");
 text(s,"空间定位收益",72,180,280,30,20,C.green,true);
 metric(s,72,224,255,"质心误差 @16","68.9 km","较 ConvLSTM 降 10.5 km",C.green);
 metric(s,347,224,255,"质心误差 @32","56.7 km","较 ConvLSTM 降 31.2 km",C.green);
 text(s,"结构保持代价",678,180,280,30,20,C.red,true);
 metric(s,678,224,220,"面积保持 @32","19.7%","多数强雨区消失",C.red);
 metric(s,918,224,220,"能量保持 @32","17.5%","强度持续衰减",C.red);
 line(s,72,410,1066,C.grid,2);
 text(s,"因果链",72,438,120,30,20,C.blue,true);
 const chain=["20 步递归插值","强雨边缘被平滑","面积与能量收缩","POD / CSI32 下降"];
 chain.forEach((v,i)=>{box(s,198+i*245,430,202,82,i===3?C.pale:C.pale2,"roundRect",i===3?C.blue:C.grid,i===3?2:1);text(s,v,214+i*245,446,170,48,18,i===3?C.blue:C.ink,true,"center");if(i<3)text(s,"→",402+i*245,450,38,36,28,C.cyan,true,"center");});
 box(s,72,552,1066,54,C.navy,"roundRect",C.navy,0);text(s,"仅靠搬运无法维持或生成强降水：后续仍需要门控与强度源汇机制。",96,560,1018,38,21,C.white,true,"center");
 notes(s,[".research/history/r4b_motion_pre0788_5ep_analysis.md"]);
}

// 8
{
 const s=base("运动方向有效，真正缺失的是“是否应该移动”","07 · 运动可辨识性","FLOW DIAGNOSTICS");
 const rows=[
   ["近静止 <0.2 px","0","不应搬运"],
   ["亚像素 0.2–0.5 px","0.5","需要减弱"],
   ["明显移动 0.5–1 px","1.0","保留完整流"],
   ["快速 / 困难 >1 px","1.25","可能低估幅度"]
 ];
 text(s,"真实运动类别",78,190,310,30,17,C.gray,true);text(s,"最佳流尺度 α",500,190,220,30,17,C.gray,true,"center");text(s,"诊断解释",820,190,270,30,17,C.gray,true);
 rows.forEach((r,i)=>{const y=230+i*76;box(s,72,y,1060,58,i===0?C.pale:C.pale2,"roundRect",C.grid,1);text(s,r[0],94,y+10,330,38,19,C.navy,true);text(s,r[1],518,y+10,180,38,22,i===0?C.red:C.blue,true,"center");text(s,r[2],820,y+10,280,38,18,C.ink,false);});
 metric(s,72,560,330,"移动对象方向正确率","86%–91%","正确半平面",C.green);
 box(s,426,550,706,92,C.navy,"roundRect",C.navy,0);text(s,"结论",450,578,82,28,17,C.cyan,true);text(s,"需要条件化运动置信度，\n而不是永久乘一个固定小尺度。",540,562,560,64,20,C.white,true);
 notes(s,[".research/history/r4b_flow_scale_persistent_object_diagnostic.md"]);
}

// 9
{
 const s=base("门控已学到排序信号，但尚未通过递归预报检验","08 · R4-b2 门控","MOTION GATING");
 const stages=[
   {x:72,t:"Gate-only",v:"≈ 0.94",d:"塌缩到接近全流",c:C.red},
   {x:350,t:"Oracle 预训练",v:"0.57 → 0.71",d:"近静止与移动可分",c:C.green},
   {x:628,t:"Scale 0.5",v:"CSI 0.641",d:"FAR 下降、POD 下降",c:C.blue},
   {x:906,t:"第二小时 @32",v:"CSI 0.009",d:"强雨生存仍失败",c:C.red}
 ];
 stages.forEach(a=>{box(s,a.x,200,248,210,C.pale2,"roundRect",C.grid,1);text(s,a.t,a.x+18,220,212,28,18,C.navy,true);text(s,a.v,a.x+18,270,212,50,28,a.c,true);line(s,a.x+18,334,58,a.c,4);text(s,a.d,a.x+18,352,212,38,17,C.gray,false);});
 text(s,"决策门",72,462,150,30,20,C.blue,true);
 const checks=[["移动对象优于零流",true],["近静止对象不过移",false],["第二小时 CSI32 不下降",false]];
 checks.forEach((a,i)=>{const y=508+i*48;s.shapes.add({geometry:"ellipse",position:{left:76,top:y+4,width:28,height:28},fill:a[1]?C.green:C.white,line:{style:"solid",fill:a[1]?C.green:C.red,width:2}});text(s,a[1]?"✓":"×",76,y+1,28,32,18,a[1]?C.white:C.red,true,"center");text(s,a[0],120,y,410,34,18,C.ink,a[1]);});
 box(s,630,484,502,126,C.pale,"roundRect",C.blue,2);text(s,"R4-c 暂不进入",656,502,450,32,22,C.blue,true);text(s,"先让门控在递归预测中同时保护定位与强雨生存，再加入 source / sink。",656,542,440,52,18,C.navy,false);
 notes(s,[".research/history/r4a_r4b2_motion_gate_analysis.md",".research/history/r4b_motion_rainrate_scale05_5ep_analysis.md"]);
}

// 10
{
 const s=base("下周先解决运动置信度，再引入强度源汇机制","09 · 下一步","NEXT STEPS");
 const items=[
  ["01","增强门控输入","加入历史差分与固定光流线索"],
  ["02","条件化运动幅度","分样本、分时效或空间化置信度"],
  ["03","统一两类检验","Teacher-forced 与 20 步递归同步评估"],
  ["04","通过后进入 R4-c","加入 source / sink 或强度修正分支"]
 ];
 items.forEach((a,i)=>{const y=184+i*86;text(s,a[0],72,y,54,54,24,C.blue,true,"center");line(s,142,y+26,52,i===3?C.cyan:C.grid,3);text(s,a[1],218,y,300,30,20,C.navy,true);text(s,a[2],530,y,600,34,18,C.gray,false);});
 box(s,72,540,1060,90,C.navy,"roundRect",C.navy,0);
 text(s,"本周结论",96,570,130,30,17,C.cyan,true);
 text(s,"显式运动能够改善定位；下一步成败取决于门控能否同时保护\n近静止对象与长时强降水。",238,550,860,68,20,C.white,true);
 notes(s,[".research/run_log.md",".research/open_questions.md"]);
}

await fs.mkdir(TMP,{recursive:true});
for (const [i,s] of p.slides.items.entries()) {
 const png=await p.export({slide:s,format:"png",scale:1});
 await fs.writeFile(`${TMP}/slide-${String(i+1).padStart(2,"0")}.png`,new Uint8Array(await png.arrayBuffer()));
 const layout=await s.export({format:"layout"});
 await fs.writeFile(`${TMP}/slide-${String(i+1).padStart(2,"0")}.layout.json`,await layout.text());
}
const montage=await p.export({format:"webp",montage:true,scale:1});
await fs.writeFile(`${TMP}/montage.webp`,new Uint8Array(await montage.arrayBuffer()));
const pptx=await PresentationFile.exportPptx(p);
await pptx.save(OUT);
console.log(OUT);
