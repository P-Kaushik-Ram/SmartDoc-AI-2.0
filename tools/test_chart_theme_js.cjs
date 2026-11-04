const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let dark=false;
const listeners={},charts=[];
const canvas={dataset:{analyticsChart:'bar',chartData:'counts'}};
const data={labels:['Ready','Review'],values:[3,2]};
class Chart {
 constructor(element,config){Object.assign(this,config);charts.push(this);}
 update(mode){this.updated=mode;}
}
vm.runInNewContext(fs.readFileSync('static/core/js/app.js','utf8'),{
 document:{documentElement:{},getElementById:id=>id==='counts'?{textContent:JSON.stringify(data)}:null,
 querySelector:()=>null,querySelectorAll:selector=>selector==='[data-analytics-chart]'?[canvas]:[],
 addEventListener:(event,handler)=>listeners[event]=handler},
 window:{Chart},Chart,
 getComputedStyle:()=>({getPropertyValue:name=>(dark?'dark:':'light:')+name}),
});
assert.equal(charts.length,1);
const chart=charts[0];
assert.equal(chart.options.scales.x.ticks.color,'light:--text-secondary');
assert.equal(chart.options.plugins.tooltip.bodyColor,'light:--text-primary');
dark=true;listeners['smartdoc:themechange']();
assert.equal(chart.data.datasets[0].backgroundColor[0],'dark:--chart-1');
assert.equal(chart.options.scales.x.grid.color,'dark:--border');
assert.equal(chart.options.scales.y.ticks.color,'dark:--text-secondary');
assert.equal(chart.options.plugins.tooltip.bodyColor,'dark:--text-primary');
assert.equal(chart.options.plugins.tooltip.backgroundColor,'dark:--surface');
assert.equal(chart.updated,'none');
assert.deepEqual(JSON.parse(JSON.stringify(chart.data.datasets[0].data)),[3,2]);
console.log('PASS: recruitment chart palette, ticks, grid, tooltip surfaces/text update without changing data.');
