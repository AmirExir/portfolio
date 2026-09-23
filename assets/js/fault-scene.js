/*
 * Signal microscope: illustrative electrical features become a classification.
 * Ia/Ib/Ic/Va/Vb/Vc match fault_classifier/fault_classifier_app.py. The repository
 * CSV has no sampling timestamps, so these designed traces are not reconstructed
 * recordings. Geometry, amplitudes, feature heights and output symbols are art;
 * they do not state a fault diagnosis, measured units, accuracy or confidence.
 * The host owns canvas sizing, DPR, visibility, motion preferences and scheduling.
 */
(() => {
  'use strict';

  const kit = window.ProjectSceneKit;
  if (!kit) return;
  const TAU = Math.PI * 2;
  const GOLD = kit.colors.gold;
  const ICE = kit.colors.ice;
  const IVORY = kit.colors.ivory;
  const MUTED = kit.colors.muted;
  const LABELS = Object.freeze({ signals: 'Signal window', features: 'Feature extraction', classified: 'Fault classification' });
  const KEYS = Object.keys(LABELS);
  const CHANNELS = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'];
  const FEATURE_HEIGHTS = [0.62, 1.04, 0.77, 1.25, 0.89, 0.51];
  const clamp = (n, a = 0, b = 1) => kit.clamp(n, a, b);
  const mix = kit.lerp;
  const smooth = kit.smooth;
  const p = (x, y, z = 0) => [x, y, z];
  const rgba = (hex, a) => {
    const n = Number.parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${n >> 8 & 255},${n & 255},${a})`;
  };

  /** Return the visible chapter; explicit selections never advance automatically. */
  function getStage({ time = 0, state = 'auto' } = {}) {
    if (Object.hasOwn(LABELS, state)) return state;
    const t = Number.isFinite(time) ? Math.max(0, time) % 12 : 0;
    return KEYS[Math.floor(t / 4)];
  }

  class Studio {
    constructor(ctx, width, height, time, pointerX, pointerY) {
      this.ctx = ctx;
      this.mobile = width <= 640;
      this.width = width;
      this.height = height;
      this.items = [];
      this.camera = kit.view(width, height, {
        extent: this.mobile ? 8.3 : 16.2, heightExtent: this.mobile ? 7.5 : 4.7,
        centerY: this.mobile ? 0.56 : 0.53, pointerX, pointerY, depthX: 0.13, depthY: 0.27,
      });
      this.scale = this.camera.scale;
      this.time = time;
    }

    project(point) {
      const screen = kit.project(point, this.camera);
      return [screen[0], screen[1], point[2] + point[1] * 0.25];
    }

    line(points, color = IVORY, alpha = 0.5, width = 0.8, order = 0) {
      const screen = points.map(point => this.project(point));
      this.items.push({ kind: 'line', points: screen, color, alpha, width, depth: screen.reduce((n, point) => n + point[2], 0) / screen.length + order });
    }

    face(points, color, alpha = 1, order = 0) {
      const screen = points.map(point => this.project(point));
      this.items.push({ kind: 'face', points: screen, color, alpha, depth: screen.reduce((n, point) => n + point[2], 0) / screen.length + order });
    }

    dot(point, color = GOLD, alpha = 1, radius = 2) {
      const screen = this.project(point);
      this.items.push({ kind: 'dot', point: screen, color, alpha, radius, depth: screen[2] + 0.2 });
    }

    box(x, y, z, width, height, depth, color = ICE, alpha = 1) {
      const x0 = x - width / 2, x1 = x + width / 2, z0 = z - depth / 2, z1 = z + depth / 2, top = y + height;
      this.face([p(x0,y,z1),p(x1,y,z1),p(x1,top,z1),p(x0,top,z1)], rgba(color, 0.105), alpha);
      this.face([p(x1,y,z0),p(x1,y,z1),p(x1,top,z1),p(x1,top,z0)], rgba(color, 0.055), alpha);
      this.face([p(x0,top,z0),p(x1,top,z0),p(x1,top,z1),p(x0,top,z1)], rgba(color, 0.19), alpha);
      this.line([p(x0,y,z1),p(x1,y,z1),p(x1,top,z1),p(x0,top,z1),p(x0,y,z1)], color, alpha * 0.7, 0.65, 0.003);
      this.line([p(x0,top,z1),p(x0,top,z0),p(x1,top,z0),p(x1,top,z1)], color, alpha * 0.7, 0.65, 0.003);
      this.line([p(x1,top,z0),p(x1,y,z0),p(x1,y,z1)], color, alpha * 0.3, 0.5);
    }

    text(label, point, color = MUTED, align = 'left', fontSize = 10) {
      const screen = this.project(point);
      this.items.push({ kind: 'text', point: screen, label, color, align, fontSize, depth: 100 });
    }

    render() {
      const ctx = this.ctx;
      this.items.sort((a, b) => a.depth - b.depth);
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      for (const item of this.items) {
        if (item.kind === 'text') {
          ctx.globalAlpha = 1; ctx.fillStyle = item.color;
          ctx.font = `400 ${item.fontSize}px "Avenir Next", system-ui, sans-serif`;
          ctx.textAlign = item.align; ctx.textBaseline = 'middle';
          ctx.fillText(item.label, item.point[0], item.point[1]);
          continue;
        }
        if (item.kind === 'dot') {
          ctx.fillStyle = item.color;
          ctx.globalAlpha = item.alpha * 0.06;
          ctx.beginPath();ctx.arc(item.point[0],item.point[1],item.radius*4,0,TAU);ctx.fill();
          ctx.globalAlpha = item.alpha * 0.2;
          ctx.beginPath();ctx.arc(item.point[0],item.point[1],item.radius*2,0,TAU);ctx.fill();
          ctx.globalAlpha = item.alpha;
          ctx.beginPath();ctx.arc(item.point[0],item.point[1],item.radius*0.6,0,TAU);ctx.fill();
          continue;
        }
        ctx.globalAlpha = item.alpha;
        ctx.beginPath();ctx.moveTo(item.points[0][0],item.points[0][1]);
        for(let i=1;i<item.points.length;i+=1)ctx.lineTo(item.points[i][0],item.points[i][1]);
        if(item.kind==='face'){ctx.closePath();ctx.fillStyle=item.color;ctx.fill();}
        else{ctx.strokeStyle=item.color;ctx.lineWidth=item.width;ctx.stroke();}
      }
      ctx.globalAlpha=1;
    }
  }

  function route(scene, points, color, alpha, progress, radius = 1.8) {
    scene.line(points, color, alpha * 0.15, 3.3);
    scene.line(points, color, alpha * 0.52, 0.75);
    const lengths = points.slice(1).map((point, i) => Math.hypot(...point.map((n, j) => n - points[i][j])));
    let distance = clamp(progress) * lengths.reduce((a, b) => a + b, 0);
    for (let i = 0; i < lengths.length; i += 1) {
      if (distance <= lengths[i] || i === lengths.length - 1) {
        const t = clamp(distance / Math.max(lengths[i], 0.001));
        scene.dot(points[i].map((n,j)=>mix(n,points[i+1][j],t)),color,alpha,radius);
        break;
      }
      distance -= lengths[i];
    }
  }

  function waveform(u, channel, time) {
    const phase = channel % 3 * TAU / 3;
    const carrier = Math.sin(u * TAU * 2.7 - time * 0.7 - phase);
    const window = Math.exp(-Math.pow((u - 0.63) / 0.095, 2));
    const disturbance = channel % 3 === 0 ? Math.sin(u * TAU * 14 - time * 0.5) * window * 0.52 : 0;
    return carrier * (channel < 3 ? 0.19 : 0.15) + disturbance;
  }

  function signals(scene, time, extraction) {
    const mobile = scene.mobile;
    const left = mobile ? -3.25 : -6.65;
    const right = mobile ? 2.88 : -2.35;
    const top = mobile ? 2.52 : 1.28;
    const spacing = mobile ? 0.32 : 0.48;
    const scan = 0.52 + Math.sin(time * 0.39) * 0.12;
    const scanX = mix(left, right, scan);
    const bottom = top - spacing * 5;
    const z = -0.28;
    scene.face([p(left-0.08,bottom-0.29,z),p(right+0.1,bottom-0.29,z),p(right+0.1,top+0.28,z),p(left-0.08,top+0.28,z)],'#12252b',0.37,-1);
    for(let guide=0;guide<=12;guide+=1){const x=mix(left,right,guide/12);scene.line([p(x,bottom-0.24,z),p(x,top+0.24,z)],MUTED,0.09,0.5);}
    const endpoints=[];
    for(let channel=0;channel<6;channel+=1){
      const color=channel<3?GOLD:ICE;
      const base=top-channel*spacing;
      scene.line([p(left,base,z),p(right,base,z)],color,0.13,0.55);
      const points=[];
      for(let sample=0;sample<=100;sample+=1){const u=sample/100;points.push(p(mix(left,right,u),base+waveform(u,channel,time)*(mobile?0.56:0.86),0));}
      scene.line(points,color,0.055,5.2);
      scene.line(points,color,0.28,1.9);
      scene.line(points,color,0.83,0.85);
      scene.text(CHANNELS[channel],p(left-0.14,base,0),color,'right',mobile?9:10);
      const tip=points[points.length-1];endpoints.push(tip);
      scene.dot(p(scanX,base+waveform(scan,channel,time)*(mobile?0.56:0.86),0.07),IVORY,0.75,1.5);
      const section=[];
      for(let sample=0;sample<=22;sample+=1){const u=scan-0.09+sample/22*0.18;section.push(p(mix(left,right,u),base+waveform(u,channel,time)*(mobile?0.56:0.86),0.06+extraction*0.7));}
      scene.line(section,color,0.26+extraction*0.6,1.2);
    }
    // A thin optical gate moves across all six lanes; its depth is visible at both ends.
    const half=(right-left)*0.09;
    scene.face([p(scanX-half,bottom-0.23,0.12),p(scanX+half,bottom-0.23,0.12),p(scanX+half,top+0.22,0.12),p(scanX-half,top+0.22,0.12)],rgba(ICE,0.045),1,0.03);
    for(const x of [scanX-half,scanX+half]){
      scene.line([p(x,bottom-0.3,-0.17),p(x,bottom-0.3,0.55),p(x,top+0.32,0.55),p(x,top+0.32,-0.17)],ICE,0.7,0.7);
    }
    for(const y of [bottom-0.3,top+0.32])scene.line([p(scanX-half,y,0.55),p(scanX+half,y,0.55)],IVORY,0.77,0.9);
    scene.text('I / V',p(left,bottom-0.65,0),MUTED,'left',mobile?8:9);
    return endpoints;
  }

  function features(scene, endpoints, extraction, time) {
    const mobile=scene.mobile;
    const center=mobile?-2.05:-0.53;
    const floor=mobile?-1.58:-1.15;
    const z=mobile?0.12:0;
    const amplitude=0.15+extraction*0.85;
    scene.line([p(center-0.89,floor,z+0.48),p(center+0.91,floor,z+0.48),p(center+0.91,floor,z-0.53),p(center-0.89,floor,z-0.53),p(center-0.89,floor,z+0.48)],ICE,0.34,0.65);
    const tips=[];
    for(let channel=0;channel<6;channel+=1){
      const color=channel<3?GOLD:ICE;
      const x=center-0.74+(channel%3)*0.64+(channel>=3?0.26:0);
      const depth=z+(channel<3?0.24:-0.26);
      const height=(mobile?1:1.3)*FEATURE_HEIGHTS[channel]*(0.38+extraction*0.62);
      scene.box(x,floor,depth,0.19,height,0.28,color,0.22+extraction*0.78);
      const top=p(x,floor+height+0.04,depth);
      tips.push(top);
      scene.dot(top,color,amplitude,1.7);
      const input=endpoints[channel];
      const routePoints=mobile
        ?[input,p(input[0]+0.09,input[1],-0.38),p(3.08,0.19,-0.38),p(center+1.19,0.19,-0.38),p(x,floor+height+0.24,depth),top]
        :[input,p(-1.88,input[1],-0.33),p(x-0.19,floor+height+0.24,depth),top];
      route(scene,routePoints,color,0.09+extraction*0.45,((time*0.17+channel*0.11)%1+1)%1,1.3);
    }
    // Lifted sample planes turn through depth as the feature window is extracted.
    for(let layer=0;layer<3;layer+=1){
      const yy=floor+1.78+layer*0.10;
      const dz=z-0.2-layer*0.26;
      scene.line([p(center-0.86,yy,dz),p(center+0.86,yy,dz),p(center+0.86,yy,dz+0.36),p(center-0.86,yy,dz+0.36),p(center-0.86,yy,dz)],layer===0?GOLD:ICE,extraction*(0.30-layer*0.065),0.65);
    }
    return {tips,center,floor};
  }

  function classifier(scene, feature, classification, time) {
    const mobile=scene.mobile;
    const x=mobile?0.68:3.07;
    const y=mobile?-0.91:-0.12;
    const radius=mobile?0.66:0.87;
    const depth=0.32;
    const outline=(z)=>[
      p(x-radius*0.63,y-radius,z),p(x+radius*0.63,y-radius,z),p(x+radius,y-radius*0.62,z),p(x+radius,y+radius*0.62,z),
      p(x+radius*0.63,y+radius,z),p(x-radius*0.63,y+radius,z),p(x-radius,y+radius*0.62,z),p(x-radius,y-radius*0.62,z),
    ];
    const back=outline(-depth),front=outline(depth);
    scene.face(front,'#172a2f',0.52);
    scene.line([...back,back[0]],ICE,0.27,0.7);
    scene.line([...front,front[0]],IVORY,0.55+classification*0.35,1.05);
    for(let i=0;i<8;i+=1)scene.line([back[i],front[i]],ICE,0.31,0.65);
    for(let stripe=0;stripe<5;stripe+=1){
      const offset=(stripe-2)*radius*0.27;
      const line=[p(x-radius*0.6,y+offset,depth+0.02),p(x-radius*0.22,y+offset,depth+0.02),p(x+radius*0.12,y+offset*0.54,depth+0.02),p(x+radius*0.61,y+offset*0.54,depth+0.02)];
      route(scene,line,stripe%2?GOLD:ICE,0.20+classification*0.6,((time*0.19+stripe*.16)%1+1)%1,1.3);
    }
    scene.text('CLASSIFIER',p(x,y-radius-0.36,depth),MUTED,'center',mobile?7.5:9);
    for(let channel=0;channel<6;channel+=1){
      const tip=feature.tips[channel];
      const entry=p(x-radius,y+(channel-2.5)*radius*0.21,-0.05);
      const path=[tip,p(feature.center+1.18,feature.floor+0.24+channel*0.19,-0.2),entry];
      route(scene,path,channel<3?GOLD:ICE,0.1+classification*0.57,((time*.17+channel*.09)%1+1)%1,1.3);
    }
    const outputX=mobile?2.61:5.65;
    for(let category=0;category<3;category+=1){
      const yy=y+(category-1)*(mobile?0.68:0.86);
      const tip=p(outputX,yy,0.05);
      const color=category===1?GOLD:ICE;
      const active=category===1?classification:0;
      const r=mobile?0.17:0.24;
      scene.line([p(outputX-r,yy,0.08),p(outputX,yy+r,0.08),p(outputX+r,yy,0.08),p(outputX,yy-r,0.08),p(outputX-r,yy,0.08)],color,0.2+active*0.72,0.8+active*.5);
      if(category===1){scene.face([p(outputX-r*.72,yy,0.07),p(outputX,yy+r*.72,0.07),p(outputX+r*.72,yy,0.07),p(outputX,yy-r*.72,0.07)],rgba(GOLD,.13),active);scene.dot(tip,GOLD,active,2.4);}
      route(scene,[p(x+radius,y,depth),p(x+radius+.24,y,0.12),p(outputX-.40,yy,0.12),tip],color,0.13+active*.8,((time*.22)%1+1)%1,1.8);
    }
    scene.text('CLASS',p(outputX,y-(mobile?1.03:1.30),0.1),MUTED,'center',mobile?7.5:9);
  }

  function atmosphere(ctx,width,height,scene) {
    kit.backdrop(ctx,width,height);
    const floor=scene.mobile?-2.24:-1.66;
    const left=scene.mobile?-3.6:-7.0, right=scene.mobile?3.6:6.4;
    for(let z=-1.4;z<=1.41;z+=0.4)scene.line([p(left,floor,z),p(right,floor,z)],MUTED,0.07,0.45,-20);
    for(let x=left;x<right;x+=0.6)scene.line([p(x,floor,-1.4),p(x,floor,1.4)],MUTED,0.05,0.45,-20);
  }

  /** Draw one decorative frame in CSS pixels; never starts a timer or touches DOM. */
  function draw(ctx,{width,height,time=0,pointerX=0,pointerY=0,state='auto'}={}) {
    if(!ctx||!Number.isFinite(width)||!Number.isFinite(height)||width<=0||height<=0)return;
    const t=Number.isFinite(time)?Math.max(0,time):0;
    const cycle=t%12;
    const stage=getStage({time:t,state});
    const selected=Object.hasOwn(LABELS,state);
    const extraction=selected?(stage==='signals'?.12:1):smooth((cycle-3.35)/1.4)*(1-smooth((cycle-11.3)/.7));
    const classification=selected?(stage==='classified'?1:.07):smooth((cycle-7.35)/1.4)*(1-smooth((cycle-11.3)/.7));
    const scene=new Studio(ctx,width,height,t,Number.isFinite(pointerX)?clamp(pointerX,-1,1):0,Number.isFinite(pointerY)?clamp(pointerY,-1,1):0);
    ctx.save();
    try{
      ctx.clearRect(0,0,width,height);
      atmosphere(ctx,width,height,scene);
      const endpoints=signals(scene,t,extraction);
      const feature=features(scene,endpoints,extraction,t);
      classifier(scene,feature,classification,t);
      scene.render();
    }finally{ctx.restore();}
  }

  window.FaultScene=Object.freeze({types:Object.freeze(['fault']),draw,getStage,labels:LABELS});
})();
