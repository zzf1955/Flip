const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "zzf621";
pres.title = "VideoEdit 项目进展汇报";

// Color palette - Ocean Gradient + Dark theme
const C = {
  bg: "0B1120",        // deep dark
  bgCard: "131D35",    // card background
  accent: "00A8E8",    // bright blue accent
  accent2: "007EA7",   // teal
  accent3: "00CED1",   // turquoise
  white: "FFFFFF",
  gray: "94A3B8",
  lightBg: "F0F4F8",
  darkText: "1E293B",
  green: "10B981",
  yellow: "F59E0B",
  red: "EF4444",
  orange: "F97316",
};

const mkShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.3 });

// ============================================================
// SLIDE 1: Title
// ============================================================
let s1 = pres.addSlide();
s1.background = { color: C.bg };

// Decorative top bar
s1.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent }
});

// Large title
s1.addText("VideoEdit", {
  x: 0.8, y: 1.2, w: 8.4, h: 1.2,
  fontSize: 54, fontFace: "Arial Black", color: C.accent,
  bold: true, margin: 0
});

s1.addText("逆向生成数据合成 — 仿人机器人视频编辑系统", {
  x: 0.8, y: 2.3, w: 8.4, h: 0.6,
  fontSize: 20, fontFace: "Calibri", color: C.gray,
  margin: 0
});

// Separator line
s1.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 3.2, w: 2.5, h: 0.04, fill: { color: C.accent }
});

s1.addText("项目进展汇报", {
  x: 0.8, y: 3.5, w: 4, h: 0.6,
  fontSize: 28, fontFace: "Calibri", color: C.white, bold: true, margin: 0
});

s1.addText("2026.03.30", {
  x: 0.8, y: 4.3, w: 4, h: 0.4,
  fontSize: 16, fontFace: "Calibri", color: C.gray, margin: 0
});

// ============================================================
// SLIDE 2: Core Idea
// ============================================================
let s2 = pres.addSlide();
s2.background = { color: C.lightBg };

s2.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent }
});

s2.addText("核心思路", {
  x: 0.8, y: 0.3, w: 8, h: 0.6,
  fontSize: 32, fontFace: "Arial Black", color: C.darkText, bold: true, margin: 0
});

// Problem vs Solution - two columns
// Left column - traditional
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.2, w: 4.2, h: 3.6,
  fill: { color: "FFFFFF" }, shadow: mkShadow()
});
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.2, w: 4.2, h: 0.5, fill: { color: "DC2626" }
});
s2.addText("传统方向 (存在域差)", {
  x: 0.5, y: 1.2, w: 4.2, h: 0.5,
  fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle"
});
s2.addText([
  { text: "人类视频  →  合成机器人视频", options: { fontSize: 16, bold: true, color: C.darkText, breakLine: true } },
  { text: "", options: { fontSize: 10, breakLine: true } },
  { text: "输出是合成视频，与真实机器人视频存在域差", options: { fontSize: 13, color: "64748B", breakLine: true } },
  { text: "", options: { fontSize: 10, breakLine: true } },
  { text: "训练出的模型在真实场景表现受限", options: { fontSize: 13, color: "64748B" } }
], { x: 0.8, y: 1.95, w: 3.6, h: 2.5, valign: "top" });

// Right column - our approach
s2.addShape(pres.shapes.RECTANGLE, {
  x: 5.3, y: 1.2, w: 4.2, h: 3.6,
  fill: { color: "FFFFFF" }, shadow: mkShadow()
});
s2.addShape(pres.shapes.RECTANGLE, {
  x: 5.3, y: 1.2, w: 4.2, h: 0.5, fill: { color: C.accent2 }
});
s2.addText("本项目方向 (零域差)", {
  x: 5.3, y: 1.2, w: 4.2, h: 0.5,
  fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle"
});
s2.addText([
  { text: "G1机器人视频  →  合成人类视频", options: { fontSize: 16, bold: true, color: C.accent2, breakLine: true } },
  { text: "", options: { fontSize: 10, breakLine: true } },
  { text: "输出为真实G1视频，零输出域差", options: { fontSize: 13, color: "64748B", breakLine: true } },
  { text: "", options: { fontSize: 10, breakLine: true } },
  { text: "输入可以是不完美的合成人类视频", options: { fontSize: 13, color: "64748B", breakLine: true } },
  { text: "", options: { fontSize: 10, breakLine: true } },
  { text: "一对多数据增强：1段G1视频 → N种人类外观", options: { fontSize: 13, color: "64748B" } }
], { x: 5.6, y: 1.95, w: 3.6, h: 2.5, valign: "top" });

// Arrow between
s2.addText("VS", {
  x: 4.3, y: 2.6, w: 1.4, h: 0.5,
  fontSize: 18, fontFace: "Arial Black", color: C.accent, align: "center", valign: "middle"
});

// ============================================================
// SLIDE 3: Technical Roadmap (3 phases)
// ============================================================
let s3 = pres.addSlide();
s3.background = { color: C.bg };

s3.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent }
});

s3.addText("技术路线", {
  x: 0.8, y: 0.3, w: 8, h: 0.6,
  fontSize: 32, fontFace: "Arial Black", color: C.white, bold: true, margin: 0
});

// Three phase cards
const phases = [
  {
    title: "Phase 1", sub: "关键帧编辑",
    items: ["提取 G1 视频关键帧", "图像编辑替换为人类", "一对多外观增强"],
    color: C.green, status: "进行中"
  },
  {
    title: "Phase 2", sub: "视频合成",
    items: ["提取控制信号(深度/边缘)", "Cosmos Transfer 2.5", "关键帧锚定生成"],
    color: C.yellow, status: "计划中"
  },
  {
    title: "Phase 3", sub: "规模化",
    items: ["端到端自动化流水线", "批量处理多变体生成", "数据质量评估"],
    color: C.gray, status: "计划中"
  }
];

phases.forEach((p, i) => {
  const x = 0.5 + i * 3.15;
  // Card
  s3.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.2, w: 2.9, h: 3.8,
    fill: { color: C.bgCard }, shadow: mkShadow()
  });
  // Top accent
  s3.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.2, w: 2.9, h: 0.06, fill: { color: p.color }
  });
  // Phase title
  s3.addText(p.title, {
    x: x + 0.2, y: 1.45, w: 2.5, h: 0.4,
    fontSize: 20, fontFace: "Arial Black", color: p.color, bold: true, margin: 0
  });
  s3.addText(p.sub, {
    x: x + 0.2, y: 1.85, w: 2.5, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, margin: 0
  });
  // Items
  const textArr = p.items.map((item, j) => ({
    text: item,
    options: { bullet: true, color: C.gray, fontSize: 12, breakLine: j < p.items.length - 1 }
  }));
  s3.addText(textArr, {
    x: x + 0.2, y: 2.4, w: 2.5, h: 1.8,
    paraSpaceAfter: 8
  });
  // Status badge
  s3.addShape(pres.shapes.RECTANGLE, {
    x: x + 0.2, y: 4.35, w: 1.2, h: 0.35,
    fill: { color: p.color, transparency: 80 }
  });
  s3.addText(p.status, {
    x: x + 0.2, y: 4.35, w: 1.2, h: 0.35,
    fontSize: 11, fontFace: "Calibri", color: p.color, bold: true,
    align: "center", valign: "middle"
  });
});

// ============================================================
// SLIDE 4: Completed Work
// ============================================================
let s4 = pres.addSlide();
s4.background = { color: C.lightBg };

s4.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent }
});

s4.addText("已完成工作", {
  x: 0.8, y: 0.3, w: 8, h: 0.6,
  fontSize: 32, fontFace: "Arial Black", color: C.darkText, bold: true, margin: 0
});

const tasks = [
  {
    id: "Task 001", title: "LEVERB 关键帧提取",
    desc: "从 LEVERB 数据集提取关键帧\n4个视频片段 → 12张关键帧",
    file: "extract_keyframes.py", status: "Review"
  },
  {
    id: "Task 002", title: "Flux Fill 图像编辑",
    desc: "ComfyUI + Flux Fill Inpaint\n自动 mask / 手动 mask 两种模式",
    file: "comfyui_flux_inpaint.py", status: "Review"
  },
  {
    id: "Task 003", title: "Qwen 文本引导编辑",
    desc: "Qwen2.5-VL 文本引导编辑\n无需 mask，支持批量处理",
    file: "comfyui_qwen_edit.py", status: "Review"
  }
];

tasks.forEach((t, i) => {
  const y = 1.15 + i * 1.4;
  // Card
  s4.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: y, w: 9, h: 1.2,
    fill: { color: "FFFFFF" }, shadow: mkShadow()
  });
  // Left accent bar
  s4.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: y, w: 0.07, h: 1.2, fill: { color: C.green }
  });
  // Task ID
  s4.addText(t.id, {
    x: 0.8, y: y + 0.1, w: 1.5, h: 0.3,
    fontSize: 11, fontFace: "Calibri", color: C.accent2, bold: true, margin: 0
  });
  // Title
  s4.addText(t.title, {
    x: 0.8, y: y + 0.35, w: 3, h: 0.35,
    fontSize: 16, fontFace: "Calibri", color: C.darkText, bold: true, margin: 0
  });
  // Desc
  s4.addText(t.desc, {
    x: 0.8, y: y + 0.7, w: 4, h: 0.4,
    fontSize: 11, fontFace: "Calibri", color: "64748B", margin: 0
  });
  // File
  s4.addText(t.file, {
    x: 5.5, y: y + 0.4, w: 2.5, h: 0.35,
    fontSize: 11, fontFace: "Consolas", color: C.accent2, margin: 0
  });
  // Status badge
  s4.addShape(pres.shapes.RECTANGLE, {
    x: 8.2, y: y + 0.4, w: 1, h: 0.35,
    fill: { color: C.green, transparency: 85 }
  });
  s4.addText(t.status, {
    x: 8.2, y: y + 0.4, w: 1, h: 0.35,
    fontSize: 11, fontFace: "Calibri", color: C.green, bold: true,
    align: "center", valign: "middle"
  });
});

// ============================================================
// SLIDE 5: Data & Key Metrics
// ============================================================
let s5 = pres.addSlide();
s5.background = { color: C.bg };

s5.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent }
});

s5.addText("数据与关键指标", {
  x: 0.8, y: 0.3, w: 8, h: 0.6,
  fontSize: 32, fontFace: "Arial Black", color: C.white, bold: true, margin: 0
});

// Key metrics - big numbers
const metrics = [
  { num: "3", label: "数据集", sub: "LEVERB / First Humanoid\nPick & Place" },
  { num: "12", label: "已提取关键帧", sub: "4个视频片段\n每片段3帧" },
  { num: "2", label: "编辑方法", sub: "Flux Fill Inpaint\nQwen2.5 Text-guided" },
  { num: "~766", label: "代码行数", sub: "4个核心脚本\n+ 共享客户端库" },
];

metrics.forEach((m, i) => {
  const x = 0.5 + i * 2.35;
  s5.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.2, w: 2.1, h: 2.2,
    fill: { color: C.bgCard }, shadow: mkShadow()
  });
  s5.addText(m.num, {
    x: x, y: 1.35, w: 2.1, h: 0.8,
    fontSize: 48, fontFace: "Arial Black", color: C.accent3, bold: true,
    align: "center", valign: "middle"
  });
  s5.addText(m.label, {
    x: x, y: 2.15, w: 2.1, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: C.white, bold: true,
    align: "center", valign: "middle"
  });
  s5.addText(m.sub, {
    x: x, y: 2.55, w: 2.1, h: 0.7,
    fontSize: 10, fontFace: "Calibri", color: C.gray,
    align: "center", valign: "top"
  });
});

// Data table
s5.addText("数据集详情", {
  x: 0.8, y: 3.7, w: 4, h: 0.4,
  fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, margin: 0
});

const tableRows = [
  [
    { text: "数据集", options: { fill: { color: C.accent2 }, color: C.white, bold: true, fontSize: 11 } },
    { text: "大小", options: { fill: { color: C.accent2 }, color: C.white, bold: true, fontSize: 11 } },
    { text: "状态", options: { fill: { color: C.accent2 }, color: C.white, bold: true, fontSize: 11 } }
  ],
  [
    { text: "LEVERB", options: { fill: { color: C.bgCard }, color: C.gray, fontSize: 11 } },
    { text: "312 MB (4 chunks)", options: { fill: { color: C.bgCard }, color: C.gray, fontSize: 11 } },
    { text: "已提取关键帧", options: { fill: { color: C.bgCard }, color: C.green, fontSize: 11 } }
  ],
  [
    { text: "First Humanoid", options: { fill: { color: "0F1729" }, color: C.gray, fontSize: 11 } },
    { text: "113 MB (1 chunk)", options: { fill: { color: "0F1729" }, color: C.gray, fontSize: 11 } },
    { text: "待处理", options: { fill: { color: "0F1729" }, color: C.yellow, fontSize: 11 } }
  ],
  [
    { text: "Pick & Place", options: { fill: { color: C.bgCard }, color: C.gray, fontSize: 11 } },
    { text: "435 MB (1 chunk)", options: { fill: { color: C.bgCard }, color: C.gray, fontSize: 11 } },
    { text: "待处理", options: { fill: { color: C.bgCard }, color: C.yellow, fontSize: 11 } }
  ]
];

s5.addTable(tableRows, {
  x: 0.5, y: 4.15, w: 9, h: 1.2,
  border: { pt: 0.5, color: "1E3050" },
  colW: [3, 3, 3]
});

// ============================================================
// SLIDE 6: Next Steps
// ============================================================
let s6 = pres.addSlide();
s6.background = { color: C.lightBg };

s6.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent }
});

s6.addText("下一步计划", {
  x: 0.8, y: 0.3, w: 8, h: 0.6,
  fontSize: 32, fontFace: "Arial Black", color: C.darkText, bold: true, margin: 0
});

const nextItems = [
  {
    pri: "P0", title: "关键帧编辑方法对比实验",
    desc: "对比 Flux Fill、Qwen Edit 等多种方法的编辑质量，确定最优方案",
    color: C.red
  },
  {
    pri: "P0", title: "控制信号提取",
    desc: "从 G1 视频提取 depth、edge、segmentation 控制信号",
    color: C.red
  },
  {
    pri: "P1", title: "Cosmos Transfer 视频生成",
    desc: "关键帧锚定 + 控制信号引导，生成完整人类视频",
    color: C.orange
  },
  {
    pri: "P1", title: "编辑质量对视频生成的影响评估",
    desc: "不同质量关键帧输入对最终视频合成效果的影响",
    color: C.orange
  }
];

nextItems.forEach((item, i) => {
  const y = 1.15 + i * 1.05;
  // Card
  s6.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: y, w: 9, h: 0.85,
    fill: { color: "FFFFFF" }, shadow: mkShadow()
  });
  // Priority badge
  s6.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: y + 0.22, w: 0.6, h: 0.4,
    fill: { color: item.color, transparency: 85 }
  });
  s6.addText(item.pri, {
    x: 0.7, y: y + 0.22, w: 0.6, h: 0.4,
    fontSize: 12, fontFace: "Calibri", color: item.color, bold: true,
    align: "center", valign: "middle"
  });
  // Title
  s6.addText(item.title, {
    x: 1.5, y: y + 0.08, w: 7.5, h: 0.35,
    fontSize: 15, fontFace: "Calibri", color: C.darkText, bold: true, margin: 0
  });
  // Description
  s6.addText(item.desc, {
    x: 1.5, y: y + 0.45, w: 7.5, h: 0.3,
    fontSize: 11, fontFace: "Calibri", color: "64748B", margin: 0
  });
});

// ============================================================
// SLIDE 7: Thank you
// ============================================================
let s7 = pres.addSlide();
s7.background = { color: C.bg };

s7.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent }
});

s7.addText("Thank You", {
  x: 0, y: 1.5, w: 10, h: 1.2,
  fontSize: 54, fontFace: "Arial Black", color: C.accent,
  bold: true, align: "center", valign: "middle"
});

s7.addText("VideoEdit — 逆向生成数据合成", {
  x: 0, y: 2.8, w: 10, h: 0.5,
  fontSize: 18, fontFace: "Calibri", color: C.gray,
  align: "center", valign: "middle"
});

s7.addShape(pres.shapes.RECTANGLE, {
  x: 4.2, y: 3.6, w: 1.6, h: 0.04, fill: { color: C.accent }
});

s7.addText("Q & A", {
  x: 0, y: 3.9, w: 10, h: 0.5,
  fontSize: 20, fontFace: "Calibri", color: C.white, bold: true,
  align: "center", valign: "middle"
});

// Write file
const outPath = "/Users/zzf/share/VideoEdit/VideoEdit_Progress.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("DONE: " + outPath);
}).catch(err => {
  console.error("ERROR:", err);
});
