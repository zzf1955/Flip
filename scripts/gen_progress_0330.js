const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10" x 5.625"
pres.author = "zzf621";
pres.title = "VideoEdit 进度汇报 2026.03.30";

// Minimalist palette
const BLACK = "1A1A1A";
const BODY = "4A4A4A";
const MUTED = "999999";
const LIGHT_LINE = "DDDDDD";
const PLACEHOLDER_BG = "F0F0F0";
const ACCENT = "2563EB"; // subtle blue for emphasis only

// ============================================================
// SLIDE 1: Title
// ============================================================
const s1 = pres.addSlide();
s1.background = { color: "FFFFFF" };

s1.addText("VideoEdit", {
  x: 0.8, y: 1.5, w: 8, h: 1.0,
  fontSize: 44, fontFace: "Arial Black", color: BLACK, bold: true, margin: 0,
});
s1.addText("仿人机器人视频生成", {
  x: 0.8, y: 2.4, w: 8, h: 0.6,
  fontSize: 28, fontFace: "Calibri", color: BLACK, margin: 0,
});
s1.addText("基于逆向生成数据合成的跨具身视频编辑", {
  x: 0.8, y: 3.1, w: 8, h: 0.5,
  fontSize: 16, fontFace: "Calibri", color: MUTED, margin: 0,
});

// thin rule
s1.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 3.8, w: 2, h: 0.02, fill: { color: BLACK },
});

s1.addText("2026.03.30", {
  x: 0.8, y: 4.0, w: 4, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: MUTED, margin: 0,
});

// ============================================================
// SLIDE 2: Research Status
// ============================================================
const s2 = pres.addSlide();
s2.background = { color: "FFFFFF" };

s2.addText("Robot-to-Human 研究现状", {
  x: 0.8, y: 0.35, w: 8, h: 0.55,
  fontSize: 28, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 0.95, w: 8.4, h: 0.01, fill: { color: LIGHT_LINE },
});

// Left column
s2.addText("机械臂方向（工作丰富）", {
  x: 0.8, y: 1.2, w: 4, h: 0.35,
  fontSize: 16, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s2.addText([
  { text: "Mitty (CVPR 2025)", options: { bold: true, fontSize: 14, color: BLACK, breakLine: true } },
  { text: "基于 Wan2.2 扩散模型", options: { bullet: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "人类视频 → 机器人视频，端到端生成", options: { bullet: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "无需关键点/轨迹等中间表征", options: { bullet: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "双向注意力融合人类和机器人特征", options: { bullet: true, fontSize: 13, color: BODY } },
], { x: 0.8, y: 1.65, w: 4, h: 2.2, paraSpaceAfter: 4 });

// Vertical divider
s2.addShape(pres.shapes.RECTANGLE, {
  x: 5.0, y: 1.2, w: 0.01, h: 2.8, fill: { color: LIGHT_LINE },
});

// Right column
s2.addText("人形机器人方向（工作较少）", {
  x: 5.3, y: 1.2, w: 4, h: 0.35,
  fontSize: 16, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s2.addText([
  { text: "X-Humanoid (2512)", options: { bold: true, fontSize: 14, color: BLACK, breakLine: true } },
  { text: "UE5 渲染合成 17h+ 训练数据", options: { bullet: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "Wan2.2 微调做 human → humanoid", options: { bullet: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "代码/数据未开源", options: { bullet: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "存在 sim2real gap", options: { bullet: true, fontSize: 13, color: BODY } },
], { x: 5.3, y: 1.65, w: 4.2, h: 2.2, paraSpaceAfter: 4 });

// Bottom summary
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 4.3, w: 8.4, h: 0.01, fill: { color: LIGHT_LINE },
});
s2.addText("人形机器人形态与人类更接近、匹配更自然；该方向研究空白大，是机会窗口", {
  x: 0.8, y: 4.5, w: 8.4, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: MUTED, italic: true, margin: 0,
});

// ============================================================
// SLIDE 3: Our Method
// ============================================================
const s3 = pres.addSlide();
s3.background = { color: "FFFFFF" };

s3.addText("方法：逆向数据合成 Pipeline", {
  x: 0.8, y: 0.35, w: 8, h: 0.55,
  fontSize: 28, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s3.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 0.95, w: 8.4, h: 0.01, fill: { color: LIGHT_LINE },
});

// Comparison table
const tblRows = [
  [
    { text: "", options: { fill: { color: "F5F5F5" }, bold: true, fontSize: 12, color: BLACK } },
    { text: "传统方向", options: { fill: { color: "F5F5F5" }, bold: true, fontSize: 12, color: BLACK } },
    { text: "我们的方向", options: { fill: { color: "F5F5F5" }, bold: true, fontSize: 12, color: BLACK } },
  ],
  [
    { text: "数据流", options: { bold: true, fontSize: 11, color: BODY } },
    { text: "人类视频 → 合成机器人视频", options: { fontSize: 11, color: BODY } },
    { text: "G1机器人视频 → 合成人类视频", options: { fontSize: 11, color: ACCENT, bold: true } },
  ],
  [
    { text: "输出端", options: { bold: true, fontSize: 11, color: BODY } },
    { text: "合成的，有 domain gap", options: { fontSize: 11, color: BODY } },
    { text: "真实 G1 视频，零 domain gap", options: { fontSize: 11, color: ACCENT, bold: true } },
  ],
  [
    { text: "数据增强", options: { bold: true, fontSize: 11, color: BODY } },
    { text: "1:1", options: { fontSize: 11, color: BODY } },
    { text: "1:N（多种人类外观）", options: { fontSize: 11, color: ACCENT, bold: true } },
  ],
];
s3.addTable(tblRows, {
  x: 0.8, y: 1.15, w: 8.4, h: 1.3,
  border: { pt: 0.5, color: LIGHT_LINE },
  colW: [1.5, 3.45, 3.45],
});

// Pipeline steps
s3.addText([
  { text: "Step 1 — 关键帧编辑", options: { bold: true, fontSize: 14, color: BLACK, breakLine: true } },
  { text: "G1 视频 → 提取关键帧 → Mask + Flux Fill Inpaint → 人类关键帧", options: { fontSize: 12, color: BODY, breakLine: true } },
  { text: "", options: { fontSize: 8, breakLine: true } },
  { text: "Step 2 — 视频合成", options: { bold: true, fontSize: 14, color: BLACK, breakLine: true } },
  { text: "G1 视频 → 提取控制信号 (depth/edge/seg) → Cosmos Transfer 2.5 + 关键帧锚定 → 合成人类视频", options: { fontSize: 12, color: BODY, breakLine: true } },
  { text: "", options: { fontSize: 8, breakLine: true } },
  { text: "Step 3 — 下游模型训练", options: { bold: true, fontSize: 14, color: BLACK, breakLine: true } },
  { text: "(合成人类视频, 真实G1视频) 配对 → Wan2.2 LoRA 微调 → Video-to-Video 编辑模型", options: { fontSize: 12, color: BODY } },
], { x: 0.8, y: 2.7, w: 8.4, h: 2.2, paraSpaceAfter: 2 });

// Advantages
s3.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 5.0, w: 8.4, h: 0.01, fill: { color: LIGHT_LINE },
});
s3.addText("核心优势：零域差输出 · 一对多增强 · 无需配对采集 · 无需3D资产", {
  x: 0.8, y: 5.1, w: 8.4, h: 0.35,
  fontSize: 13, fontFace: "Calibri", color: MUTED, margin: 0,
});

// ============================================================
// SLIDE 4: Data Status
// ============================================================
const s4 = pres.addSlide();
s4.background = { color: "FFFFFF" };

s4.addText("第三人称人形机器人数据现状", {
  x: 0.8, y: 0.35, w: 8, h: 0.55,
  fontSize: 28, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s4.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 0.95, w: 8.4, h: 0.01, fill: { color: LIGHT_LINE },
});

// First person
s4.addText("第一人称视角数据", {
  x: 0.8, y: 1.2, w: 8, h: 0.4,
  fontSize: 16, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s4.addText([
  { text: "数量丰富（DROID, OXE 等），主要用于机器人控制", options: { bullet: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "视角不适合做图像编辑 — 看不到机器人全身", options: { bullet: true, fontSize: 13, color: BODY } },
], { x: 0.8, y: 1.65, w: 8.4, h: 0.7, paraSpaceAfter: 4 });

// Third person
s4.addText("第三人称视角数据", {
  x: 0.8, y: 2.5, w: 8, h: 0.4,
  fontSize: 16, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s4.addText([
  { text: "控制领域：几乎全是第一人称，极少第三人称", options: { bullet: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "姿态估计领域：找到 1 个可用数据集 — LEVERB", options: { bullet: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "  1080×1920 竖版视频，AV1 编码，G1 室内操作任务", options: { fontSize: 12, color: MUTED, breakLine: true } },
  { text: "其他数据集：多为固定机械臂（Franka / UR5），非人形", options: { bullet: true, fontSize: 13, color: BODY } },
], { x: 0.8, y: 2.95, w: 8.4, h: 1.3, paraSpaceAfter: 4 });

// Conclusion
s4.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 4.5, w: 8.4, h: 0.01, fill: { color: LIGHT_LINE },
});
s4.addText("结论：第三人称人形机器人数据极度稀缺 → 凸显数据合成方法的必要性和价值", {
  x: 0.8, y: 4.7, w: 8.4, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});

// ============================================================
// SLIDE 5: Results (placeholder for images)
// ============================================================
const s5 = pres.addSlide();
s5.background = { color: "FFFFFF" };

s5.addText("初步结果：Flux Fill Mask 编辑", {
  x: 0.8, y: 0.35, w: 8, h: 0.55,
  fontSize: 28, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s5.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 0.95, w: 8.4, h: 0.01, fill: { color: LIGHT_LINE },
});

// Left group placeholder
s5.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 1.2, w: 3.8, h: 2.8,
  fill: { color: PLACEHOLDER_BG }, line: { color: LIGHT_LINE, width: 1 },
});
s5.addText("原始 G1 关键帧 → 编辑后人类图像\n（在此插入第 1 组对比图）", {
  x: 0.8, y: 1.2, w: 3.8, h: 2.8,
  fontSize: 13, fontFace: "Calibri", color: MUTED,
  align: "center", valign: "middle",
});

// Right group placeholder
s5.addShape(pres.shapes.RECTANGLE, {
  x: 5.4, y: 1.2, w: 3.8, h: 2.8,
  fill: { color: PLACEHOLDER_BG }, line: { color: LIGHT_LINE, width: 1 },
});
s5.addText("原始 G1 关键帧 → 编辑后人类图像\n（在此插入第 2 组对比图）", {
  x: 5.4, y: 1.2, w: 3.8, h: 2.8,
  fontSize: 13, fontFace: "Calibri", color: MUTED,
  align: "center", valign: "middle",
});

// Parameters
s5.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 4.3, w: 8.4, h: 0.01, fill: { color: LIGHT_LINE },
});
s5.addText([
  { text: "编辑方法: ", options: { bold: true, fontSize: 12, color: BODY } },
  { text: "Flux Fill Inpaint (GGUF Q8)    ", options: { fontSize: 12, color: BODY } },
  { text: "参数: ", options: { bold: true, fontSize: 12, color: BODY } },
  { text: "steps=28, cfg=1.0, euler, denoise=0.85, guidance=30.0", options: { fontSize: 12, color: BODY, breakLine: true } },
  { text: "Prompt: ", options: { bold: true, fontSize: 12, color: BODY } },
  { text: "\"a human performing the same action, realistic, third person view\"", options: { fontSize: 12, color: MUTED, italic: true } },
], { x: 0.8, y: 4.45, w: 8.4, h: 0.8 });

// ============================================================
// SLIDE 6: Next Steps
// ============================================================
const s6 = pres.addSlide();
s6.background = { color: "FFFFFF" };

s6.addText("下一步计划", {
  x: 0.8, y: 0.35, w: 8, h: 0.55,
  fontSize: 28, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s6.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 0.95, w: 8.4, h: 0.01, fill: { color: LIGHT_LINE },
});

// P0
s6.addText("P0 — 核心可行性验证", {
  x: 0.8, y: 1.2, w: 8, h: 0.4,
  fontSize: 16, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s6.addText([
  { text: "关键帧编辑方法对比", options: { bullet: true, bold: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "  Flux Fill 整体 mask / Flux Fill 分段 mask / Qwen Edit 无 mask / Flux Dev img2img", options: { fontSize: 12, color: MUTED, breakLine: true } },
  { text: "评估指标：人体自然度、背景保留、人物交互合理性", options: { bullet: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "控制信号提取", options: { bullet: true, bold: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "  DepthAnything V2 (深度) + Canny (边缘) + SAM2 (分割)", options: { fontSize: 12, color: MUTED } },
], { x: 0.8, y: 1.65, w: 8.4, h: 1.6, paraSpaceAfter: 4 });

// P1
s6.addText("P1 — Pipeline 核心", {
  x: 0.8, y: 3.4, w: 8, h: 0.4,
  fontSize: 16, fontFace: "Calibri", color: BLACK, bold: true, margin: 0,
});
s6.addText([
  { text: "Cosmos Transfer 2.5 视频合成 + 关键帧锚定", options: { bullet: true, bold: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "  锚定策略：首帧 / 首尾帧 / 均匀多帧 (K=8/16/32)", options: { fontSize: 12, color: MUTED, breakLine: true } },
  { text: "控制信号组合消融", options: { bullet: true, bold: true, fontSize: 13, color: BODY, breakLine: true } },
  { text: "  depth only / +edge / +seg / SalientObject", options: { fontSize: 12, color: MUTED, breakLine: true } },
  { text: "不同编辑方法 × 视频合成质量的交叉对比", options: { bullet: true, fontSize: 13, color: BODY } },
], { x: 0.8, y: 3.85, w: 8.4, h: 1.5, paraSpaceAfter: 4 });

// Write file
const outPath = "/Users/zzf/share/VideoEdit/VideoEdit_Progress_0330.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("DONE: " + outPath);
}).catch((err) => {
  console.error("ERROR:", err);
});
