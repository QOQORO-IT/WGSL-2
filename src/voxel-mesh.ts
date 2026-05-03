// @ts-nocheck
/**
 * visible_block_faces mesh generation algorithm
 * Inspired by https://github.com/bonsairobo/block-mesh-rs
 */

const VOXEL_GRID_X = 256;
const VOXEL_GRID_Y = 128;
const VOXEL_GRID_Z = 256;
const VOXEL_SCALE = 0.125;

export interface CameraState {
  position: [number, number, number];
  yaw: number;
  pitch: number;
  fov: number;
}

export let camera: CameraState = {
  position: [16, 10, 22],
  yaw: -1.8,
  pitch: -0.35,
  fov: Math.PI / 2.2,
};

let keys: Record<string, boolean> = {};
let mouseLocked = false;

const VOXEL_COLORS: Record<number, [number, number, number, number]> = {
  1:  [0.35, 0.65, 0.25, 1.0],
  2:  [0.45, 0.30, 0.15, 1.0],
  3:  [0.55, 0.55, 0.50, 1.0],
  4:  [0.25, 0.55, 0.80, 0.7],
  5:  [0.45, 0.25, 0.10, 1.0],
  6:  [0.15, 0.40, 0.15, 1.0],
  7:  [0.95, 0.95, 1.00, 1.0],
  8:  [0.90, 0.80, 0.50, 1.0],
  9:  [0.60, 0.60, 0.55, 1.0],
  11: [0.75, 0.20, 0.15, 1.0],
  12: [0.95, 0.70, 0.85, 1.0],
  13: [0.80, 0.70, 0.25, 1.0],
  15: [0.95, 0.95, 0.90, 1.0],
};

function getVoxelColor(type: number): [number, number, number, number] {
  return VOXEL_COLORS[type] || [0.5, 0.5, 0.5, 1.0];
}

const FACE_NORMALS = [[-1,0,0],[0,-1,0],[0,0,-1],[1,0,0],[0,1,0],[0,0,1]];
const FACE_U_AXES = [[0,0,1],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[1,0,0]];
const FACE_V_AXES = [[0,1,0],[0,0,1],[0,1,0],[0,1,0],[0,0,1],[0,1,0]];
const NEIGHBOR_OFFSETS = [[-1,0,0],[0,-1,0],[0,0,-1],[1,0,0],[0,1,0],[0,0,1]];
const QUAD_LOCAL = [[0,0,0],[1,0,0],[0,1,0],[1,1,0]];
const QUAD_INDICES = [0,1,2,2,1,3];

function generateVoxelWorld(sx: number, sy: number, sz: number): Uint8Array {
  const data = new Uint8Array(sx * sy * sz);
  for (let x = 0; x < sx; x++) {
    for (let z = 0; z < sz; z++) {
      const height = Math.max(1, Math.min(sy - 4, Math.floor(
        Math.sin(x * 0.02) * Math.cos(z * 0.02) * 12 +
        Math.sin(x * 0.04) * Math.sin(z * 0.04) * 6 +
        Math.sin(x * 0.01 + z * 0.015) * 8 +
        Math.sin(x * 0.075 + z * 0.06) * 2 + 44
      )));
      for (let y = 0; y <= height; y++) {
        const idx = x * sy * sz + y * sz + z;
        if (y === height) data[idx] = y > 80 ? 7 : y < 20 ? 8 : 1;
        else if (y > height - 3) data[idx] = 2;
        else data[idx] = 3;
      }
      const waterLevel = 24;
      if (height < waterLevel) {
        for (let y = height + 1; y <= waterLevel; y++) {
          const idx = x * sy * sz + y * sz + z;
          if (idx < data.length) data[idx] = 4;
        }
      }
    }
  }
  buildJapaneseScene(data, sx, sy, sz);
  return data;
}

function generateVisibleBlockFaces(data: Uint8Array, sx: number, sy: number, sz: number) {
  const vertices: number[] = [];
  const indices: number[] = [];
  let vertexCount = 0;
  const minX = 1, maxX = sx - 2, minY = 1, maxY = sy - 2, minZ = 1, maxZ = sz - 2;

  for (let x = minX; x <= maxX; x++) {
    for (let y = minY; y <= maxY; y++) {
      for (let z = minZ; z <= maxZ; z++) {
        const idx = x * sy * sz + y * sz + z;
        const voxelType = data[idx];
        if (voxelType === 0) continue;
        const color = getVoxelColor(voxelType);

        for (let face = 0; face < 6; face++) {
          const [nx, ny, nz] = NEIGHBOR_OFFSETS[face];
          const nxIdx = (x + nx) * sy * sz + (y + ny) * sz + (z + nz);
          const neighborType = data[nxIdx];
          let faceNeedsMesh = neighborType === 0;
          if (!faceNeedsMesh && neighborType === 4 && voxelType !== 4) faceNeedsMesh = true;

          if (faceNeedsMesh) {
            const normal = FACE_NORMALS[face];
            const uAxis = FACE_U_AXES[face];
            const vAxis = FACE_V_AXES[face];
            for (let vi = 0; vi < 4; vi++) {
              const lp = QUAD_LOCAL[vi];
              let wx = x, wy = y, wz = z;
              if (face >= 3) { wx += normal[0]; wy += normal[1]; wz += normal[2]; }
              wx += lp[0] * uAxis[0] + lp[1] * vAxis[0];
              wy += lp[0] * uAxis[1] + lp[1] * vAxis[1];
              wz += lp[0] * uAxis[2] + lp[1] * vAxis[2];
              vertices.push(
                wx * VOXEL_SCALE, wy * VOXEL_SCALE, wz * VOXEL_SCALE,
                normal[0], normal[1], normal[2],
                color[0], color[1], color[2], color[3],
                lp[0], lp[1]
              );
            }
            for (let ii = 0; ii < 6; ii++) indices.push(vertexCount + QUAD_INDICES[ii]);
            vertexCount += 4;
          }
        }
      }
    }
  }
  return { vertices: new Float32Array(vertices), indices: new Uint32Array(indices), vertexCount, indexCount: indices.length };
}

// Scene builders (unchanged)
function buildJapaneseScene(data: Uint8Array, sx: number, sy: number, sz: number) {
  const cx = 128, cz = 128;
  const groundH = findGroundHeight(data, sx, sy, sz, cx, cz);
  buildPond(data, sx, sy, sz, cx - 40, cz + 20, 36, 24);
  buildStonePath(data, sx, sy, sz, cx, cz, cx - 20, cz + 40);
  buildToriiGate(data, sx, sy, sz, cx - 20, groundH + 1, cz + 40);
  buildTemple(data, sx, sy, sz, cx, groundH + 1, cz);
  buildLantern(data, sx, sy, sz, cx + 20, groundH + 1, cz + 8);
  buildLantern(data, sx, sy, sz, cx - 20, groundH + 1, cz + 8);
  buildLantern(data, sx, sy, sz, cx + 20, groundH + 1, cz - 8);
  buildLantern(data, sx, sy, sz, cx - 20, groundH + 1, cz - 8);
  buildFence(data, sx, sy, sz, cx, groundH + 1, cz, 28);
  buildWheatField(data, sx, sy, sz, cx - 60, cz - 50, 30, 40);
  buildWheatField(data, sx, sy, sz, cx + 30, cz - 60, 35, 35);
  buildSakuraTree(data, sx, sy, sz, cx + 35, findGroundHeight(data, sx, sy, sz, cx + 35, cz + 25), cz + 25);
  buildSakuraTree(data, sx, sy, sz, cx - 30, findGroundHeight(data, sx, sy, sz, cx - 30, cz + 35), cz + 35);
  buildSakuraTree(data, sx, sy, sz, cx + 45, findGroundHeight(data, sx, sy, sz, cx + 45, cz - 20), cz - 20);
  buildSakuraTree(data, sx, sy, sz, cx - 45, findGroundHeight(data, sx, sy, sz, cx - 45, cz - 15), cz - 15);
  buildSakuraTree(data, sx, sy, sz, cx - 55, findGroundHeight(data, sx, sy, sz, cx - 55, cz + 45), cz + 45);
  buildSakuraTree(data, sx, sy, sz, cx + 55, findGroundHeight(data, sx, sy, sz, cx + 55, cz + 50), cz + 50);
  buildPineTree(data, sx, sy, sz, cx - 50, findGroundHeight(data, sx, sy, sz, cx - 50, cz - 30), cz - 30);
  buildPineTree(data, sx, sy, sz, cx + 60, findGroundHeight(data, sx, sy, sz, cx + 60, cz - 40), cz - 40);
  buildSteps(data, sx, sy, sz, cx, groundH + 1, cz + 14, 6, 6);
}
function buildTemple(data: Uint8Array, sx: number, sy: number, sz: number, cx: number, baseY: number, cz: number) {
  for (let x = cx - 16; x <= cx + 16; x++) for (let z = cz - 14; z <= cz + 14; z++) for (let y = baseY; y <= baseY + 2; y++) { const idx = x * sy * sz + y * sz + z; if (idx < data.length) data[idx] = 9; }
  for (let x = cx - 14; x <= cx + 14; x++) for (let z = cz - 12; z <= cz + 12; z++) for (let y = baseY + 3; y <= baseY + 14; y++) { const idx = x * sy * sz + y * sz + z; if (idx >= data.length) continue; const isWall = x === cx - 14 || x === cx + 14 || z === cz - 12 || z === cz + 12; const isEntrance = z === cz + 12 && x >= cx - 3 && x <= cx + 3 && y < baseY + 10; if (isWall && !isEntrance) data[idx] = (y === baseY + 10 || y === baseY + 11) ? 9 : 5; }
  const pillars = [[cx - 14, cz - 12], [cx + 14, cz - 12], [cx - 14, cz + 12], [cx + 14, cz + 12]];
  for (const [px, pz] of pillars) for (let y = baseY; y <= baseY + 20; y++) { const idx = px * sy * sz + y * sz + pz; if (idx < data.length) data[idx] = 5; }
  for (let y = baseY + 15; y <= baseY + 18; y++) { const layer = y - (baseY + 15); const extend = 6 - layer; for (let x = cx - 14 - extend; x <= cx + 14 + extend; x++) for (let z = cz - 12 - extend; z <= cz + 12 + extend; z++) { const idx = x * sy * sz + y * sz + z; if (idx < data.length) data[idx] = 5; } }
  for (let x = cx - 10; x <= cx + 10; x++) for (let z = cz - 8; z <= cz + 8; z++) for (let y = baseY + 19; y <= baseY + 26; y++) { const idx = x * sy * sz + y * sz + z; if (idx >= data.length) continue; const isWall = x === cx - 10 || x === cx + 10 || z === cz - 8 || z === cz + 8; if (isWall) data[idx] = 5; }
  for (let y = baseY + 27; y <= baseY + 30; y++) { const layer = y - (baseY + 27); const extend = 5 - layer; for (let x = cx - 10 - extend; x <= cx + 10 + extend; x++) for (let z = cz - 8 - extend; z <= cz + 8 + extend; z++) { const idx = x * sy * sz + y * sz + z; if (idx < data.length) data[idx] = 5; } }
  for (let x = cx - 6; x <= cx + 6; x++) for (let z = cz - 4; z <= cz + 4; z++) for (let y = baseY + 31; y <= baseY + 34; y++) { const idx = x * sy * sz + y * sz + z; if (idx >= data.length) continue; const isWall = x === cx - 6 || x === cx + 6 || z === cz - 4 || z === cz + 4; if (isWall) data[idx] = 5; }
  for (let y = baseY + 35; y <= baseY + 38; y++) { const layer = y - (baseY + 35); const extend = 4 - layer; for (let x = cx - 6 - extend; x <= cx + 6 + extend; x++) for (let z = cz - 4 - extend; z <= cz + 4 + extend; z++) { const idx = x * sy * sz + y * sz + z; if (idx < data.length) data[idx] = 5; } }
  for (let y = baseY + 39; y <= baseY + 46; y++) { const idx = cx * sy * sz + y * sz + cz; if (idx < data.length) data[idx] = 8; }
  for (let y = baseY + 41; y <= baseY + 44; y += 2) for (let x = cx - 2; x <= cx + 2; x++) for (let z = cz - 2; z <= cz + 2; z++) { if (x === cx && z === cz) continue; const idx = x * sy * sz + y * sz + z; if (idx < data.length) data[idx] = 8; }
}
function buildToriiGate(data: Uint8Array, sx: number, sy: number, sz: number, cx: number, baseY: number, cz: number) {
  const leftPillar = cx - 6, rightPillar = cx + 6;
  for (let y = baseY; y <= baseY + 28; y++) { let idx = leftPillar * sy * sz + y * sz + cz; if (idx < data.length) data[idx] = 11; idx = (leftPillar + 1) * sy * sz + y * sz + cz; if (idx < data.length) data[idx] = 11; idx = rightPillar * sy * sz + y * sz + cz; if (idx < data.length) data[idx] = 11; idx = (rightPillar - 1) * sy * sz + y * sz + cz; if (idx < data.length) data[idx] = 11; }
  for (let x = leftPillar - 1; x <= leftPillar + 2; x++) for (let z = cz - 2; z <= cz + 2; z++) for (let y = baseY; y <= baseY + 1; y++) { const idx = x * sy * sz + y * sz + z; if (idx < data.length) data[idx] = 9; }
  for (let x = rightPillar - 2; x <= rightPillar + 1; x++) for (let z = cz - 2; z <= cz + 2; z++) for (let y = baseY; y <= baseY + 1; y++) { const idx = x * sy * sz + y * sz + z; if (idx < data.length) data[idx] = 9; }
  for (let x = leftPillar; x <= rightPillar; x++) { const y = baseY + 16; const idx = x * sy * sz + y * sz + cz; if (idx < data.length) data[idx] = 11; }
  for (let x = leftPillar - 4; x <= rightPillar + 4; x++) { const dx = x - cx; const curve = Math.abs(dx) * 0.3; const y = Math.floor(baseY + 24 + curve); const idx = x * sy * sz + y * sz + cz; if (idx < data.length) data[idx] = 11; const idx2 = x * sy * sz + (y + 1) * sz + cz; if (idx2 < data.length) data[idx2] = 11; }
  for (let x = cx - 2; x <= cx + 2; x++) for (let y = baseY + 20; y <= baseY + 23; y++) { const idx = x * sy * sz + y * sz + cz; if (idx < data.length) data[idx] = 15; }
}
function buildSakuraTree(data: Uint8Array, sx: number, sy: number, sz: number, tx: number, baseY: number, tz: number) {
  if (baseY <= 0 || baseY >= sy - 20) return;
  for (let h = 0; h < 14; h++) { const curveX = Math.floor(Math.sin(h * 0.2) * 1.5); const px = tx + curveX; const py = baseY + h; for (let ox = 0; ox <= 1; ox++) for (let oz = 0; oz <= 1; oz++) { const idx = (px + ox) * sy * sz + py * sz + (tz + oz); if (idx < data.length) data[idx] = 5; } }
  const branchStart = baseY + 8;
  for (let b = 0; b < 6; b++) { const angle = (b / 6) * Math.PI * 2; const len = 4 + Math.floor(((tx * 13 + tz * 7) % 17) / 17 * 4); const bx = Math.cos(angle); const bz = Math.sin(angle); for (let l = 0; l < len; l++) { const px = Math.floor(tx + 0.5 + bx * l); const pz = Math.floor(tz + 0.5 + bz * l); const py = branchStart + Math.floor(l * 0.5); const idx = px * sy * sz + py * sz + pz; if (idx < data.length) data[idx] = 5; } }
  const canopyCenterY = baseY + 12;
  for (let lx = tx - 10; lx <= tx + 10; lx++) for (let ly = canopyCenterY - 4; ly <= canopyCenterY + 6; ly++) for (let lz = tz - 10; lz <= tz + 10; lz++) { const dx = lx - tx; const dy = ly - canopyCenterY; const dz = lz - tz; const dist = Math.sqrt(dx * dx + dy * dy * 2.0 + dz * dz); if (dist <= 8) { const idx = lx * sy * sz + ly * sz + lz; if (idx < data.length && (data[idx] === 0 || data[idx] === 4)) { const isEdge = dist > 6; const isBranch = (((tx * 7 + tz * 13 + ly * 3) % 17) / 17) > 0.85; if (isBranch && !isEdge) data[idx] = 5; else data[idx] = 12; } } }
  for (let gx = tx - 8; gx <= tx + 8; gx++) for (let gz = tz - 8; gz <= tz + 8; gz++) { const gIdx = gx * sy * sz + (baseY - 1) * sz + gz; if (gIdx >= 0 && gIdx < data.length && data[gIdx] === 1) if ((((gx * 11 + gz * 7) % 19) / 19) > 0.92) data[gIdx] = 12; }
}
function buildPineTree(data: Uint8Array, sx: number, sy: number, sz: number, tx: number, baseY: number, tz: number) {
  if (baseY <= 0 || baseY >= sy - 20) return;
  for (let h = 0; h < 18; h++) { const idx = tx * sy * sz + (baseY + h) * sz + tz; if (idx < data.length) data[idx] = 5; }
  const tiers = [[16, 4], [13, 5], [10, 6], [7, 5]];
  for (const [h, r] of tiers) { const py = baseY + h; for (let lx = tx - r; lx <= tx + r; lx++) for (let lz = tz - r; lz <= tz + r; lz++) { const d = Math.sqrt((lx - tx) * (lx - tx) + (lz - tz) * (lz - tz)); if (d <= r) { const idx = lx * sy * sz + py * sz + lz; if (idx < data.length && (data[idx] === 0 || data[idx] === 4)) data[idx] = 6; } } }
}
function buildPond(data: Uint8Array, sx: number, sy: number, sz: number, cx: number, cz: number, rx: number, rz: number) {
  for (let x = cx - rx; x <= cx + rx; x++) for (let z = cz - rz; z <= cz + rz; z++) { const dx = (x - cx) / rx; const dz = (z - cz) / rz; if (dx * dx + dz * dz <= 1.0) { for (let y = 0; y < 30; y++) { const idx = x * sy * sz + y * sz + z; if (idx < data.length) { if (y < 18) data[idx] = 4; else if (y < 22) data[idx] = 8; } } if (dx * dx + dz * dz > 0.7 && (((x * 7 + z * 13 + 18 * 3) % 17) / 17) > 0.85) { const surfaceIdx = x * sy * sz + 18 * sz + z; if (surfaceIdx < data.length) data[surfaceIdx] = 9; } } }
}
function buildWheatField(data: Uint8Array, sx: number, sy: number, sz: number, fx: number, fz: number, fw: number, fl: number) {
  for (let x = fx; x < Math.min(sx, fx + fw); x++) for (let z = fz; z < Math.min(sz, fz + fl); z++) { const gh = findGroundHeight(data, sx, sy, sz, x, z); if (gh > 0 && data[x * sy * sz + gh * sz + z] === 1) { const height = 1 + Math.floor(Math.sin(x * 0.3) * Math.cos(z * 0.3) * 1.5 + 2); for (let h = 0; h < height; h++) { const idx = x * sy * sz + (gh + h) * sz + z; if (idx < data.length) data[idx] = 13; } } }
}
function buildStonePath(data: Uint8Array, sx: number, sy: number, sz: number, x1: number, z1: number, x2: number, z2: number) { const dx = x2 - x1; const dz = z2 - z1; const steps = Math.max(Math.abs(dx), Math.abs(dz)) * 2; for (let i = 0; i <= steps; i++) { const t = i / steps; const wx = Math.floor(x1 + dx * t); const wz = Math.floor(z1 + dz * t); const gh = findGroundHeight(data, sx, sy, sz, wx, wz); for (let ox = -1; ox <= 1; ox++) for (let oz = -1; oz <= 1; oz++) { const idx = (wx + ox) * sy * sz + gh * sz + (wz + oz); if (idx >= 0 && idx < data.length) data[idx] = ((wx + ox + wz + oz) % 3 === 0) ? 9 : 3; } } }
function buildSteps(data: Uint8Array, sx: number, sy: number, sz: number, cx: number, baseY: number, cz: number, width: number, count: number) { for (let i = 0; i < count; i++) { const z = cz + i; const y = baseY + i; for (let x = cx - width; x <= cx + width; x++) { const idx = x * sy * sz + y * sz + z; if (idx < data.length) data[idx] = 9; } } }
function buildFence(data: Uint8Array, sx: number, sy: number, sz: number, cx: number, baseY: number, cz: number, radius: number) { for (let angle = 0; angle < 360; angle += 6) { const rad = (angle * Math.PI) / 180; const px = Math.floor(cx + Math.cos(rad) * radius); const pz = Math.floor(cz + Math.sin(rad) * radius); for (let y = baseY; y <= baseY + 4; y++) { const idx = px * sy * sz + y * sz + pz; if (idx < data.length) data[idx] = 5; } } }
function buildLantern(data: Uint8Array, sx: number, sy: number, sz: number, lx: number, baseY: number, lz: number) { for (let y = baseY; y <= baseY + 1; y++) for (let x = lx - 1; x <= lx + 1; x++) for (let z = lz - 1; z <= lz + 1; z++) { const idx = x * sy * sz + y * sz + z; if (idx < data.length) data[idx] = 9; } for (let y = baseY + 2; y <= baseY + 5; y++) { const idx = lx * sy * sz + y * sz + lz; if (idx < data.length) data[idx] = 9; } for (let x = lx - 1; x <= lx + 1; x++) for (let z = lz - 1; z <= lz + 1; z++) for (let y = baseY + 6; y <= baseY + 8; y++) { const idx = x * sy * sz + y * sz + z; if (idx < data.length) data[idx] = (x === lx && z === lz) ? 15 : 9; } for (let x = lx - 2; x <= lx + 2; x++) for (let z = lz - 2; z <= lz + 2; z++) { const idx = x * sy * sz + (baseY + 9) * sz + z; if (idx < data.length) data[idx] = 9; } }
function findGroundHeight(data: Uint8Array, sx: number, sy: number, sz: number, x: number, z: number): number { for (let y = sy - 1; y >= 0; y--) { const idx = x * sy * sz + y * sz + z; if (data[idx] !== 0 && data[idx] !== 4) return y; } return 0; }

// ==================== WEBGPU MESH RENDERER ====================

export class VoxelMeshRenderer {
  device: GPUDevice | null = null;
  context: GPUCanvasContext | null = null;
  pipeline: GPURenderPipeline | null = null;
  vertexBuffer: GPUBuffer | null = null;
  indexBuffer: GPUBuffer | null = null;
  cameraBuffer: GPUBuffer | null = null;
  uniformBindGroup: GPUBindGroup | null = null;
  depthTexture: GPUTexture | null = null;
  depthTextureView: GPUTextureView | null = null;
  indexCount = 0;
  width = 0;
  height = 0;
  initialized = false;
  sunDirection: [number, number, number] = [0.3, -0.8, 0.5];

  async init(canvas: HTMLCanvasElement): Promise<boolean> {
    console.error("=== INIT START ===");
    if (!navigator.gpu) throw new Error("WebGPU is not supported in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("Failed to get WebGPU adapter.");
    this.device = await adapter.requestDevice({ requiredFeatures: [], requiredLimits: {} });
    console.error("Device acquired");
    this.context = canvas.getContext("webgpu");
    if (!this.context) throw new Error("Failed to get WebGPU context.");
    this.width = canvas.width; this.height = canvas.height;
    console.error("Canvas size:", this.width, this.height);
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: canvasFormat, alphaMode: "opaque", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    console.error("Context configured, format:", canvasFormat);

    console.time("generateVoxelWorld");
    const voxelData = generateVoxelWorld(VOXEL_GRID_X, VOXEL_GRID_Y, VOXEL_GRID_Z);
    console.timeEnd("generateVoxelWorld");
    console.time("generateVisibleBlockFaces");
    const mesh = generateVisibleBlockFaces(voxelData, VOXEL_GRID_X, VOXEL_GRID_Y, VOXEL_GRID_Z);
    console.timeEnd("generateVisibleBlockFaces");
    console.error(`Generated ${mesh.vertexCount} vertices, ${mesh.indexCount} indices, ${mesh.indexCount / 6} quads`);
    this.indexCount = mesh.indexCount;

    const vertexStride = 12 * 4;
    this.vertexBuffer = this.device.createBuffer({ label: "Vertex Buffer", size: mesh.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.vertexBuffer, 0, mesh.vertices);
    console.error("Vertex buffer created:", mesh.vertices.byteLength, "bytes");
    this.indexBuffer = this.device.createBuffer({ label: "Index Buffer", size: mesh.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.indexBuffer, 0, mesh.indices);
    console.error("Index buffer created:", mesh.indices.byteLength, "bytes");

    this.cameraBuffer = this.device.createBuffer({ label: "Camera Uniforms", size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    console.error("Fetching shader...");
    const response = await fetch("/mesh-shader.wgsl");
    console.error("Shader response status:", response.status);
    const shaderCode = await response.text();
    console.error("Shader loaded, length:", shaderCode.length);
    const shaderModule = this.device.createShaderModule({ label: "Mesh Shader", code: shaderCode });

    this.createDepthTexture();
    console.error("Depth texture created:", !!this.depthTextureView);

    const bindGroupLayout = this.device.createBindGroupLayout({ label: "Uniform Bind Group Layout", entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }] });
    this.uniformBindGroup = this.device.createBindGroup({ label: "Uniform Bind Group", layout: bindGroupLayout, entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }] });
    console.error("Bind group created");

    const pipelineLayout = this.device.createPipelineLayout({ label: "Pipeline Layout", bindGroupLayouts: [bindGroupLayout] });
    this.pipeline = this.device.createRenderPipeline({
      label: "Mesh Render Pipeline", layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: "vs_main", buffers: [{ arrayStride: vertexStride, attributes: [{ format: "float32x3", offset: 0, shaderLocation: 0 }, { format: "float32x3", offset: 12, shaderLocation: 1 }, { format: "float32x4", offset: 24, shaderLocation: 2 }, { format: "float32x2", offset: 40, shaderLocation: 3 }] }] },
      fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format: canvasFormat, blend: { color: { operation: "add", srcFactor: "one", dstFactor: "one-minus-src-alpha" }, alpha: { operation: "add", srcFactor: "one", dstFactor: "one-minus-src-alpha" } } }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    });
    console.error("Pipeline created");
    this.initialized = true;
    console.error("=== INIT COMPLETE ===");
    return true;
  }

  createDepthTexture() {
    if (!this.device) return;
    if (this.width <= 0 || this.height <= 0) { console.warn("Cannot create depth texture with size", this.width, this.height); return; }
    this.depthTexture = this.device.createTexture({ label: "Depth Texture", size: { width: this.width, height: this.height }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depthTextureView = this.depthTexture.createView();
  }

  updateCameraBuffer() {
    if (!this.device || !this.cameraBuffer) return;
    const aspect = this.width / this.height;
    const fov = camera.fov;
    const near = 0.1;
    const far = 200.0;
    const view = this.lookAt(camera.position, [camera.position[0] + Math.cos(camera.yaw) * Math.cos(camera.pitch), camera.position[1] + Math.sin(camera.pitch), camera.position[2] + Math.sin(camera.yaw) * Math.cos(camera.pitch)], [0, 1, 0]);
    const proj = this.perspective(fov, aspect, near, far);
    const viewProj = this.mat4Multiply(view, proj);
    const data = new Float32Array(20);
    data.set(viewProj, 0);
    data.set(this.sunDirection, 16);
    this.device.queue.writeBuffer(this.cameraBuffer, 0, data);
  }

  lookAt(eye: [number, number, number], target: [number, number, number], up: [number, number, number]): number[] {
    const z = this.normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
    const x = this.normalize(this.cross(up, z));
    const y = this.cross(z, x);
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -this.dot(x, eye), -this.dot(y, eye), -this.dot(z, eye), 1];
  }

  perspective(fov: number, aspect: number, near: number, far: number): number[] {
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1.0 / (near - far);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, near * far * nf, 0];
  }

  mat4Multiply(a: number[], b: number[]): number[] {
    const out = new Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { out[i * 4 + j] = 0; for (let k = 0; k < 4; k++) out[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j]; }
    return out;
  }

  normalize(v: [number, number, number]): [number, number, number] { const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); return [v[0] / len, v[1] / len, v[2] / len]; }
  cross(a: [number, number, number], b: [number, number, number]): [number, number, number] { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  dot(a: [number, number, number], b: [number, number, number]): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

  render() {
    if (!this.initialized || !this.device || !this.context || !this.pipeline || !this.vertexBuffer || !this.indexBuffer || !this.uniformBindGroup || !this.depthTextureView) {
      console.error("RENDER BLOCKED:", { initialized: this.initialized, device: !!this.device, context: !!this.context, pipeline: !!this.pipeline, vbuf: !!this.vertexBuffer, ibuf: !!this.indexBuffer, bindGroup: !!this.uniformBindGroup, depthView: !!this.depthTextureView });
      return;
    }
    try {
      this.updateCameraBuffer();
      const canvasTexture = this.context.getCurrentTexture();
      console.error("RENDERING FRAME", "canvas:", canvasTexture.width, "x", canvasTexture.height, "quads:", this.indexCount / 6);
      const commandEncoder = this.device.createCommandEncoder({ label: "Render Encoder" });
      const textureView = canvasTexture.createView();
      const renderPass = commandEncoder.beginRenderPass({
        label: "Render Pass",
        colorAttachments: [{ view: textureView, clearValue: { r: 0.4, g: 0.6, b: 0.9, a: 1 }, loadOp: "clear", storeOp: "store" }],
        depthStencilAttachment: { view: this.depthTextureView!, depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
      });
      renderPass.setPipeline(this.pipeline);
      renderPass.setBindGroup(0, this.uniformBindGroup);
      renderPass.setVertexBuffer(0, this.vertexBuffer);
      renderPass.setIndexBuffer(this.indexBuffer, "uint32");
      renderPass.drawIndexed(this.indexCount, 1, 0, 0, 0);
      renderPass.end();
      this.device.queue.submit([commandEncoder.finish()]);
      console.error("FRAME SUBMITTED OK");
    } catch (e) { console.error("RENDER EXCEPTION:", e); }
  }

  resize(width: number, height: number) { this.width = width; this.height = height; this.createDepthTexture(); }
}

export function setupInput(canvas: HTMLCanvasElement) {
  window.addEventListener("keydown", (e: KeyboardEvent) => { keys[e.code] = true; if (e.code === "Escape") { mouseLocked = false; document.exitPointerLock(); } });
  window.addEventListener("keyup", (e: KeyboardEvent) => { keys[e.code] = false; });
  canvas.addEventListener("click", () => { if (!mouseLocked) canvas.requestPointerLock(); });
  document.addEventListener("pointerlockchange", () => { mouseLocked = document.pointerLockElement === canvas; });
  document.addEventListener("mousemove", (e: MouseEvent) => { if (!mouseLocked) return; camera.yaw += e.movementX * 0.002; camera.pitch -= e.movementY * 0.002; camera.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, camera.pitch)); });
}

export function updateCamera(deltaTime: number) {
  const speed = 8.0 * deltaTime;
  const forward: [number, number, number] = [Math.cos(camera.yaw) * Math.cos(camera.pitch), Math.sin(camera.pitch), Math.sin(camera.yaw) * Math.cos(camera.pitch)];
  const right: [number, number, number] = [Math.cos(camera.yaw + Math.PI / 2), 0, Math.sin(camera.yaw + Math.PI / 2)];
  if (keys["KeyW"] || keys["ArrowUp"]) { camera.position[0] += forward[0] * speed; camera.position[1] += forward[1] * speed; camera.position[2] += forward[2] * speed; }
  if (keys["KeyS"] || keys["ArrowDown"]) { camera.position[0] -= forward[0] * speed; camera.position[1] -= forward[1] * speed; camera.position[2] -= forward[2] * speed; }
  if (keys["KeyA"] || keys["ArrowLeft"]) { camera.position[0] -= right[0] * speed; camera.position[2] -= right[2] * speed; }
  if (keys["KeyD"] || keys["ArrowRight"]) { camera.position[0] += right[0] * speed; camera.position[2] += right[2] * speed; }
  if (keys["Space"]) camera.position[1] += speed;
  if (keys["ShiftLeft"] || keys["ControlLeft"]) camera.position[1] -= speed;
  const worldSizeX = VOXEL_GRID_X * VOXEL_SCALE;
  const worldSizeY = VOXEL_GRID_Y * VOXEL_SCALE;
  const worldSizeZ = VOXEL_GRID_Z * VOXEL_SCALE;
  camera.position[0] = Math.max(0.1, Math.min(worldSizeX - 0.1, camera.position[0]));
  camera.position[1] = Math.max(0.1, Math.min(worldSizeY - 0.1, camera.position[1]));
  camera.position[2] = Math.max(0.1, Math.min(worldSizeZ - 0.1, camera.position[2]));
}
