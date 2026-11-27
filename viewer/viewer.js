/* global DATA */
function startViewer(DATA) {

  "use strict";

  const meta = DATA.meta;
  const series = DATA.series;
  const agents = meta.agents;
  const A = agents.length;
  const N = series.resources.length;
  const FLOOR = meta.resource_floor;
  const GIFT = 0;
  const REPAY = 1;
  const TRADE = 2;
  const SPEEDS = [1, 2, 4, 8, 16];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const MONO = getComputedStyle(document.documentElement).getPropertyValue("--mono").trim();
  const SANS = getComputedStyle(document.documentElement).getPropertyValue("--sans").trim();
  const TWO_PI = Math.PI * 2;

  // Printed-ledger palette. Keep drawing colors here so the canvas reads like
  // ink and flat pigment instead of a glossy dashboard.
  const PALETTE = {
    ink: "#1c1b18",
    paper: "#f4efe6",
    paperDeep: "#efe8da",
    meadow: "#dce3ce",
    meadowPatch: "#cfd8be",
    meadowDeep: "#c3cfb2",
    town: "#f2e2ce",
    townEdge: "#ecd8bd",
    plaza: "#e9d8bc",
    plazaLine: "#d7c1a0",
    gold: "#c88a1e",
    coin: "#d99b26",
    water: "#2f6f73",
    sage: "#4e6b50",
    sageLight: "#6b8260",
    sageDeep: "#456046",
    sageDry: "#8a8a66",
    alert: "#b84a39",
    stone: "#8c857b",
    coffer: "#c79a62",
    cofferLid: "#d9b585",
    cofferInner: "#f7f2e7",
    hutWall: "#ede3d0",
    door: "#6b4b2f",
    skin: "#e6bd94",
    shadow: "#d8cebc",
    rule: "#d8d0c0",
    muted: "#666055",
    white: "#fbf7ef"
  };

  if (!A || !N) {
    document.getElementById("story").textContent = "There is nothing to replay.";
    return;
  }

  const world = document.getElementById("world");
  const worldWrap = document.getElementById("worldWrap");
  const ctx = world.getContext("2d");
  const worldNote = document.getElementById("worldNote");
  const tooltip = document.getElementById("tooltip");
  const shockBanner = document.getElementById("shockBanner");
  const storyEl = document.getElementById("story");
  const selectedPanel = document.getElementById("selectedPanel");
  const feed = document.getElementById("feed");
  const feedTick = document.getElementById("feedTick");
  const playBtn = document.getElementById("playBtn");
  const playIcon = document.getElementById("playIcon");
  const speedBtn = document.getElementById("speedBtn");
  const scrub = document.getElementById("scrub");
  const readout = document.getElementById("readout");
  const readoutTotal = document.getElementById("readoutTotal");
  const statIteration = document.getElementById("statIteration");
  const statAlive = document.getElementById("statAlive");
  const statStock = document.getElementById("statStock");
  const stockBar = document.getElementById("stockBar");
  const statCoins = document.getElementById("statCoins");
  const statGini = document.getElementById("statGini");
  const liveStatus = document.getElementById("liveStatus");
  const helpModal = document.getElementById("helpModal");
  const deathLegend = document.getElementById("deathLegend");

  const state = {
    t: 0,
    playing: false,
    speed: 4,
    selected: null,
    hovered: null,
    dragging: false,
    lastUiTick: -1,
    lastUiTime: 0
  };

  const naturalEnd = Math.max(0, N - 1);
  const extinctionTick = meta.extinction_iteration;
  const end = extinctionTick != null && extinctionTick >= 0
    ? Math.min(naturalEnd, extinctionTick)
    : naturalEnd;

  const initialTotal = Math.max(1, Number(series.total[0]) || meta.initial_resources * A);
  const maxCoins = (function () {
    let maximum = Math.max(1, meta.initial_resources);
    for (let t = 0; t < N; t += 1) {
      const row = series.resources[t];
      for (let i = 0; i < A; i += 1) {
        if (row[i] > maximum) maximum = row[i];
      }
    }
    return maximum;
  })();

  const eventsByTick = Array.from({ length: N }, function () { return []; });
  for (const event of DATA.events) {
    const tick = event[0] | 0;
    if (tick >= 0 && tick < N) eventsByTick[tick].push(event);
  }

  const deathsByTick = Array.from({ length: N }, function () { return []; });
  let hasDeaths = false;
  if (Array.isArray(DATA.deaths) && DATA.deaths.length) {
    for (const death of DATA.deaths) {
      if (death.tick >= 0 && death.tick < N) {
        deathsByTick[death.tick].push(death.index);
        hasDeaths = true;
      }
    }
  } else {
    for (let t = 1; t < N; t += 1) {
      for (let i = 0; i < A; i += 1) {
        if (series.resources[t - 1][i] > FLOOR && series.resources[t][i] <= FLOOR) {
          deathsByTick[t].push(i);
          hasDeaths = true;
        }
      }
    }
  }

  const shocksByTick = Array.from({ length: N }, function () { return []; });
  for (const shock of DATA.shocks) {
    const start = Math.max(0, shock.iteration);
    const stop = Math.min(N - 1, shock.iteration + shock.duration - 1);
    for (let t = start; t <= stop; t += 1) shocksByTick[t].push(shock);
  }

  // Concentric layout: the market square sits at the centre of an organic
  // town clearing, huts ring the square, and the commons meadow wraps the
  // outside. There are no fixed pathways; villagers walk directly.
  const CENTER = { x: 0.5, y: 0.5 };
  const TOWN = { rx: 0.425, ry: 0.405 };
  const HUT_RING = { rx: 0.305, ry: 0.278 };
  const MARKET = { x: 0.5, y: 0.5 };

  const homes = agents.map(function (_, index) {
    const spread = A === 1 ? 0 : (index / A) * TWO_PI;
    const angle = -Math.PI / 2 + spread + Math.sin(index * 2.399963 + 0.7) * 0.055;
    const radiusJitter = 1 + Math.sin(index * 1.618034 + 1.3) * 0.045;
    const x = CENTER.x + Math.cos(angle) * HUT_RING.rx * radiusJitter;
    const y = CENTER.y + Math.sin(angle) * HUT_RING.ry * radiusJitter;
    const inset = 0.10 + (index % 3) * 0.012;
    return {
      angle: angle,
      x: x,
      y: y,
      homeX: x + (CENTER.x - x) * inset,
      homeY: y + (CENTER.y - y) * inset
    };
  });

  // Berry bushes and ponds sit in the meadow around the outside of the town
  // clearing. Their ripeness and water level follow the recorded commons
  // percentage, so the outer ring thins out as the stock is spent.
  const commonsSites = (function buildCommonsSites() {
    const sites = [];

    // Main bushes with varied sizes and slightly uneven spacing.
    const bushCount = 11;
    for (let i = 0; i < bushCount; i += 1) {
      const angle = -Math.PI / 2 + (i / bushCount) * TWO_PI + (i % 3 - 1) * 0.085;
      const rx = 0.428 + ((i * 3) % 5) * 0.011;
      const ry = 0.408 + ((i * 5) % 4) * 0.010;
      sites.push({
        kind: "bush",
        x: CENTER.x + Math.cos(angle) * rx,
        y: CENTER.y + Math.sin(angle) * ry,
        size: 0.026 + ((i * 7) % 5) * 0.005,
        berries: 4 + (i % 4) * 2,
        threshold: 0.06 + (i % 6) * 0.055,
        tone: i % 3,
        seed: i * 1.37
      });
    }

    // A second scatter of smaller berry clusters between the main bushes.
    const clusterCount = 7;
    for (let i = 0; i < clusterCount; i += 1) {
      const angle = -Math.PI / 2 + ((i + 0.5) / clusterCount) * TWO_PI + 0.22;
      const rx = 0.442 + ((i * 2) % 4) * 0.010;
      const ry = 0.420 + ((i * 3) % 3) * 0.011;
      sites.push({
        kind: "bush",
        x: CENTER.x + Math.cos(angle) * rx,
        y: CENTER.y + Math.sin(angle) * ry,
        size: 0.013 + (i % 3) * 0.003,
        berries: 2 + (i % 3),
        threshold: 0.04 + (i % 5) * 0.05,
        tone: (i + 1) % 3,
        seed: 50 + i * 1.91
      });
    }

    const pondCount = 5;
    for (let i = 0; i < pondCount; i += 1) {
      const angle = -Math.PI / 2 + Math.PI * 0.27 +
        (i / pondCount) * TWO_PI + (i % 2 ? 0.055 : -0.04);
      const rx = 0.436 + ((i * 3) % 4) * 0.011;
      const ry = 0.416 + ((i * 2) % 3) * 0.011;
      sites.push({
        kind: "pond",
        x: CENTER.x + Math.cos(angle) * rx,
        y: CENTER.y + Math.sin(angle) * ry,
        size: 0.013 + ((i * 5) % 4) * 0.0035,
        threshold: 0.04 + (i % 4) * 0.05,
        seed: 30 + i * 2.11
      });
    }
    return sites;
  })();

  // A single pre-rendered coin sprite keeps the per-frame draw cheap.
  const coinSprite = document.createElement("canvas");
  coinSprite.width = 32;
  coinSprite.height = 32;
  (function buildCoinSprite() {
    const c = coinSprite.getContext("2d");
    c.fillStyle = PALETTE.gold;
    c.strokeStyle = PALETTE.ink;
    c.lineWidth = 2;
    c.beginPath();
    c.arc(16, 16, 13, 0, TWO_PI);
    c.fill();
    c.stroke();
    c.strokeStyle = "rgba(28, 27, 24, 0.55)";
    c.lineWidth = 1.2;
    c.beginPath();
    c.arc(16, 16, 8, 0, TWO_PI);
    c.stroke();
  })();

  // A brighter coin for the static box stacks. The moving coins keep the
  // slightly deeper ochre so the animation reads as motion, not decoration.
  const cofferCoinSprite = document.createElement("canvas");
  cofferCoinSprite.width = 32;
  cofferCoinSprite.height = 32;
  (function buildCofferCoinSprite() {
    const c = cofferCoinSprite.getContext("2d");
    c.fillStyle = PALETTE.coin;
    c.strokeStyle = PALETTE.ink;
    c.lineWidth = 1.6;
    c.beginPath();
    c.arc(16, 16, 13, 0, TWO_PI);
    c.fill();
    c.stroke();
    c.strokeStyle = "rgba(28, 27, 24, 0.35)";
    c.lineWidth = 1;
    c.beginPath();
    c.arc(16, 16, 8.5, 0, TWO_PI);
    c.stroke();
  })();

  let W = 0;
  let H = 0;
  let DPR = 1;
  let backgroundCanvas = null;
  let lastTime = performance.now();

  const targetCache = new Map();

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function smoothStep(value) {
    const x = clamp(value, 0, 1);
    return x * x * (3 - 2 * x);
  }

  function formatNumber(value, digits) {
    if (!Number.isFinite(value)) return "0";
    const decimals = digits === undefined ? (Math.abs(value) < 10 ? 1 : 0) : digits;
    return Number(value).toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: decimals
    });
  }

  function nameFor(index) {
    return index == null || !agents[index] ? "Unknown" : agents[index].label;
  }

  function aliveAt(index, tick) {
    return series.resources[clamp(tick, 0, N - 1)][index] > FLOOR;
  }

  function resourceAt(index, time) {
    const low = Math.floor(clamp(time, 0, N - 1));
    const high = Math.min(N - 1, low + 1);
    const fraction = clamp(time, 0, N - 1) - low;
    const a = series.resources[low][index];
    const b = series.resources[high][index];
    return a + (b - a) * fraction;
  }

  function totalAt(time) {
    const low = Math.floor(clamp(time, 0, N - 1));
    const high = Math.min(N - 1, low + 1);
    const fraction = clamp(time, 0, N - 1) - low;
    const a = series.total[low];
    const b = series.total[high];
    return a + (b - a) * fraction;
  }

  function stockRatio(time) {
    return clamp(totalAt(time) / initialTotal, 0, 1);
  }

  function screenPoint(point) {
    const padX = Math.min(56, W * 0.06);
    const padY = Math.min(48, H * 0.07);
    return {
      x: padX + point.x * Math.max(1, W - padX * 2),
      y: padY + point.y * Math.max(1, H - padY * 2)
    };
  }

  function unit() {
    return Math.min(W, H);
  }

  function agentRadius() {
    return clamp(unit() * 0.025, 9, 19);
  }

  function hutSize() {
    return clamp(unit() * 0.044, 18, 33);
  }

  function marketScale() {
    return clamp(unit() * 0.040, 16, 32);
  }

  function blobPath(context, cx, cy, rx, ry, seed, wobble) {
    const points = 12;
    const xs = [];
    const ys = [];
    for (let i = 0; i < points; i += 1) {
      const angle = (i / points) * TWO_PI;
      const radius = 1 +
        Math.sin(angle * 3 + seed) * wobble +
        Math.cos(angle * 5 + seed * 1.7) * wobble * 0.6;
      xs.push(cx + Math.cos(angle) * rx * radius);
      ys.push(cy + Math.sin(angle) * ry * radius);
    }
    context.beginPath();
    context.moveTo((xs[points - 1] + xs[0]) / 2, (ys[points - 1] + ys[0]) / 2);
    for (let i = 0; i < points; i += 1) {
      const next = (i + 1) % points;
      context.quadraticCurveTo(
        xs[i],
        ys[i],
        (xs[i] + xs[next]) / 2,
        (ys[i] + ys[next]) / 2
      );
    }
    context.closePath();
  }

  function visitorSpot(receiver, giver) {
    const home = homes[receiver];
    const inward = Math.atan2(CENTER.y - home.homeY, CENTER.x - home.homeX);
    const spread = ((giver * 0.61803398875) % 1) - 0.5;
    const angle = inward + spread * 1.7;
    const radius = 0.045 + (giver % 2) * 0.012;
    return {
      x: home.homeX + Math.cos(angle) * radius,
      y: home.homeY + Math.sin(angle) * radius * 0.74
    };
  }

  // Separate rings for buyers and sellers keep a crowded market legible.
  function marketSlot(index, seller) {
    const angle = -Math.PI / 2 + ((index + (seller ? 0.5 : 0)) / A) * TWO_PI;
    const rx = seller ? 0.09 : 0.062;
    const ry = seller ? 0.074 : 0.055;
    return {
      x: MARKET.x + Math.cos(angle) * rx,
      y: MARKET.y + Math.sin(angle) * ry
    };
  }

  function computeTargets(tick) {
    const x = new Float32Array(A);
    const y = new Float32Array(A);

    for (let i = 0; i < A; i += 1) {
      const home = homes[i];
      if (!aliveAt(i, tick)) {
        x[i] = home.x;
        y[i] = home.y;
      } else {
        x[i] = home.homeX + Math.sin(tick * 0.57 + i * 1.7) * 0.008;
        y[i] = home.homeY + Math.cos(tick * 0.43 + i * 1.3) * 0.006;
      }
    }

    const events = eventsByTick[tick] || [];
    for (const event of events) {
      const type = event[1];
      const from = event[2];
      const to = event[3];
      if (type === TRADE) {
        if (aliveAt(from, tick)) {
          const spot = marketSlot(from, false);
          x[from] = spot.x;
          y[from] = spot.y;
        }
        if (aliveAt(to, tick)) {
          const spot = marketSlot(to, true);
          x[to] = spot.x;
          y[to] = spot.y;
        }
      } else {
        if (aliveAt(from, tick)) {
          const spot = visitorSpot(to, from);
          x[from] = spot.x;
          y[from] = spot.y;
        }
        if (aliveAt(to, tick)) {
          x[to] = homes[to].homeX;
          y[to] = homes[to].homeY;
        }
      }
    }

    return { x: x, y: y };
  }

  function targetsFor(tick) {
    let entry = targetCache.get(tick);
    if (!entry) {
      if (targetCache.size > 24) targetCache.clear();
      entry = computeTargets(tick);
      targetCache.set(tick, entry);
    }
    return entry;
  }

  function positionAt(index, time) {
    const low = Math.floor(clamp(time, 0, N - 1));
    const high = Math.min(N - 1, low + 1);
    const fraction = smoothStep(clamp(time, 0, N - 1) - low);
    const a = targetsFor(low);
    const b = targetsFor(high);
    return screenPoint({
      x: a.x[index] + (b.x[index] - a.x[index]) * fraction,
      y: a.y[index] + (b.y[index] - a.y[index]) * fraction
    });
  }

  function eventTypeName(type) {
    if (type === TRADE) return "trade";
    if (type === REPAY) return "repayment";
    return "gift";
  }

  function eventIcon(type) {
    if (type === TRADE) return "T";
    if (type === REPAY) return "R";
    return "G";
  }

  function eventText(event) {
    const type = event[1];
    const from = event[2];
    const to = event[3];
    const amount = event[4];
    const quantity = event[5];
    if (type === TRADE) {
      return nameFor(from) + " bought " + formatNumber(quantity, 1) +
        " goods from " + nameFor(to) + " for " + formatNumber(amount, 1) + " coins";
    }
    if (type === REPAY) {
      return nameFor(from) + " repaid " + formatNumber(amount, 1) +
        " coins to " + nameFor(to);
    }
    return nameFor(from) + " gave " + formatNumber(amount, 1) +
      " coins to " + nameFor(to);
  }

  function recentEventFor(index, tick) {
    for (let t = tick; t >= Math.max(0, tick - 40); t -= 1) {
      const events = eventsByTick[t] || [];
      for (const event of events) {
        if (event[2] === index || event[3] === index) return event;
      }
    }
    return null;
  }

  function activeShocks(tick) {
    return shocksByTick[clamp(tick, 0, N - 1)] || [];
  }

  function shockLabel(shock) {
    if (shock.type === "scarcity") {
      return "scarcity: production x" + formatNumber(Math.max(0, 1 - shock.magnitude), 2);
    }
    return "information: reputation noise +" + formatNumber(shock.magnitude, 2);
  }

  function roundedRect(context, x, y, width, height, radius) {
    const r = Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2);
    context.beginPath();
    context.moveTo(x + r, y);
    context.arcTo(x + width, y, x + width, y + height, r);
    context.arcTo(x + width, y + height, x, y + height, r);
    context.arcTo(x, y + height, x, y, r);
    context.arcTo(x, y, x + width, y, r);
    context.closePath();
  }

  function drawTag(context, text, x, y, fill, color, border, mono) {
    context.save();
    context.font = "500 10px " + (mono ? MONO : SANS);
    const width = context.measureText(text).width + 12;
    context.fillStyle = fill || PALETTE.paper;
    context.strokeStyle = border || PALETTE.ink;
    context.lineWidth = 1;
    context.fillRect(x - width / 2, y - 8, width, 16);
    context.strokeRect(
      x - width / 2 + 0.5,
      y - 7.5,
      Math.max(1, width - 1),
      15
    );
    context.fillStyle = color || PALETTE.ink;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, x, y + 0.5);
    context.restore();
  }

  function drawHut(context, index, point) {
    const s = hutSize();
    const color = agents[index].color;
    context.save();
    context.translate(point.x, point.y);

    // Solid offset shadow, like a two-colour print.
    context.fillStyle = PALETTE.shadow;
    context.fillRect(-s * 0.5 + 2, -s * 0.25 + 2, s, s * 0.88);

    context.fillStyle = PALETTE.hutWall;
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1.5;
    context.fillRect(-s * 0.5, -s * 0.25, s, s * 0.88);
    context.strokeRect(-s * 0.5, -s * 0.25, s, s * 0.88);

    context.fillStyle = PALETTE.door;
    context.fillRect(-s * 0.13, s * 0.25, s * 0.26, s * 0.38);
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1;
    context.strokeRect(-s * 0.13, s * 0.25, s * 0.26, s * 0.38);

    context.fillStyle = PALETTE.gold;
    context.fillRect(-s * 0.38, s * 0.03, s * 0.15, s * 0.16);
    context.strokeRect(-s * 0.38, s * 0.03, s * 0.15, s * 0.16);

    // Roof, with the same solid offset as the wall.
    context.fillStyle = PALETTE.shadow;
    context.beginPath();
    context.moveTo(-s * 0.66 + 2, -s * 0.25 + 2);
    context.lineTo(2, -s * 0.92 + 2);
    context.lineTo(s * 0.66 + 2, -s * 0.25 + 2);
    context.closePath();
    context.fill();

    context.fillStyle = color;
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(-s * 0.66, -s * 0.25);
    context.lineTo(0, -s * 0.92);
    context.lineTo(s * 0.66, -s * 0.25);
    context.closePath();
    context.fill();
    context.stroke();

    context.strokeStyle = "rgba(28, 27, 24, 0.55)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(-s * 0.42, -s * 0.31);
    context.lineTo(0, -s * 0.78);
    context.stroke();

    context.restore();

  }

  function drawHutLabels(context, tick) {
    const s = hutSize();
    for (let index = 0; index < A; index += 1) {
      const point = screenPoint(homes[index]);
      const angle = homes[index].angle;
      const alive = aliveAt(index, tick);
      context.save();
      context.globalAlpha = alive ? 1 : 0.58;
      drawTag(context, agents[index].label,
        point.x + Math.cos(angle) * s * 1.02,
        point.y + Math.sin(angle) * s * 1.02,
        PALETTE.paper,
        alive ? PALETTE.ink : PALETTE.stone,
        alive ? PALETTE.ink : PALETTE.rule,
        true);
      context.restore();
    }
  }

  function drawMarket(context) {
    const p = screenPoint(MARKET);
    const s = marketScale();
    const open = meta.market_enabled;
    const width = s * 8.8;
    const height = s * 5.2;
    const left = p.x - width / 2;
    const top = p.y - height / 2;

    context.save();

    // A crisp stone square in the middle of the clearing.
    context.fillStyle = PALETTE.shadow;
    context.fillRect(left + 3, top + 3, width, height);
    context.fillStyle = PALETTE.plaza;
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1.5;
    context.fillRect(left, top, width, height);
    context.strokeRect(left, top, width, height);

    context.strokeStyle = PALETTE.plazaLine;
    context.lineWidth = 1;
    context.strokeRect(left + 7, top + 7, width - 14, height - 14);

    // Sparse cobble marks, kept flat and quiet.
    context.strokeStyle = "rgba(28, 27, 24, 0.18)";
    for (let i = 1; i < 5; i += 1) {
      const y = top + (height / 5) * i;
      context.beginPath();
      context.moveTo(left + 12, y);
      context.lineTo(left + width - 12, y);
      context.stroke();
    }
    for (let i = 1; i < 4; i += 1) {
      const x = left + (width / 4) * i;
      context.beginPath();
      context.moveTo(x, top + 10);
      context.lineTo(x, top + height - 10);
      context.stroke();
    }

    const awning = open
      ? [PALETTE.gold, PALETTE.water, PALETTE.alert]
      : [PALETTE.stone, "#9b9488", "#a39c90"];
    for (let k = -1; k <= 1; k += 1) {
      const x = p.x + k * s * 2.45;
      const y = p.y + Math.abs(k) * s * 0.16;

      // Stall table.
      context.fillStyle = PALETTE.coffer;
      context.strokeStyle = PALETTE.ink;
      context.lineWidth = 1.2;
      context.fillRect(x - s * 0.58, y - s * 0.06, s * 1.16, s * 0.58);
      context.strokeRect(x - s * 0.58, y - s * 0.06, s * 1.16, s * 0.58);

      // A little produce on the table.
      context.fillStyle = open ? PALETTE.sage : PALETTE.sageDry;
      for (let i = -1; i <= 1; i += 1) {
        context.beginPath();
        context.arc(x + i * s * 0.25, y + s * 0.18, s * 0.09, 0, TWO_PI);
        context.fill();
      }

      // Awning and posts.
      context.fillStyle = awning[k + 1];
      context.beginPath();
      context.moveTo(x - s * 0.74, y - s * 0.06);
      context.lineTo(x + s * 0.74, y - s * 0.06);
      context.lineTo(x + s * 0.54, y - s * 0.60);
      context.lineTo(x - s * 0.54, y - s * 0.60);
      context.closePath();
      context.fill();
      context.strokeStyle = PALETTE.ink;
      context.lineWidth = 1.2;
      context.stroke();

      context.fillStyle = PALETTE.ink;
      context.fillRect(x - s * 0.70, y - s * 0.06, s * 0.07, s * 0.42);
      context.fillRect(x + s * 0.63, y - s * 0.06, s * 0.07, s * 0.42);
    }

    if (!open) {
      // A muted wash and a small ledger stamp read as closed without the
      // harsh diagonal alert line.
      context.fillStyle = "rgba(244, 239, 230, 0.38)";
      context.fillRect(left, top, width, height);

      context.save();
      context.translate(left + width * 0.78, top + height * 0.82);
      context.rotate(-0.035);
      const stampWidth = s * 2.1;
      const stampHeight = s * 0.78;
      context.strokeStyle = "rgba(102, 96, 85, 0.45)";
      context.lineWidth = 1;
      context.strokeRect(
        -stampWidth / 2,
        -stampHeight / 2,
        stampWidth,
        stampHeight
      );
      context.fillStyle = "rgba(102, 96, 85, 0.72)";
      context.font = "500 " + Math.max(9, Math.round(s * 0.33)) + "px " + SANS;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText("Closed", 0, 0.5);
      context.restore();
    }

    context.restore();
    drawTag(context, open ? "Market" : "Market - closed",
      p.x, top - s * 0.34,
      PALETTE.paper,
      open ? PALETTE.ink : PALETTE.muted,
      open ? PALETTE.ink : PALETTE.stone,
      false);
  }

  function buildBackground() {
    if (!backgroundCanvas) backgroundCanvas = document.createElement("canvas");
    backgroundCanvas.width = Math.max(1, Math.round(W * DPR));
    backgroundCanvas.height = Math.max(1, Math.round(H * DPR));
    const bg = backgroundCanvas.getContext("2d");
    bg.setTransform(DPR, 0, 0, DPR, 0, 0);
    bg.clearRect(0, 0, W, H);

    const padX = Math.min(56, W * 0.06);
    const padY = Math.min(48, H * 0.07);
    const usableW = Math.max(1, W - padX * 2);
    const usableH = Math.max(1, H - padY * 2);
    const townCenter = screenPoint(CENTER);
    const townRx = TOWN.rx * usableW;
    const townRy = TOWN.ry * usableH;

    // The outer terrain is a muted meadow, not a paper panel.
    bg.fillStyle = PALETTE.meadow;
    bg.fillRect(0, 0, W, H);

    const patches = [
      [0.10, 0.16, 0.16, 0.10, 1.3],
      [0.86, 0.14, 0.13, 0.09, 2.7],
      [0.15, 0.84, 0.18, 0.11, 4.1],
      [0.91, 0.78, 0.12, 0.09, 5.6],
      [0.50, 0.06, 0.14, 0.055, 7.2],
      [0.48, 0.94, 0.17, 0.06, 8.9]
    ];
    for (let i = 0; i < patches.length; i += 1) {
      const patch = patches[i];
      const point = screenPoint({ x: patch[0], y: patch[1] });
      bg.fillStyle = i % 2 ? PALETTE.meadowPatch : PALETTE.meadowDeep;
      bg.globalAlpha = 0.48;
      blobPath(bg, point.x, point.y, patch[2] * W, patch[3] * H, patch[4], 0.15);
      bg.fill();
    }
    bg.globalAlpha = 1;

    let seed = 90210;
    function random() {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    }

    // Sparse grass and flower marks, always as flat ink or pigment.
    bg.save();
    for (let i = 0; i < 150; i += 1) {
      const x = random() * W;
      const y = random() * H;
      bg.globalAlpha = 0.10 + random() * 0.12;
      bg.fillStyle = i % 5 === 0 ? PALETTE.gold : PALETTE.sage;
      if (random() > 0.66) {
        bg.strokeStyle = PALETTE.sage;
        bg.lineWidth = 1;
        bg.beginPath();
        bg.moveTo(x - 2, y + 2);
        bg.quadraticCurveTo(x, y - 2, x + 1, y - 5);
        bg.moveTo(x + 1, y + 2);
        bg.quadraticCurveTo(x + 2, y - 1, x + 4, y - 4);
        bg.stroke();
      } else {
        bg.beginPath();
        bg.arc(x, y, 0.5 + random() * 1.2, 0, TWO_PI);
        bg.fill();
      }
    }
    bg.restore();

    // The town clearing is an organic warm-parchment patch inside the meadow.
    // Stacked translucent blobs soften its edge without a dashboard gradient.
    bg.save();
    bg.fillStyle = PALETTE.townEdge;
    bg.globalAlpha = 0.24;
    blobPath(bg, townCenter.x, townCenter.y, townRx * 1.10, townRy * 1.11, 12.4, 0.055);
    bg.fill();
    bg.globalAlpha = 0.38;
    blobPath(bg, townCenter.x, townCenter.y, townRx * 1.055, townRy * 1.06, 13.1, 0.05);
    bg.fill();
    bg.globalAlpha = 1;
    bg.fillStyle = PALETTE.town;
    blobPath(bg, townCenter.x, townCenter.y, townRx, townRy, 14.7, 0.045);
    bg.fill();

    // No hard outline: a loose band of grass tufts straddles the edge so the
    // peach ground dissolves into the meadow.
    for (let i = 0; i < 54; i += 1) {
      const angle = (i / 54) * TWO_PI + 0.17;
      const wobble = 1 +
        Math.sin(angle * 3 + 14.7) * 0.045 +
        Math.cos(angle * 5 + 21.4) * 0.03;
      const scale = 0.992 + ((i * 7) % 5) * 0.007;
      const x = townCenter.x + Math.cos(angle) * townRx * wobble * scale;
      const y = townCenter.y + Math.sin(angle) * townRy * wobble * scale;
      const tilt = ((i % 5) - 2) * 0.35;
      bg.globalAlpha = 0.20 + ((i * 3) % 4) * 0.06;
      bg.strokeStyle = i % 4 === 0 ? PALETTE.sageLight : PALETTE.sage;
      bg.lineWidth = 1;
      bg.beginPath();
      bg.moveTo(x - 1.8, y + 1.6);
      bg.quadraticCurveTo(x - 0.5, y - 1.5, x + 0.4, y - 4.2);
      bg.moveTo(x + 0.4, y + 1.6);
      bg.quadraticCurveTo(x + 1.4 + tilt, y - 1.2, x + 2.2 + tilt, y - 3.6);
      bg.stroke();
    }
    bg.restore();

    // Central market square. It is drawn here because its open state does not
    // change during a replay; the stalls and awnings sit over the plaza.
    drawMarket(bg);

    bg.save();
    bg.fillStyle = PALETTE.muted;
    bg.font = "500 10px " + SANS;
    bg.textAlign = "left";
    bg.textBaseline = "middle";
    bg.fillText("N", 18, H - 22);
    bg.beginPath();
    bg.moveTo(18, H - 38);
    bg.lineTo(21.5, H - 29);
    bg.lineTo(18, H - 31);
    bg.lineTo(14.5, H - 29);
    bg.closePath();
    bg.fill();
    bg.restore();
  }

  function drawCommonsRing(context, ratio) {
    for (const site of commonsSites) {
      if (site.kind === "pond") {
        drawCommonsPond(context, site, ratio);
      } else {
        drawCommonsBush(context, site, ratio);
      }
    }

    const labelPoint = screenPoint({ x: 0.5, y: 0.055 });
    drawTag(context, "Commons " + Math.round(ratio * 100) + "%",
      labelPoint.x, labelPoint.y,
      PALETTE.paper, PALETTE.ink, PALETTE.ink, false);
  }

  function drawCommonsBush(context, site, ratio) {
    if (ratio <= site.threshold) return;

    const point = screenPoint(site);
    const lush = clamp(
      (ratio - site.threshold) / Math.max(0.01, 0.88 - site.threshold),
      0,
      1
    );
    const radius = unit() * site.size * (0.58 + 0.42 * lush);
    const livingColor = site.tone === 0
      ? PALETTE.sageLight
      : site.tone === 1
        ? PALETTE.sage
        : PALETTE.sageDeep;

    context.save();
    context.fillStyle = mixColor(PALETTE.sageDry, livingColor, lush);
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1.2;
    blobPath(context, point.x, point.y, radius, radius * 0.84, site.seed, 0.17);
    context.fill();
    context.stroke();

    context.strokeStyle = "rgba(28, 27, 24, 0.30)";
    context.lineWidth = 1;
    for (let i = 0; i < 3; i += 1) {
      const angle = site.seed + i * 2.1;
      const distance = radius * 0.34;
      context.beginPath();
      context.moveTo(
        point.x + Math.cos(angle) * distance,
        point.y + Math.sin(angle) * distance
      );
      context.quadraticCurveTo(
        point.x + Math.cos(angle + 0.5) * distance * 1.5,
        point.y + Math.sin(angle + 0.5) * distance * 1.5,
        point.x + Math.cos(angle + 0.8) * distance * 1.8,
        point.y + Math.sin(angle + 0.8) * distance * 1.8
      );
      context.stroke();
    }

    const berries = Math.round(site.berries * lush);
    for (let i = 0; i < berries; i += 1) {
      const angle = i * 2.399963 + site.seed;
      const distance = radius * (0.25 + (i % 2) * 0.5);
      context.fillStyle = i % 3 === 0 ? PALETTE.gold : PALETTE.alert;
      context.beginPath();
      context.arc(
        point.x + Math.cos(angle) * distance,
        point.y + Math.sin(angle) * distance * 0.82,
        1.5,
        0,
        TWO_PI
      );
      context.fill();
    }
    context.restore();
  }

  function drawCommonsPond(context, site, ratio) {
    const point = screenPoint(site);
    const base = unit() * site.size;
    const level = clamp(
      (ratio - site.threshold) / Math.max(0.01, 1 - site.threshold),
      0,
      1
    );
    const water = ratio > site.threshold
      ? base * (0.22 + 0.78 * level)
      : 0;

    context.save();

    // The mud bank remains visible even when the water is nearly gone.
    context.fillStyle = mixColor(PALETTE.meadowPatch, PALETTE.townEdge, 0.55);
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1;
    blobPath(context, point.x, point.y, base * 1.12, base * 0.72, site.seed, 0.12);
    context.fill();
    context.stroke();

    if (water > 1.5) {
      context.fillStyle = PALETTE.water;
      context.strokeStyle = PALETTE.ink;
      context.lineWidth = 1.2;
      blobPath(context, point.x, point.y, water, water * 0.60, site.seed + 4.2, 0.08);
      context.fill();
      context.stroke();

      context.strokeStyle = "rgba(244, 239, 230, 0.50)";
      context.lineWidth = 1;
      context.beginPath();
      context.ellipse(
        point.x - water * 0.16,
        point.y - water * 0.10,
        water * 0.34,
        water * 0.14,
        0,
        Math.PI * 0.85,
        Math.PI * 1.7
      );
      context.stroke();
    }

    // Reeds thin out with the water level.
    const reeds = Math.round(5 * level);
    context.strokeStyle = PALETTE.sageDry;
    context.lineWidth = 1.2;
    for (let i = 0; i < reeds; i += 1) {
      const angle = site.seed + i * 2.1;
      const x = point.x + Math.cos(angle) * base * 0.92;
      const y = point.y + Math.sin(angle) * base * 0.52;
      context.beginPath();
      context.moveTo(x, y);
      context.quadraticCurveTo(x + 1, y - 5, x + 3, y - 8);
      context.stroke();
    }
    context.restore();
  }

  function mixColor(from, to, amount) {
    const t = clamp(amount, 0, 1);
    const parse = function (hex) {
      return [
        parseInt(hex.slice(1, 3), 16),
        parseInt(hex.slice(3, 5), 16),
        parseInt(hex.slice(5, 7), 16)
      ];
    };
    const a = parse(from);
    const b = parse(to);
    const channel = function (index) {
      return Math.round(a[index] + (b[index] - a[index]) * t);
    };
    return "rgb(" + channel(0) + "," + channel(1) + "," + channel(2) + ")";
  }

  function drawCoinBox(context, index, hutPoint, ratio) {
    const s = hutSize();
    const angle = homes[index].angle;
    const tangentX = -Math.sin(angle);
    const tangentY = Math.cos(angle);
    const cx = hutPoint.x + tangentX * s * 1.36;
    const cy = hutPoint.y + tangentY * s * 1.36 + s * 0.12;
    const width = s * 0.88;
    const height = s * 0.70;
    const left = cx - width / 2;
    const top = cy - height / 2;

    context.save();

    // Solid two-colour print shadow.
    context.fillStyle = PALETTE.shadow;
    context.fillRect(left + 2, top + 2, width, height);

    // Chest body.
    context.fillStyle = PALETTE.coffer;
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1.5;
    context.fillRect(left, top, width, height);
    context.strokeRect(left, top, width, height);

    // Bright cream interior, then coin stacks rising from the bottom up.
    const innerLeft = left + width * 0.08;
    const innerTop = top + height * 0.24;
    const innerWidth = width * 0.84;
    const innerHeight = height * 0.64;
    const ratioClamped = clamp(ratio, 0, 1);
    context.fillStyle = PALETTE.cofferInner;
    context.fillRect(innerLeft, innerTop, innerWidth, innerHeight);

    const columns = 4;
    const rows = 3;
    const slots = columns * rows;
    const filled = Math.ceil(ratioClamped * slots);
    const stepX = innerWidth / columns;
    const stepY = innerHeight / rows;
    const coinR = Math.min(stepX, stepY) * 0.40;

    for (let slot = 0; slot < filled; slot += 1) {
      const row = Math.floor(slot / columns);
      const column = slot % columns;
      const coinX = innerLeft + stepX * (column + 0.5);
      const coinY = innerTop + innerHeight - stepY * (row + 0.5);
      context.drawImage(
        cofferCoinSprite,
        coinX - coinR,
        coinY - coinR,
        coinR * 2,
        coinR * 2
      );
    }

    // Lid band and a small latch keep the box readable at any size.
    context.fillStyle = PALETTE.cofferLid;
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1.2;
    context.fillRect(left, top, width, height * 0.18);
    context.strokeRect(left, top, width, height * 0.18);

    context.fillStyle = PALETTE.gold;
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1;
    context.fillRect(cx - width * 0.075, top + height * 0.12,
      width * 0.15, height * 0.17);
    context.strokeRect(cx - width * 0.075, top + height * 0.12,
      width * 0.15, height * 0.17);

    context.restore();
  }

  function drawTombstone(context, x, y, color, hovered) {
    const s = agentRadius() * 1.25;
    context.save();
    context.translate(x, y);

    context.fillStyle = PALETTE.shadow;
    context.beginPath();
    context.ellipse(2, s * 0.84, s * 0.82, s * 0.26, 0, 0, TWO_PI);
    context.fill();

    context.globalAlpha = hovered ? 1 : 0.9;
    context.fillStyle = PALETTE.stone;
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(-s * 0.55, s * 0.45);
    context.lineTo(-s * 0.55, -s * 0.1);
    context.arc(0, -s * 0.1, s * 0.55, Math.PI, 0);
    context.lineTo(s * 0.55, s * 0.45);
    context.closePath();
    context.fill();
    context.stroke();

    context.strokeStyle = PALETTE.paper;
    context.lineWidth = 1.4;
    context.beginPath();
    context.moveTo(-s * 0.2, -s * 0.23);
    context.lineTo(s * 0.2, -s * 0.23);
    context.moveTo(0, -s * 0.43);
    context.lineTo(0, s * 0.05);
    context.stroke();

    context.fillStyle = color;
    context.beginPath();
    context.arc(-s * 0.58, s * 0.58, s * 0.13, 0, TWO_PI);
    context.fill();
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1;
    context.stroke();
    context.restore();
  }

  function drawVillage(context, tick, time) {
    for (let index = 0; index < A; index += 1) {
      if (!aliveAt(index, tick)) continue;
      const point = screenPoint(homes[index]);
      drawHut(context, index, point);
      const ratio = clamp(resourceAt(index, time) / maxCoins, 0, 1);
      drawCoinBox(context, index, point, ratio);
    }
  }

  function drawAgent(context, index, position, time, now) {
    const R = agentRadius();
    const agent = agents[index];
    const alive = aliveAt(index, Math.floor(clamp(time, 0, N - 1)));
    const hovered = state.hovered === index;
    const selected = state.selected === index;

    if (!alive) {
      drawTombstone(context, position.x, position.y, agent.color, hovered || selected);
      return;
    }

    const previous = positionAt(index, Math.max(0, time - 0.025));
    const dx = position.x - previous.x;
    const distance = Math.abs(dx) + Math.abs(position.y - previous.y);
    const moving = distance > 0.12 && !reduceMotion;
    const phase = moving ? (state.playing ? now * 0.012 : 0) + index * 1.7 : 0;
    const bounce = moving ? Math.abs(Math.sin(phase)) * R * 0.08 : 0;
    const y = position.y - bounce;
    const face = dx < -0.01 ? -1 : 1;

    context.save();

    context.fillStyle = PALETTE.shadow;
    context.beginPath();
    context.ellipse(position.x + 2, y + R * 1.12, R * 0.78, R * 0.22, 0, 0, TWO_PI);
    context.fill();

    const stride = moving ? Math.sin(phase) * R * 0.36 : 0;
    const leftLift = moving ? Math.max(0, Math.sin(phase)) * R * 0.18 : 0;
    const rightLift = moving ? Math.max(0, -Math.sin(phase)) * R * 0.18 : 0;
    const footY = y + R * 1.04;
    const leftFootX = position.x - R * 0.2 + stride;
    const rightFootX = position.x + R * 0.2 - stride;

    context.strokeStyle = PALETTE.ink;
    context.lineWidth = Math.max(1.5, R * 0.14);
    context.lineCap = "round";
    context.beginPath();
    context.moveTo(position.x - R * 0.17, y + R * 0.3);
    context.lineTo(leftFootX, footY - leftLift);
    context.moveTo(position.x + R * 0.17, y + R * 0.3);
    context.lineTo(rightFootX, footY - rightLift);
    context.stroke();

    context.fillStyle = PALETTE.ink;
    context.beginPath();
    context.ellipse(leftFootX, footY - leftLift, R * 0.17, R * 0.09, 0, 0, TWO_PI);
    context.fill();
    context.beginPath();
    context.ellipse(rightFootX, footY - rightLift, R * 0.17, R * 0.09, 0, 0, TWO_PI);
    context.fill();

    context.fillStyle = agent.color;
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1.5;
    roundedRect(context, position.x - R * 0.43, y - R * 0.68, R * 0.86, R * 1.05, R * 0.22);
    context.fill();
    context.stroke();

    context.fillStyle = PALETTE.skin;
    context.strokeStyle = PALETTE.ink;
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(position.x, y - R * 1.05, R * 0.42, 0, TWO_PI);
    context.fill();
    context.stroke();

    context.fillStyle = PALETTE.ink;
    context.beginPath();
    context.arc(position.x, y - R * 1.15, R * 0.43, Math.PI, 0);
    context.fill();

    context.beginPath();
    context.arc(position.x - R * 0.14 + face * R * 0.04, y - R * 1.06, 0.9, 0, TWO_PI);
    context.arc(position.x + R * 0.14 + face * R * 0.04, y - R * 1.06, 0.9, 0, TWO_PI);
    context.fill();

    context.fillStyle = PALETTE.door;
    context.beginPath();
    context.arc(position.x + face * R * 0.3, y - R * 0.99, 0.8, 0, TWO_PI);
    context.fill();

    if (selected || hovered) {
      context.strokeStyle = selected ? PALETTE.gold : PALETTE.ink;
      context.lineWidth = selected ? 2 : 1.5;
      context.setLineDash(selected ? [4, 3] : []);
      context.beginPath();
      context.ellipse(position.x, y + R * 0.15, R * 1.55, R * 1.85, 0, 0, TWO_PI);
      context.stroke();
      context.setLineDash([]);
    }

    context.restore();

    if (selected || hovered) {
      drawTag(context, agent.label, position.x, y - R * 2.05,
        PALETTE.paper, PALETTE.ink, PALETTE.ink, true);
    }
  }

  function drawAgents(context, time, now) {
    const order = [];
    for (let i = 0; i < A; i += 1) {
      order.push({ index: i, y: positionAt(i, time).y });
    }
    order.sort(function (a, b) { return a.y - b.y; });

    for (const item of order) {
      drawAgent(context, item.index, positionAt(item.index, time), time, now);
    }
  }

  function handPoint(position, toward) {
    const dx = toward.x - position.x;
    const side = dx >= 0 ? 1 : -1;
    return {
      x: position.x + side * agentRadius() * 0.42,
      y: position.y - agentRadius() * 0.3
    };
  }

  function quadraticPoint(from, control, to, progress) {
    const inverse = 1 - progress;
    return {
      x: inverse * inverse * from.x + 2 * inverse * progress * control.x + progress * progress * to.x,
      y: inverse * inverse * from.y + 2 * inverse * progress * control.y + progress * progress * to.y
    };
  }

  function drawEventParticles(context, time) {
    const tick = Math.floor(clamp(time, 0, N - 1));
    const events = eventsByTick[tick] || [];
    if (!events.length) return;

    const fraction = clamp(time, 0, N - 1) - tick;
    const progress = state.playing
      ? smoothStep(clamp(fraction / 0.7, 0, 1))
      : 0.55;

    for (let i = 0; i < events.length; i += 1) {
      const event = events[i];
      const type = event[1];
      const from = positionAt(event[2], time);
      const to = positionAt(event[3], time);
      const start = handPoint(from, to);
      const end = { x: to.x, y: to.y - agentRadius() * 0.15 };
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const length = Math.hypot(dx, dy) || 1;
      const bend = Math.min(38, length * 0.22) * (i % 2 === 0 ? 1 : -1);
      const control = {
        x: (start.x + end.x) / 2 - (dy / length) * bend,
        y: (start.y + end.y) / 2 + (dx / length) * bend
      };
      const point = quadraticPoint(start, control, end, progress);
      const amount = Number(event[4]) || 0;
      const coinR = clamp(2.8 + Math.sqrt(amount) * 1.1, 3, 6.5);

      // A faint printed trail behind the coin.
      context.save();
      context.strokeStyle = "rgba(200, 138, 30, 0.34)";
      context.lineWidth = 1.5;
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.quadraticCurveTo(control.x, control.y, point.x, point.y);
      context.stroke();
      context.restore();

      context.drawImage(coinSprite, point.x - coinR, point.y - coinR, coinR * 2, coinR * 2);

      if (type === TRADE) {
        const goodsProgress = 1 - progress;
        const goodsPoint = quadraticPoint(start, control, end, goodsProgress);
        context.save();
        context.fillStyle = PALETTE.paperDeep;
        context.strokeStyle = PALETTE.ink;
        context.lineWidth = 1.5;
        roundedRect(context, goodsPoint.x + 7, goodsPoint.y - 4, 8, 8, 0);
        context.fill();
        context.stroke();
        context.beginPath();
        context.moveTo(goodsPoint.x + 7, goodsPoint.y);
        context.lineTo(goodsPoint.x + 15, goodsPoint.y);
        context.moveTo(goodsPoint.x + 11, goodsPoint.y - 4);
        context.lineTo(goodsPoint.x + 11, goodsPoint.y + 4);
        context.stroke();
        context.restore();
      }
    }
  }

  function drawShockTint(tick) {
    const shocks = activeShocks(tick);
    if (!shocks.length) return;
    const scarcity = shocks.some(function (shock) { return shock.type === "scarcity"; });
    ctx.save();
    ctx.fillStyle = scarcity
      ? "rgba(184, 74, 57, 0.05)"
      : "rgba(47, 111, 115, 0.045)";
    ctx.fillRect(0, 0, W, H);
    ctx.restore();
  }

  function renderWorld(now) {
    if (!W || !H || !backgroundCanvas) return;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(backgroundCanvas, 0, 0, W, H);
    const tick = Math.floor(clamp(state.t, 0, N - 1));
    drawShockTint(tick);
    drawCommonsRing(ctx, stockRatio(state.t));
    drawVillage(ctx, tick, state.t);
    drawHutLabels(ctx, tick);
    drawAgents(ctx, state.t, now);
    drawEventParticles(ctx, state.t);
  }

  function resizeWorld() {
    const rect = worldWrap.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return;
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = rect.width;
    H = rect.height;
    world.width = Math.max(1, Math.round(W * DPR));
    world.height = Math.max(1, Math.round(H * DPR));
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    buildBackground();
    renderWorld(performance.now());
  }

  function screenToAgent(mouseX, mouseY) {
    const R = agentRadius();
    let best = null;
    let bestDistance = Infinity;
    for (let i = 0; i < A; i += 1) {
      const position = positionAt(i, state.t);
      const dead = !aliveAt(i, Math.floor(state.t));
      const centerY = position.y - (dead ? 0 : R * 0.4);
      const hitRadius = Math.max(15, R * (dead ? 1.4 : 1.5));
      const distance = Math.hypot(mouseX - position.x, mouseY - centerY);
      if (distance <= hitRadius && distance < bestDistance) {
        best = i;
        bestDistance = distance;
      }
    }
    return best;
  }

  function tooltipHtml(index, tick) {
    const agent = agents[index];
    const coins = resourceAt(index, state.t);
    const alive = aliveAt(index, tick);
    const ratio = clamp(coins / maxCoins, 0, 1);
    const parts = [
      "<strong>" + agent.label + "</strong>",
      (alive ? "alive" : "dead") + " - " + formatNumber(coins, 1) + " coins",
      "<span class=\"muted\">coin box " + Math.round(ratio * 100) + "% full</span>"
    ];
    const recent = recentEventFor(index, tick);
    if (recent) {
      parts.push("<span class=\"muted\">last: " + eventTypeName(recent[1]) + " at iter " + recent[0] + "</span>");
    }
    parts.push("<span class=\"muted\">click to inspect</span>");
    return parts.join("<br>");
  }

  function updateTooltip(event) {
    const rect = world.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const hit = screenToAgent(x, y);
    state.hovered = hit;
    if (hit == null) {
      tooltip.style.display = "none";
      return;
    }
    tooltip.innerHTML = tooltipHtml(hit, Math.floor(state.t));
    tooltip.style.display = "block";
    const tooltipRect = tooltip.getBoundingClientRect();
    let left = x + 16;
    let top = y + 16;
    if (left + tooltipRect.width > W - 8) left = x - tooltipRect.width - 16;
    if (top + tooltipRect.height > H - 8) top = y - tooltipRect.height - 16;
    tooltip.style.left = Math.max(8, left) + "px";
    tooltip.style.top = Math.max(8, top) + "px";
  }

  function describeTick(tick) {
    if (tick === 0) {
      return "The village begins with " + formatNumber(initialTotal, 0) +
        " coins and a full commons.";
    }
    const alive = series.alive[tick];
    if (alive === 0) {
      return "The last villager is gone. Only the empty meadow remains.";
    }
    const deaths = deathsByTick[tick] || [];
    if (deaths.length) {
      return nameFor(deaths[0]) + " ran out of coins and left the village.";
    }
    const shocks = activeShocks(tick);
    if (shocks.length) {
      return shockLabel(shocks[0]) + ".";
    }
    const events = eventsByTick[tick] || [];
    if (events.length === 1) {
      const event = events[0];
      if (event[1] === TRADE) {
        return eventText(event) + " in the central square.";
      }
      if (event[1] === REPAY) {
        return eventText(event) + ".";
      }
      return eventText(event) + " at " + nameFor(event[3]) + "'s hut.";
    }
    if (events.length > 1) {
      let gifts = 0;
      let trades = 0;
      let repayments = 0;
      for (const event of events) {
        if (event[1] === TRADE) trades += 1;
        else if (event[1] === REPAY) repayments += 1;
        else gifts += 1;
      }
      const parts = [];
      if (gifts) parts.push(gifts + " gift" + (gifts === 1 ? "" : "s"));
      if (trades) parts.push(trades + " market trade" + (trades === 1 ? "" : "s"));
      if (repayments) parts.push(repayments + " repayment" + (repayments === 1 ? "" : "s"));
      const summary = parts.length > 1
        ? parts.slice(0, -1).join(", ") + " and " + parts[parts.length - 1]
        : parts[0];
      return summary + " this iteration.";
    }
    const cost = Number(meta.living_cost) || 0;
    if (cost > 0) {
      return "A quiet iteration. " + alive + " villagers paid " +
        formatNumber(cost * alive, 0) + " coins to live, and the commons shrank.";
    }
    return "A quiet iteration with no exchanges.";
  }

  function renderHeader(tick) {
    const alive = series.alive[tick];
    const gini = series.gini[tick];
    const total = series.total[tick];
    const ratio = stockRatio(tick);

    statIteration.textContent = formatNumber(tick, 0);
    statAlive.textContent = alive + " / " + A;
    statStock.textContent = Math.round(ratio * 100) + "%";
    stockBar.style.width = (ratio * 100).toFixed(1) + "%";
    statCoins.textContent = formatNumber(total, 0);
    statGini.textContent = Number(gini).toFixed(3);
    readout.textContent = formatNumber(tick, 0);
    readoutTotal.textContent = "/ " + formatNumber(end, 0);
    storyEl.textContent = describeTick(tick);

    const shocks = activeShocks(tick);
    if (shocks.length) {
      const shock = shocks[0];
      shockBanner.textContent = shock.type === "scarcity"
        ? "Scarcity - production x" + formatNumber(Math.max(0, 1 - shock.magnitude), 2)
        : "Information shock - noisy reputation";
      shockBanner.classList.add("visible");
      shockBanner.classList.toggle("info", shock.type !== "scarcity");
    } else {
      shockBanner.classList.remove("visible");
    }

    liveStatus.textContent = "Iteration " + tick + " of " + end + ". " +
      alive + " of " + A + " villagers alive. Commons " +
      Math.round(ratio * 100) + " percent. Total coins " + formatNumber(total, 0) + ".";
  }

  function renderInspector(tick) {
    if (state.selected == null) {
      selectedPanel.innerHTML =
        "<div class=\"panel-head\"><div class=\"panel-title\">Villager</div></div>" +
        "<p class=\"selected-empty\">Click any villager in the village, or a row in this iteration's feed, to see their coins, coin box, and latest exchange.</p>";
      return;
    }

    const index = state.selected;
    const agent = agents[index];
    const coins = resourceAt(index, state.t);
    const ratio = clamp(coins / maxCoins, 0, 1);
    const alive = aliveAt(index, tick);
    const recent = recentEventFor(index, tick);
    const goods = series.goods.length ? series.goods[clamp(tick, 0, N - 1)][index] : null;
    const reputation = series.reputation.length ? series.reputation[clamp(tick, 0, N - 1)][index] : null;
    const risk = series.risk && series.risk.length ? series.risk[clamp(tick, 0, N - 1)][index] : agent.risk;

    let cells =
      "<div class=\"inspect-grid\">" +
        "<div><div class=\"k\">Coin box</div><div class=\"v\">" + Math.round(ratio * 100) + "% full</div></div>" +
        "<div><div class=\"k\">Risk</div><div class=\"v\">" + formatNumber(risk, 2) + "</div></div>";
    if (goods != null) {
      cells += "<div><div class=\"k\">Goods</div><div class=\"v\">" + formatNumber(goods, 1) + "</div></div>";
    }
    if (reputation != null) {
      cells += "<div><div class=\"k\">Reputation</div><div class=\"v\">" + formatNumber(reputation, 2) + "</div></div>";
    }
    cells += "</div>";

    const recentHtml = recent
      ? "<div class=\"recent\">" + eventText(recent) +
        "<span class=\"muted\">iteration " + recent[0] + "</span></div>"
      : "<div class=\"recent\"><span class=\"muted\">No exchange in the last 40 iterations.</span></div>";

    selectedPanel.innerHTML =
      "<div class=\"panel-head\">" +
        "<div class=\"agent-head\"><span class=\"swatch\" style=\"background:" + agent.color + "\"></span>" +
          "<span>" + agent.id + "</span>" +
          "<span class=\"status " + (alive ? "" : "dead") + "\">" + (alive ? "alive" : "dead") + "</span>" +
        "</div>" +
        "<button class=\"clear-btn\" id=\"clearSelection\">Clear</button>" +
      "</div>" +
      "<div class=\"big-value\">" + formatNumber(coins, 1) + "<small>coins</small></div>" +
      "<div class=\"coin-meter\"><i style=\"width:" + (ratio * 100).toFixed(1) + "%\"></i></div>" +
      cells +
      recentHtml;

    const clearButton = document.getElementById("clearSelection");
    if (clearButton) {
      clearButton.addEventListener("click", function () { selectAgent(null); });
    }
  }

  function renderFeed(tick) {
    feedTick.textContent = "iter " + formatNumber(tick, 0);
    const rows = [];

    for (const shock of DATA.shocks) {
      if (shock.iteration === tick) {
        rows.push({
          agent: null,
          icon: "!",
          iconClass: "shock",
          main: "<b>" + (shock.type === "scarcity" ? "Scarcity shock" : "Information shock") + "</b>",
          sub: shockLabel(shock) + " - lasts " + shock.duration + " iterations"
        });
      }
    }

    for (const index of deathsByTick[tick] || []) {
      rows.push({
        agent: index,
        icon: "x",
        iconClass: "death",
        main: "<b>" + nameFor(index) + "</b> ran out of coins",
        sub: "at or below the resource floor"
      });
    }

    for (const event of eventsByTick[tick] || []) {
      rows.push({
        agent: event[2],
        icon: eventIcon(event[1]),
        iconClass: event[1] === TRADE ? "trade" : event[1] === REPAY ? "repay" : "",
        main: eventText(event),
        sub: event[1] === TRADE
          ? formatNumber(event[5], 1) + " goods for " + formatNumber(event[4], 1) + " coins"
          : eventTypeName(event[1]) + " - iteration " + tick
      });
    }

    if (!rows.length) {
      const cost = Number(meta.living_cost) || 0;
      feed.innerHTML = "<div class=\"feed-row\"><span class=\"feed-icon quiet\">-</span>" +
        "<span><span class=\"feed-main\">No exchanges this iteration.</span>" +
        "<span class=\"feed-sub\">" +
        (cost > 0
          ? "Every living villager paid " + formatNumber(cost, 1) + " coins of living cost."
          : "The village is quiet.") +
        "</span></span></div>";
      return;
    }

    const visible = rows.slice(0, 7);
    let html = "";
    for (let i = 0; i < visible.length; i += 1) {
      const row = visible[i];
      html += "<button class=\"feed-row\" data-agent=\"" +
        (row.agent == null ? "" : row.agent) + "\">" +
        "<span class=\"feed-icon " + row.iconClass + "\">" + row.icon + "</span>" +
        "<span><span class=\"feed-main\">" + row.main + "</span>" +
        "<span class=\"feed-sub\">" + row.sub + "</span></span>" +
      "</button>";
    }
    if (rows.length > visible.length) {
      html += "<div class=\"feed-row\"><span class=\"feed-icon quiet\">+</span>" +
        "<span class=\"feed-main\">" + (rows.length - visible.length) +
        " more event" + (rows.length - visible.length === 1 ? "" : "s") +
        " this iteration.</span></div>";
    }
    feed.innerHTML = html;
  }

  function renderUi(force) {
    const now = performance.now();
    if (!force && state.playing && now - state.lastUiTime < 70) return;
    const tick = Math.floor(clamp(state.t, 0, N - 1));
    if (!force && state.lastUiTick === tick) return;
    state.lastUiTime = now;
    state.lastUiTick = tick;

    renderHeader(tick);
    renderInspector(tick);
    renderFeed(tick);

    if (!state.dragging) scrub.value = String(tick);
  }

  function updatePlayButton() {
    playBtn.setAttribute("aria-label", state.playing ? "Pause animation" : "Play animation");
    playIcon.innerHTML = state.playing
      ? "<path d=\"M3 1.5h2.4v9H3v-9Zm3.6 0H9v9H6.6v-9Z\" fill=\"currentColor\"/>"
      : "<path d=\"M3 1.5v9L10.5 6 3 1.5Z\" fill=\"currentColor\"/>";
  }

  function togglePlay() {
    if (end === 0) return;
    if (!state.playing && state.t >= end - 0.001) state.t = 0;
    state.playing = !state.playing;
    updatePlayButton();
    renderUi(true);
  }

  function setTime(value, force) {
    state.t = clamp(value, 0, end);
    renderUi(force === true);
  }

  function step(delta) {
    state.playing = false;
    state.t = clamp(Math.floor(state.t) + delta, 0, end);
    updatePlayButton();
    renderUi(true);
  }

  function cycleSpeed() {
    const index = SPEEDS.indexOf(state.speed);
    state.speed = SPEEDS[(index + 1) % SPEEDS.length];
    speedBtn.textContent = state.speed + "x";
    speedBtn.title = "Change playback speed (" + state.speed + " iterations per second)";
  }

  function selectAgent(index) {
    state.selected = index;
    renderUi(true);
  }

  function frame(now) {
    const delta = Math.min(0.1, (now - lastTime) / 1000);
    lastTime = now;

    if (state.playing) {
      state.t += delta * state.speed;
      if (state.t >= end) {
        state.t = end;
        state.playing = false;
        updatePlayButton();
      }
    }

    renderWorld(now);
    renderUi(false);
    requestAnimationFrame(frame);
  }

  function openHelp() {
    helpModal.classList.add("visible");
    document.getElementById("helpClose").focus();
  }

  function closeHelp() {
    helpModal.classList.remove("visible");
  }

  function exportFrameData() {
    const tick = Math.floor(clamp(state.t, 0, N - 1));
    return {
      iteration: tick,
      agents: agents.map(function (agent, index) {
        return {
          id: agent.id,
          resources: series.resources[tick][index],
          alive: aliveAt(index, tick)
        };
      })
    };
  }

  function init() {
    deathLegend.style.display = hasDeaths ? "grid" : "none";
    scrub.max = String(end);
    readoutTotal.textContent = "/ " + formatNumber(end, 0);
    worldNote.textContent = meta.market_enabled
      ? "Click a villager - gifts, repayments and central-market trades are animated"
      : "Click a villager - the market is closed in this run";

    resizeWorld();
    renderUi(true);
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () {
        buildBackground();
        renderWorld(performance.now());
      });
    }
    if (!reduceMotion && end > 0) state.playing = true;
    updatePlayButton();
    requestAnimationFrame(frame);
  }

  world.addEventListener("pointermove", updateTooltip);
  world.addEventListener("pointerleave", function () {
    state.hovered = null;
    tooltip.style.display = "none";
  });
  world.addEventListener("click", function (event) {
    const rect = world.getBoundingClientRect();
    const hit = screenToAgent(event.clientX - rect.left, event.clientY - rect.top);
    selectAgent(hit);
  });

  feed.addEventListener("click", function (event) {
    const button = event.target.closest("[data-agent]");
    if (!button) return;
    const value = button.getAttribute("data-agent");
    if (value === "" || value == null) return;
    selectAgent(Number(value));
  });

  playBtn.addEventListener("click", togglePlay);
  document.getElementById("rewindBtn").addEventListener("click", function () { setTime(0, true); });
  document.getElementById("prevBtn").addEventListener("click", function () { step(-1); });
  document.getElementById("nextBtn").addEventListener("click", function () { step(1); });
  speedBtn.addEventListener("click", cycleSpeed);

  scrub.addEventListener("pointerdown", function () {
    state.dragging = true;
    if (state.playing) {
      state.playing = false;
      updatePlayButton();
    }
  });
  scrub.addEventListener("input", function () {
    setTime(Number(scrub.value), true);
  });
  scrub.addEventListener("change", function () { state.dragging = false; });
  scrub.addEventListener("pointerup", function () { state.dragging = false; });

  document.getElementById("helpBtn").addEventListener("click", openHelp);
  document.getElementById("helpClose").addEventListener("click", closeHelp);
  helpModal.addEventListener("click", function (event) {
    if (event.target === helpModal) closeHelp();
  });

  document.addEventListener("keydown", function (event) {
    const tag = event.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (helpModal.classList.contains("visible")) {
      if (event.key === "Escape") closeHelp();
      return;
    }
    if (event.key === "Escape") {
      selectAgent(null);
    } else if (event.key === " " || event.code === "Space") {
      event.preventDefault();
      togglePlay();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      step(event.shiftKey ? -10 : -1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      step(event.shiftKey ? 10 : 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      setTime(0, true);
    } else if (event.key === "End") {
      event.preventDefault();
      setTime(end, true);
    }
  });

  if (window.ResizeObserver) {
    const observer = new ResizeObserver(function () {
      resizeWorld();
      renderWorld(performance.now());
    });
    observer.observe(worldWrap);
  } else {
    window.addEventListener("resize", function () {
      resizeWorld();
      renderWorld(performance.now());
    });
  }

  window.commonsReplay = {
    data: DATA,
    state: state,
    end: end,
    frame: exportFrameData,
    setTime: setTime,
    render: function () {
      renderWorld(performance.now());
    },
    redraw: function () {
      renderWorld(performance.now());
      renderUi(true);
    }
  };

  init();

  return window.commonsReplay;
}
