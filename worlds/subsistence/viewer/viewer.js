/* Canvas village and PPF chart for the subsistence replay. */
(function (global) {
  "use strict";

  var LOGICAL_W = 1000;
  var LOGICAL_H = 700;
  var CENTER = { x: 500, y: 350 };
  var COMMONS_RADIUS = 280;
  var WILD_RADIUS = 340;

  var SEASONS = {
    spring: {
      sky: "#e8efdc",
      canvas: "#dfe6d2",
      deep: "#d3dcc4",
      ground: "rgba(244, 239, 230, 0.45)",
      leaf: "#4f7450",
      leafLight: "#78906c",
      wild: "#5c7d58",
      note: "berries regrow"
    },
    summer: {
      sky: "#eef0d6",
      canvas: "#e4e6c6",
      deep: "#d9dbaf",
      ground: "rgba(246, 240, 220, 0.45)",
      leaf: "#486c3f",
      leafLight: "#7d9257",
      wild: "#4f7038",
      note: "berries flourish"
    },
    autumn: {
      sky: "#efe6d2",
      canvas: "#e8dcc3",
      deep: "#ddcfae",
      ground: "rgba(247, 238, 221, 0.45)",
      leaf: "#a8703f",
      leafLight: "#c1955a",
      wild: "#9a6a3a",
      note: "berries thin"
    },
    winter: {
      sky: "#e5ebef",
      canvas: "#dde5ea",
      deep: "#cfd9df",
      ground: "rgba(240, 244, 247, 0.55)",
      leaf: "#7d8b8f",
      leafLight: "#9aa8ab",
      wild: "#6f7f84",
      note: "berries dormant"
    },
    temperate: {
      sky: "#e7ecdb",
      canvas: "#dfe6d2",
      deep: "#d3dcc4",
      ground: "rgba(244, 239, 230, 0.42)",
      leaf: "#4e6b50",
      leafLight: "#78906c",
      wild: "#54735a",
      note: "steady growth"
    }
  };

  var COLORS = {
    paper: "#f4efe6",
    paperDeep: "#efe8da",
    canvas: "#dfe6d2",
    canvasDeep: "#d4ddc5",
    ink: "#1c1b18",
    muted: "#666055",
    rule: "#d8d0c0",
    ruleSoft: "#e4dccd",
    berry: "#b23b4b",
    berryDark: "#7e2433",
    wood: "#9a6a3a",
    woodDark: "#6f4725",
    leaf: "#4e6b50",
    leafLight: "#78906c",
    water: "#2f6f73",
    alert: "#b84a39",
    stone: "#8c857b",
    gold: "#c88a1e"
  };

  var FEED_ICONS = {
    gather: "&#9679;",
    craft: "&#9881;",
    gift: "&#9829;",
    trade: "&#8646;",
    loan: "&#8594;",
    repay: "&#8634;",
    default: "&#9888;",
    late: "&#9203;",
    shock: "&#9889;",
    spoil: "&#10005;",
    death: "&#8226;",
    quiet: "&#183;"
  };

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  function easeInOut(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }

  function formatNumber(value, digits) {
    if (value == null || Number.isNaN(Number(value))) return "0";
    return Number(value).toFixed(digits == null ? 1 : digits);
  }

  function formatPercent(value) {
    return Math.round(clamp(value, 0, 1) * 100) + "%";
  }

  function mixColor(from, to, amount) {
    var a = parseColor(from);
    var b = parseColor(to);
    var t = clamp(amount, 0, 1);
    return "rgb(" + Math.round(lerp(a[0], b[0], t)) + "," +
      Math.round(lerp(a[1], b[1], t)) + "," +
      Math.round(lerp(a[2], b[2], t)) + ")";
  }

  function parseColor(value) {
    if (value.charAt(0) === "#") {
      var hex = value.slice(1);
      if (hex.length === 3) {
        hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
      }
      return [
        parseInt(hex.slice(0, 2), 16),
        parseInt(hex.slice(2, 4), 16),
        parseInt(hex.slice(4, 6), 16)
      ];
    }
    var match = value.match(/\d+/g);
    return match ? match.map(Number) : [0, 0, 0];
  }

  function roundedRect(ctx, x, y, width, height, radius) {
    var r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function ellipse(ctx, x, y, rx, ry) {
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
  }

  function randomFor(seed) {
    var value = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
    return value - Math.floor(value);
  }

  function startSubsistenceViewer(view) {
    var meta = view.meta;
    var series = view.series;
    var agents = meta.agents;
    var N = meta.iterations + 1;
    var A = agents.length;

    var canvas = document.getElementById("world");
    var ctx = canvas.getContext("2d");
    var tooltip = document.getElementById("tooltip");
    var worldNote = document.getElementById("worldNote");

    var homes = buildHomes(A);
    var berrySites = buildBerrySites();
    var treeSites = buildTreeSites();
    var wildTrees = buildWildTrees();

    var state = {
      tick: 0,
      selected: null,
      hover: null,
      playing: true,
      speedIndex: 2,
      speeds: [1, 2, 4, 8, 16],
      lastFrame: 0,
      tickAccumulator: 0,
      particles: [],
      dpr: 1,
      scale: 1,
      offsetX: 0,
      offsetY: 0,
      ppfAll: false,
      ppfZoom: false,
      ppfOpen: false,
      startedAt: 0
    };

    // A village of `count` huts arranged in up to three jittered rings around
    // the market square. Bigger villages use more rings and smaller huts.
    function hutScale(count) {
      return clamp(1.02 - count * 0.0042, 0.58, 1);
    }

    function buildHomes(count) {
      var rings = count <= 12
        ? [172]
        : count <= 32
          ? [138, 196]
          : count <= 72
            ? [132, 180, 224]
            : [122, 164, 205, 242];
      var scale = hutScale(count);
      var perRing = Math.ceil(count / rings.length);
      var result = [];
      for (var ring = 0; ring < rings.length; ring += 1) {
        var inRing = Math.min(perRing, count - ring * perRing);
        if (inRing <= 0) break;
        var radius = rings[ring];
        var step = (Math.PI * 2) / inRing;
        for (var slot = 0; slot < inRing; slot += 1) {
          var index = ring * perRing + slot;
          var seed = index + 1;
          var angle = -Math.PI / 2 + slot * step +
            (randomFor(seed * 3.1) - 0.5) * step * 0.22;
          var distance = radius + (randomFor(seed * 7.7) - 0.5) * 16;
          result.push({
            x: CENTER.x + Math.cos(angle) * distance,
            y: CENTER.y + Math.sin(angle) * distance * 0.93,
            angle: angle,
            scale: scale
          });
        }
      }
      return result;
    }

    function scatteredSites(count, startDegrees, endDegrees, radius, seedBase) {
      var sites = [];
      for (var i = 0; i < count; i += 1) {
        var t = count <= 1 ? 0.5 : i / (count - 1);
        var base = startDegrees + (endDegrees - startDegrees) * t;
        var angle =
          ((base + (randomFor((seedBase + i) * 1.7) - 0.5) * 9) * Math.PI) /
          180;
        var distance = radius + (randomFor((seedBase + i) * 4.3) - 0.5) * 22;
        sites.push({
          x: CENTER.x + Math.cos(angle) * distance,
          y: CENTER.y + Math.sin(angle) * distance * 0.9,
          seed: seedBase + i
        });
      }
      return sites;
    }

    function buildBerrySites() {
      return scatteredSites(8, 118, 242, COMMONS_RADIUS, 1);
    }

    function buildTreeSites() {
      return scatteredSites(8, -58, 62, COMMONS_RADIUS, 21);
    }

    // Decorative forest beyond the worked commons. These trees never change
    // and are never harvested; they give the village a horizon.
    function buildWildTrees() {
      var trees = [];
      var count = 52;
      for (var i = 0; i < count; i += 1) {
        var angle =
          -Math.PI / 2 +
          (i * Math.PI * 2) / count +
          (randomFor(i * 2.3) - 0.5) * 0.07;
        var distance = WILD_RADIUS + (randomFor(i * 5.1) - 0.5) * 30;
        trees.push({
          x: CENTER.x + Math.cos(angle) * distance,
          y: CENTER.y + Math.sin(angle) * distance * 0.86,
          size: 0.72 + randomFor(i * 9.7) * 0.55,
          seed: i + 41
        });
      }
      return trees;
    }

    function seasonInfoAt(tick) {
      var seasons = meta.seasons;
      if (!seasons || !seasons.enabled) return null;
      var cycle = seasons.cycle || [];
      if (!cycle.length) return null;
      var raw = series.season
        ? series.season[clamp(tick, 0, N - 1)]
        : 0;
      var index = Number(raw);
      if (!Number.isFinite(index) || index < 0) index = 0;
      index = ((index % cycle.length) + cycle.length) % cycle.length;
      var entry = cycle[index] || {};
      return {
        index: index,
        name: String(entry.name || "season " + (index + 1)),
        berry: entry.berry_regen == null ? 1 : Number(entry.berry_regen),
        wood: entry.wood_regen == null ? 1 : Number(entry.wood_regen)
      };
    }

    function seasonStyleAt(tick) {
      var info = seasonInfoAt(tick);
      return SEASONS[info ? info.name : "temperate"] || SEASONS.temperate;
    }

    function stateAt(index, tick) {
      var t = clamp(tick, 0, N - 1);
      return {
        health: series.health[t][index],
        berries: series.berries[t][index],
        wood: series.wood[t][index],
        crates: series.crates[t][index],
        crateDurability: series.crateDurability[t][index],
        craftProgress: series.craftProgress[t][index],
        priceBelief: series.priceBelief[t][index],
        trust: series.trust[t][index],
        debtWood: series.debtWood[t][index],
        debtBerries: series.debtBerries[t][index],
        debtCrates: series.debtCrates[t][index],
        creditOutWood: series.creditOutWood[t][index],
        creditOutBerries: series.creditOutBerries[t][index],
        creditOutCrates: series.creditOutCrates[t][index],
        storage: series.storage[t][index],
        laborB: series.laborB[t][index],
        laborW: series.laborW[t][index],
        laborC: series.laborC[t][index],
        maxB: series.maxB[t][index],
        maxW: series.maxW[t][index],
        plannedB: series.plannedB[t][index],
        plannedW: series.plannedW[t][index],
        plannedCrates: series.plannedCrates[t][index],
        woodProgress: series.woodProgress[t][index],
        gatheredB: series.gatheredB[t][index],
        gatheredW: series.gatheredW[t][index],
        postTradeB: series.postTradeB[t][index],
        postTradeW: series.postTradeW[t][index],
        alive: series.alive[t][index]
      };
    }

    function metricsAt(tick) {
      var t = clamp(tick, 0, N - 1);
      return {
        alive: series.aliveCount[t],
        meanHealth: series.meanHealth[t],
        totalBerries: series.totalBerries[t],
        totalWood: series.totalWood[t],
        totalCrates: series.totalCrates[t],
        crateDurability: series.crateDurability[t],
        cratesCrafted: series.cratesCrafted[t],
        cratesWorn: series.cratesWorn[t],
        commonsBerries: series.commonsBerries[t],
        commonsWood: series.commonsWood[t],
        commonsBerriesPct: series.commonsBerriesPct[t],
        commonsWoodPct: series.commonsWoodPct[t],
        price: series.price[t],
        lastCratePrice: series.lastCratePrice[t],
        meanPriceBelief: series.meanPriceBelief[t],
        marketParticipants: series.marketParticipants[t],
        marketTraders: series.marketTraders[t],
        failedTrades: series.failedTrades[t],
        spoiled: series.spoiled[t],
        tradeBerries: series.tradeBerries[t],
        tradeWood: series.tradeWood[t],
        crateTrades: series.crateTrades[t],
        ppfEfficiency: series.ppfEfficiency[t],
        storageUtilization: series.storageUtilization[t],
        specialisation: series.specialisation[t],
        giniBerries: series.giniBerries[t],
        giniWood: series.giniWood[t],
        gifts: series.gifts[t],
        unmetFood: series.unmetFood[t],
        berriesGathered: series.berriesGathered[t],
        woodGathered: series.woodGathered[t],
        loans: series.loans[t],
        loanPrincipalWood: series.loanPrincipalWood[t],
        loanPrincipalBerries: series.loanPrincipalBerries[t],
        loanPrincipalCrates: series.loanPrincipalCrates[t],
        meanInterest: series.meanInterest[t],
        loansRepaid: series.loansRepaid[t],
        loansDefaulted: series.loansDefaulted[t],
        creditOutstandingWood: series.creditOutstandingWood[t],
        creditOutstandingBerries: series.creditOutstandingBerries[t],
        creditOutstandingCrates: series.creditOutstandingCrates[t],
        meanTrust: series.meanTrust[t],
        shockActive: series.shockActive[t],
        shockCount: series.shockCount[t],
        giftObligations: series.giftObligations[t]
      };
    }

    function eventsAt(tick) {
      return view.eventsByTick[tick] || [];
    }

    function aliveAt(index, tick) {
      return Boolean(series.alive[clamp(tick, 0, N - 1)][index]);
    }

    // ---------------------------------------------------------------- resize
    function resizeWorld() {
      var rect = canvas.parentElement.getBoundingClientRect();
      var dpr = global.devicePixelRatio || 1;
      state.dpr = dpr;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      state.scale = Math.min(rect.width / LOGICAL_W, rect.height / LOGICAL_H);
      state.offsetX = (rect.width - LOGICAL_W * state.scale) / 2;
      state.offsetY = (rect.height - LOGICAL_H * state.scale) / 2;
    }

    function toCanvas(x, y) {
      return {
        x: (state.offsetX + x * state.scale) * state.dpr,
        y: (state.offsetY + y * state.scale) * state.dpr
      };
    }

    function toLogical(x, y) {
      return {
        x: (x - state.offsetX) / state.scale,
        y: (y - state.offsetY) / state.scale
      };
    }

    // --------------------------------------------------------------- drawing
    function drawBackground() {
      var style = seasonStyleAt(state.tick);
      var gradient = ctx.createLinearGradient(0, 0, 0, LOGICAL_H);
      gradient.addColorStop(0, style.sky);
      gradient.addColorStop(0.55, style.canvas);
      gradient.addColorStop(1, style.deep);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, LOGICAL_W, LOGICAL_H);

      // Subtle field contours.
      ctx.save();
      ctx.globalAlpha = 0.16;
      ctx.strokeStyle = "#b9c4a9";
      ctx.lineWidth = 1;
      for (var i = 0; i < 8; i += 1) {
        ellipse(ctx, CENTER.x, CENTER.y, 116 + i * 52, 86 + i * 38);
        ctx.stroke();
      }
      ctx.restore();

      // Village ground.
      ctx.save();
      ellipse(ctx, CENTER.x, CENTER.y, 306, 268);
      ctx.fillStyle = style.ground;
      ctx.fill();
      ctx.strokeStyle = "rgba(184, 174, 154, 0.5)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    }

    function drawWildTrees() {
      var style = seasonStyleAt(state.tick);
      ctx.save();
      ctx.globalAlpha = 0.42;
      wildTrees.forEach(function (tree) {
        drawTreeShape(
          tree.x,
          tree.y,
          0.72,
          style.wild,
          mixColor(style.wild, "#f4efe6", 0.25),
          tree.size
        );
      });
      ctx.restore();
    }

    function drawPaths() {
      ctx.save();
      ctx.strokeStyle = "rgba(178, 168, 148, 0.55)";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 7]);
      homes.forEach(function (home) {
        ctx.beginPath();
        ctx.moveTo(home.x, home.y);
        ctx.lineTo(CENTER.x, CENTER.y);
        ctx.stroke();
      });
      ctx.restore();
    }

    function drawCommons(metrics) {
      var berryRatio = meta.initialCommons.berries > 0
        ? metrics.commonsBerries / meta.initialCommons.berries
        : 0;
      var woodRatio = meta.initialCommons.wood > 0
        ? metrics.commonsWood / meta.initialCommons.wood
        : 0;

      berrySites.forEach(function (site) {
        drawBerryBush(site, clamp(berryRatio, 0, 1.2));
      });
      treeSites.forEach(function (site) {
        drawTree(site, clamp(woodRatio, 0, 1.2));
      });

      // Commons labels.
      ctx.save();
      ctx.fillStyle = "rgba(44, 54, 40, 0.6)";
      ctx.font = "10px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText("BERRY BUSHES", CENTER.x - 250, LOGICAL_H - 62);
      ctx.fillText("FOREST", CENTER.x + 250, LOGICAL_H - 62);
      ctx.restore();
    }

    function drawBerryBush(site, ratio) {
      var style = seasonStyleAt(state.tick);
      var size = 15 + 9 * clamp(ratio, 0, 1.2);
      ctx.save();
      ctx.globalAlpha = 0.62 + 0.38 * clamp(ratio, 0, 1);
      var wobble = randomFor(site.seed) * 0.5;
      for (var i = 0; i < 4; i += 1) {
        var angle = (i / 4) * Math.PI * 2 + wobble;
        ellipse(
          ctx,
          site.x + Math.cos(angle) * size * 0.48,
          site.y + Math.sin(angle) * size * 0.34,
          size * 0.7,
          size * 0.55
        );
        ctx.fillStyle = i % 2 === 0
          ? mixColor(style.leaf, "#3f5a3a", 0.25)
          : mixColor(style.leafLight, "#55764f", 0.3);
        ctx.fill();
      }
      var berries = Math.round(clamp(ratio, 0, 1) * 9);
      for (var b = 0; b < berries; b += 1) {
        var rx = site.x - size * 0.55 + randomFor(site.seed * 13 + b) * size * 1.1;
        var ry = site.y - size * 0.25 + randomFor(site.seed * 17 + b) * size * 0.7;
        ellipse(ctx, rx, ry, 2.1, 2.1);
        ctx.fillStyle = COLORS.berry;
        ctx.fill();
      }
      ctx.restore();
    }

    function drawTreeShape(x, y, ratio, leaf, leafLight, scale) {
      var s = scale == null ? 1 : scale;
      var height = (28 + 22 * clamp(ratio, 0, 1.2)) * s;
      var width = (9 + 4 * clamp(ratio, 0, 1)) * s;
      ctx.fillStyle = COLORS.woodDark;
      ctx.fillRect(x - width / 2, y, width, height * 0.72);
      var canopy = (13 + 12 * clamp(ratio, 0, 1.2)) * s;
      ellipse(ctx, x, y - canopy * 0.12, canopy, canopy * 0.72);
      ctx.fillStyle = leaf;
      ctx.fill();
      ellipse(ctx, x - canopy * 0.34, y + canopy * 0.16, canopy * 0.62, canopy * 0.5);
      ctx.fillStyle = leafLight;
      ctx.fill();
      ellipse(ctx, x + canopy * 0.34, y + canopy * 0.2, canopy * 0.56, canopy * 0.46);
      ctx.fillStyle = leaf;
      ctx.fill();
    }

    function drawTree(site, ratio) {
      var style = seasonStyleAt(state.tick);
      ctx.save();
      ctx.globalAlpha = 0.55 + 0.45 * clamp(ratio, 0, 1);
      drawTreeShape(
        site.x,
        site.y,
        ratio,
        style.leaf,
        mixColor(style.leafLight, "#6f9466", 0.35),
        1
      );
      ctx.restore();
    }

    function drawMarket(metrics) {
      var x = CENTER.x;
      var y = CENTER.y;
      ctx.save();

      // Stall shadow.
      ellipse(ctx, x, y + 34, 86, 22);
      ctx.fillStyle = "rgba(72, 62, 48, 0.18)";
      ctx.fill();

      // Table.
      roundedRect(ctx, x - 74, y - 2, 148, 38, 5);
      ctx.fillStyle = "#c79a62";
      ctx.fill();
      ctx.strokeStyle = "#8b6234";
      ctx.lineWidth = 1.4;
      ctx.stroke();

      // Posts.
      ctx.fillStyle = "#8b6234";
      ctx.fillRect(x - 76, y - 34, 5, 36);
      ctx.fillRect(x + 71, y - 34, 5, 36);

      // Awning.
      ctx.beginPath();
      ctx.moveTo(x - 88, y - 34);
      ctx.lineTo(x, y - 58);
      ctx.lineTo(x + 88, y - 34);
      ctx.closePath();
      ctx.fillStyle = "rgba(47, 111, 115, 0.88)";
      ctx.fill();
      ctx.strokeStyle = "#25595c";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Berry bowl.
      ellipse(ctx, x - 28, y + 11, 20, 9);
      ctx.fillStyle = "#b8875a";
      ctx.fill();
      var berries = clamp(Math.round(metrics.tradeBerries * 2 + 2), 2, 9);
      for (var i = 0; i < berries; i += 1) {
        ellipse(ctx, x - 38 + i * 5, y + 8 + (i % 2) * 3, 2.6, 2.6);
        ctx.fillStyle = COLORS.berry;
        ctx.fill();
      }

      // Wood pile.
      for (var w = 0; w < 4; w += 1) {
        roundedRect(ctx, x + 12 + (w % 2) * 9, y + 4 + w * 6, 22, 5, 2.5);
        ctx.fillStyle = w % 2 === 0 ? "#9a6a3a" : "#ad7a45";
        ctx.fill();
        ctx.strokeStyle = "#6f4725";
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }

      // Sign.
      roundedRect(ctx, x - 48, y - 28, 96, 20, 3);
      ctx.fillStyle = "rgba(244, 239, 230, 0.94)";
      ctx.fill();
      ctx.strokeStyle = "#9b8d77";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = COLORS.ink;
      ctx.font = "9px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(
        "1 wood = " + formatNumber(metrics.price, 2) + " berries",
        x,
        y - 14
      );
      ctx.restore();
    }

    function healthColor(fraction) {
      if (fraction >= 0.6) return mixColor("#c88a1e", "#4e6b50", (fraction - 0.6) / 0.4);
      return mixColor("#b84a39", "#c88a1e", fraction / 0.6);
    }

    function drawHut(index, home, now) {
      var agent = agents[index];
      var agentState = stateAt(index, state.tick);
      var selected = state.selected === index;
      var hovered = state.hover === index;
      var x = home.x;
      var y = home.y;
      var scale = home.scale || 1;
      var maxHealth = meta.health.max || 4;
      var healthFraction = clamp(agentState.health / maxHealth, 0, 1);
      var alive = agentState.alive;
      var labelled = A <= 40 || selected || hovered;

      ctx.save();
      ctx.translate(x, y);
      ctx.scale(scale, scale);

      if (!alive) {
        drawTombstone(0, 0, agent.color, selected || hovered);
        ctx.restore();
        if (labelled) drawHutLabel(agent, x, y, scale, selected);
        return;
      }

      // Selection glow.
      if (selected || hovered) {
        ctx.save();
        ctx.globalAlpha = selected ? 0.5 : 0.28;
        ellipse(ctx, 0, 8, 38, 27);
        ctx.fillStyle = agent.color;
        ctx.fill();
        ctx.restore();
      }

      // Health ring.
      ctx.save();
      ctx.lineWidth = 5;
      ctx.strokeStyle = "rgba(28, 27, 24, 0.12)";
      ctx.beginPath();
      ctx.arc(0, -2, 30, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = healthColor(healthFraction);
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.arc(
        0,
        -2,
        30,
        -Math.PI / 2,
        -Math.PI / 2 + Math.PI * 2 * healthFraction
      );
      ctx.stroke();
      ctx.restore();

      // Hut body.
      roundedRect(ctx, -15, -4, 30, 22, 3);
      ctx.fillStyle = "#e4d3b3";
      ctx.fill();
      ctx.strokeStyle = "#9b7d55";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Roof.
      ctx.beginPath();
      ctx.moveTo(-23, -4);
      ctx.lineTo(0, -24);
      ctx.lineTo(23, -4);
      ctx.closePath();
      ctx.fillStyle = agent.color;
      ctx.fill();
      ctx.strokeStyle = "rgba(28, 27, 24, 0.28)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Door.
      roundedRect(ctx, -4, 6, 8, 12, 1.5);
      ctx.fillStyle = "#7b5a37";
      ctx.fill();

      // Berry basket.
      var berryFill = clamp(agentState.berries / Math.max(4, meta.health.food_per_iteration * 4), 0, 1);
      ctx.save();
      ctx.translate(-31, 14);
      ctx.beginPath();
      ctx.moveTo(-8, 0);
      ctx.lineTo(8, 0);
      ctx.lineTo(6, 8);
      ctx.lineTo(-6, 8);
      ctx.closePath();
      ctx.fillStyle = "#b8875a";
      ctx.fill();
      ctx.strokeStyle = "#7b5a37";
      ctx.lineWidth = 1;
      ctx.stroke();
      var dots = Math.round(berryFill * 5);
      for (var i = 0; i < dots; i += 1) {
        ellipse(ctx, -4 + i * 2.2, 1 + (i % 2) * 1.6, 1.6, 1.6);
        ctx.fillStyle = COLORS.berry;
        ctx.fill();
      }
      ctx.restore();

      // Wood pile.
      var woodCount = Math.round(clamp(agentState.wood / 8, 0, 1) * 4);
      ctx.save();
      ctx.translate(24, 15);
      for (var w = 0; w < woodCount; w += 1) {
        roundedRect(ctx, -8 + (w % 2) * 5, -w * 4, 13, 3.4, 1.7);
        ctx.fillStyle = w % 2 === 0 ? COLORS.wood : "#ad7a45";
        ctx.fill();
        ctx.strokeStyle = COLORS.woodDark;
        ctx.lineWidth = 0.6;
        ctx.stroke();
      }
      ctx.restore();

      // Storage crate.
      if (agentState.storage > 0) {
        var crateWidth = 20;
        var crateHeight = 9;
        var cx = -crateWidth / 2;
        var cy = 24;
        roundedRect(ctx, cx, cy, crateWidth, crateHeight, 2);
        ctx.fillStyle = "rgba(199, 154, 98, 0.9)";
        ctx.fill();
        ctx.strokeStyle = "#8b6234";
        ctx.lineWidth = 1;
        ctx.stroke();
        var stored = clamp(agentState.berries / Math.max(agentState.storage, 0.001), 0, 1);
        var storedDots = Math.round(stored * 4);
        for (var d = 0; d < storedDots; d += 1) {
          ellipse(ctx, cx + 4 + d * 4, cy + 4.5, 1.4, 1.4);
          ctx.fillStyle = COLORS.berry;
          ctx.fill();
        }
        if (agentState.berries > agentState.storage + 0.01) {
          ctx.fillStyle = COLORS.alert;
          ctx.font = "bold 9px 'IBM Plex Mono', monospace";
          ctx.textAlign = "left";
          ctx.fillText("!", cx + crateWidth + 2, cy + crateHeight);
        }
      }

      ctx.restore();
      if (labelled) drawHutLabel(agent, x, y, scale, selected);
    }

    function drawHutLabel(agent, x, y, scale, selected) {
      ctx.save();
      ctx.fillStyle = selected ? COLORS.ink : "rgba(28, 27, 24, 0.72)";
      ctx.font = (selected ? "600 " : "") + "9px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(agent.label, x, y + 46 * scale + 2);
      ctx.restore();
    }

    function drawTombstone(x, y, color, highlighted) {
      ctx.save();
      if (highlighted) {
        ctx.globalAlpha = 0.3;
        ellipse(ctx, x, y + 12, 28, 18);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      roundedRect(ctx, x - 11, y - 10, 22, 26, 10);
      ctx.fillStyle = COLORS.stone;
      ctx.fill();
      ctx.strokeStyle = "#6f6a61";
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.fillStyle = "rgba(28, 27, 24, 0.45)";
      ctx.font = "11px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText("+", x, y + 8);
      ctx.restore();
    }

    // ------------------------------------------------------------- workers
    function commonsDensity(kind, metrics) {
      var value = kind === "berries"
        ? metrics.commonsBerriesPct
        : metrics.commonsWoodPct;
      if (value == null) {
        var initial = kind === "berries"
          ? meta.initialCommons.berries
          : meta.initialCommons.wood;
        var stock = kind === "berries"
          ? metrics.commonsBerries
          : metrics.commonsWood;
        value = initial > 0 ? stock / initial : 0;
      }
      var numeric = Number(value);
      if (!Number.isFinite(numeric)) numeric = 1;
      return clamp(numeric, 0, 1);
    }

    function routeForAgent(index, kind, density) {
      var sites = kind === "berries" ? berrySites : treeSites;
      var scarcity = 1 - clamp(density, 0, 1);
      // Full commons: one stop. Mild depletion already adds a second stop;
      // very thin commons sends the worker around up to six sites.
      var visits = 1 + Math.round(scarcity * 6);
      visits = Math.min(sites.length, Math.max(1, visits));
      var start = index % sites.length;
      var step = index % 2 === 0 ? 1 : sites.length - 1;
      var route = [];
      for (var v = 0; v < visits; v += 1) {
        route.push(sites[(start + v * step) % sites.length]);
      }
      return route;
    }

    function assignmentForAgent(index, agentState, metrics) {
      if (agentState.laborB <= 0.001 && agentState.laborW <= 0.001) {
        return null;
      }
      var kind = agentState.laborB >= agentState.laborW ? "berries" : "wood";
      var density = commonsDensity(kind, metrics);
      return {
        kind: kind,
        density: density,
        route: routeForAgent(index, kind, density)
      };
    }

    function drawWorkers(now) {
      var metrics = metricsAt(state.tick);
      homes.forEach(function (home, index) {
        var agentState = stateAt(index, state.tick);
        if (!agentState.alive) return;
        var assignment = assignmentForAgent(index, agentState, metrics);
        if (!assignment) return;
        var route = assignment.route;
        var scarcity = 1 - assignment.density;
        // Sparse commons also makes the whole search cycle take longer.
        var cycle = 1800 + scarcity * 1200;
        var phase = ((now - state.startedAt) / cycle + index * 0.137) % 1;
        var x;
        var y;
        var working = false;
        if (phase < 0.26) {
          var walkOut = easeInOut(phase / 0.26);
          x = lerp(home.x, route[0].x, walkOut);
          y = lerp(home.y, route[0].y, walkOut) - Math.sin(walkOut * Math.PI) * 14;
        } else if (phase < 0.74) {
          var search = (phase - 0.26) / 0.48;
          var segmentSize = 1 / route.length;
          var segment = Math.min(
            route.length - 1,
            Math.floor(search / segmentSize)
          );
          var local = (search - segment * segmentSize) / segmentSize;
          var current = route[segment];
          var next = route[segment + 1];
          if (next && local >= 0.58) {
            // Walk to the next sparse bush/tree: visible search time.
            var travel = easeInOut((local - 0.58) / 0.42);
            x = lerp(current.x, next.x, travel);
            y = lerp(current.y, next.y, travel) - Math.sin(travel * Math.PI) * 8;
          } else {
            var workLocal = next ? local / 0.58 : local;
            x = current.x + Math.cos(workLocal * Math.PI * 4) * 2.4;
            y = current.y + 9 + Math.sin(workLocal * Math.PI * 8) * 0.8;
            working = true;
          }
        } else {
          var walkBack = easeInOut((phase - 0.74) / 0.26);
          var last = route[route.length - 1];
          x = lerp(last.x, home.x, walkBack);
          y = lerp(last.y, home.y, walkBack) - Math.sin(walkBack * Math.PI) * 14;
        }
        drawWorker(
          x, y, agents[index].color, assignment.kind, working, index, now
        );
      });
    }

    function drawWorker(x, y, color, kind, working, index, now) {
      var bob = working ? Math.sin((now / 110) + index) * 1.2 : Math.sin((now / 180) + index) * 0.6;
      ctx.save();
      ctx.translate(x, y + bob);

      // Shadow.
      ellipse(ctx, 0, 7, 5.5, 2.2);
      ctx.fillStyle = "rgba(60, 52, 42, 0.22)";
      ctx.fill();

      // Legs.
      ctx.strokeStyle = "#5d4b38";
      ctx.lineWidth = 1.8;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(-1.8, 2);
      ctx.lineTo(-2.6, 7);
      ctx.moveTo(1.8, 2);
      ctx.lineTo(2.6, 7);
      ctx.stroke();

      // Body.
      roundedRect(ctx, -4, -5, 8, 9, 3);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "rgba(28, 27, 24, 0.3)";
      ctx.lineWidth = 0.9;
      ctx.stroke();

      // Head.
      ellipse(ctx, 0, -9, 3.4, 3.4);
      ctx.fillStyle = "#d9b18c";
      ctx.fill();
      ctx.strokeStyle = "rgba(28, 27, 24, 0.25)";
      ctx.lineWidth = 0.7;
      ctx.stroke();

      // Hat.
      ctx.beginPath();
      ctx.moveTo(-4.2, -10.5);
      ctx.lineTo(4.2, -10.5);
      ctx.lineTo(2.6, -13.5);
      ctx.lineTo(-2.6, -13.5);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();

      if (kind === "berries") {
        // Basket.
        ctx.save();
        ctx.translate(5.2, 0);
        ctx.beginPath();
        ctx.moveTo(-3, 0);
        ctx.lineTo(3, 0);
        ctx.lineTo(2.3, 4);
        ctx.lineTo(-2.3, 4);
        ctx.closePath();
        ctx.fillStyle = "#b8875a";
        ctx.fill();
        ctx.strokeStyle = "#7b5a37";
        ctx.lineWidth = 0.7;
        ctx.stroke();
        ellipse(ctx, -1, 0.5, 1, 1);
        ctx.fillStyle = COLORS.berry;
        ctx.fill();
        ellipse(ctx, 1.2, 1.2, 1, 1);
        ctx.fillStyle = COLORS.berry;
        ctx.fill();
        ctx.restore();
      } else {
        // Axe, swinging while working.
        var swing = working ? Math.sin((now / 120) + index) * 0.55 : 0.25;
        ctx.save();
        ctx.translate(5, -1);
        ctx.rotate(-0.5 + swing);
        ctx.strokeStyle = "#6f4725";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(0, 4);
        ctx.lineTo(0, -5);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(-2.5, -5);
        ctx.lineTo(2.5, -5);
        ctx.lineTo(1.4, -8);
        ctx.lineTo(-1.4, -8);
        ctx.closePath();
        ctx.fillStyle = "#8c959c";
        ctx.fill();
        ctx.strokeStyle = "#555e64";
        ctx.lineWidth = 0.6;
        ctx.stroke();
        ctx.restore();
      }
      ctx.restore();
    }

    // ------------------------------------------------------------- particles
    function spawnTickEvents(tick) {
      state.particles = [];
      var events = eventsAt(tick);
      var now = global.performance ? global.performance.now() : Date.now();
      events.forEach(function (event, index) {
        if (event.type === "trade") {
          var from = homes[event.actor];
          var to = homes[event.target];
          if (!from || !to) return;
          state.particles.push({
            kind: "berry",
            from: { x: from.x, y: from.y },
            to: { x: to.x, y: to.y },
            control: { x: CENTER.x, y: CENTER.y - 30 },
            start: now,
            duration: 900,
            fade: 160
          });
          state.particles.push({
            kind: "wood",
            from: { x: to.x, y: to.y },
            to: { x: from.x, y: from.y },
            control: { x: CENTER.x, y: CENTER.y + 30 },
            start: now + 110,
            duration: 900,
            fade: 160
          });
        } else if (event.type === "gift") {
          var giver = homes[event.actor];
          var receiver = homes[event.target];
          if (!giver || !receiver) return;
          state.particles.push({
            kind: "berry",
            from: { x: giver.x, y: giver.y },
            to: { x: receiver.x, y: receiver.y },
            control: {
              x: (giver.x + receiver.x) / 2,
              y: (giver.y + receiver.y) / 2 - 50
            },
            start: now,
            duration: 850,
            fade: 160
          });
        } else if (event.type === "loan" || event.type === "repay") {
          var creditFrom = homes[event.actor];
          var creditTo = homes[event.target];
          if (!creditFrom || !creditTo) return;
          state.particles.push({
            kind: event.resource === "berries" ? "berry" : "wood",
            from: { x: creditFrom.x, y: creditFrom.y },
            to: { x: creditTo.x, y: creditTo.y },
            control: {
              x: (creditFrom.x + creditTo.x) / 2,
              y: (creditFrom.y + creditTo.y) / 2 + 44
            },
            start: now,
            duration: 950,
            fade: 180
          });
        } else if (event.type === "spoil") {
          var spoiledHome = homes[event.actor];
          if (!spoiledHome) return;
          state.particles.push({
            kind: "spoil",
            from: { x: spoiledHome.x, y: spoiledHome.y - 8 },
            to: { x: spoiledHome.x + 8, y: spoiledHome.y - 48 },
            control: { x: spoiledHome.x - 12, y: spoiledHome.y - 30 },
            start: now,
            duration: 900,
            fade: 260
          });
        }
      });
    }

    function drawParticles(now) {
      state.particles.forEach(function (particle) {
        var elapsed = now - particle.start;
        if (elapsed < 0) return;
        var t = clamp(elapsed / particle.duration, 0, 1);
        var eased = easeOutCubic(t);
        var x = quadratic(
          particle.from.x,
          particle.control.x,
          particle.to.x,
          eased
        );
        var y = quadratic(
          particle.from.y,
          particle.control.y,
          particle.to.y,
          eased
        );
        var alpha = 1;
        if (particle.duration - elapsed < particle.fade) {
          alpha = clamp((particle.duration - elapsed) / particle.fade, 0, 1);
        }
        ctx.save();
        ctx.globalAlpha = alpha;
        if (particle.kind === "berry") {
          ellipse(ctx, x, y, 3.2, 3.2);
          ctx.fillStyle = COLORS.berry;
          ctx.fill();
          ctx.strokeStyle = "rgba(255,255,255,0.65)";
          ctx.lineWidth = 0.8;
          ctx.stroke();
        } else if (particle.kind === "wood") {
          ctx.translate(x, y);
          ctx.rotate(0.4);
          roundedRect(ctx, -5, -2, 10, 4, 2);
          ctx.fillStyle = COLORS.wood;
          ctx.fill();
          ctx.strokeStyle = COLORS.woodDark;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        } else {
          ctx.globalAlpha = alpha * 0.7;
          ellipse(ctx, x, y, 4.5 + t * 4, 3 + t * 3);
          ctx.fillStyle = COLORS.alert;
          ctx.fill();
        }
        ctx.restore();
      });
    }

    function quadratic(a, b, c, t) {
      var inverse = 1 - t;
      return inverse * inverse * a + 2 * inverse * t * b + t * t * c;
    }

    function renderWorld(now) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.setTransform(
        state.dpr * state.scale,
        0,
        0,
        state.dpr * state.scale,
        state.dpr * state.offsetX,
        state.dpr * state.offsetY
      );

      var metrics = metricsAt(state.tick);
      drawBackground();
      drawWildTrees();
      drawCommons(metrics);
      drawPaths();
      drawMarket(metrics);

      var order = homes.map(function (home, index) {
        return { home: home, index: index };
      });
      order.sort(function (left, right) {
        return left.home.y - right.home.y;
      });
      order.forEach(function (entry) {
        drawHut(entry.index, entry.home, now);
      });

      drawWorkers(now);
      drawParticles(now);

      // Commons percentage tags.
      ctx.save();
      ctx.fillStyle = "rgba(28, 27, 24, 0.55)";
      ctx.font = "10px 'IBM Plex Mono', monospace";
      ctx.textAlign = "left";
      var seasonInfo = seasonInfoAt(state.tick);
      var tag = "berries " + formatPercent(
        meta.initialCommons.berries > 0
          ? metrics.commonsBerries / meta.initialCommons.berries
          : 0
      ) + "  ·  wood " + formatPercent(
        meta.initialCommons.wood > 0
          ? metrics.commonsWood / meta.initialCommons.wood
          : 0
      );
      if (seasonInfo) {
        tag = seasonInfo.name + "  ·  " + tag;
      }
      ctx.fillText(tag, 22, 30);
      ctx.restore();
    }

    // ------------------------------------------------------------------- ui
    function renderHeader(metrics) {
      document.getElementById("statIteration").textContent = state.tick;
      document.getElementById("statAlive").textContent = metrics.alive;
      document.getElementById("statHealth").textContent = formatNumber(metrics.meanHealth, 1);
      document.getElementById("statPrice").textContent = formatNumber(metrics.price, 2);
      var commonsRatio = meta.initialCommons.berries > 0
        ? metrics.commonsBerries / meta.initialCommons.berries
        : 0;
      document.getElementById("statCommons").textContent = formatPercent(commonsRatio);
      document.getElementById("commonsBar").style.width = formatPercent(commonsRatio);
      document.getElementById("statCommonsNote").textContent =
        "wood " + formatPercent(
          meta.initialCommons.wood > 0
            ? metrics.commonsWood / meta.initialCommons.wood
            : 0
        );
      var seasonEl = document.getElementById("statSeason");
      if (seasonEl) {
        var season = seasonInfoAt(state.tick);
        var style = season ? SEASONS[season.name] : SEASONS.temperate;
        seasonEl.textContent = season ? season.name : "temperate";
        document.getElementById("statSeasonNote").textContent = style
          ? style.note
          : "steady growth";
        seasonEl.style.color = season && season.berry <= 0
          ? COLORS.alert
          : COLORS.ink;
      }
      document.getElementById("readout").textContent = state.tick;
      document.getElementById("readoutTotal").textContent = "/ " + meta.iterations;
      document.getElementById("scrub").value = state.tick;
      document.getElementById("feedTick").textContent = "iter " + state.tick;
    }

    function describeTick(metrics) {
      if (state.tick === 0) {
        return "A village with a starter kit of berries, one crate, and its skills.";
      }
      if (metrics.alive === 0) {
        return "The village is empty. " + formatNumber(metrics.totalBerries, 0) +
          " berries and " + formatNumber(metrics.totalWood, 0) +
          " wood remain in the world.";
      }
      var parts = [];
      if (metrics.crateTrades > 0.05) {
        parts.push(
          formatNumber(metrics.crateTrades, 0) + " crates traded at " +
          formatNumber(metrics.lastCratePrice, 2) + " berries each"
        );
      } else if (metrics.tradeBerries > 0.05) {
        parts.push(
          formatNumber(metrics.tradeBerries, 0) + " berries changed hands at " +
          formatNumber(metrics.price, 2) + " berries per wood"
        );
      }
      if (metrics.spoiled > 0.05) {
        parts.push(formatNumber(metrics.spoiled, 0) + " berries spoiled");
      }
      if (metrics.loans > 0.05) {
        parts.push(
          formatNumber(metrics.loans, 0) + " loan" +
          (metrics.loans === 1 ? "" : "s") + " made"
        );
      }
      if (metrics.loansDefaulted > 0.05) {
        parts.push(formatNumber(metrics.loansDefaulted, 0) + " defaulted");
      }
      if (metrics.shockActive > 0.05) {
        parts.push("a commons shock is active");
      }
      if (metrics.unmetFood > 0.05) {
        parts.push(formatNumber(metrics.unmetFood, 0) + " berries short of food");
      }
      if (!parts.length) {
        parts.push(
          formatNumber(metrics.berriesGathered, 0) + " berries and " +
          formatNumber(metrics.woodGathered, 0) + " wood gathered"
        );
      }
      return parts.join(". ") + ".";
    }

    function renderInspector(metrics) {
      var panel = document.getElementById("selectedPanel");
      if (state.selected == null) {
        var social = !!(meta.features.CREDIT || meta.features.RECIPROCITY);
        var creditDetails = social
          ? detail("Credit", formatNumber(metrics.creditOutstandingWood, 0) + " w · " + formatNumber(metrics.creditOutstandingBerries, 0) + " b · " + formatNumber(metrics.creditOutstandingCrates, 0) + " c") +
            detail("Loans", formatNumber(metrics.loans, 0) + " new · " + formatNumber(metrics.loansDefaulted, 0) + " defaulted") +
            detail("Mean trust", formatPercent(metrics.meanTrust))
          : "";
        panel.innerHTML =
          "<div class=\"panel-head\"><div class=\"panel-title\">A villager</div>" +
          "<div class=\"panel-note\">click a hut</div></div>" +
          "<p class=\"muted\" style=\"margin:0 0 10px;font-size:12px\">" +
          "Click a villager to inspect their health, stores, skills, and " +
          "production possibilities. The whole village currently holds " +
          formatNumber(metrics.totalBerries, 0) + " berries, " +
          formatNumber(metrics.totalWood, 0) + " raw wood, and " +
          formatNumber(metrics.totalCrates, 0) + " crates.</p>" +
          "<div class=\"detail-grid\">" +
          detail("Trade", formatNumber(metrics.tradeBerries, 0) + " berries") +
          creditDetails +
          detail("Crate trades", formatNumber(metrics.crateTrades, 0)) +
          detail("Spoiled", formatNumber(metrics.spoiled, 0)) +
          detail("Food shortfall", formatNumber(metrics.unmetFood, 0)) +
          detail("PPF efficiency", formatPercent(metrics.ppfEfficiency)) +
          "</div>";
        return;
      }
      var agent = agents[state.selected];
      var agentState = stateAt(state.selected, state.tick);
      var maxHealth = meta.health.max || 4;
      var healthFraction = clamp(agentState.health / maxHealth, 0, 1);
      var berryScale = Math.max(4, series.storage[state.tick][state.selected] || 4);
      var berryFraction = clamp(agentState.berries / berryScale, 0, 1);
      var woodFraction = clamp(agentState.wood / 10, 0, 1);
      var storageFraction = agentState.storage > 0
        ? clamp(agentState.berries / agentState.storage, 0, 1)
        : 0;
      var craftMinutes = agent.craftMinutes > 0 ? agent.craftMinutes : 100;
      var craftFraction = clamp(agentState.craftProgress / craftMinutes, 0, 1);
      var opportunity = agentState.maxW > 0
        ? agentState.maxB / agentState.maxW
        : null;
      var potentialB = agentState.maxB;
      var potentialW = agentState.maxW;
      var belief = agentState.priceBelief;
      var craftShare = agent.labour > 0 ? agentState.laborC / agent.labour : 0;
      var social = !!(meta.features.CREDIT || meta.features.RECIPROCITY);
      var creditDetails = social
        ? detail("Trust", formatPercent(agentState.trust)) +
          detail("Debt", formatNumber(agentState.debtWood, 0) + " w · " + formatNumber(agentState.debtBerries, 0) + " b · " + formatNumber(agentState.debtCrates, 0) + " c") +
          detail("Lent out", formatNumber(agentState.creditOutWood, 0) + " w · " + formatNumber(agentState.creditOutBerries, 0) + " b · " + formatNumber(agentState.creditOutCrates, 0) + " c")
        : "";

      panel.innerHTML =
        "<div class=\"selected-hero\">" +
        "<span class=\"selected-swatch\" style=\"background:" + agent.color + "\"></span>" +
        "<div><div class=\"selected-name\">" + agent.label + "</div>" +
        "<div class=\"muted\" style=\"font-size:11px\">berry skill " +
        formatNumber(agent.berrySkill, 2) + " · wood skill " +
        formatNumber(agent.woodSkill, 2) + " · craft " +
        formatNumber(craftMinutes, 0) + " min</div></div>" +
        "<div class=\"selected-status" + (agentState.alive ? "" : " dead") + "\">" +
        (agentState.alive ? "alive" : "dead") + "</div></div>" +
        meter("Health", healthFraction, formatNumber(agentState.health, 0) + " / " + formatNumber(maxHealth, 0), "health") +
        meter("Berries", berryFraction, formatNumber(agentState.berries, 0), "berries") +
        meter("Raw wood", woodFraction, formatNumber(agentState.wood, 0), "wood") +
        meter("Crates", clamp(agentState.crates / 3, 0, 1), formatNumber(agentState.crates, 0) + " · " + formatPercent(agentState.crateDurability), "crate") +
        meter("Storage", storageFraction, formatNumber(Math.min(agentState.berries, agentState.storage), 0) + " / " + formatNumber(agentState.storage, 0), "crate") +
        meter("Craft", craftFraction, formatPercent(craftFraction), "craft") +
        "<div class=\"detail-grid\">" +
        detail("Price belief", formatNumber(belief, 0) + " berries / wood") +
        detail("Market price", formatNumber(metrics.price, 2) + " berries / wood") +
        detail("Opportunity cost", opportunity == null ? "-" : formatNumber(opportunity, 2) + " berries / wood") +
        detail("Effective max", formatNumber(potentialB, 1) + " b · " + formatNumber(potentialW, 1) + " w") +
        detail("Harvest density", formatPercent(metrics.commonsBerriesPct) + " b · " + formatPercent(metrics.commonsWoodPct) + " w") +
        detail("Gathered", formatNumber(agentState.gatheredB, 0) + " b · " + formatNumber(agentState.gatheredW, 0) + " w") +
        detail("Chopping", formatNumber(agentState.woodProgress, 2) + " of next block") +
        creditDetails +
        "</div>" +
        "<div class=\"labour-legend\"><span>berry " + formatPercent(agent.labour > 0 ? agentState.laborB / agent.labour : 0) + "</span>" +
        "<span>wood " + formatPercent(agent.labour > 0 ? agentState.laborW / agent.labour : 0) + "</span>" +
        "<span>craft " + formatPercent(craftShare) + "</span></div>" +
        "<div class=\"labour-bar\">" +
        "<i class=\"berry-labour\" style=\"width:" + (agent.labour > 0 ? (agentState.laborB / agent.labour) * 100 : 0) + "%\"></i>" +
        "<i class=\"wood-labour\" style=\"width:" + (agent.labour > 0 ? (agentState.laborW / agent.labour) * 100 : 0) + "%\"></i>" +
        "<i class=\"craft-labour\" style=\"width:" + craftShare * 100 + "%\"></i>" +
        "</div>" +
        "<button class=\"ppf-button\" id=\"inspectorPpf\">Production possibilities</button>";
      var button = document.getElementById("inspectorPpf");
      if (button) button.addEventListener("click", openPpf);
    }

    function detail(label, value) {
      return "<div class=\"detail\"><span>" + label + "</span><b>" + value + "</b></div>";
    }

    function meter(label, fraction, value, kind) {
      return "<div class=\"meter-row\"><span class=\"meter-label\">" + label + "</span>" +
        "<span class=\"meter " + kind + "\"><i style=\"width:" +
        clamp(fraction, 0, 1) * 100 + "%\"></i></span>" +
        "<span class=\"meter-value\">" + value + "</span></div>";
    }

    function renderFeed() {
      var feed = document.getElementById("feed");
      var events = eventsAt(state.tick);
      var rows = [];
      var gather = {};

      events.forEach(function (event) {
        if (event.type === "gather") {
          var key = event.actor;
          if (!gather[key]) {
            gather[key] = { berries: 0, wood: 0 };
          }
          if (event.resource === "berries") gather[key].berries += event.amount;
          else gather[key].wood += event.amount;
        } else if (event.type === "trade") {
          var isCrate = event.good === "crate";
          rows.push({
            order: 1,
            icon: "trade",
            title: agents[event.actor].label + " traded " +
              formatNumber(event.berries, 0) + " berries for " +
              formatNumber(isCrate ? event.crates : event.wood, 0) +
              (isCrate ? " crate" : " wood") + " with " +
              agents[event.target].label,
            sub: formatNumber(event.price, 2) +
              (isCrate ? " berries per crate" : " berries per wood"),
            select: event.actor
          });
        } else if (event.type === "craft") {
          rows.push({
            order: 1,
            icon: "craft",
            title: agents[event.actor].label + " crafted " +
              formatNumber(event.amount, 0) + " crate",
            sub: formatNumber(event.wood, 0) + " raw wood worked",
            select: event.actor
          });
        } else if (event.type === "gift") {
          rows.push({
            order: 2,
            icon: "gift",
            title: agents[event.actor].label + " gave " +
              formatNumber(event.amount, 0) + " berries to " +
              agents[event.target].label,
            sub: "an in-kind gift",
            select: event.target
          });
        } else if (event.type === "loan") {
          rows.push({
            order: 2,
            icon: "loan",
            title: agents[event.actor].label + " lent " +
              formatNumber(event.principal, 0) + " " +
              (event.resource === "wood" ? "wood" :
                event.resource === "crates" ? "crate" : "berries") +
              " to " + agents[event.target].label,
            sub: "repay " + formatNumber(event.repayment, 0) + " in " +
              formatNumber(event.dueIteration, 0) +
              (event.interestRate > 0.001
                ? " · " + formatPercent(event.interestRate) + " interest"
                : " · reciprocal"),
            select: event.target
          });
        } else if (event.type === "repay") {
          rows.push({
            order: 2,
            icon: "repay",
            title: agents[event.actor].label + " repaid " +
              formatNumber(event.amount, 0) + " " +
              (event.resource === "wood" ? "wood" :
                event.resource === "crates" ? "crate" : "berries") +
              " to " + agents[event.target].label,
            sub: "principal " + formatNumber(event.principal, 0),
            select: event.actor
          });
        } else if (event.type === "default") {
          rows.push({
            order: 1,
            icon: "default",
            title: agents[event.actor].label + " defaulted to " +
              agents[event.target].label,
            sub: "lost " + formatNumber(event.loss, 0) + " " +
              (event.resource === "wood" ? "wood" :
                event.resource === "crates" ? "crate" : "berries") +
              (event.collateralSeized > 0
                ? " · " + formatNumber(event.collateralSeized, 0) + " crate seized"
                : "") +
              (event.collateralExcess > 0.0001
                ? " · " + formatNumber(event.collateralExcess, 1) +
                  " collateral excess"
                : ""),
            select: event.actor
          });
        } else if (event.type === "write_off") {
          rows.push({
            order: 2,
            icon: "quiet",
            title: "A claim on " + agents[event.actor].label + " was written off",
            sub: formatNumber(event.loss, 0) + " " +
              (event.resource === "wood" ? "wood" :
                event.resource === "crates" ? "crate" : "berries") +
              " · " + (event.reason === "creditor_death"
                ? "creditor died"
                : "no surviving holder"),
            select: event.actor
          });
        } else if (event.type === "late") {
          rows.push({
            order: 2,
            icon: "late",
            title: agents[event.actor].label + " missed a payment to " +
              agents[event.target].label,
            sub: "grace period extended",
            select: event.actor
          });
        } else if (event.type === "shock") {
          rows.push({
            order: 0,
            icon: "shock",
            title: "A " + event.shockType + " reduced commons regrowth",
            sub: formatNumber(event.duration, 0) + " days · berries " +
              formatPercent(event.berryFactor) + " · wood " +
              formatPercent(event.woodFactor),
            select: state.selected == null ? 0 : state.selected
          });
        } else if (event.type === "spoil") {
          rows.push({
            order: 3,
            icon: "spoil",
            title: agents[event.actor].label + " lost " +
              formatNumber(event.amount, 0) + " berries to spoilage",
            sub: "outside storage",
            select: event.actor
          });
        } else if (event.type === "death") {
          rows.push({
            order: 0,
            icon: "death",
            title: agents[event.actor].label + " died of hunger",
            sub: "health reached zero",
            select: event.actor
          });
        }
      });

      Object.keys(gather).forEach(function (key) {
        var index = Number(key);
        var entry = gather[index];
        if (entry.berries <= 0 && entry.wood <= 0) return;
        var pieces = [];
        if (entry.berries > 0.01) pieces.push(formatNumber(entry.berries, 0) + " berries");
        if (entry.wood > 0.01) pieces.push(formatNumber(entry.wood, 0) + " wood");
        var agentState = stateAt(index, state.tick);
        rows.push({
          order: 4,
          icon: entry.berries >= entry.wood ? "gather" : "wood",
          title: agents[index].label + " gathered " + pieces.join(" and "),
          sub: "labour: " + formatPercent(agentState.laborB / Math.max(agents[index].labour, 0.001)) +
            " berries · " + formatPercent(agentState.laborW / Math.max(agents[index].labour, 0.001)) + " wood",
          select: index
        });
      });

      rows.sort(function (left, right) { return left.order - right.order; });
      if (!rows.length) {
        feed.innerHTML =
          "<div class=\"feed-row\"><span class=\"feed-icon quiet\">" + FEED_ICONS.quiet + "</span>" +
          "<div class=\"feed-main\"><b>Nothing happened</b><div class=\"feed-sub\">" +
          "no gather, trade, gift, credit, shock, spoilage, or death was recorded.</div></div></div>";
        return;
      }
      var shown = rows.slice(0, 14);
      feed.innerHTML = shown.map(function (row) {
        return "<div class=\"feed-row\" data-select=\"" + row.select + "\">" +
          "<span class=\"feed-icon " + row.icon + "\">" + (FEED_ICONS[row.icon] || FEED_ICONS.quiet) + "</span>" +
          "<div class=\"feed-main\"><b>" + row.title + "</b>" +
          "<div class=\"feed-sub\">" + row.sub + "</div></div></div>";
      }).join("") +
        (rows.length > shown.length
          ? "<div class=\"feed-row\"><span class=\"feed-icon quiet\">" + FEED_ICONS.quiet + "</span>" +
            "<div class=\"feed-main\"><b>" + (rows.length - shown.length) + " more records</b>" +
            "<div class=\"feed-sub\">step or scrub to another iteration</div></div></div>"
          : "");
      feed.querySelectorAll("[data-select]").forEach(function (element) {
        element.addEventListener("click", function () {
          selectAgent(Number(element.getAttribute("data-select")));
        });
      });
    }

    function renderUi() {
      var metrics = metricsAt(state.tick);
      renderHeader(metrics);
      document.getElementById("story").textContent = describeTick(metrics);
      renderInspector(metrics);
      renderFeed();
      if (state.ppfOpen) renderPpf();
    }

    // ------------------------------------------------------------------ ppf
    function openPpf() {
      state.ppfOpen = true;
      document.getElementById("ppfModal").classList.add("visible");
      document.getElementById("ppfAll").checked = state.ppfAll;
      document.getElementById("ppfZoom").checked = state.ppfZoom;
      renderPpf();
    }

    function closePpf() {
      state.ppfOpen = false;
      document.getElementById("ppfModal").classList.remove("visible");
    }

    function renderPpf() {
      var svg = document.getElementById("ppfChart");
      var title = document.getElementById("ppfTitle");
      var sub = document.getElementById("ppfSub");
      var legend = document.getElementById("ppfLegend");
      svg.innerHTML = "";
      legend.innerHTML = "";

      var selected = state.selected;
      if (!state.ppfAll && selected == null) {
        if (metricsAt(state.tick).alive === 0) {
          title.textContent = "No living villagers";
          sub.textContent = "The frontier is empty at this iteration.";
          return;
        }
        for (var candidate = 0; candidate < A; candidate += 1) {
          if (aliveAt(candidate, state.tick)) {
            selected = candidate;
            break;
          }
        }
        if (selected == null) selected = 0;
      }
      var indices = [];
      if (state.ppfAll) {
        for (var i = 0; i < A; i += 1) {
          if (aliveAt(i, state.tick) || i === selected) indices.push(i);
        }
      } else {
        indices.push(selected);
      }
      if (!indices.length) {
        title.textContent = "No living villagers";
        sub.textContent = "The frontier is empty at this iteration.";
        return;
      }

      var maxB = 1;
      var maxW = 1;
      indices.forEach(function (index) {
        var agent = agents[index];
        var agentState = stateAt(index, state.tick);
        var axisB = state.ppfZoom ? agentState.maxB : agent.maxBerries;
        var axisW = state.ppfZoom ? agentState.maxW : agent.maxWood;
        maxB = Math.max(
          maxB,
          axisB,
          agentState.gatheredB,
          agentState.plannedB
        );
        maxW = Math.max(
          maxW,
          axisW,
          agentState.gatheredW,
          agentState.plannedW
        );
      });
      maxB *= 1.16;
      maxW *= 1.16;

      var width = 900;
      var height = 560;
      var margin = { left: 72, right: 28, top: 34, bottom: 58 };
      var plotW = width - margin.left - margin.right;
      var plotH = height - margin.top - margin.bottom;

      function x(value) {
        return margin.left + (value / maxB) * plotW;
      }
      function y(value) {
        return margin.top + plotH - (value / maxW) * plotH;
      }

      // Grid.
      for (var g = 0; g <= 5; g += 1) {
        var gx = margin.left + (plotW * g) / 5;
        var gy = margin.top + (plotH * g) / 5;
        svgAppend(svg, "line", {
          x1: gx, y1: margin.top, x2: gx, y2: margin.top + plotH,
          stroke: "#e4dccd", "stroke-width": 1
        });
        svgAppend(svg, "line", {
          x1: margin.left, y1: gy, x2: margin.left + plotW, y2: gy,
          stroke: "#e4dccd", "stroke-width": 1
        });
        svgAppend(svg, "text", {
          x: gx, y: height - 34, "text-anchor": "middle",
          fill: "#666055", "font-size": 10, "font-family": "IBM Plex Mono, monospace"
        }, formatNumber((maxB * g) / 5, 1));
        svgAppend(svg, "text", {
          x: margin.left - 10, y: gy + 3, "text-anchor": "end",
          fill: "#666055", "font-size": 10, "font-family": "IBM Plex Mono, monospace"
        }, formatNumber((maxW * (5 - g)) / 5, 1));
      }

      svgAppend(svg, "text", {
        x: margin.left + plotW / 2, y: height - 10, "text-anchor": "middle",
        fill: "#1c1b18", "font-size": 12, "font-family": "IBM Plex Sans, sans-serif"
      }, "Berries per iteration");
      svgAppend(svg, "text", {
        x: 16, y: margin.top + plotH / 2, "text-anchor": "middle",
        fill: "#1c1b18", "font-size": 12, "font-family": "IBM Plex Sans, sans-serif",
        transform: "rotate(-90 16 " + (margin.top + plotH / 2) + ")"
      }, "Wood per iteration");

      var defs = svgAppend(svg, "defs");
      var clip = svgAppend(defs, "clipPath", { id: "ppfPlotClip" });
      svgAppend(clip, "rect", {
        x: margin.left,
        y: margin.top,
        width: plotW,
        height: plotH
      });
      var plot = svgAppend(svg, "g", { "clip-path": "url(#ppfPlotClip)" });

      var highlight = selected != null ? selected : indices[0];
      indices.forEach(function (index) {
        var agent = agents[index];
        var agentState = stateAt(index, state.tick);
        var isSelected = index === highlight;
        var opacity = isSelected ? 1 : 0.24;
        var potentialB = agentState.maxB;
        var potentialW = agentState.maxW;
        var plannedB = agentState.plannedB;
        var plannedW = agentState.plannedW;

        svgAppend(plot, "line", {
          x1: x(0), y1: y(potentialW),
          x2: x(potentialB), y2: y(0),
          stroke: agent.color,
          "stroke-width": isSelected ? 2.5 : 1.2,
          opacity: opacity,
          "stroke-linecap": "round"
        });

        if (isSelected) {
          // Price line at the current exchange ratio. In the full view it
          // passes through post-barter holdings; in the zoomed frontier view
          // it passes through the chosen production point so it stays visible.
          var price = stateAt(highlight, state.tick).priceBelief;
          if (price > 0) {
            var referenceB = plannedB;
            var referenceW = plannedW;
            var interceptB = referenceB + price * referenceW;
            var wMin = Math.max(0, (interceptB - maxB) / price);
            var wMax = Math.min(maxW, interceptB / price);
            if (wMin <= wMax) {
              svgAppend(plot, "line", {
                x1: x(interceptB - price * wMin), y1: y(wMin),
                x2: x(interceptB - price * wMax), y2: y(wMax),
                stroke: "#666055",
                "stroke-width": 1.2,
                "stroke-dasharray": "5 5",
                opacity: 0.75
              });
            }
          }
        }

        // Planned production on the frontier.
        svgAppend(plot, "circle", {
          cx: x(plannedB), cy: y(plannedW), r: isSelected ? 4.5 : 2.8,
          fill: "#fbf8f1", stroke: agent.color,
          "stroke-width": isSelected ? 2 : 1.2, opacity: opacity
        });
        // Actual gathered, inside the frontier if the commons limited it.
        svgAppend(plot, "circle", {
          cx: x(agentState.gatheredB), cy: y(agentState.gatheredW),
          r: isSelected ? 4 : 2.4,
          fill: agent.color, opacity: opacity
        });
        // After trade.
        svgAppend(plot, "rect", {
          x: x(agentState.postTradeB) - 3.5, y: y(agentState.postTradeW) - 3.5,
          width: 7, height: 7,
          fill: "#2f6f73", opacity: opacity,
          transform: "rotate(45 " + x(agentState.postTradeB) + " " + y(agentState.postTradeW) + ")"
        });
        // End of iteration.
        svgAppend(plot, "rect", {
          x: x(agentState.berries) - 2.6, y: y(agentState.wood) - 2.6,
          width: 5.2, height: 5.2,
          fill: "#4e6b50", opacity: opacity
        });

        if (isSelected) {
          svgAppend(plot, "text", {
            x: x(agentState.postTradeB) + 8, y: y(agentState.postTradeW) - 8,
            fill: agent.color, "font-size": 11,
            "font-family": "IBM Plex Mono, monospace"
          }, agent.label);
        }
      });

      var chosen = agents[highlight];
      var chosenState = stateAt(highlight, state.tick);
      var opportunity = chosenState.maxW > 0
        ? chosenState.maxB / chosenState.maxW
        : null;
      var tickMetrics = metricsAt(state.tick);
      title.textContent = state.ppfAll
        ? "All villagers"
        : chosen.label;
      var scaleNote = state.ppfZoom
        ? " Axes are scaled to the effective frontier; holdings may fall outside the plot."
        : " Axes show full-commons limits, so the current frontier contracts inside them; holdings may fall outside the plot.";
      var ratioNote =
        "The dashed line is the exchange ratio through the planned production point.";
      var densityNote =
        " Harvest density: bushes " + formatPercent(tickMetrics.commonsBerriesPct) +
        ", forest " + formatPercent(tickMetrics.commonsWoodPct) + ".";
      sub.textContent = state.ppfAll
        ? "Each line is one villager's density-adjusted frontier. Filled dots are actual gathering; diamonds are holdings after barter." + densityNote + scaleNote
        : chosen.label + ": opportunity cost " +
          (opportunity == null ? "—" : formatNumber(opportunity, 2) + " berries per wood") +
          ", market price " + formatNumber(tickMetrics.price, 2) +
          ", belief " + formatNumber(chosenState.priceBelief, 2) +
          ". " + ratioNote + densityNote + scaleNote;

      var legendHtml =
        "<span><i></i>frontier</span>" +
        "<span><i class=\"ring\"></i>planned labour point</span>" +
        "<span><i class=\"dot\"></i>actual gathering</span>";
      if (!state.ppfZoom) {
        legendHtml +=
          "<span><i class=\"diamond\"></i>after barter</span>" +
          "<span><i class=\"square\"></i>end of iteration</span>";
      }
      legendHtml += "<span><i class=\"dashed\"></i>price line</span>";
      legend.innerHTML = legendHtml;
    }

    function svgAppend(parent, tag, attrs, text) {
      var element = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.keys(attrs || {}).forEach(function (key) {
        element.setAttribute(key, attrs[key]);
      });
      if (text != null) element.textContent = text;
      parent.appendChild(element);
      return element;
    }

    // ------------------------------------------------------------- controls
    function setTime(value, force) {
      var next = clamp(Math.round(value), 0, N - 1);
      if (!force && next === state.tick) return;
      state.tick = next;
      spawnTickEvents(next);
      renderUi();
    }

    function togglePlay() {
      state.playing = !state.playing;
      updatePlayButton();
    }

    function updatePlayButton() {
      var icon = document.getElementById("playIcon");
      if (!icon) return;
      icon.innerHTML = state.playing
        ? "<path d=\"M3 1.5v9L10.5 6 3 1.5Z\" fill=\"currentColor\"/>"
        : "<path d=\"M3 1.5v9M9 1.5v9\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\"/>";
    }

    function step(delta) {
      setTime(state.tick + delta);
    }

    function selectAgent(index) {
      state.selected = index;
      renderUi();
    }

    function cycleSpeed() {
      state.speedIndex = (state.speedIndex + 1) % state.speeds.length;
      document.getElementById("speedBtn").textContent =
        state.speeds[state.speedIndex] + "x";
    }

    function screenToAgent(clientX, clientY) {
      var rect = canvas.getBoundingClientRect();
      var logical = toLogical(clientX - rect.left, clientY - rect.top);
      var best = null;
      var bestScore = Infinity;
      homes.forEach(function (home, index) {
        var scale = home.scale || 1;
        var distance = Math.hypot(
          logical.x - home.x,
          logical.y - (home.y + 6 * scale)
        );
        var reach = 34 * scale + 10;
        if (distance < reach && distance / reach < bestScore) {
          best = index;
          bestScore = distance / reach;
        }
      });
      return best;
    }

    function updateTooltip(clientX, clientY) {
      var index = screenToAgent(clientX, clientY);
      state.hover = index;
      if (index == null) {
        tooltip.classList.remove("visible");
        return;
      }
      var agent = agents[index];
      var agentState = stateAt(index, state.tick);
      tooltip.innerHTML =
        "<strong>" + agent.label + "</strong> " +
        (agentState.alive ? "" : "<span class=\"muted\">(dead)</span>") +
        "<br>health " + formatNumber(agentState.health, 0) + " / " + formatNumber(meta.health.max || 4, 0) +
        "<br>berries " + formatNumber(agentState.berries, 0) +
        " · wood " + formatNumber(agentState.wood, 0) +
        "<br>storage " + formatNumber(Math.min(agentState.berries, agentState.storage), 0) +
        " / " + formatNumber(agentState.storage, 0) +
        "<br>skills " + formatNumber(agent.berrySkill, 2) + " b · " +
        formatNumber(agent.woodSkill, 2) + " w";
      tooltip.classList.add("visible");
      var wrap = canvas.parentElement.getBoundingClientRect();
      tooltip.style.left = (clientX - wrap.left + 14) + "px";
      tooltip.style.top = (clientY - wrap.top + 12) + "px";
    }

    function frame(now) {
      var delta = state.lastFrame ? now - state.lastFrame : 16;
      state.lastFrame = now;
      if (state.playing) {
        var perTick = 900 / state.speeds[state.speedIndex];
        state.tickAccumulator += delta;
        if (state.tickAccumulator >= perTick) {
          state.tickAccumulator -= perTick;
          if (state.tick < N - 1) {
            setTime(state.tick + 1);
          } else {
            state.playing = false;
            updatePlayButton();
          }
        }
      }
      renderWorld(now);
      global.requestAnimationFrame(frame);
    }

    // ---------------------------------------------------------------- wiring
    document.getElementById("rewindBtn").addEventListener("click", function () {
      setTime(0);
    });
    document.getElementById("prevBtn").addEventListener("click", function () {
      step(-1);
    });
    document.getElementById("playBtn").addEventListener("click", togglePlay);
    document.getElementById("nextBtn").addEventListener("click", function () {
      step(1);
    });
    document.getElementById("speedBtn").addEventListener("click", cycleSpeed);
    document.getElementById("scrub").addEventListener("input", function (event) {
      setTime(Number(event.target.value), true);
    });
    document.getElementById("helpBtn").addEventListener("click", function () {
      document.getElementById("helpModal").classList.add("visible");
    });
    document.getElementById("helpClose").addEventListener("click", function () {
      document.getElementById("helpModal").classList.remove("visible");
    });
    document.getElementById("helpModal").addEventListener("click", function (event) {
      if (event.target.id === "helpModal") {
        event.currentTarget.classList.remove("visible");
      }
    });
    document.getElementById("ppfBtn").addEventListener("click", openPpf);
    document.getElementById("ppfClose").addEventListener("click", closePpf);
    document.getElementById("ppfModal").addEventListener("click", function (event) {
      if (event.target.id === "ppfModal") closePpf();
    });
    document.getElementById("ppfAll").addEventListener("change", function (event) {
      state.ppfAll = event.target.checked;
      renderPpf();
    });
    document.getElementById("ppfZoom").addEventListener("change", function (event) {
      state.ppfZoom = event.target.checked;
      renderPpf();
    });

    canvas.addEventListener("mousemove", function (event) {
      updateTooltip(event.clientX, event.clientY);
    });
    canvas.addEventListener("mouseleave", function () {
      state.hover = null;
      tooltip.classList.remove("visible");
    });
    canvas.addEventListener("click", function (event) {
      var index = screenToAgent(event.clientX, event.clientY);
      if (index != null) selectAgent(index);
    });
    global.addEventListener("keydown", function (event) {
      if (event.target && /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
      if (event.code === "Space") {
        event.preventDefault();
        togglePlay();
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        step(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        step(1);
      } else if (event.key === "p" || event.key === "P") {
        event.preventDefault();
        if (state.ppfOpen) closePpf();
        else openPpf();
      } else if (event.key === "Escape") {
        closePpf();
        document.getElementById("helpModal").classList.remove("visible");
      }
    });
    global.addEventListener("resize", function () {
      resizeWorld();
      renderWorld(global.performance ? global.performance.now() : Date.now());
    });

    // ---------------------------------------------------------------- start
    state.startedAt = global.performance ? global.performance.now() : Date.now();
    document.getElementById("scrub").max = String(N - 1);
    document.getElementById("speedBtn").textContent =
      state.speeds[state.speedIndex] + "x";
    resizeWorld();
    setTime(0, true);
    updatePlayButton();
    renderWorld(global.performance ? global.performance.now() : Date.now());
    global.requestAnimationFrame(frame);

    var note = "Click a villager to inspect them. Press P for the production frontier.";
    if (meta.extinctionIteration != null) {
      note += " Extinction at iteration " + meta.extinctionIteration + ".";
    }
    worldNote.textContent = note;
    document.getElementById("statCommonsNote").textContent =
      "wood " + formatPercent(
        meta.initialCommons.wood > 0
          ? series.commonsWood[0] / meta.initialCommons.wood
          : 0
      );
  }

  global.startSubsistenceViewer = startSubsistenceViewer;
})(window);
