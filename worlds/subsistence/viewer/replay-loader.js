/* Replay loading and view-model adaptation for the subsistence world.
 *
 * The simulation writes a versioned replay (schema 2, meta.world =
 * "subsistence"). This module turns that stable schema into the compact,
 * index-based view model the canvas and PPF chart use.
 */
(function (global) {
  "use strict";

  var SCHEMA_VERSION = 2;
  var STORAGE_KEY = "subsistenceReplayBytes";

  function assertReplay(replay) {
    if (!replay || typeof replay !== "object") {
      throw new Error("The replay file is empty or is not JSON.");
    }
    if (replay.schema_version !== SCHEMA_VERSION) {
      throw new Error(
        "Unsupported replay schema version " + replay.schema_version +
        "; this viewer reads version " + SCHEMA_VERSION + "."
      );
    }
    if (!replay.meta || replay.meta.world !== "subsistence") {
      throw new Error(
        "This is not a subsistence replay. Open a replay whose meta.world " +
        "is \"subsistence\"."
      );
    }
    if (!Array.isArray(replay.agents) || !replay.agents.length) {
      throw new Error("The replay has no agents.");
    }
    if (!Array.isArray(replay.ticks) || !replay.ticks.length) {
      throw new Error("The replay has no ticks.");
    }
    if (!Array.isArray(replay.events)) {
      throw new Error("The replay has no events list.");
    }
  }

  function buildView(replay) {
    assertReplay(replay);

    var ticks = replay.ticks;
    var agentIds = replay.agents.map(function (agent) { return agent.id; });
    var indexById = {};
    agentIds.forEach(function (id, index) { indexById[id] = index; });

    var agents = replay.agents.map(function (agent, index) {
      return {
        id: agent.id,
        label: agent.label || agent.id,
        color: agent.color || "#475569",
        berrySkill: Number(agent.berry_skill) || 0,
        woodSkill: Number(agent.wood_skill) || 0,
        labour: Number(agent.labour) || 1,
        timePerIteration: Number(agent.time_per_iteration) || 100,
        berryTimeCost: Number(agent.berry_time_cost) || 10,
        woodTimeCost: Number(agent.wood_time_cost) || 80,
        maxBerries: Number(agent.max_berries_per_day) || 0,
        maxWood: Number(agent.max_wood_per_day) || 0,
        storageWoodPerContainer: Number(agent.storage_wood_per_container) || 8,
        storageBerriesPerContainer: Number(agent.storage_berries_per_container) || 10,
        initialHealth: Number(agent.initial_health) || 0,
        index: index
      };
    });

    var N = ticks.length;
    var A = agents.length;
    var metrics = {
      aliveCount: [],
      meanHealth: [],
      totalBerries: [],
      totalWood: [],
      commonsBerries: [],
      commonsWood: [],
      price: [],
      spoiled: [],
      tradeBerries: [],
      tradeWood: [],
      ppfEfficiency: [],
      storageUtilization: [],
      specialisation: [],
      giniBerries: [],
      giniWood: [],
      gifts: [],
      unmetFood: [],
      berriesGathered: [],
      woodGathered: [],
      storageCapacity: []
    };
    var eventTicks = [];
    var deaths = [];
    var extinctionIteration = null;
    var lastAliveIteration = 0;

    for (var t = 0; t < N; t += 1) {
      var tick = ticks[t];
      var rowMetrics = tick.metrics || {};
      metrics.aliveCount.push(Number(rowMetrics.alive) || 0);
      metrics.meanHealth.push(Number(rowMetrics.mean_health) || 0);
      metrics.totalBerries.push(Number(rowMetrics.total_berries) || 0);
      metrics.totalWood.push(Number(rowMetrics.total_wood) || 0);
      metrics.commonsBerries.push(Number(rowMetrics.commons_berries) || 0);
      metrics.commonsWood.push(Number(rowMetrics.commons_wood) || 0);
      metrics.price.push(Number(rowMetrics.barter_price) || 0);
      metrics.spoiled.push(Number(rowMetrics.berries_spoiled) || 0);
      metrics.tradeBerries.push(Number(rowMetrics.trade_berries) || 0);
      metrics.tradeWood.push(Number(rowMetrics.trade_wood) || 0);
      metrics.ppfEfficiency.push(Number(rowMetrics.ppf_efficiency) || 0);
      metrics.storageUtilization.push(
        Number(rowMetrics.storage_utilization) || 0
      );
      metrics.specialisation.push(Number(rowMetrics.specialisation) || 0);
      metrics.giniBerries.push(Number(rowMetrics.gini_berries) || 0);
      metrics.giniWood.push(Number(rowMetrics.gini_wood) || 0);
      metrics.gifts.push(Number(rowMetrics.gifts) || 0);
      metrics.unmetFood.push(Number(rowMetrics.unmet_food) || 0);
      metrics.berriesGathered.push(Number(rowMetrics.berries_gathered) || 0);
      metrics.woodGathered.push(Number(rowMetrics.wood_gathered) || 0);
      metrics.storageCapacity.push(Number(rowMetrics.storage_capacity) || 0);

      if (metrics.aliveCount[t] > 0) lastAliveIteration = t;
      if (extinctionIteration == null && t > 0 && metrics.aliveCount[t] === 0) {
        extinctionIteration = t;
      }
    }

    // Build the per-agent series directly from the replay rows.
    var stateSeries = {};
    ["health", "berries", "wood", "storage", "laborB", "laborW",
     "plannedB", "plannedW", "gatheredB", "gatheredW",
     "postTradeB", "postTradeW", "alive"].forEach(
      function (key) {
        stateSeries[key] = [];
        for (var t = 0; t < N; t += 1) {
          var row = new Array(A).fill(0);
          var tick = ticks[t];
          for (var s = 0; s < tick.agents.length; s += 1) {
            var state = tick.agents[s];
            var index = indexById[state.id];
            if (index === undefined) continue;
            if (key === "health") row[index] = Number(state.health) || 0;
            else if (key === "berries") row[index] = Number(state.berries) || 0;
            else if (key === "wood") row[index] = Number(state.wood) || 0;
            else if (key === "storage") row[index] = Number(state.storage_capacity) || 0;
            else if (key === "laborB") row[index] = Number(state.labor_berries) || 0;
            else if (key === "laborW") row[index] = Number(state.labor_wood) || 0;
            else if (key === "plannedB") row[index] = Number(state.planned_berries) || 0;
            else if (key === "plannedW") row[index] = Number(state.planned_wood) || 0;
            else if (key === "gatheredB") row[index] = Number(state.gathered_berries) || 0;
            else if (key === "gatheredW") row[index] = Number(state.gathered_wood) || 0;
            else if (key === "postTradeB") row[index] = Number(state.post_trade_berries) || 0;
            else if (key === "postTradeW") row[index] = Number(state.post_trade_wood) || 0;
            else if (key === "alive") row[index] = Boolean(state.alive);
          }
          stateSeries[key].push(row);
        }
      }
    );

    for (var e = 0; e < replay.events.length; e += 1) {
      var event = replay.events[e];
      var tickNumber = Number(event.tick) | 0;
      var actor = indexById[event.actor];
      var target = event.target == null ? null : indexById[event.target];
      if (event.actor != null && actor === undefined) continue;
      if (event.target != null && target === undefined) continue;
      if (!eventTicks[tickNumber]) eventTicks[tickNumber] = [];
      eventTicks[tickNumber].push({
        type: event.type,
        actor: actor == null ? null : actor,
        target: target == null ? null : target,
        resource: event.resource || null,
        amount: Number(event.amount) || 0,
        waste: Number(event.waste) || 0,
        berries: Number(event.berries) || 0,
        wood: Number(event.wood) || 0,
        price: Number(event.price) || 0,
        cause: event.cause || null
      });
      if (event.type === "death") {
        deaths.push({ tick: tickNumber, index: actor, cause: event.cause });
      }
    }

    var maxValues = {
      berries: 1,
      wood: 1,
      health: 4,
      storage: 1,
      potentialBerries: 1,
      potentialWood: 1
    };
    for (var i = 0; i < A; i += 1) {
      maxValues.potentialBerries = Math.max(
        maxValues.potentialBerries,
        agents[i].maxBerries
      );
      maxValues.potentialWood = Math.max(
        maxValues.potentialWood,
        agents[i].maxWood
      );
    }
    for (var t2 = 0; t2 < N; t2 += 1) {
      for (var i2 = 0; i2 < A; i2 += 1) {
        maxValues.berries = Math.max(maxValues.berries, stateSeries.berries[t2][i2]);
        maxValues.wood = Math.max(maxValues.wood, stateSeries.wood[t2][i2]);
        maxValues.storage = Math.max(maxValues.storage, stateSeries.storage[t2][i2]);
        maxValues.health = Math.max(maxValues.health, stateSeries.health[t2][i2]);
      }
    }

    return {
      meta: {
        world: replay.meta.world,
        iterations: replay.meta.iterations,
        seed: replay.meta.seed,
        agentCount: A,
        features: replay.meta.features || {},
        health: replay.meta.health || { max: 4, food_per_iteration: 6 },
        resources: replay.meta.resources || {},
        initialCommons: replay.meta.initial_commons || { berries: 0, wood: 0 },
        initialTotals: replay.meta.initial_totals || { berries: 0, wood: 0 },
        extinctionIteration: extinctionIteration,
        lastAliveIteration: lastAliveIteration,
        agents: agents
      },
      series: Object.assign({}, metrics, stateSeries),
      eventsByTick: eventTicks,
      deaths: deaths,
      maxValues: maxValues
    };
  }

  function isGzip(bytes) {
    return bytes.length > 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
  }

  async function decompressGzip(bytes) {
    if (typeof global.DecompressionStream !== "function") {
      throw new Error(
        "This browser cannot decompress .json.gz files. Use an uncompressed " +
        "replay JSON file or a current Chrome, Edge, Firefox, or Safari."
      );
    }
    var stream = new Blob([bytes]).stream().pipeThrough(
      new global.DecompressionStream("gzip")
    );
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  async function parseReplayBytes(bytes, label) {
    var jsonBytes = isGzip(bytes) ? await decompressGzip(bytes) : bytes;
    var text = new TextDecoder("utf-8").decode(jsonBytes);
    var replay;
    try {
      replay = JSON.parse(text);
    } catch (error) {
      throw new Error(
        "Could not parse " + (label || "the replay") + " as JSON: " +
        error.message
      );
    }
    assertReplay(replay);
    return replay;
  }

  async function fetchReplay(url) {
    var response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(
        "Could not load " + url + " (HTTP " + response.status + "). " +
        "Serve the viewer over HTTP, or open a replay file directly."
      );
    }
    var buffer = await response.arrayBuffer();
    return parseReplayBytes(new Uint8Array(buffer), url);
  }

  function bytesToBase64(bytes) {
    var chunks = [];
    var size = 0x8000;
    for (var i = 0; i < bytes.length; i += size) {
      chunks.push(String.fromCharCode.apply(null, bytes.subarray(i, i + size)));
    }
    return btoa(chunks.join(""));
  }

  function base64ToBytes(value) {
    var binary = atob(value);
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  function takeStoredReplay() {
    var raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(STORAGE_KEY);
    var stored = JSON.parse(raw);
    return parseReplayBytes(base64ToBytes(stored.bytes), stored.name);
  }

  function showEmptyState() {
    var story = document.getElementById("story");
    if (story) {
      story.textContent =
        "Open a replay file, or add ?replay=path/to/replay.json.gz to the URL.";
    }
    var note = document.getElementById("worldNote");
    if (note) note.textContent = "No replay loaded yet.";
  }

  function showError(error) {
    var story = document.getElementById("story");
    if (story) story.textContent = error.message;
    var note = document.getElementById("worldNote");
    if (note) note.textContent = "The replay could not be loaded.";
    if (global.console) console.error(error);
  }

  function bindFileInput() {
    var input = document.getElementById("replayFile");
    var button = document.getElementById("openReplayBtn");
    if (!input || !button) return;
    button.addEventListener("click", function () { input.click(); });
    input.addEventListener("change", async function () {
      var file = input.files && input.files[0];
      if (!file) return;
      try {
        var bytes = new Uint8Array(await file.arrayBuffer());
        sessionStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ name: file.name, bytes: bytesToBase64(bytes) })
        );
        global.location.reload();
      } catch (error) {
        showError(new Error(
          "Could not open " + file.name + ": " + error.message
        ));
      }
    });
  }

  async function boot() {
    bindFileInput();
    try {
      var stored = await takeStoredReplay();
      if (stored) {
        global.startSubsistenceViewer(buildView(stored));
        return;
      }
      if (global.__REPLAY__) {
        global.startSubsistenceViewer(buildView(global.__REPLAY__));
        return;
      }
      var url = new URLSearchParams(global.location.search).get("replay");
      if (url) {
        global.startSubsistenceViewer(buildView(await fetchReplay(url)));
        return;
      }
      showEmptyState();
    } catch (error) {
      showError(error);
    }
  }

  global.SubsistenceReplay = {
    SCHEMA_VERSION: SCHEMA_VERSION,
    assertReplay: assertReplay,
    buildView: buildView,
    parseReplayBytes: parseReplayBytes,
    fetchReplay: fetchReplay,
    boot: boot
  };

  global.addEventListener("load", boot);
})(window);
