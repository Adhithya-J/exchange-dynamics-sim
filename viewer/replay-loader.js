/* Replay loading and view-model adaptation for The Commons.
 *
 * The simulation writes a versioned replay (see replay.py). This module turns
 * that stable schema into the compact, index-based view model the canvas
 * renderer expects, so the renderer never needs to know about schemas, agents
 * by id, or JSON transport details.
 */
(function (global) {
  "use strict";

  var SCHEMA_VERSION = 1;
  var STORAGE_KEY = "commonsReplayBytes";
  var GIFT = 0;
  var REPAY = 1;
  var TRADE = 2;

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
    if (!Array.isArray(replay.agents) || !replay.agents.length) {
      throw new Error("The replay has no agents.");
    }
    if (!Array.isArray(replay.ticks) || !replay.ticks.length) {
      throw new Error("The replay has no ticks.");
    }
    if (!Array.isArray(replay.events)) {
      throw new Error("The replay has no events list.");
    }
    if (!replay.meta || typeof replay.meta !== "object") {
      throw new Error("The replay has no meta section.");
    }
  }

  function buildView(replay) {
    assertReplay(replay);

    var ticks = replay.ticks;
    var agentIds = replay.agents.map(function (agent) { return agent.id; });
    var indexById = {};
    agentIds.forEach(function (id, index) { indexById[id] = index; });

    var A = agentIds.length;
    var N = ticks.length;
    var resources = [];
    var goods = [];
    var risk = [];
    var reputation = [];
    var gini = [];
    var alive = [];
    var total = [];

    for (var t = 0; t < N; t += 1) {
      var tick = ticks[t];
      var rowResources = new Array(A);
      var rowGoods = new Array(A);
      var rowRisk = new Array(A);
      var rowReputation = new Array(A);

      for (var s = 0; s < tick.agents.length; s += 1) {
        var state = tick.agents[s];
        var index = indexById[state.id];
        if (index === undefined) continue;
        rowResources[index] = Number(state.resources) || 0;
        rowGoods[index] = Number(state.goods) || 0;
        rowRisk[index] = Number(state.risk_tolerance) || 0;
        rowReputation[index] = state.reputation == null
          ? null
          : Number(state.reputation);
      }

      var metrics = tick.metrics || {};
      resources.push(rowResources);
      goods.push(rowGoods);
      risk.push(rowRisk);
      reputation.push(rowReputation);
      gini.push(Number(metrics.gini) || 0);
      alive.push(
        metrics.alive == null
          ? rowResources.filter(function (value) {
              return value > (replay.meta.resource_floor || 0);
            }).length
          : Number(metrics.alive)
      );
      total.push(Number(metrics.total) || 0);
    }

    var extinctionIteration = null;
    var lastAliveIteration = 0;
    for (var i = 0; i < N; i += 1) {
      if (alive[i] > 0) lastAliveIteration = i;
      if (extinctionIteration == null && i > 0 && alive[i] === 0) {
        extinctionIteration = i;
      }
    }

    var events = [];
    var deaths = [];
    var shocks = [];
    for (var e = 0; e < replay.events.length; e += 1) {
      var event = replay.events[e];
      var tickNumber = Number(event.tick) | 0;
      var actor = indexById[event.actor];
      var target = indexById[event.target];
      if (event.type === "gift" || event.type === "repayment") {
        if (actor === undefined || target === undefined) continue;
        events.push([
          tickNumber,
          event.type === "gift" ? GIFT : REPAY,
          actor,
          target,
          Number(event.amount) || 0,
          0
        ]);
      } else if (event.type === "trade") {
        if (actor === undefined || target === undefined) continue;
        events.push([
          tickNumber,
          TRADE,
          actor,
          target,
          Number(event.amount) || 0,
          Number(event.quantity) || 0
        ]);
      } else if (event.type === "death") {
        if (actor === undefined) continue;
        deaths.push({ tick: tickNumber, index: actor, cause: event.cause });
      } else if (event.type === "shock") {
        shocks.push({
          iteration: tickNumber,
          duration: Number(event.duration) || 1,
          type: event.shock_type,
          magnitude: Number(event.magnitude) || 0
        });
      }
    }
    events.sort(function (left, right) {
      return left[0] - right[0] || left[1] - right[1];
    });

    var features = replay.meta.features || {};
    var config = replay.meta.config || {};
    var marketConfig = config.MARKET || {};
    var agents = replay.agents.map(function (agent) {
      return {
        id: agent.id,
        label: agent.label || agent.id,
        color: agent.color,
        generosity: Number(agent.generosity) || 0,
        acceptance: Number(agent.acceptance) || 0,
        risk: Number(agent.initial_risk_tolerance) || 0,
        production: Number(agent.production_rate) || 0,
        consumption: Number(agent.consumption_need) || 0
      };
    });

    var marketEnabled = Boolean(features.market);
    var trustEnabled = Boolean(features.trust);
    var shocksEnabled = Boolean(features.shocks);

    return {
      meta: {
        iterations: replay.meta.iterations,
        seed: replay.meta.seed,
        agents: agents,
        initial_resources: Number(replay.meta.initial_resources) || 0,
        resource_floor: Number(replay.meta.resource_floor) || 0,
        living_cost: Number(replay.meta.living_cost) || 0,
        transfer_range: replay.meta.transfer_range || [0, 0],
        market_enabled: marketEnabled,
        market_price: Number(marketConfig.PRICE) || 1,
        trust_enabled: trustEnabled,
        shocks_enabled: shocksEnabled,
        reciprocity_enabled: Boolean(features.reciprocity),
        extinction_iteration: extinctionIteration,
        last_alive_iteration: lastAliveIteration
      },
      series: {
        resources: resources,
        goods: marketEnabled ? goods : [],
        risk: trustEnabled || shocksEnabled ? risk : [],
        reputation: trustEnabled ? reputation : [],
        gini: gini,
        alive: alive,
        total: total
      },
      events: events,
      deaths: deaths,
      shocks: shocks
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
      chunks.push(String.fromCharCode.apply(
        null,
        bytes.subarray(i, i + size)
      ));
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
      var stored = takeStoredReplay();
      if (stored) {
        global.startViewer(buildView(stored));
        return;
      }
      if (global.__REPLAY__) {
        global.startViewer(buildView(global.__REPLAY__));
        return;
      }
      var url = new URLSearchParams(global.location.search).get("replay");
      if (url) {
        global.startViewer(buildView(await fetchReplay(url)));
        return;
      }
      showEmptyState();
    } catch (error) {
      showError(error);
    }
  }

  global.ReplayView = {
    SCHEMA_VERSION: SCHEMA_VERSION,
    assertReplay: assertReplay,
    buildView: buildView,
    parseReplayBytes: parseReplayBytes,
    fetchReplay: fetchReplay,
    boot: boot
  };

  global.addEventListener("load", boot);
})(window);
