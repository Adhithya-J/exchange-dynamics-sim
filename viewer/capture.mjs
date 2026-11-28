#!/usr/bin/env node
/* Deterministic frame capture for The Commons replay viewer.
 *
 * Serves viewer/ and the replay over a short-lived local HTTP server, opens
 * the viewer in Playwright, pauses playback, then steps the timeline at a
 * fixed frame rate and saves PNGs. If --video is passed, frames are handed to
 * FFmpeg afterwards.
 *
 * Usage:
 *   node viewer/capture.mjs --replay replay.json.gz --out capture
 *   node viewer/capture.mjs --replay replay.json.gz --out capture \
 *     --fps 30 --seconds 20 --video capture/commons.mp4
 */

import { chromium } from "playwright";
import { createServer } from "node:http";
import { existsSync, mkdirSync, createReadStream, statSync } from "node:fs";
import { extname, join, resolve, sep, basename } from "node:path";
import { spawnSync } from "node:child_process";

const options = parseArgs(process.argv.slice(2));
const replayPath = resolve(options.replay || "replay.json.gz");
const outDir = resolve(options.out || "capture");
const fps = positiveNumber(options.fps, 30);
const seconds = positiveNumber(options.seconds, 20);
const width = positiveNumber(options.width, 1440);
const height = positiveNumber(options.height, 900);
const videoPath = options.video === true
  ? join(outDir, "commons.mp4")
  : options.video;
const canvasOnly = Boolean(options["canvas-only"]);
const tickLimit = options.ticks ? positiveNumber(options.ticks, Infinity) : Infinity;

if (!existsSync(replayPath)) {
  console.error("Replay not found: " + replayPath);
  console.error("Create one with: python main.py --replay replay.json.gz");
  process.exit(1);
}

const viewerRoot = resolve(process.cwd(), "viewer");
const mime = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".gz": "application/gzip",
  ".png": "image/png"
};

const server = createServer((request, response) => {
  const url = new URL(request.url || "/", "http://127.0.0.1");
  const pathname = decodeURIComponent(url.pathname);
  if (pathname === "/") {
    response.writeHead(302, { Location: "/viewer/" });
    response.end();
    return;
  }
  if (pathname === "/replay") {
    sendFile(response, replayPath);
    return;
  }
  if (pathname.startsWith("/viewer/")) {
    const relative = pathname.slice("/viewer/".length) || "index.html";
    const filePath = resolve(viewerRoot, relative);
    if (filePath !== viewerRoot && !filePath.startsWith(viewerRoot + sep)) {
      response.writeHead(403);
      response.end("Forbidden");
      return;
    }
    sendFile(response, filePath);
    return;
  }
  response.writeHead(404);
  response.end("Not found");
});

function sendFile(response, filePath) {
  if (!existsSync(filePath) || !statSync(filePath).isFile()) {
    response.writeHead(404);
    response.end("Not found");
    return;
  }
  response.writeHead(200, {
    "Content-Type": mime[extname(filePath).toLowerCase()] || "application/octet-stream",
    "Cache-Control": "no-store"
  });
  createReadStream(filePath).pipe(response);
}

function positiveNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith("--")) continue;
    const key = argument.slice(2);
    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      parsed[key] = next;
      index += 1;
    } else {
      parsed[key] = true;
    }
  }
  return parsed;
}

function ffmpegFrames(fpsValue, framesDir, output) {
  const input = join(framesDir, "frame-%05d.png");
  const result = spawnSync(
    "ffmpeg",
    [
      "-y",
      "-framerate", String(fpsValue),
      "-i", input,
      "-c:v", "libx264",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
      output
    ],
    { stdio: "inherit" }
  );
  if (result.error || result.status !== 0) {
    console.warn(
      "FFmpeg was not able to build the video. Frames are still in " + framesDir
    );
    console.warn(
      "Try: ffmpeg -y -framerate " + fpsValue + " -i \"" + input +
      "\" -c:v libx264 -pix_fmt yuv420p \"" + output + "\""
    );
    return false;
  }
  return true;
}

async function main() {
  await new Promise((ready) => server.listen(0, "127.0.0.1", ready));
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  const url = "http://127.0.0.1:" + port + "/viewer/?replay=/replay";

  mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width, height },
    deviceScaleFactor: 1,
    reducedMotion: "reduce"
  });
  const page = await context.newPage();
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });

  console.log("Loading " + basename(replayPath) + " ...");
  await page.goto(url, { waitUntil: "load", timeout: 30000 });
  await page.waitForFunction(
    () => window.commonsReplay && typeof window.commonsReplay.setTime === "function",
    null,
    { timeout: 20000 }
  );
  await page.evaluate(() => document.fonts && document.fonts.ready);

  const end = await page.evaluate(() => {
    const viewer = window.commonsReplay;
    viewer.state.playing = false;
    viewer.setTime(0, true);
    viewer.render();
    return viewer.end;
  });
  if (browserErrors.length) {
    console.error("Viewer errors:\n" + browserErrors.join("\n"));
    await browser.close();
    server.close();
    process.exit(1);
  }
  if (!Number.isFinite(end) || end <= 0) {
    console.error("The replay has no playable iterations.");
    await browser.close();
    server.close();
    process.exit(1);
  }

  const lastTick = Math.min(end, tickLimit);
  const frameCount = Math.max(2, Math.round(fps * seconds) + 1);

  console.log(
    "Capturing " + frameCount + " frames at " + fps + " fps over ticks 0-" +
    lastTick + " ..."
  );
  for (let frame = 0; frame < frameCount; frame += 1) {
    const time = lastTick * (frame / (frameCount - 1));
    await page.evaluate((value) => {
      const viewer = window.commonsReplay;
      viewer.state.playing = false;
      viewer.setTime(value, true);
      viewer.render();
    }, time);
    await page.evaluate(() => new Promise((ready) =>
      requestAnimationFrame(() => requestAnimationFrame(ready))
    ));

    const file = join(
      outDir,
      "frame-" + String(frame).padStart(5, "0") + ".png"
    );
    if (canvasOnly) {
      await page.locator("#world").screenshot({ path: file });
    } else {
      await page.screenshot({ path: file });
    }
    if (frame === 0 || (frame + 1) % Math.max(1, Math.round(fps)) === 0) {
      process.stdout.write(
        "\r  frame " + (frame + 1) + "/" + frameCount
      );
    }
  }
  process.stdout.write("\n");

  if (videoPath) {
    console.log("Writing " + videoPath + " ...");
    ffmpegFrames(fps, outDir, resolve(videoPath));
  }

  await browser.close();
  server.close();
  console.log("Frames written to " + outDir);
}

main().catch((error) => {
  console.error(error);
  server.close();
  process.exit(1);
});
