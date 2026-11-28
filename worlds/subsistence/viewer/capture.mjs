#!/usr/bin/env node
/* Record a subsistence replay as a video of the village.
 *
 * Serves the viewer and a replay over a short-lived local HTTP server, then
 * walks the timeline from one iteration to another over a fixed wall-clock
 * duration with Playwright video recording. The output is a WebM file.
 *
 * Usage:
 *   node worlds/subsistence/viewer/capture.mjs --replay replay.json.gz \
 *     --out docs/subsistence-year.webm --from 1 --to 181 --seconds 60
 */

import { createServer } from "node:http";
import {
  createReadStream,
  existsSync,
  mkdirSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync
} from "node:fs";
import { spawnSync } from "node:child_process";
import { basename, dirname, extname, join, resolve, sep } from "node:path";

// The repo keeps Playwright in viewer/node_modules for the money-world capture
// script. Prefer a normal resolution, then fall back to that install so this
// script works from anywhere in the checkout.
async function loadPlaywright() {
  try {
    return await import("playwright");
  } catch (error) {
    const fallback = new URL(
      "../../../viewer/node_modules/playwright/index.mjs",
      import.meta.url
    );
    return await import(fallback.href);
  }
}

const { chromium } = await loadPlaywright();

const options = parseArgs(process.argv.slice(2));
const replayPath = resolve(options.replay || "replay-subsistence.json.gz");
const outPath = resolve(options.out || "subsistence-year.webm");
const viewerRoot = resolve(
  options.viewer || join(process.cwd(), "worlds", "subsistence", "viewer")
);
const from = nonNegativeInt(options.from, 1);
const to = nonNegativeInt(options.to, from + 180);
const seconds = positiveNumber(options.seconds, 60);
const width = positiveNumber(options.width, 1280);
const height = positiveNumber(options.height, 800);
const trim = options["no-trim"] ? false : true;

if (!existsSync(replayPath)) {
  console.error("Replay not found: " + replayPath);
  console.error("Create one with: python -m worlds.subsistence --agents 50 --iterations 5000 --replay replay.json.gz");
  process.exit(1);
}
if (to <= from) {
  console.error("--to must be greater than --from");
  process.exit(1);
}

const mime = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".gz": "application/gzip",
  ".png": "image/png"
};

const server = createServer((request, response) => {
  const url = new URL(request.url || "/", "http://127.0.0.1");
  const pathname = decodeURIComponent(url.pathname);
  if (pathname === "/replay" || pathname === "/replay.json.gz") {
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

function positiveNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function nonNegativeInt(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? Math.round(number) : fallback;
}

function findFfmpeg() {
  if (process.env.FFMPEG_PATH) return process.env.FFMPEG_PATH;
  const root = join(
    process.env.LOCALAPPDATA || process.env.HOME || "",
    "ms-playwright"
  );
  if (!existsSync(root)) return null;
  const candidates = readdirSync(root)
    .filter((name) => name.startsWith("ffmpeg-"))
    .sort()
    .reverse();
  for (const name of candidates) {
    const binary = join(root, name, "ffmpeg-win64.exe");
    if (existsSync(binary)) return binary;
    const unixBinary = join(root, name, "ffmpeg-linux");
    if (existsSync(unixBinary)) return unixBinary;
  }
  return null;
}

async function main() {
  await new Promise((ready) => server.listen(0, "127.0.0.1", ready));
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  const url = "http://127.0.0.1:" + port + "/viewer/?replay=/replay";
  const videoDir = join(
    dirname(outPath),
    "." + basename(outPath).replace(/\.[^.]+$/, "") + "-frames"
  );
  rmSync(videoDir, { recursive: true, force: true });
  mkdirSync(videoDir, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width, height },
    recordVideo: { dir: videoDir, size: { width, height } },
    reducedMotion: "no-preference"
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  const created = Date.now();
  console.log("Loading replay ...");
  await page.goto(url, { waitUntil: "load", timeout: 120000 });
  await page.waitForFunction(
    () => document.getElementById("scrub") && document.getElementById("scrub").max !== "0",
    null,
    { timeout: 120000 }
  );
  await page.evaluate(() => document.fonts && document.fonts.ready);
  // Start paused on the first frame of the window. Click through the DOM so
  // the page does not scroll the footer into view inside the recording.
  await page.evaluate(() => {
    const button = document.getElementById("playBtn");
    if (button) button.click();
    window.scrollTo(0, 0);
  });
  await page.evaluate((tick) => {
    const scrub = document.getElementById("scrub");
    scrub.value = String(tick);
    scrub.dispatchEvent(new Event("input"));
  }, from);
  const prerollMs = Date.now() - created;
  console.log("Recording iterations " + from + "-" + to + " over " + seconds + "s ...");

  const ticks = Math.max(1, to - from);
  const stepMs = (seconds * 1000) / ticks;
  let next = Date.now();
  for (let index = 1; index <= ticks; index += 1) {
    const tick = from + index;
    await page.evaluate((value) => {
      const scrub = document.getElementById("scrub");
      scrub.value = String(value);
      scrub.dispatchEvent(new Event("input"));
    }, tick);
    next += stepMs;
    const delay = next - Date.now();
    if (delay > 0) {
      await new Promise((ready) => setTimeout(ready, delay));
    }
  }

  if (errors.length) {
    console.error("Viewer errors:\n" + errors.join("\n"));
  }
  await context.close();
  await browser.close();
  server.close();

  const recorded = readdirSync(videoDir).find((name) => name.endsWith(".webm"));
  if (!recorded) {
    console.error("Playwright produced no video in " + videoDir);
    process.exit(1);
  }
  const rawVideo = join(videoDir, recorded);
  mkdirSync(resolve(outPath, ".."), { recursive: true });

  const ffmpeg = findFfmpeg();
  if (trim && ffmpeg && prerollMs > 500) {
    const startSeconds = (prerollMs / 1000).toFixed(2);
    console.log("Trimming " + startSeconds + "s of page load from the recording ...");
    const result = spawnSync(
      ffmpeg,
      [
        "-y",
        "-ss",
        String(Math.max(0, Number(startSeconds) - 0.4)),
        "-i",
        rawVideo,
        "-t",
        String(seconds),
        "-an",
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "0",
        "-crf",
        "38",
        "-row-mt",
        "1",
        "-cpu-used",
        "4",
        outPath
      ],
      { stdio: "inherit" }
    );
    if (result.status !== 0) {
      console.warn("FFmpeg could not trim the video; keeping the raw recording.");
      rmSync(outPath, { force: true });
      renameSync(rawVideo, outPath);
    }
  } else {
    rmSync(outPath, { force: true });
    renameSync(rawVideo, outPath);
  }
  rmSync(videoDir, { recursive: true, force: true });
  const sizeMb = (statSync(outPath).size / 1024 / 1024).toFixed(2);
  console.log("Video written to " + outPath + " (" + sizeMb + " MB)");
}

main().catch((error) => {
  console.error(error);
  server.close();
  process.exit(1);
});
