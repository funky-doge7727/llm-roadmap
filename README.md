# Startup Guide

How to run the LLM Roadmap app locally.

## Prerequisites

- **Node.js 18+** and **npm** (check with `node -v` and `npm -v`)
- An internet connection the first time you click "Run" on a demo that uses a model (it downloads a small model file once, then caches it in the browser — after that it works offline)

## 1. Install dependencies

From the project root:

```bash
npm install
```

This installs `vite` (dev server/bundler), `gpt-tokenizer` (real BPE tokenizer), and `prismjs` (syntax highlighting). The `@xenova/transformers` library is *not* an npm dependency — it's loaded from its CDN build at runtime by design (see note at the bottom).

## 2. Start the dev server

```bash
npm run dev
```

Vite will print a local URL, typically:

```
➜  Local:   http://localhost:5173/
```

Open that URL in your browser. The page hot-reloads as you edit files in `src/`.

## 3. Use it

- Scroll through the 9 stages, or jump via the top nav.
- Click **"Lesson, runnable code & videos"** on any stage to expand it.
- Use the **"Run in browser"** / **"Production reference (Python)"** tabs to switch between the live demo and the real-world reference code.
- Click **▶ Run** to actually execute the JS demo. A few stages (Foundations, Prompt Engineering, Retrieval & Memory, Evaluation & Safety) load a small model on first use — expect a one-time download (~23–60MB) with progress shown in the output panel; every run after that is instant.
- Checkbox progress is saved to `localStorage`, so it persists across reloads.

## Building for production

```bash
npm run build      # outputs static files to dist/
npm run preview     # serves the built dist/ folder locally to sanity-check it
```

`dist/` is a plain static site afterward — you can host it anywhere that serves static files (no server-side code involved).

## Stopping the server

`Ctrl+C` in the terminal running `npm run dev` (or `npm run preview`).

## Troubleshooting

| Problem | Fix |
|---|---|
| Port 5173 already in use | `npm run dev -- --port 5500` (or any free port) |
| A "Run" demo seems stuck on "Downloading…" | Check your internet connection — model files are fetched from Hugging Face's CDN on first use only |
| Want to re-download/reset a cached model | Open DevTools → Application → Storage → "Clear site data" for the page, then reload |
| Blank page / console error on load | Hard-refresh (`Ctrl+Shift+R`). If it persists, delete `node_modules` and re-run `npm install` |

### Why `@xenova/transformers` isn't bundled

That library statically pulls in `onnxruntime-web`, whose build doesn't survive Vite's dev-server module transform and crashes the page on load. Loading it from its CDN build at call time (inside `src/models.js`) sidesteps that bundler incompatibility — this is the pattern the library's own "vanilla JS, no bundler" docs recommend. `gpt-tokenizer` has no such issue and is bundled normally.
