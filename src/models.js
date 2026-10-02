// @xenova/transformers statically pulls in onnxruntime-web, whose UMD build
// doesn't survive Vite's dev-server module transform (crashes the whole page
// on import). Loading it from its own CDN build at call time sidesteps that
// bundler incompatibility entirely -- this is the pattern the library's own
// "vanilla JS, no bundler" docs recommend.
const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

let transformersModule = null;
async function loadTransformers() {
  if (!transformersModule) {
    transformersModule = await import(/* @vite-ignore */ TRANSFORMERS_CDN);
    transformersModule.env.allowLocalModels = false;
  }
  return transformersModule;
}

let embedderPromise = null;
let generatorPromise = null;

function withProgress(onProgress) {
  return (data) => {
    if (!onProgress) return;
    if (data.status === "progress" && data.file) {
      const pct = data.total ? Math.round((data.loaded / data.total) * 100) : null;
      onProgress(`Downloading ${data.file}${pct !== null ? ` — ${pct}%` : ""}`);
    } else if (data.status === "ready") {
      onProgress("Model ready.");
    } else if (data.status === "done" && data.file) {
      onProgress(`Cached ${data.file}.`);
    }
  };
}

// Small sentence-embedding model (~23MB quantized), shared by the Foundations
// and Retrieval & Memory demos.
export function getEmbedder(onProgress) {
  if (!embedderPromise) {
    embedderPromise = loadTransformers().then(({ pipeline }) =>
      pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
        progress_callback: withProgress(onProgress)
      })
    );
  } else if (onProgress) {
    onProgress("Reusing already-loaded embedding model.");
  }
  return embedderPromise;
}

export async function embed(text, onProgress) {
  const model = await getEmbedder(onProgress);
  const output = await model(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

export function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// Small instruction-tuned text2text model (~60MB quantized), shared by the
// Prompt Engineering, Retrieval & Memory, and Evaluation & Safety demos.
// It's tiny compared to GPT-4o/Claude, so treat its answers as illustrative
// of the *mechanism*, not as production-quality output.
export function getGenerator(onProgress) {
  if (!generatorPromise) {
    generatorPromise = loadTransformers().then(({ pipeline }) =>
      pipeline("text2text-generation", "Xenova/flan-t5-small", {
        progress_callback: withProgress(onProgress)
      })
    );
  } else if (onProgress) {
    onProgress("Reusing already-loaded generation model.");
  }
  return generatorPromise;
}

export async function generate(prompt, onProgress, options = {}) {
  const model = await getGenerator(onProgress);
  const output = await model(prompt, { max_new_tokens: 80, ...options });
  return output[0].generated_text.trim();
}
