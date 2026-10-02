import { encode, decode } from "gpt-tokenizer";
import { embed, cosineSim, generate } from "./models.js";

export const STAGES = [
  {
    title: "Foundations",
    level: "beginner",
    desc: "The vocabulary and intuition everything else builds on.",
    topics: [
      ["What an LLM actually is", "Next-token prediction over a probability distribution — not a database or a search engine."],
      ["Tokens & tokenization", "Why text becomes subword pieces (BPE), and why token count ≠ word count."],
      ["Embeddings", "How words/tokens become vectors that capture meaning and relationships."],
      ["Neural network basics", "Layers, weights, activations, backpropagation — just enough to follow the rest."],
      ["Parameters & scale", "What \"7B\" or \"70B\" parameters means, and why scale changes behavior."]
    ],
    lesson: "An LLM is, mechanically, a function that takes a sequence of tokens and outputs a probability distribution over the next token — repeated one token at a time. Everything downstream (chat, reasoning, code) is this loop plus clever training. Two ideas unlock most of the rest of the field: tokenization (text → integers the model can process) and embeddings (integers → vectors that place similar meanings near each other in space). Run the demo below to see both happen for real, in your browser.",
    pythonCode: {
      label: "tokenization_and_embeddings.py",
      text: `# 1. Tokenization: see how text becomes integer tokens
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Large language models predict the next token.")
print(tokens)                              # [37309, 4221, 4211, ...]
print(len(tokens), "tokens for", len("Large language models predict the next token."), "characters")
print([enc.decode([t]) for t in tokens])   # see the actual sub-word pieces

# 2. Embeddings: similar meaning -> nearby vectors
import numpy as np
from openai import OpenAI

client = OpenAI()

def embed(text):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding)

a = embed("The cat sat on the mat.")
b = embed("A kitten rested on the rug.")
c = embed("Stock markets fell sharply today.")

cosine = lambda x, y: x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
print("related pair:  ", cosine(a, b))   # high similarity, e.g. ~0.7
print("unrelated pair:", cosine(a, c))   # low similarity, e.g. ~0.1`
    },
    modelNote: "Uses a real BPE tokenizer instantly, then a small local embedding model (~23MB, downloaded once and cached).",
    run: async (ctx) => {
      const text = "Large language models predict the next token.";
      const tokens = encode(text);
      ctx.log(`Text: "${text}"`);
      ctx.log(`Token IDs: [${tokens.join(", ")}]`);
      ctx.log(`${tokens.length} tokens for ${text.length} characters`);
      ctx.log(`Pieces: ${tokens.map((t) => JSON.stringify(decode([t]))).join(" ")}`);
      ctx.log("");
      ctx.log("Loading embedding model (first run downloads, then it's cached)...");
      const a = await embed("The cat sat on the mat.", ctx.status);
      const b = await embed("A kitten rested on the rug.", ctx.status);
      const c = await embed("Stock markets fell sharply today.", ctx.status);
      ctx.status("");
      ctx.log(`related pair similarity ("cat/mat" vs "kitten/rug"):   ${cosineSim(a, b).toFixed(3)}`);
      ctx.log(`unrelated pair similarity ("cat/mat" vs "stock market"): ${cosineSim(a, c).toFixed(3)}`);
    }
  },
  {
    title: "The Transformer",
    level: "beginner",
    desc: "The architecture that made modern LLMs possible.",
    topics: [
      ["Self-attention", "How a model decides which other tokens matter when processing each token."],
      ["Multi-head attention", "Running several attention patterns in parallel to capture different relationships."],
      ["Positional encoding", "How order is injected into a model with no inherent sense of sequence."],
      ["Encoder vs. decoder vs. decoder-only", "Why GPT-style models dropped the encoder entirely."],
      ["Context window", "What it is, why it's finite, and how it limits a single conversation."]
    ],
    lesson: "Self-attention is the core trick: for every token, the model computes how much it should \"attend to\" every other token, then blends their representations accordingly. Stack that (multi-head, multiple layers), add positional encoding so order isn't lost, and you have a transformer. The demo below computes real scaled dot-product attention over 4 random token vectors, entirely in JavaScript — no model download needed.",
    pythonCode: {
      label: "self_attention.py",
      text: `import torch
import torch.nn.functional as F

def self_attention(x, Wq, Wk, Wv):
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)  # how much each token relates to every other
    weights = F.softmax(scores, dim=-1)                       # turn scores into attention weights
    return weights @ V                                        # blend values by those weights

seq_len, d_model = 4, 8
x = torch.randn(seq_len, d_model)                  # 4 tokens, 8-dim embeddings
Wq, Wk, Wv = [torch.randn(d_model, d_model) for _ in range(3)]

out = self_attention(x, Wq, Wk, Wv)
print(out.shape)  # torch.Size([4, 8]) -- each token's new vector now carries context from the others`
    },
    run: async (ctx) => {
      const seqLen = 4, dModel = 8;
      const randn = () => Math.random() * 2 - 1;
      const randMat = (r, c) => Array.from({ length: r }, () => Array.from({ length: c }, randn));
      const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
      const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
      const softmax = (row) => {
        const m = Math.max(...row);
        const exps = row.map((v) => Math.exp(v - m));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map((e) => e / sum);
      };

      const x = randMat(seqLen, dModel);
      const Wq = randMat(dModel, dModel), Wk = randMat(dModel, dModel), Wv = randMat(dModel, dModel);
      const Q = matmul(x, Wq), K = matmul(x, Wk), V = matmul(x, Wv);
      const scores = matmul(Q, transpose(K)).map((row) => row.map((v) => v / Math.sqrt(dModel)));
      const weights = scores.map(softmax);
      const out = matmul(weights, V);

      ctx.log("Attention weights (row i = how much token i attends to every other token):");
      weights.forEach((row, i) => ctx.log(`  token ${i}: [${row.map((v) => v.toFixed(2)).join(", ")}]`));
      ctx.log("");
      ctx.log("Output (each token's new vector, blended by those weights):");
      out.forEach((row, i) => ctx.log(`  token ${i}: [${row.map((v) => v.toFixed(2)).join(", ")}]`));
    },
    videos: [
      { title: "But what is a GPT? Visual intro to transformers", channel: "3Blue1Brown", url: "https://www.youtube.com/watch?v=wjZofJX0v4M" },
      { title: "Attention in transformers, visually explained", channel: "3Blue1Brown", url: "https://www.youtube.com/watch?v=eMlx5fFNoYc" },
      { title: "Let's build GPT: from scratch, in code, spelled out.", channel: "Andrej Karpathy", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY" }
    ]
  },
  {
    title: "Training & Alignment",
    level: "intermediate",
    desc: "How a raw pattern-matcher becomes a helpful assistant.",
    topics: [
      ["Pretraining", "Learning language structure from massive unlabeled text corpora."],
      ["Supervised fine-tuning (SFT)", "Teaching instruction-following from curated input/output examples."],
      ["RLHF & preference tuning", "Using human (or AI) preference signals to shape tone, safety, and helpfulness (RLHF, DPO)."],
      ["Instruction tuning", "Why a base model and a chat model behave so differently."],
      ["Catastrophic forgetting", "Why fine-tuning on narrow data can degrade general ability."]
    ],
    lesson: "A base (pretrained) model just completes text plausibly. Supervised fine-tuning (SFT) teaches it the instruction → response shape using curated examples, shifting what it generates toward that format. A full neural SFT run needs a GPU and the Python stack below — but the underlying idea (bias generation toward your examples' shape) is visible even in a tiny character-level model you can train instantly in the browser.",
    pythonCode: {
      label: "sft_finetune.py",
      text: `from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# A small instruction-following dataset: {"prompt": ..., "completion": ...} pairs
dataset = load_dataset("trl-lib/Capybara", split="train[:1%]")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="sft-demo", max_steps=50, per_device_train_batch_size=2),
)
trainer.train()  # the base model starts learning to follow instructions, not just continue text`
    },
    modelNote: "Trains a tiny character-level bigram model live, in your browser — a toy stand-in for real neural fine-tuning, not a neural network itself.",
    run: async (ctx) => {
      const pretrainCorpus =
        "the quick brown fox jumps over the lazy dog the dog barks at the fox the fox runs away into the " +
        "forest while the dog chases it through the trees and bushes near the river bank ";
      const sftCorpus =
        "question what is the capital of france answer the capital of france is paris " +
        "question what is two plus two answer two plus two is four " +
        "question who wrote hamlet answer william shakespeare wrote hamlet ";

      const buildBigram = (corpus) => {
        const model = {};
        for (let i = 0; i < corpus.length - 1; i++) {
          const cur = corpus[i], next = corpus[i + 1];
          model[cur] = model[cur] || {};
          model[cur][next] = (model[cur][next] || 0) + 1;
        }
        return model;
      };

      const sample = (model, start, length) => {
        let out = start;
        let cur = start[start.length - 1];
        for (let i = 0; i < length; i++) {
          const options = model[cur];
          if (!options) break;
          const entries = Object.entries(options);
          const total = entries.reduce((s, [, c]) => s + c, 0);
          let r = Math.random() * total;
          let next = entries[0][0];
          for (const [ch, c] of entries) {
            if (r < c) { next = ch; break; }
            r -= c;
          }
          out += next;
          cur = next;
        }
        return out;
      };

      const pretrained = buildBigram(pretrainCorpus);
      ctx.log('Pretrained only on generic text. Sampling a continuation of "th":');
      ctx.log(`  "${sample(pretrained, "th", 40)}"`);
      ctx.log("");

      // "Fine-tuning": the same model, now additionally trained on a small
      // instruction-formatted dataset (repeated, the way a few fine-tuning
      // epochs would emphasize it relative to the huge pretraining corpus).
      const finetuned = buildBigram(pretrainCorpus + sftCorpus.repeat(4));
      ctx.log('Fine-tuned on a Q/A-formatted dataset. Sampling a continuation of "qu":');
      ctx.log(`  "${sample(finetuned, "qu", 60)}"`);
      ctx.log("");
      ctx.log("The fine-tuned model drifts toward the question/answer shape of its extra data --");
      ctx.log("that's the essence of SFT, even though a real LLM fine-tune uses gradient descent");
      ctx.log("over a neural network instead of character counts.");
    },
    videos: [
      { title: "Reinforcement Learning from Human Feedback (RLHF) — Explained in 10 minutes", channel: "YouTube", url: "https://www.youtube.com/watch?v=GSqZ8oQ6s50" }
    ]
  },
  {
    title: "Prompt Engineering",
    level: "beginner",
    desc: "Getting reliable, high-quality output without touching weights.",
    topics: [
      ["Zero-shot vs. few-shot prompting", "When examples in the prompt help, and when they just add noise."],
      ["Chain-of-thought", "Asking a model to reason step-by-step before answering."],
      ["System prompts & role framing", "Setting persistent behavior, tone, and constraints."],
      ["Structured output", "Getting reliable JSON/schema-conformant responses."],
      ["Prompt sensitivity & iteration", "Why small wording changes can swing output quality — and how to test for it."]
    ],
    lesson: "The model's weights don't change between calls — everything you control happens through the prompt. Chain-of-thought (asking the model to reason before answering) reliably improves accuracy on multi-step problems because it gives the model \"room\" to work through sub-steps instead of jumping straight to a guess. The demo runs both styles on a real (tiny) local model.",
    pythonCode: {
      label: "zero_shot_vs_cot.py",
      text: `from openai import OpenAI
client = OpenAI()

question = ("A juggler has 16 balls. Half of the balls are golf balls, and half "
            "of the golf balls are blue. How many blue golf balls are there?")

# Zero-shot: straight to the answer
zero_shot = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": question}],
)

# Chain-of-thought: ask it to reason first
cot = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": question + " Think step by step, then give the final answer."}],
)

print("Zero-shot:", zero_shot.choices[0].message.content)
print("CoT:      ", cot.choices[0].message.content)  # more consistently correct on arithmetic/logic`
    },
    modelNote: "Uses a small local instruction-tuned model (~60MB, downloaded once and cached). It's far weaker than GPT-4o — judge the pattern, not the answer quality.",
    run: async (ctx) => {
      const question =
        "A juggler has 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. " +
        "How many blue golf balls are there?";
      ctx.log("Loading a small local instruction model (first run downloads, then it's cached)...");
      const zeroShot = await generate(question, ctx.status);
      ctx.status("");
      ctx.log(`Zero-shot:        ${zeroShot}`);
      const cot = await generate(`${question} Think step by step, then give the final answer.`, ctx.status);
      ctx.status("");
      ctx.log(`Chain-of-thought: ${cot}`);
      ctx.log("");
      ctx.log("This is a 77M-parameter demo model -- don't expect GPT-4o-level reasoning.");
      ctx.log("Swap the same prompts into the Python reference above for production quality.");
    }
  },
  {
    title: "Retrieval & Memory",
    level: "intermediate",
    desc: "Giving a model knowledge and context it wasn't trained on.",
    topics: [
      ["Vector databases", "Storing embeddings for fast similarity search (e.g. Pinecone, pgvector, FAISS)."],
      ["RAG (Retrieval-Augmented Generation)", "Fetching relevant documents at query time and injecting them into the prompt."],
      ["Chunking strategies", "Splitting documents so retrieval returns useful, coherent context."],
      ["Long-context vs. RAG trade-offs", "When a bigger context window beats retrieval, and vice versa."],
      ["Conversation memory", "Summarization and state management across long sessions."]
    ],
    lesson: "RAG solves a simple problem: the model doesn't know your private data and you can't fit it all in a prompt. So instead, embed your documents once, embed the user's query at request time, find the nearest document vectors, and stuff just those into the prompt as context. The demo below does exactly that — real embeddings, real cosine similarity, real (small) generation — on three sentences.",
    pythonCode: {
      label: "mini_rag.py",
      text: `import numpy as np
from openai import OpenAI

client = OpenAI()
docs = [
    "The Eiffel Tower is in Paris, France.",
    "Mount Everest is the tallest mountain on Earth.",
    "Python was created by Guido van Rossum.",
]

def embed(text):
    return np.array(client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding)

doc_vectors = [embed(d) for d in docs]  # do this once, offline, for your real knowledge base

def retrieve(query, k=1):
    qv = embed(query)
    sims = [qv @ dv / (np.linalg.norm(qv) * np.linalg.norm(dv)) for dv in doc_vectors]
    top = np.argsort(sims)[::-1][:k]
    return [docs[i] for i in top]

query = "Who built the Python programming language?"
context = retrieve(query)

answer = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Context: {context}\\n\\nQuestion: {query}\\nAnswer using only the context."}],
)
print(answer.choices[0].message.content)  # "Guido van Rossum" -- grounded in retrieved text, not memorized`
    },
    modelNote: "Reuses the embedding model from Foundations plus the small generator from Prompt Engineering (cached if you already ran those demos).",
    run: async (ctx) => {
      const docs = [
        "The Eiffel Tower is in Paris, France.",
        "Mount Everest is the tallest mountain on Earth.",
        "Python was created by Guido van Rossum.",
      ];
      ctx.log("Embedding documents...");
      const docVecs = [];
      for (const d of docs) docVecs.push(await embed(d, ctx.status));
      ctx.status("");

      const query = "Who built the Python programming language?";
      const qVec = await embed(query);
      const sims = docVecs.map((v) => cosineSim(qVec, v));
      const bestIdx = sims.indexOf(Math.max(...sims));

      docs.forEach((d, i) => ctx.log(`  sim=${sims[i].toFixed(3)}  "${d}"`));
      ctx.log("");
      ctx.log(`Retrieved: "${docs[bestIdx]}"`);

      ctx.log("Generating a grounded answer...");
      const answer = await generate(
        `Context: ${docs[bestIdx]} Question: ${query} Answer using only the context.`,
        ctx.status
      );
      ctx.status("");
      ctx.log(`Answer: ${answer}`);
    },
    videos: [
      { title: "Embeddings & Vector Databases Explained", channel: "YouTube", url: "https://www.youtube.com/watch?v=rw1YfQQttfo" },
      { title: "RAG Tutorial 2026: Complete Introduction to Retrieval Augmented Generation", channel: "YouTube", url: "https://www.youtube.com/watch?v=vAK0iqA6-QI" }
    ]
  },
  {
    title: "Tool Use & Agents",
    level: "intermediate",
    desc: "Moving from a model that talks to a system that acts.",
    topics: [
      ["Function / tool calling", "How a model requests structured actions instead of just generating text."],
      ["Agent loops", "Plan → act → observe → repeat, until a goal is reached."],
      ["Multi-agent systems", "Orchestrating specialized agents that collaborate or hand off work."],
      ["MCP (Model Context Protocol)", "A standard way to connect models to external tools and data sources."],
      ["Planning & reflection", "Letting a model critique and revise its own intermediate steps."]
    ],
    lesson: "Tool calling is the foundation of agents: you describe functions (name, description, JSON-schema parameters), and the model replies with structured data like \"call get_weather(city='Tokyo')\" instead of executing anything itself. Your code runs the real function and feeds the result back. Reliable structured decisions like this need a real LLM (tiny local models are too unreliable at it), so the demo below simulates the model's decision with a simple parser and runs the real tool-execution loop around it.",
    pythonCode: {
      label: "tool_calling.py",
      text: `from openai import OpenAI
import json

client = OpenAI()

def get_weather(city: str) -> str:
    return f"72F and sunny in {city}"  # stub -- call a real weather API here

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)

call = response.choices[0].message.tool_calls[0]   # the model asked to call a tool, didn't run it
args = json.loads(call.function.arguments)
result = get_weather(**args)                        # we execute it
print(result)  # feed this back as a "tool" role message to let the model use it in its final answer`
    },
    modelNote: "No model download — the LLM's tool-call decision is simulated with a regex so the demo is instant; everything after that decision (executing the tool, using its result) is the real pattern.",
    run: async (ctx) => {
      const getWeather = (city) => `72F and sunny in ${city}`;
      const userMessage = "What's the weather in Tokyo?";
      ctx.log(`User: "${userMessage}"`);

      // A real LLM call returns a structured tool_calls response (see the
      // Python reference). We stand in for that decision with a regex so
      // this demo runs with zero downloads and zero API key.
      const match = userMessage.match(/weather in ([a-zA-Z ]+)\??/i);
      if (!match) {
        ctx.log("Model: no tool needed, answering directly.");
        return;
      }
      const city = match[1].trim();
      ctx.log(`Model decided to call: get_weather(city="${city}")`);

      const result = getWeather(city);
      ctx.log(`Tool executed. Result: "${result}"`);
      ctx.log(`Model's final answer: "It's ${result.toLowerCase()}."`);
      ctx.log("");
      ctx.log("In production, swap the regex for a real response.choices[0].message.tool_calls");
      ctx.log("from the API -- the execute-and-feed-back loop stays identical.");
    },
    videos: [
      { title: "How LLM Tool Calling Works", channel: "YouTube", url: "https://www.youtube.com/watch?v=QiRdYCNXAxk" }
    ]
  },
  {
    title: "Evaluation & Safety",
    level: "advanced",
    desc: "Knowing whether a model (or your system) is actually good.",
    topics: [
      ["Benchmarks", "MMLU, HumanEval, GSM8K and what general-purpose scores do and don't tell you."],
      ["Hallucination", "Why models state falsehoods confidently, and mitigation strategies."],
      ["LLM-as-judge", "Using a model to grade another model's output, and its failure modes."],
      ["Red-teaming & jailbreaks", "Adversarial testing to find where safety behavior breaks."],
      ["Bias & alignment limits", "Why \"aligned\" doesn't mean neutral, correct, or safe by default."]
    ],
    lesson: "Public benchmarks (MMLU, GSM8K, HumanEval) measure general capability but rarely reflect how a model does on your specific task — which is why most real teams build small custom eval sets. \"LLM-as-judge\" uses a second model call to grade outputs against a reference, which scales far better than human review but inherits its own blind spots. The demo below runs a real (tiny) model as the judge.",
    pythonCode: {
      label: "llm_as_judge.py",
      text: `from openai import OpenAI
client = OpenAI()

def llm_judge(question, answer, reference):
    prompt = f"""Question: {question}
Reference answer: {reference}
Model answer: {answer}

Does the model answer match the reference in meaning? Reply only "CORRECT" or "INCORRECT"."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

cases = [
    {"q": "Capital of Australia?", "ref": "Canberra", "answer": "Sydney"},
    {"q": "Capital of Australia?", "ref": "Canberra", "answer": "It's Canberra."},
]

for c in cases:
    verdict = llm_judge(c["q"], c["answer"], c["ref"])
    print(c["q"], "->", c["answer"], "->", verdict)
# First case: INCORRECT (a classic hallucination -- Sydney is NOT the capital)
# Second case: CORRECT`
    },
    modelNote: "Reuses the small local generator as the judge. It's a far weaker judge than GPT-4o — this demonstrates the pattern, not production-grade accuracy.",
    run: async (ctx) => {
      const judge = async (question, answer, reference) => {
        const prompt =
          `Question: ${question} Reference answer: ${reference} Model answer: ${answer} ` +
          `Does the model answer match the reference in meaning? Reply only CORRECT or INCORRECT.`;
        return (await generate(prompt, ctx.status)).trim();
      };

      const cases = [
        { q: "Capital of Australia?", ref: "Canberra", answer: "Sydney" },
        { q: "Capital of Australia?", ref: "Canberra", answer: "Canberra" },
      ];

      ctx.log("Loading judge model (first run downloads, then it's cached)...");
      for (const c of cases) {
        const verdict = await judge(c.q, c.answer, c.ref);
        ctx.status("");
        ctx.log(`"${c.answer}" vs. reference "${c.ref}" -> ${verdict}`);
      }
      ctx.log("");
      ctx.log("A real hallucination case (Sydney isn't Australia's capital) should read INCORRECT.");
    },
    videos: [
      { title: "LLM Benchmarks: HELM, Open LLM Leaderboard, MMLU Explained", channel: "YouTube", url: "https://www.youtube.com/watch?v=JOIV1LdRnP8" },
      { title: "What is LLM Red Teaming? How Generative AI Safety Testing Works", channel: "YouTube", url: "https://www.youtube.com/watch?v=-OUmHDuaPPA" }
    ]
  },
  {
    title: "Deployment & Efficiency",
    level: "advanced",
    desc: "Running models fast, cheap, and reliably in production.",
    topics: [
      ["Quantization", "Trading numerical precision for smaller, faster models (int8, int4, GGUF)."],
      ["Inference optimization", "KV caching, batching, speculative decoding."],
      ["Serving infrastructure", "Latency, throughput, and cost trade-offs across providers and self-hosting."],
      ["Distillation", "Training a smaller model to mimic a larger one's behavior."],
      ["Prompt caching", "Reusing computed context across requests to cut latency and cost."]
    ],
    lesson: "A 7B-parameter model in full precision (fp16) needs ~14GB of memory just to load. Quantization stores those weights in fewer bits (int8, int4/NF4) — roughly a 2-4x memory cut with modest quality loss. You can see the core math instantly: the demo quantizes 1,000 random \"weights\" to int8 with a min-max scale and measures the compression ratio and reconstruction error.",
    pythonCode: {
      label: "quantized_inference.py",
      text: `from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=quant_config,
    device_map="auto",
)  # ~4x smaller memory footprint than fp16, with modest quality loss
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

inputs = tokenizer("Explain quantization in one sentence.", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))`
    },
    run: async (ctx) => {
      const n = 1000;
      const weights = Float32Array.from({ length: n }, () => Math.random() * 2 - 1);
      const min = Math.min(...weights), max = Math.max(...weights);
      const scale = (max - min) / 255;
      const quantized = Uint8Array.from(weights, (w) => Math.round((w - min) / scale));
      const dequantized = Float32Array.from(quantized, (q) => q * scale + min);

      let maxErr = 0, sumErr = 0;
      for (let i = 0; i < n; i++) {
        const e = Math.abs(weights[i] - dequantized[i]);
        maxErr = Math.max(maxErr, e);
        sumErr += e;
      }

      ctx.log(`${n} weights, fp32: ${weights.byteLength} bytes`);
      ctx.log(`${n} weights, int8: ${quantized.byteLength} bytes  (${(weights.byteLength / quantized.byteLength).toFixed(1)}x smaller)`);
      ctx.log(`mean reconstruction error: ${(sumErr / n).toFixed(5)}`);
      ctx.log(`max reconstruction error:  ${maxErr.toFixed(5)}`);
      ctx.log("");
      ctx.log("Real quantization methods (GPTQ, AWQ, bitsandbytes) use smarter per-channel scales,");
      ctx.log("but the core trade -- fewer bits per weight for a small accuracy cost -- is the same math.");
    },
    videos: [
      { title: "LLM Quantization Explained: GPTQ, AWQ, QLoRA, GGUF and More", channel: "YouTube", url: "https://www.youtube.com/watch?v=WmvZwR4rKJg" }
    ]
  },
  {
    title: "Frontier Topics",
    level: "advanced",
    desc: "Where the field is moving right now.",
    topics: [
      ["Mixture of Experts (MoE)", "Routing each token to a subset of specialized sub-networks for efficient scale."],
      ["Reasoning models", "Models trained to spend extra inference-time compute \"thinking\" before answering."],
      ["Multimodal models", "Unified handling of text, image, audio, and video in one model."],
      ["Long-context architectures", "Techniques pushing context windows from thousands to millions of tokens."],
      ["Model evaluation at scale", "Building internal eval suites as models and prompts evolve over time."]
    ],
    lesson: "Mixture of Experts (MoE) replaces one big feed-forward block with many smaller \"expert\" blocks plus a router that sends each token to only a few of them (e.g. 2 of 8). You get the capacity of a much larger model while only paying the compute cost of the active experts. The demo below routes 5 random token vectors through a real top-k router, entirely in JavaScript.",
    pythonCode: {
      label: "mini_moe.py",
      text: `import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)
        self.top_k = top_k

    def forward(self, x):
        scores = self.router(x)                        # which experts fit this token?
        weights, idx = scores.topk(self.top_k, dim=-1)  # route to only the top_k experts
        weights = torch.softmax(weights, dim=-1)

        out = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = idx[..., i]
            for e in expert_idx.unique():
                mask = expert_idx == e
                out[mask] += weights[mask, i:i+1] * self.experts[e](x[mask])
        return out  # most experts are untouched per token -> large capacity, cheap compute

layer = MoELayer(d_model=16, n_experts=8, top_k=2)
x = torch.randn(5, 16)   # 5 tokens
print(layer(x).shape)    # torch.Size([5, 16])`
    },
    run: async (ctx) => {
      const dModel = 16, nExperts = 8, topK = 2, nTokens = 5;
      const randn = () => Math.random() * 2 - 1;
      const randMat = (r, c) => Array.from({ length: r }, () => Array.from({ length: c }, randn));
      const matvec = (M, v) => M.map((row) => row.reduce((s, w, i) => s + w * v[i], 0));

      const router = randMat(nExperts, dModel);
      const tokens = randMat(nTokens, dModel);

      const softmaxTopK = (scores, k) => {
        const idx = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]).slice(0, k);
        const vals = idx.map((i) => scores[i]);
        const m = Math.max(...vals);
        const exps = vals.map((v) => Math.exp(v - m));
        const sum = exps.reduce((a, b) => a + b, 0);
        return idx.map((i, j) => ({ expert: i, weight: exps[j] / sum }));
      };

      tokens.forEach((t, ti) => {
        const scores = matvec(router, t);
        const chosen = softmaxTopK(scores, topK);
        ctx.log(`token ${ti}: routed to ${chosen.map((c) => `expert #${c.expert} (weight=${c.weight.toFixed(2)})`).join(", ")}`);
      });
      ctx.log("");
      ctx.log(`Only ${topK} of ${nExperts} experts ran per token -- the rest of the network stayed idle for that token.`);
    },
    videos: [
      { title: "A Visual Guide to Mixture of Experts (MoE) in LLMs", channel: "YouTube", url: "https://www.youtube.com/watch?v=sOPDGQjFcuM" },
      { title: "Will Test-Time Compute Reasoning Scale to AGI? (OpenAI o3, DeepSeek-R1)", channel: "YouTube", url: "https://www.youtube.com/watch?v=mcUu-MXrjhc" }
    ]
  }
];
