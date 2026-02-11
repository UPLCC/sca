const { vocab, stoi, itos, w } = vector;

const SEP = "<sep>";
const END = "<eos>";
const d = 64;
const maxLen = 64;

function tokenize(text) {
  return text.toLowerCase().replace(/[^\w\s?.!]/g, "").split(/\s+/).filter(Boolean);
}

const V = vocab.length;

class Mat {
  constructor(r, c, src) {
    this.rows = r;
    this.cols = c;
    this.w = src ? new Float32Array(src) : new Float32Array(r * c);
    this.dw = new Float32Array(r * c);
    this.m = new Float32Array(r * c);
    this.v = new Float32Array(r * c);
  }
  get(i, j) { return this.w[i * this.cols + j]; }
  set(i, j, v) { this.w[i * this.cols + j] = v; }
}

class Graph {
  constructor(bp = true) { this.bp = bp; this.fns = []; }
  backward() { for (let i = this.fns.length - 1; i >= 0; i--) this.fns[i](); }
  matmul(a, b) {
    const o = new Mat(a.rows, b.cols);
    for (let i = 0; i < a.rows; i++) for (let j = 0; j < b.cols; j++) {
      let s = 0; for (let k = 0; k < a.cols; k++) s += a.w[i * a.cols + k] * b.w[k * b.cols + j];
      o.w[i * b.cols + j] = s;
    }
    if (this.bp) this.fns.push(() => {
      for (let i = 0; i < a.rows; i++) for (let j = 0; j < b.cols; j++) {
        const g = o.dw[i * b.cols + j]; if (!g) continue;
        for (let k = 0; k < a.cols; k++) {
          a.dw[i * a.cols + k] += b.w[k * b.cols + j] * g;
          b.dw[k * b.cols + j] += a.w[i * a.cols + k] * g;
        }
      }
    });
    return o;
  }
  add(a, b) {
    const o = new Mat(a.rows, a.cols);
    for (let i = 0; i < o.w.length; i++) o.w[i] = a.w[i] + b.w[i];
    if (this.bp) this.fns.push(() => { for (let i = 0; i < o.w.length; i++) { a.dw[i] += o.dw[i]; b.dw[i] += o.dw[i]; } });
    return o;
  }
  embed(m, i) {
    const o = new Mat(1, m.cols);
    for (let j = 0; j < m.cols; j++) o.w[j] = m.w[i * m.cols + j];
    if (this.bp) this.fns.push(() => { for (let j = 0; j < m.cols; j++) m.dw[i * m.cols + j] += o.dw[j]; });
    return o;
  }
  tanh(m) {
    const o = new Mat(m.rows, m.cols);
    for (let i = 0; i < m.w.length; i++) o.w[i] = Math.tanh(m.w[i]);
    if (this.bp) this.fns.push(() => { for (let i = 0; i < m.w.length; i++) { const y = o.w[i]; m.dw[i] += (1 - y * y) * o.dw[i]; } });
    return o;
  }
  softmax(m) {
    const o = new Mat(m.rows, m.cols);
    for (let i = 0; i < m.rows; i++) {
      let max = -1e9; for (let j = 0; j < m.cols; j++) max = Math.max(max, m.get(i, j));
      let sum = 0; for (let j = 0; j < m.cols; j++) { const e = Math.exp(m.get(i, j) - max); o.set(i, j, e); sum += e; }
      for (let j = 0; j < m.cols; j++) o.set(i, j, o.get(i, j) / sum);
    }
    if (this.bp) this.fns.push(() => {
      for (let i = 0; i < m.rows; i++) for (let j = 0; j < m.cols; j++) {
        const g = o.dw[i * m.cols + j]; if (!g) continue;
        for (let k = 0; k < m.cols; k++) { const pj = o.get(i, j), pk = o.get(i, k); m.dw[i * m.cols + k] += g * pk * ((j === k ? 1 : 0) - pj); }
      }
    });
    return o;
  }
}

const model = {
  wte: new Mat(V, d, w.wte),
  wpe: new Mat(maxLen, d, w.wpe),
  wq:  new Mat(d, d, w.wq),
  wk:  new Mat(d, d, w.wk),
  wv:  new Mat(d, d, w.wv),
  wo:  new Mat(d, d, w.wo),
  w1:  new Mat(d, d * 4, w.w1),
  w2:  new Mat(d * 4, d, w.w2),
  wh:  new Mat(d, V, w.wh)
};

function forward(ids, g) {
  const n = ids.length;
  const x = new Mat(n, d);
  for (let i = 0; i < n; i++) {
    const e = g.add(g.embed(model.wte, ids[i]), g.embed(model.wpe, i));
    for (let j = 0; j < d; j++) x.w[i * d + j] = e.w[j];
  }
  const Q = g.matmul(x, model.wq);
  const K = g.matmul(x, model.wk);
  const Vp = g.matmul(x, model.wv);
  const att = new Mat(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) { let s = 0; for (let k = 0; k < d; k++) s += Q.get(i, k) * K.get(j, k); att.set(i, j, s / Math.sqrt(d)); }
    for (let j = i + 1; j < n; j++) att.set(i, j, -1e9);
  }
  const aw = g.softmax(att);
  const ao = g.matmul(aw, Vp);
  const r1 = g.add(x, g.matmul(ao, model.wo));
  const r2 = g.add(r1, g.matmul(g.tanh(g.matmul(r1, model.w1)), model.w2));
  return g.matmul(r2, model.wh);
}

function generate(text, n = 32) {
  let seq = [...tokenize(text), SEP].map(t => stoi[t]).filter(x => x != null);
  for (let i = 0; i < n; i++) {
    const ctx = seq.slice(-maxLen);
    const g = new Graph(false);
    const logits = forward(ctx, g);
    const off = (ctx.length - 1) * V;
    let max = -1e9; for (let v = 0; v < V; v++) max = Math.max(max, logits.w[off + v]);
    let sum = 0; const p = []; for (let v = 0; v < V; v++) { const x = Math.exp(logits.w[off + v] - max); p.push(x); sum += x; }
    let r = Math.random() * sum, idx = 0; for (; idx < V; idx++) { r -= p[idx]; if (r <= 0) break; }
    if (itos[idx] === END) break;
    seq.push(idx);
  }
  return seq.map(i => itos[i]).filter(t => t !== SEP).join(" ");
}

function ask(msg) { return generate(msg); }