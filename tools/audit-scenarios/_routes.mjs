// One scenario per demo route for tools/audit-demo-models.mjs. A scenario DRIVES the page to a real result and returns evidence of it
// (the rendered text, a measured value). Throwing fails the route. ⚠️ Assert on the RENDERED RESULT, never on a console line: a page's logging
// is incidental, its result is the fact (see tools/drive-ml-pages.cs for why).
//
// h (see audit-demo-models.mjs): ev(js) waitFor(js, ms, label) click(sel) clickText(sel, /re/) type(sel, text) typeNth(sel, n, text) count(sel)
//   text(sel) upload(sel, file) shot(name) sleep(ms) log(msg) base
// Route options: allowHuggingFace (the page legitimately talks to huggingface.co, e.g. a model BROWSER), expectConsole: [RegExp] (errors the page
// is expected to log; anything else fails the route).

export const defaultRoutes = [
  '/', '/getting-started', '/pipelines',
  '/classify', '/clip', '/depth', '/depth-voxel', '/detect', '/explain', '/face', '/pose', '/remove-bg', '/style', '/super-res',
  '/sentiment', '/embeddings', '/text-gen', '/whisper', '/tts', '/ai-chat',
  '/generate', '/image-to-3d', '/text-to-3d',
  '/inspector', '/models', '/cache', '/benchmark', '/train', '/snake', '/tests',
];

