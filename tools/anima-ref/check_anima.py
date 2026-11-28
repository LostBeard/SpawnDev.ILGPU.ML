"""Check every Anima stage of anima_ref.py against Lore's ComfyUI taps / dumps.

Usage:  python check_anima.py [--skip-full] [--weights PATH] [--out JSON]
Prints max-abs and relRMS (||a-b|| / ||b||) per stage; writes the table as JSON.

Each stage is fed its OWN tapped inputs (errors do not compound), except the end-to-end rows (DiT call_0 /
call_1 from input + timestep + c_crossattn, adapter from Qwen3 states + T5 ids), which chain everything.
RoPE / sinusoid tables are always computed here in float64 (never taken from the taps), so the block rows
also test them. The "alt:" rows are red-checks: a wrong convention must show a large error, which proves the
taps can tell the conventions apart.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anima_ref as A  # noqa: E402
from st_reader import SafeTensors  # noqa: E402
from taps import tap, dump, TAPS, DUMPS  # noqa: E402

ROWS = []


def report(stage, ours, ref, note=''):
    ours = np.asarray(ours, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert ours.shape == ref.shape, (stage, ours.shape, ref.shape)
    d = ours - ref
    mx = float(np.abs(d).max())
    nr = float(np.linalg.norm(ref))
    rr = float(np.linalg.norm(d) / nr) if nr > 0 else float(np.linalg.norm(d))
    ROWS.append({'stage': stage, 'max_abs': mx, 'rel_rms': rr, 'ref_max': float(np.abs(ref).max()), 'note': note})
    print(f'{stage:<58} max_abs {mx:10.3e}  relRMS {rr:10.3e}  {note}', flush=True)
    return rr


class Weights:
    """Weight getter with a small cache for non-block tensors; block tensors are dropped by drop_block()."""

    def __init__(self, path):
        self.st = SafeTensors(path)
        self.cache = {}

    def __call__(self, name):
        w = self.cache.get(name)
        if w is None:
            w = self.st.get(name)
            self.cache[name] = w
        return w

    def drop(self, prefix):
        for k in [k for k in self.cache if k.startswith(prefix)]:
            del self.cache[k]


def check_adapter(w):
    print('--- LLM adapter (positive prompt) ---')
    ids = tap('llm_adapter_embed_0_pre-in0_0')
    report('adapter.embed (gather rows of embed.weight)', w('net.llm_adapter.embed.weight')[ids], tap('llm_adapter_embed_0_out0'))
    report('adapter.in_proj (Identity)', tap('llm_adapter_in_proj_0_pre-in0_0'), tap('llm_adapter_in_proj_0_out0'))
    pos = tap('llm_adapter_rotary_emb_0_pre-in1_0')
    assert np.array_equal(pos[0], np.arange(pos.shape[1])), 'adapter position ids are not 0..M-1'
    M = pos.shape[1]
    cx, sx = A.adapter_rope(M)
    report('adapter.rotary_emb cos [1,M,64] (T5 side)', cx[None], tap('llm_adapter_rotary_emb_0_out0'))
    report('adapter.rotary_emb sin [1,M,64] (T5 side)', sx[None], tap('llm_adapter_rotary_emb_0_out1'))
    N = tap('llm_adapter_blocks_0_0_pre-in1_0').shape[1]
    cc, sc = A.adapter_rope(N)
    report('adapter rotary cos [1,N,64] (Qwen3 side)', cc[None], tap('llm_adapter_blocks_0_0_pre-kw-position_embeddings_context0'))
    report('adapter rotary sin [1,N,64] (Qwen3 side)', sc[None], tap('llm_adapter_blocks_0_0_pre-kw-position_embeddings_context1'))
    for i in range(A.A_BLOCKS):
        x = tap(f'llm_adapter_blocks_{i}_0_pre-in0_0')
        ctx = tap(f'llm_adapter_blocks_{i}_0_pre-in1_0')
        y = A.adapter_block(x, ctx, w, i, cx, sx, cc, sc)
        ref = tap(f'llm_adapter_blocks_{i}_0_out0')
        report(f'adapter.blocks.{i} (own tapped inputs)', y, ref)
        report(f'adapter.blocks.{i} delta (out - in)', y - x, ref - x)
    x = tap('llm_adapter_out_proj_0_pre-in0_0')
    y = A.linear(x, w('net.llm_adapter.out_proj.weight'), w('net.llm_adapter.out_proj.bias'))
    report('adapter.out_proj (Linear + bias)', y, tap('llm_adapter_out_proj_0_out0'))
    x = tap('llm_adapter_norm_0_pre-in0_0')
    report('adapter.norm (RMSNorm eps 1e-6, AFTER out_proj)', A.rms_norm(x, w('net.llm_adapter.norm.weight'), A.EPS_A),
           tap('llm_adapter_norm_0_out0'))

    # alt (red-check) conventions on block 0 / norm
    x0 = tap('llm_adapter_blocks_0_0_pre-in0_0')
    ctx0 = tap('llm_adapter_blocks_0_0_pre-in1_0')
    ref0 = tap('llm_adapter_blocks_0_0_out0')
    report('alt: adapter.blocks.0 eps 1e-5', A.adapter_block(x0, ctx0, w, 0, cx, sx, cc, sc, eps=1e-5), ref0, 'should be worse')
    report('alt: adapter.blocks.0 GELU tanh', A.adapter_block(x0, ctx0, w, 0, cx, sx, cc, sc, gelu=A.gelu_tanh), ref0, 'should be worse')
    report('alt: adapter.blocks.0 no RoPE in cross-attn', A.adapter_block(x0, ctx0, w, 0, cx, sx, cc, sc, cross_rope=False), ref0, 'should be worse')
    report('alt: adapter.norm eps 1e-5', A.rms_norm(x, w('net.llm_adapter.norm.weight'), 1e-5), tap('llm_adapter_norm_0_out0'), 'should be worse')

    # end to end, positive and negative prompts vs c_crossattn rows (call_0 batch: row 0 = negative, row 1 = positive)
    ca = dump('call_0_c_c_crossattn')
    for label, row, qname, idname in [('positive', 1, 'pos_0_cond', 'pos_0_t5xxl_ids'), ('negative', 0, 'neg_0_cond', 'neg_0_t5xxl_ids')]:
        q = dump(qname)
        t5 = np.rint(dump(idname)).astype(np.int64)[None]
        out = A.adapter_forward(q, t5, w)
        Mx = t5.shape[1]
        if label == 'positive':
            report('adapter END-TO-END positive vs adapter tap out', out, tap('llm_adapter_0_out0'))
        report(f'adapter END-TO-END {label} vs c_crossattn[{row}, :{Mx}]', out[0], ca[row, :Mx])
        padded = A.pad_context(out)
        report(f'adapter {label} padded to 512 vs c_crossattn[{row}]', padded[0], ca[row])


def check_dit_stages(w):
    print('--- DiT stages (call_0, batch [uncond, cond]) ---')
    t = tap('t_embedder_0_0_pre-in0_0')[:, 0]
    tp, adaln, emb = A.t_embedder(t, w)
    report('t_embedder.0 sinusoid (t = sigma, [cos, sin])', tp, tap('t_embedder_0_0_out0')[:, 0])
    report('t_embedder.1 out1 adaLN-LoRA vector [B,6144]', adaln, tap('t_embedder_1_0_out1')[:, 0])
    report('t_embedder.1 out0 (= its input, t_proj)', tp, tap('t_embedder_1_0_out0')[:, 0])
    report('t_embedding_norm (RMSNorm eps 1e-6) from tapped input',
           A.rms_norm(tap('t_embedding_norm_0_pre-in0_0'), w('net.t_embedding_norm.weight'), A.EPS_TNORM), tap('t_embedding_norm_0_out0'))
    report('t_embedding_norm chained from t', emb, tap('t_embedding_norm_0_out0')[:, 0])
    report('alt: t_embedding_norm eps 1e-5',
           A.rms_norm(tap('t_embedding_norm_0_pre-in0_0'), w('net.t_embedding_norm.weight'), 1e-5), tap('t_embedding_norm_0_out0'), 'should be worse')
    report('alt: t sinusoid with t*1000', A.timestep_proj(t * 1000.0), tap('t_embedder_0_0_out0')[:, 0], 'should be worse')

    x17 = tap('x_embedder_0_pre-in0_0')
    assert np.all(x17[:, 16] == 0.0)
    report('x_embedder.proj (patch 1x2x2, (c,pt,ph,pw) order)', A.x_embedder(x17, w), tap('x_embedder_0_out0'))
    report('pad_mask_channel(call_0 input) vs x_embedder input', A.pad_mask_channel(dump('call_0_input')), x17)

    ang = A.rope_angles(1, 32, 32)
    R = tap('pos_embedder_0_out0')
    report('pos_embedder rotation matrices [1024,64,2,2]', A.rope_matrices(ang), R)
    report('alt: rope matrices with Diffusers default scale (1,1)', A.rope_matrices(A.rope_angles(1, 32, 32, scale_hw=1.0)), R, 'should be worse')
    cos, sin = A.dit_rope_cos_sin(ang)

    for i in (0, 13, 27):
        x = tap(f'blocks_{i}_0_pre-in0_0')
        B = x.shape[0]
        e = tap(f'blocks_{i}_0_pre-in1_0')[:, 0]
        ctx = tap(f'blocks_{i}_0_pre-in2_0')
        ad = tap(f'blocks_{i}_0_pre-kw-adaln_lora_B_T_3D0')[:, 0]
        y = A.dit_block(x.reshape(B, -1, A.D_MODEL), e, ctx, ad, cos, sin, w, i).reshape(x.shape)
        ref = tap(f'blocks_{i}_0_out0')
        report(f'blocks.{i} (own tapped inputs) both rows', y, ref)
        report(f'blocks.{i} row 0 (uncond)', y[0], ref[0])
        report(f'blocks.{i} row 1 (cond)', y[1], ref[1])
        # the residual stream carries large outlier channels (ref_max in the JSON), which can hide errors in the
        # block's own update; the delta (out - in) row measures the block's contribution alone
        report(f'blocks.{i} delta (out - in)', y - x, ref - x, f'|x|max {np.abs(x).max():.3g}')
        if i == 0:
            xf = x.reshape(B, -1, A.D_MODEL)
            report('alt: blocks.0 q/k RMSNorm eps 1e-5 (Diffusers default)',
                   A.dit_block(xf, e, ctx, ad, cos, sin, w, 0, eps_qk=1e-5).reshape(x.shape), ref, 'should be worse')
            report('alt: blocks.0 interleaved RoPE pairs (2j,2j+1)',
                   A.dit_block(xf, e, ctx, ad, np.cos(ang), np.sin(ang), w, 0, rope_mode='interleaved').reshape(x.shape), ref, 'should be worse')
            report('alt: blocks.0 GELU tanh',
                   A.dit_block(xf, e, ctx, ad, cos, sin, w, 0, gelu=A.gelu_tanh).reshape(x.shape), ref, 'should be worse')
            # zero padding rows are NOT masked: they give k = v = 0, score 0, and still take softmax mass
            m_real = [int(np.count_nonzero(np.abs(ctx[b]).sum(-1))) for b in range(B)]
            y_np = np.concatenate([A.dit_block(xf[b:b + 1], e[b:b + 1], ctx[b:b + 1, :m_real[b]], ad[b:b + 1], cos, sin, w, 0)
                                   for b in range(B)]).reshape(x.shape)
            report(f'alt: blocks.0 context NOT padded (real rows {m_real} only)', y_np, ref, 'should be worse')
            report('alt: blocks.0 adaLN LayerNorm eps 1e-5',
                   A.dit_block(xf, e, ctx, ad, cos, sin, w, 0, eps_ln=1e-5).reshape(x.shape), ref, 'should be worse')
        w.drop(f'net.blocks.{i}.')

    x = tap('final_layer_0_pre-in0_0')
    B = x.shape[0]
    y = A.final_layer(x.reshape(B, -1, A.D_MODEL), tap('final_layer_0_pre-in1_0')[:, 0], tap('final_layer_0_pre-kw-adaln_lora_B_T_3D0')[:, 0], w)
    report('final_layer (own tapped inputs) [2,1,32,32,64]', y.reshape(tap('final_layer_0_out0').shape), tap('final_layer_0_out0'))


def run_dit_full(w, folder, call, intermediate):
    x = dump(f'{call}_input', folder)
    t = dump(f'{call}_timestep', folder)
    ctx = dump(f'{call}_c_c_crossattn', folder)
    ref = dump(f'{call}_output', folder)
    B = x.shape[0]
    taps_at = {}
    if intermediate:
        for i in (0, 13, 27):
            taps_at[i] = tap(f'blocks_{i}_0_out0').reshape(B, -1, A.D_MODEL)

    def hook(i, h):
        if i in taps_at:
            report(f'  {call} chained: after blocks.{i}', h, taps_at[i])
        w.drop(f'net.blocks.{i}.')
        print(f'    block {i} done {time.time() - t0:.1f}s', flush=True)

    t0 = time.time()
    v = A.dit_forward(x, t, ctx, w, block_hook=hook)   # the DiT's raw output (flow velocity)
    np.save(os.path.join(SCRATCH, f'dit_{call}_out_f64.npy'), v)
    if intermediate:
        fl = tap('final_layer_0_out0').reshape(B, -1, 64)
        report(f'  {call} chained: DiT raw output vs unpatchify(final_layer tap)', v, A.unpatchify(fl, 1, x.shape[3] // 2, x.shape[4] // 2))
    # The dumped call output is ComfyUI's DENOISED prediction (CONST / flow): out = input - sigma * v.
    s = t[:, None, None, None, None]
    den = A.denoise_flow(x, t, v)
    report(f'DiT END-TO-END {call} denoised x - t*v vs dumped output (t={t[0]:.7f})', den, ref)
    report(f'DiT END-TO-END {call} row 0 (uncond) denoised', den[0], ref[0])
    report(f'DiT END-TO-END {call} row 1 (cond) denoised', den[1], ref[1])
    v_ref = (x - ref) / s
    report(f'DiT END-TO-END {call} raw v vs implied (input - output)/t', v, v_ref, 'fp32 cancellation in the dump')
    report(f'alt: {call} raw v vs dumped output directly', v, ref, 'should be worse (dump is denoised)')
    return v


SCRATCH = r'D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\_scratch\anima-ref'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=r'D:\Projects\anima\anima-preview3-base.safetensors')
    ap.add_argument('--skip-full', action='store_true')
    ap.add_argument('--only', choices=['adapter', 'dit', 'full', 'all'], default='all')
    ap.add_argument('--out', default=os.path.join(SCRATCH, 'check_results.json'))
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    w = Weights(a.weights)
    t0 = time.time()
    if a.only in ('adapter', 'all'):
        check_adapter(w)
        w.drop('net.llm_adapter.')
    if a.only in ('dit', 'all'):
        check_dit_stages(w)
    if a.only in ('full', 'all') and not a.skip_full:
        print('--- DiT end to end (28 blocks) ---')
        # run3 call_0 == run1 call_0 (same input / timestep / context), so the call_0 chain can be compared to the block taps too
        run_dit_full(w, DUMPS, 'call_0', intermediate=True)
        run_dit_full(w, DUMPS, 'call_1', intermediate=False)
    print(f'total {time.time() - t0:.1f}s')
    prev = []
    if os.path.exists(a.out) and a.only != 'all':
        with open(a.out) as f:
            prev = [r for r in json.load(f) if r['stage'] not in {x['stage'] for x in ROWS}]
    with open(a.out, 'w') as f:
        json.dump(prev + ROWS, f, indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
