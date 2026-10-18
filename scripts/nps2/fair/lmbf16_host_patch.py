"""lmbf16_host_patch.py: demo.py side of MPK_LM_BF16 (BF16 LM head) in the fair lane. Idempotent."""
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "/home/user/fleet-lanes/fair/demo/gpt_oss/demo.py"
s = open(p).read()
if "MPK_LM_BF16" in s:
    print("already patched")
    sys.exit(0)
start = '        print(">>> cat done, quantize", flush=True)\n'
end = ('              f"-> {lm_head_packed.shape} packed ({lm_head_packed.numel()/1e6:.0f} MB)")\n')
i = s.index(start)
j = s.index(end, i) + len(end)
assert s.count(start) == 1, "anchor not unique"
old = s[i:j]
new = (
    '        if os.environ.get("MPK_LM_BF16", "0") == "1":\n'
    '            # MPK_LM_BF16: the checkpoint\'s BF16 LM head, as vLLM runs it. K-major records\n'
    '            # [n_wgs, 4 tiles, K/32 steps, 64 lanes, 8 bf16], lane = 16 * (K8 group) + row.\n'
    '            _lm_nw = lm_head_weight.shape[0] // lm_head_output_per_wg\n'
    '            _lm_k = lm_head_weight.shape[1]\n'
    '            assert lm_head_weight.dtype == torch.bfloat16 and _lm_k % 32 == 0\n'
    '            lm_head_packed = (lm_head_weight.reshape(_lm_nw, 4, 16, _lm_k // 32, 4, 8)\n'
    '                              .permute(0, 1, 3, 4, 2, 5).contiguous()\n'
    '                              .view(torch.uint8).reshape(_lm_nw, -1))\n'
    '            print(f"LM head BF16 (MPK_LM_BF16): {lm_head_weight.shape} -> {lm_head_packed.shape} "\n'
    '                  f"K-major records ({lm_head_packed.numel()/1e6:.0f} MB)", flush=True)\n'
    '        else:\n'
    + "".join("    " + ln if ln.strip() else ln for ln in old.splitlines(keepends=True))
)
s = s[:i] + new + s[j:]
open(p, "w").write(s)
print("patched demo.py")
