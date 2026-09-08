# Reproducing the NPS2 arm sweep

Measured Sep 8. Every arm is gated on `rmsnorm_out=ff1bdbe7ad2c4ed3`; a fast run
with a different hash is not a result.

## Results

SPX+NPS2, patched driver, `xcp_nc=Y flag_mtype=2`, 5 reps x 400 layers per arm,
median over all `[OPROJ_INNER]` samples (~3,200 per run, 8 XCDs):

| arm | total | bar | slicewait | mfma | gate |
| --- | --- | --- | --- | --- | --- |
| bare (no flags) | 22.60 | 6.64 | 4.44 | 4.80 | green |
| `--aid=0 --split=0 --lsplit=0 --hrdv=0` | 22.80 | 6.84 | 4.40 | 5.12 | green |
| `--coherent=1` alone | 22.84 | 6.84 | 4.52 | 5.16 | green |
| `--aid=1` | 10.72 | 2.64 | 0.40 | 3.92 | green |
| `--coherent=1 --aid=1` | **10.52** | 2.64 | 0.40 | 3.96 | green |

SPX+NPS1 baseline on the **stock** driver, n=25,600: total **9.800**,
slicewait 0.400, mfma 3.960, bar 2.000, rmsnorm_router 1.520, topk 1.960.

These reproduce the three reference rows measured the day before:

| config | bar w/ poll | bar no poll | poll costs | total (ref) | total (repro) |
| --- | --- | --- | --- | --- | --- |
| NPS1 flat | 1.92 | 1.52 | 0.40 | 9.76 | 9.800 |
| NPS2 flat (NC sync line) | 6.76 | 1.44 | 5.32 | 22.8 | 22.60 - 22.84 |
| NPS2 best | 2.40 | 2.00 | 0.40 | 10.32 | 10.52 |

## What the arms show

**`--aid=1` is the operative knob, not `--coherent=1`.** On its own
`--coherent=1` measures 22.84, i.e. no improvement over bare. `--aid=1` alone
gets 10.72; adding `--coherent=1` on top trims a further 0.20 to 10.52.
`--aid=1` is what allocates the sync buffer AID_LOCAL + EXT_COHERENT so the
driver's FLAGMTYPE knob (`aid_local_flag_mtype=2`) can give it **CC**. Without
it the sync line is **NC** and the barrier poll costs 5.32 us.

**`--aid` defaults to 0**, so bare defaults (22.60) are the slow NC-sync-line
case. Note the committed repro line in `RESULTS-NPS2.md` section 3 passes
`--aid=0` explicitly, which is the 22.8 arm, *not* the ~10.3 arm. Do not read
that command as the "default path" number.

**The residual gap is entirely in `bar`.** NPS2 best 10.52 vs NPS1 9.800 is
+0.72 us, of which bar is +0.64 (2.00 -> 2.64). `slicewait` (0.40) and `mfma`
(3.96) are at exact NPS1 parity, so the 5.32 us poll cost is fully recovered and
what remains is the barrier itself.

## Driver setup

NPS1 runs on the **stock** driver. The patched driver is needed only for
SPX+NPS2 -- do not use it for the NPS1 baseline.

For SPX+NPS2 use the canonical path, which ends by asserting SPX via sysfs:

```bash
~/nps1/load_perbo.sh        # -> load_perbo_gen.sh 1 2 (xcp_nc=1, flag_mtype=2)
                            # -> ends in set_spx.sh (per-BDF sysfs write)
```

Never use `rocm-smi --setmemorypartition`: it restarts amdgpu, which reloads the
**stock** DKMS module, discarding the patched build and leaving the box in
DPX+NPS2 with SPX absent from `available_compute_partition`.

Gate before measuring -- absent parameters mean the stock module is loaded:

```bash
cat /sys/module/amdgpu/parameters/aid_local_xcp_nc        # Y
cat /sys/module/amdgpu/parameters/aid_local_flag_mtype    # 2
cat /sys/module/amdgpu/parameters/aid_local_spx_nps2      # Y
for b in 05 15 65 75 85 95 e5 f5; do
  cat /sys/bus/pci/devices/0000:$b:00.0/current_compute_partition   # SPX
  cat /sys/bus/pci/devices/0000:$b:00.0/current_memory_partition    # NPS2
done
```

A `PERBO:` line in dmesg is **not** proof the patched module is live -- dmesg
survives reloads and the line may be hours stale. Only the sysfs parameters are
reliable.

## Running it from the Windows box

PowerShell strips single quotes before `ssh` sees them and leaves CRLF in
uploaded files, so write the script locally, pipe it over stdin, and strip `\r`
on arrival. Do not inline `$(...)` in the remote command -- PowerShell expands it
locally.

```powershell
$K = "$env:USERPROFILE\.ssh\id_ed25519"
$H = "schowdha@rainier-login"
((Get-Content -Raw .\coherent_arm.sh) -replace "`r","") |
  ssh -o BatchMode=yes -i $K -o IdentitiesOnly=yes $H `
      "cat > /tmp/coh.sh && tr -d '\r' < /tmp/coh.sh > ~/nps1/coherent_arm.sh"
ssh -o BatchMode=yes -i $K -o IdentitiesOnly=yes $H `
    "srun --jobid=<JOBID> --overlap timeout 2000 bash ~/nps1/coherent_arm.sh 2>&1 | tail -25"
```

`salloc` must run on the login node (`rainier-login`); direct SSH to thor-2 is
gated by `pam_slurm_adopt` until a job exists. As of Sep 8 there are no
reservations, so omit `--reservation`. Reuse a running job with
`srun --jobid=<id> --overlap`.

## The sweep script

`~/nps1/coherent_arm.sh`:

```bash
#!/bin/bash
set -u
FL=$HOME/fleet-chiplet-megakernel
cd "$FL" || exit 1

printf '  %s / %s   xcp_nc=%s flag_mtype=%s\n\n' \
  "$(cat /sys/bus/pci/devices/0000:05:00.0/current_compute_partition)" \
  "$(cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition)" \
  "$(cat /sys/module/amdgpu/parameters/aid_local_xcp_nc)" \
  "$(cat /sys/module/amdgpu/parameters/aid_local_flag_mtype)"

run() {
  local name=$1; shift
  local log=/tmp/c_$name.log
  : > "$log"
  for r in 1 2 3 4 5; do
    timeout 300 ./drive_phase7 --layers=400 --tiles=23 "$@" \
      --tag="$name$r" >> "$log" 2>&1
  done
  python3 - "$log" "$name" <<'PY'
import re, sys, statistics as st
log, name = sys.argv[1], sys.argv[2]
t = open(log).read()
g = lambda k: (st.median([float(x) for x in re.findall(rf'\b{k}=([0-9.]+)', t)])
               if re.search(rf'\b{k}=', t) else float('nan'))
h = re.search(r'rmsnorm_out=([0-9a-f]+)', t)
h = h.group(1) if h else 'NONE'
print(f"  {name:<22} total={g('total'):6.2f}  bar={g('bar'):5.2f} "
      f"slice={g('slicewait'):5.2f} mfma={g('mfma'):5.2f}  "
      f"{'green' if h=='ff1bdbe7ad2c4ed3' else 'RED '+h[:8]}")
PY
}

run bare
run aid0_explicit  --aid=0 --split=0 --lsplit=0 --hrdv=0
run aid1           --aid=1
run coherent1      --coherent=1
run coh_aid        --coherent=1 --aid=1
```

Medians come from the per-XCD `[OPROJ_INNER]` lines, which report
`slicewait / mfma / bar / rmsnorm_router / topk / total`. `mfma` sits at 3.96 in
a clean run; if it spreads (e.g. 1.76-4.80) the GPU is still busy from a previous
job and the numbers are contaminated -- let it settle and re-run.


## Rendezvous sweep: split / lsplit / hrdv are not where the gap lives

Two sweeps, both SPX+NPS2 on the patched driver, all hashes green, all 8 XCDs
dispatched.

**Without `--aid=1`** (6 reps, so the sync line stays NC):

| arm | total | bar | slicewait | mfma |
| --- | --- | --- | --- | --- |
| defaults | 22.84 | 6.68 | 4.40 | 4.96 |
| `--split=1` | 22.36 | 6.68 | 4.36 | 4.88 |
| `--lsplit=1` | 22.12 | 6.56 | 4.28 | 4.68 |
| `--hrdv=1` | 20.08 | 5.40 | 5.48 | 3.88 |
| `--aid=1` | **10.56** | 2.64 | 0.40 | 3.92 |
| all four | 10.88 | 2.60 | 0.40 | 3.96 |

**Layered on `--aid=1 --coherent=1`** (6 reps):

| arm | total | vs NPS1 | bar | vs NPS1 |
| --- | --- | --- | --- | --- |
| `--aid=1` | 10.72 | +0.92 | 2.64 | +0.64 |
| `+ --coherent=1` | 10.40 | +0.60 | 2.64 | +0.64 |
| `+ --split=1` | 10.40 | +0.60 | 2.64 | +0.64 |
| `+ --lsplit=1` | **10.36** | **+0.56** | 2.64 | +0.64 |
| `+ --hrdv=1` | 10.40 | +0.60 | 2.60 | +0.60 |
| `+ --split --lsplit` | 10.40 | +0.60 | 2.64 | +0.64 |
| all four | 10.52 | +0.72 | **2.52** | +0.52 |

### Conclusion

The residual NPS2 gap is **not** in the `split` / `lsplit` / `hrdv` rendezvous
machinery. Two independent directions say so:

* Without `--aid=1`, none of the three rescues the NC sync line -- 22.36, 22.12
  and 20.08 against a 22.84 baseline. Only `hrdv` moves at all.
* With `--aid=1`, `bar` is pinned at 2.52-2.64 across every combination; the
  whole span of the three knobs is 0.12 us.

`--aid=1` is the single decisive knob, worth ~12 us, because it replicates the
barrier sync buffer per AID (`alloc_in_aid(sync_bytes, 0/1, coherent)`) so each
die polls a flag line homed on its own die. `--coherent=1` is worth a further
~0.2-0.3 by marking that buffer EXT_COHERENT so `aid_local_flag_mtype=2` gives
it CC; on its own it is worth nothing, because the `if (use_aid)` block that
calls `alloc_in_aid` is skipped entirely.

Best NPS2 is **10.36** (`--aid=1 --coherent=1 --lsplit=1`) against NPS1 **9.800**,
i.e. **+0.56 us**, inside the previously recorded 10.28-10.52 band.

What remains is `bar` at ~2.6 vs NPS1's 2.00, insensitive to every knob above.
Working hypothesis, not yet measured: in NPS1 the sync line is `MTYPE_RW`,
cacheable and coherent within a single memory partition, while the best
available in NPS2 is `MTYPE_CC`, device-wide coherent through the DF-CS shadow
tags, which costs more per barrier round. If that holds, NPS1 parity is not
reachable while the barrier must be device-coherent across two AIDs, and the
remaining 0.56 us is a hardware property rather than a placement bug. The test
is a direct CC-vs-RW latency probe on the barrier line.

### Source notes

From `drive_phase7.cu`:

```
int use_aid = 0, split_on = 1, lsplit_on = 0, hrdv_on = 0, bsplit_on = 0;
...
if (use_aid) {
  void *fa = alloc_in_aid(sync_bytes, 0, g_coherent);
  void *fb = alloc_in_aid(sync_bytes, 1, g_coherent);
```

`--aid` defaults to **0** and `--split` defaults to **1**, so `--split=1` adds
nothing over bare defaults -- it is already on.
