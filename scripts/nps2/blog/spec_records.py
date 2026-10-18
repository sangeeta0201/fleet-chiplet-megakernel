"""spec_records(lines): the kernel's [SPEC] counter records from a served log, including the ones another process's
output cut in two. In a served log the kernel's [SPEC] line and the launch-placement summary ([RESID], same log) are
written at the same moment, so a record can be split: its head ("[SPEC] passes=117 deco") ends where the other line
starts, and its tail ("de_passes=89 drafted=216 accepted=167 final_step=423 k=5 corrupt=0") appears on a later line of
its own; or the record loses its "[S" prefix ("PEC] passes=68 ..."). A complete record ends in " k=", so a number cut
short ("final_step=3" + "89 k=5 ...") is never read as complete.
Returns [(line_index, (passes, decode_passes, drafted, accepted, final_step))] and the heads left without a tail."""
import re

FULL = re.compile(r"passes=(\d+) decode_passes=(\d+) drafted=(\d+) accepted=(\d+) final_step=(\d+) k=")
HEAD = re.compile(r"\[SPEC\] (passes=\d+[^\[]*)")


def spec_records(lines):
    recs, unresolved, used = [], [], set()
    for i, ln in enumerate(lines):
        m = FULL.search(ln)
        if m:
            recs.append((i, tuple(map(int, m.groups()))))
            continue
        h = HEAD.search(ln)
        if not h:
            continue
        for j in range(i + 1, min(i + 400, len(lines))):
            t = lines[j]
            if j in used or t.startswith("[") or "corrupt=" not in t:
                continue
            m2 = FULL.search(h.group(1) + t)
            if m2:
                used.add(j)
                recs.append((i, tuple(map(int, m2.groups()))))
                break
        else:
            unresolved.append((i, ln[:120]))
    return recs, unresolved


def distinct_full(recs, min_tokens=200):
    """One record per question. The bench reruns a question right after itself (on a long stream gap), and a rerun
    repeats its question's counters exactly (greedy decode), so a record equal to the one before it is the rerun and
    is dropped. Records are not deduplicated otherwise: different questions can share all five counters (DFlash: the
    same prompt length and number of decode passes, seen twice in one GSM8K-128 serve). Requests shorter than
    min_tokens (the bench's closing 1-token requests) are excluded."""
    out, prev = [], None
    for i, k in recs:
        if k[3] + k[1] < min_tokens:
            continue
        if k != prev:
            out.append((i, k))
        prev = k
    return out

