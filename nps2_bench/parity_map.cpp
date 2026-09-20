// The parity-flex MoE dispatch map, verified exhaustively.
//
// Constraint that drives the whole design: every XCD does exactly
// 184/8 == 23 groups, always -- today's perfect load balance, preserved
// unconditionally. Locality is then maximised WITHIN that budget rather than
// being bought with load imbalance.
//
// An earlier formulation let each AID cover however many experts it drew.
// Verified correct, but it cost 46 ranks/XCD on a 4-0 split (2.00x the work,
// with four XCDs idle) and 31.9 ranks/XCD in expectation -- a 1.39x inflation
// that would have eaten the 2.5x read win it was meant to buy.
//
// The map:
//   * parity-sort the top-4, so the n0 even-id experts occupy slots
//     0..n0-1 and the odd-id ones occupy slots n0..3. Slot order is a free
//     permutation, so this costs nothing.
//   * work item i == slot * 46 + group, i in [0,184). Items [0, n0*46) are
//     therefore exactly the even-parity (AID0-resident) ones.
//   * AID0's four XCDs take items [0,92), AID1's take [92,184), 23 each.
//
// Because the split point n0*46 moves with the routing while the AID boundary
// sits at 92, a favourable split puts every item in the AID that owns its
// weights, and an unfavourable one degrades gracefully to today's 50% rather
// than to idle silicon.
//
// Expert e's weights live in AID (e & 1) with no replication, so this is
// zero extra memory.
//
// Host-only, no GPU needed.
#include <cstdio>
#include <vector>

static constexpr int kXcds = 8;
static constexpr int kXcdsPerAid = 4;
static constexpr int kTopK = 4;
static constexpr int kGroups = 46;
static constexpr int kTotalWork = kTopK * kGroups;          // 184
static constexpr int kPerXcd = kTotalWork / kXcds;          // 23
static constexpr int kPerAid = kTotalWork / 2;              // 92

struct Decoded {
  bool active;
  int slot;   // 0..3, index into the parity-sorted top-4
  int group;  // 0..45
  bool local; // does this XCD's AID own this expert's weights?
};

static Decoded decode(int xcd, int phase_rank, int n0) {
  Decoded d{false, -1, -1, false};
  if (phase_rank >= kPerXcd) {
    return d; // padding rank; early-returns in ~0 cycles
  }
  int const aid = xcd / kXcdsPerAid;
  int const local_xcd = xcd % kXcdsPerAid;
  int const idx = aid * kPerAid + local_xcd * kPerXcd + phase_rank;
  d.active = true;
  d.slot = idx / kGroups;
  d.group = idx % kGroups;
  // Items below n0*46 are the even-parity experts, resident in AID0.
  int const owner_aid = (idx < n0 * kGroups) ? 0 : 1;
  d.local = (owner_aid == aid);
  return d;
}

int main() {
  printf("=== parity-flex MoE dispatch map (constant 23 groups/XCD) ===\n");
  printf("  %d XCDs, top-%d, %d groups/expert, %d work items, %d per XCD\n\n",
         kXcds, kTopK, kGroups, kTotalWork, kPerXcd);

  bool all_ok = true;
  double exp_local = 0.0;
  // P(n0) for top-4 of 128 with balanced parity ~ Binomial(4, 1/2)
  double const prob[5] = {0.0625, 0.25, 0.375, 0.25, 0.0625};

  printf("  split  P      ranks/XCD  covered  dup  missing  local%%  verdict\n");
  for (int n0 = 0; n0 <= kTopK; n0++) {
    std::vector<int> count((size_t)kTotalWork, 0);
    int n_local = 0, n_active = 0, max_rank = 0;

    for (int xcd = 0; xcd < kXcds; xcd++) {
      for (int r = 0; r < kPerXcd; r++) {
        Decoded const d = decode(xcd, r, n0);
        if (!d.active) {
          continue;
        }
        n_active++;
        if (d.local) {
          n_local++;
        }
        if (r + 1 > max_rank) {
          max_rank = r + 1;
        }
        count[(size_t)d.slot * kGroups + d.group]++;
      }
    }

    int covered = 0, dup = 0, missing = 0;
    for (int i = 0; i < kTotalWork; i++) {
      if (count[i] == 1) {
        covered++;
      } else if (count[i] > 1) {
        dup++;
      } else {
        missing++;
      }
    }
    double const pct = 100.0 * (double)n_local / (double)n_active;
    exp_local += prob[n0] * pct;
    bool const ok = (dup == 0 && missing == 0 && covered == kTotalWork);
    all_ok = all_ok && ok;
    printf("  %d-%d    %.4f %8d   %6d  %3d  %7d  %5.1f  %s\n", n0,
           kTopK - n0, prob[n0], max_rank, covered, dup, missing, pct,
           ok ? "OK" : "BROKEN");
  }

  printf("\n  expected locality: %.1f%%  (today's interleave: ~50%%)\n",
         exp_local);
  printf("  groups per XCD: %d in EVERY split -- no load inflation, and the\n"
         "  launcher's existing 23-ranks/XCD budget is unchanged.\n", kPerXcd);
  printf("  worst case (4-0 / 0-4) is 50%% local == today, never worse.\n");
  printf("\n  %s\n", all_ok ? "ALL SPLITS COVER EXACTLY ONCE -- safe to integrate"
                            : "COVERAGE BROKEN -- do not integrate");
  return all_ok ? 0 : 1;
}

