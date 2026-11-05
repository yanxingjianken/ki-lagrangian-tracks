# Figure audits (fresh reader, image only)

## f1_conditional_ki_3d.png — intended message: the 3-D conditional KI keeps its skill in unseen flows
- 2026-11-05 round 1: FAIL. Title and y-axis label clipped; the reader noticed that the flows outside the training
  ranges have the smallest errors for straight motion too and the figure did not say why. Fix: wider figure, shorter
  y label, title says those flows are slower.
- 2026-11-05 round 2: PASS. Reader: "in unseen flows the conditional KI has lower error than straight motion at every
  horizon, and the gap holds for flows outside the training ranges"; title numbers consistent (pooled over forecast
  and backcast).

## f2_by_level.png — intended message: at every start level the KI cuts the 24-h error of straight motion
- 2026-11-05 round 1 (first version had 6 bars per level; replaced before the audit by KI vs straight motion for the
  in-distribution case only): PASS with caveats: "cuts ... to X %" could read as a reduction of X % -> title now says
  the KI error is X % of straight motion's; legend tick made smaller. Reader's note kept for the README: at sigma
  0.968 the KI vertical error (5.2 hPa) exceeds the true 24-h pressure change (4.8 hPa).
