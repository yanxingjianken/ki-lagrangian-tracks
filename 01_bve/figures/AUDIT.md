# Figure audits (fresh reader, image only)

## f1_abilities.png — intended message: in the training flow, gaps are filled to about a kilometre, while 24-h forecasts and backcasts are only modestly better than straight motion
- 2026-11-05 round 1: PASS. Reader: "a trained KI fills 24-h gaps to 1.5 km, but its 24-h forecast and backcast errors
  are 585 and 802 km, only modestly better than straight motion". Notes: KI not spelled out, "patch" undefined →
  legend now says "Kinematic Interpolant (KI)", title says "8 patches of 1000 parcels".

## f2_unseen_flows.png — intended message: a model trained on one flow does not transfer to other regions or flows
- 2026-11-05 round 1: FAIL. Title claimed "worse than no learning in 3 of 3 unseen cases", true only for gap filling;
  in backcasts the model beats straight motion in 2 of 3 unseen flows. Fix: title counts the cases per ability from
  the data; legend clipped → wider figure.
- 2026-11-05 round 2: PASS. Every title number matches the bars. Suggestion adopted: "does not transfer its gap
  filling" (backcasts are mixed).

## f3_conditional_ki.png — intended message: the conditional KI keeps most of its skill in flows it never saw
- 2026-11-05 round 1: PASS. Reader: "in every flow category and at every horizon the conditional KI error is well
  below straight motion, by roughly 40-60 % at 24 h, and it holds in unseen flows"; title numbers checked against the
  lines. Note: "KI" undefined -> legend now says "conditional Kinematic Interpolant (KI)". Earlier drafts had a
  duplicated legend and a clipped title (fixed before the audit).
