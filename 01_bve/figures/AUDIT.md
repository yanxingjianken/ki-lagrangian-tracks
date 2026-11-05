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
