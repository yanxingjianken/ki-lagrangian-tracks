# 02_isca — three-dimensional parcel tracks from a dry baroclinic Isca run

Planned. Same question as `01_bve`, in three dimensions: can a kinematic-interpolant model trained on part of a
population of air-parcel tracks fill any missing segment of an unseen parcel's track, and how far can it forecast?

- Model: Isca dry dynamical core (Held–Suarez-type forcing), output of u, v, ω and surface pressure on model levels.
- Tracks: LAGRANTO (`../lagranto`) in full 3-D mode, with the vertical wind.
- Learning: the `01_bve` model with a 3-D state (horizontal position plus pressure or potential temperature).

Nothing here has been run yet.
