# 02_dinosaur — three-dimensional parcel tracks in baroclinic life cycles (Dinosaur, JAX)

Planned. Same question as `01_bve`, in three dimensions: can a kinematic-interpolant model, given the observed tracks
of the other parcels of a patch, fill a common gap in their tracks, forecast and backcast them, and does it work in
flows it never saw?

- Model: the JAX primitive-equation dynamical core Dinosaur, set up as for the Thorncroft, Hoskins & McIntyre (1993)
  baroclinic life cycles, with a family of jet configurations and perturbation wavenumbers (training and unseen
  combinations, as in `01_bve`).
- Tracks: LAGRANTO (`../lagranto`) in full 3-D mode (u, v, ω, surface pressure on sigma levels).
- Learning: the `01_bve` context model with a 3-D state (horizontal position plus pressure or potential
  temperature), trained with DDP on at least 1e5 tracks.

Nothing here has been run yet.
