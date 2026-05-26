# Bushings Function

Surfaces of revolution generated from Bushing's function — derived from the
coherent states of the anisotropic harmonic oscillator.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
make test
```

## Project Structure

```
src/bushings_function/
  core/          — Bushing's function evaluation, incomplete gamma series
  geometry/      — Surface of revolution, Delaunay triangulation, coordinates
  plotting/      — Matplotlib visualisation (future)
```

## Parameters

The surface of revolution is wholly described by three parameters:

| Parameter | Type  | Description                    |
|-----------|-------|--------------------------------|
| η         | float | Gaussian spread parameter      |
| λ         | int   | Anisotropy parameter           |
| nF        | int   | Occupancy (Fermi) parameter    |

Two surface types are generated — **cigar** and **pancake** — corresponding to
the two orientations of the anisotropic harmonic oscillator.

## References

* Busch et al., "Inhibition of spontaneous emission in Fermi gases",
  Europhys. Lett. 44, 1 (1998).
  [doi:10.1209/epl/i1998-00426-2](https://doi.org/10.1209/epl/i1998-00426-2)

* O'Sullivan & Busch, "Spontaneous Emission in ultra-cold spin-polarised
  anisotropic Fermi Seas", Phys. Rev. A 79, 033602 (2009).
  [arXiv:0810.0231](https://arxiv.org/abs/0810.0231)

* O'Sullivan, "Spatial and energetic mode dynamics of cold atomic systems",
  PhD thesis (2012).
  [cora.ucc.ie](https://cora.ucc.ie/handle/10468/963)
