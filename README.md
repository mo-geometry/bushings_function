# Surfaces of revolution from Bushing's function

***Python scripts to animate a pair of 3D surfaces of revolution, and calculate their first and second derivatives.

<center>
<img src="https://user-images.githubusercontent.com/62537514/85952592-57d91700-b962-11ea-831d-745cc6264454.png" width="linewidth"/>
</center>


# About

Bushing's function is derived from the coherent
states of the anisotropic harmonic oscillator.

<center>
<img src="https://user-images.githubusercontent.com/62537514/85952459-5f4bf080-b961-11ea-83d7-054e922ebb40.png" width="linewidth"/>
</center>

γ(a, x) and Γ(a) are the lower incomplete gamma function, and gamma function respectively [(see wolfram)](https://mathworld.wolfram.com/IncompleteGammaFunction.html).

θ ranges from 0 to 2π in Bushing's function, and from 0 to π for the surface of revolution.

The functions α(θ) and β(θ) are given by either 
```
- cigar harmonic oscillator:
α(θ) = η^2sin^2(θ) and β(θ) = η^2cos^2(θ)

- pancake harmonic oscillator:
α(θ) = η^2cos^2(θ) and β(θ) = η^2sin^2(θ)
```
Note: The surface of revolution and its derivatives are wholly described by the parameters (η, λ, nF).

```
η (float): gaussian spread parameter
λ (int): anisotropy parameter
nF (int): occupancy parameter
```

For chosen values of (λ,nF) bushing's surfaces are animated for a range of η values.



<center>
<img src="https://user-images.githubusercontent.com/62537514/85954158-5e20c080-b96d-11ea-8c8f-725fbcb181c5.gif" width="linewidth"/>
</center>


### References

* Th. Busch, J. R. Anglin, J. I. Cirac and P. Zoller, ”Inhibition of spontaneous emission in Fermi gases”, 
Euro-phys. Letters 44, 1 (1998). [(doi)](https://doi.org/10.1209/epl/i1998-00426-2)

* B. O'Sullivan & Th. Busch, ”Spontaneous Emission in ultra-cold spin-polarised anisotropic Fermi Seas”, Phys. Rev. A 79, 033602 (2009). 
([arXiv:0810.0231](https://arxiv.org/abs/0810.0231) | [doi](https://doi.org/10.1103/PhysRevA.79.033602))

* Brian O'Sullivan, ”Spatial and energetic mode dynamics of cold atomic systems” PhD thesis (2012). ([cora](https://cora.ucc.ie/handle/10468/963))
  - see chapter 2.3 for a derivation of bushing's equation (2.112).
