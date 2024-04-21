### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ b199813e-ac0f-40a1-b823-dbc8d11399a7
begin
	println("This just calls the packages I am using")
	using Symbolics
	using PlutoUI
	using Distributions
	using Printf
	using Plots
	using Random
	using LinearAlgebra
	using BenchmarkTools
	using Graphs
	using GraphPlot
	using FileIO
	using JLD2
	using LaTeXStrings
	using StatsBase
	using EmpiricalDistributions
end

# ╔═╡ 08f9424c-87cc-442e-9b6e-5f1951597f50
md"""
# Introduction

This Julia notebook appompanies the manuscript: "Emergence of metastability in frustrated oscillatory networks: the key role of hierarchical modularity".

If opened from a `.html` file, this notebook may be edited and run online via binder, see top right corner. Otherwise, you should be able to see the previously ran output.

A few useful information to read the notebook:

- in this first section, we just call the necessary packages, using binder you should not have issues running the whole notebook.
- on the right of the screen a table of contents should appear, you can navigate the sections easily from there.
- most functions are defined at the end of the notebook, each function has various dependencies and may not work if outside the notebook. If you want to write your own simulations, it may be easier to copy paste all the functions in a new notebook or just modify this notebook.
- most analytical calculations are included in the appendix of the manuscript, here you wil find the implementation of the model and some useful examples.
- all figures in the manuscript can be reproduced from this notebook, however it may take a long time to average over many initial conditions. As a default, we include only single runs here.
- last but not least, **no need to know how to program to play with the models**, there are many interactive buttons that automatically re-run the experiments after you click submit.
"""

# ╔═╡ 2c3103e0-ad76-11ee-0f91-392fd41b20f4
PlutoUI.TableOfContents()

# ╔═╡ cb0f05ab-4b84-43e0-8e10-6486aa719645
md"""
Do you have Kuramodel installed locally? $(@bind kuramodel_intro_1Q confirm(CheckBox(default=false)))
"""

# ╔═╡ 5e06750a-c50d-4d16-897a-0aec36bb135a
begin
	using Pkg
	if kuramodel_intro_1Q
		Pkg.add(path="/Users/ec627/.julia/dev/Kuramodel")
		using Kuramodel
		greet()
		println("This loads the local Kuramodel Package and keeps it updated.")
	else
		# define required functions
		"""
		```jldoctest
			macro_op(θs::AbstractMatrix)
		```
		Function to calculate the instantaneous macroscopic parameter for each row
		of the instantaneous phases stored in θs.
		
			Input:
				θs::AbstractMatrix	(tot_num_timesteps × N) matrix, each row
									contains the instantaneous phases of all
									N oscillators.
			Output:
				R::AbstractArray	Instantaneous oreder parameter for each 
									timestep.
		
		"""
		function macro_op(θs::AbstractMatrix)
			# num of cols = N
			N = length(θs[1, : ])
		    map(θ -> abs(sum(exp.(im * θ) / N)), eachrow(θs))
		end

		"""
		    fun_pattern(A::AbstractMatrix, θs::AbstractArray, t_span)
		
		Function used to calculate the functional pattern (correlation matrix)
		of a system of oscillators.
		
		    Inputs:
		        A::AbstractMatrix   adjacency matrix of the underlying network
		        θs::AbstractArray   instantaneous phases at each timestep
		        t_span              time window of interest (can be a tuple or vector of order 2)
		
		    Output:
		        R::AbstractMatrix   functional pattern matrix for time window t_span
		"""
		function fun_pattern(A::AbstractMatrix, θs::AbstractArray, t_span)
		    
		    R = zeros(size(A))
		
		    for i in 1:length(A[1,:])
		        for j in 1:length(A[1,:])
		            local corr_sum = 0
		            for t in t_span[1]:t_span[2]
		                corr_sum += cos(θs[t,j] - θs[t,i])
		            end
		            R[i,j] = corr_sum / (t_span[2] - t_span[1]+1)
		        end
		    end
		
		    return R
		end

		println("This cell adds `macro_op` and `fun_pattern` functions from the Kuramodel package without having to add it via the `Pkg` manager.")
	end
end

# ╔═╡ b3fdf394-2da6-4fed-b787-db5e8460e436
md"""
# Methods

In this section we describe the main formulas used to reproduce the results. It is divided into three subsections (as you should be able to see in the Table of Contents on the right):
- **Network Model**, where we include some examples and some sanity checks;
- **Dynamical Model**, where we describe the dynamical equations used in this work;
- **Renormalization Approach**, this is mostly just a sum up of the work of Villegas and collaborators [[1]](https://doi.org/10.1103/PhysRevResearch.4.033196) and [[2]](https://doi.org/10.1038/s41567-022-01866-8) (if you click on the references you should be directed to the original papers).

## Network Model
The probability of connection between nodes in the same block at layer $l$ are given by (see manuscript Appendix A for full derivation):
```math
\begin{align}
p_1(H) &= 1 - \left(\frac{1 - H}{2}\right)\left(\frac{n_1 \gamma}{n_1 - 1}\right), \\
p_2(H) &= \left(\frac{1+H}{2}\right)\gamma,\\
p_3(H) &= \left(\frac{1-H}{2}\right)\gamma,
\end{align}
```

such that the average degree of the network is:

$\langle{k}\rangle = (n_1-1)p_1(H) + n_1(n_2 - 1)p_2(H) + n_2n_1(n_3 - 1)p_3(H)$

plugging in the expressions for $p_i(H)$ we obtain (using `Symbolics.jl`):
"""

# ╔═╡ e280f836-d7b4-4694-85ff-cc5369c6af85
let
	@variables H n₁ n₂ n₃ γ 

	p₁ = (n₁*H*γ) / (2*(n₁ - 1)) + 1 - (n₁*γ) / (2*(n₁ - 1))
	p₂ = ((1+H)/2) * γ
	p₃ = ((1-H)/2) * γ

	k = p₁ * (n₁ - 1) + p₂*(n₁*(n₂ - 1)) + p₃*n₁*n₂ |> simplify
	
	println("Computation check using Symbolics.jl")

	k
end

# ╔═╡ 81e9e58d-474c-4b3c-84e9-4e41586b6088
md"""
Below we check numerically that the model indeed generates hierarchically modular networks with the predicted analytical average degree, independently of the choice of control parameter $H$.
"""

# ╔═╡ 4f3e3b64-4568-405f-93d3-863309b7f4df
md"""
### Network Generator Example

We generate networks using the function

	SBMvar(n::AbstractArray, B::AbstractArray, H::Number, k::Number; K = 1)

Where the parameter $\gamma$ is calculated using the given average degree, then the constructor is called.

**NOTE**
You can use the live docs (bottom right of the screen) to see more information about this function, or take a look at the code in the Functions section.
"""

# ╔═╡ b37f1ede-c82c-448e-8e21-f9cdb62c1eb7
md"""
Select H: $(@bind H_methods_1_input confirm(NumberField(0.00:0.01:1, default=0.25)))
"""

# ╔═╡ 77440c0b-490f-4279-b55d-3030ffadfecd
md"""
Select $k$: $(@bind k_methods_1_input confirm(NumberField(16:1:127, default=25)))
"""

# ╔═╡ 090fa835-acd3-4532-b557-e527ea5e6de5
begin
	n_methods_1 = [16, 8, 2]
	B_methods_1 = [16, 2, 1]
	N_methods_1 = prod(n_methods_1) # number of nodes
	K_methods_1 = 1
	H_methods_1 = H_methods_1_input
	k_methods_1 = k_methods_1_input
	println("Use this cell to change other network parameters")
end

# ╔═╡ 5645229e-0cb7-4f8a-a40f-74884f642c2b
md"""
Save figure? $(@bind save_methods_1Q confirm(CheckBox(default=false)))
"""

# ╔═╡ a6e25db2-c9b5-4278-9f70-678098863bad
md"""
### Average Degree
"""

# ╔═╡ 67bd2260-c60d-438c-a7f5-3ac214ac792d
md"""
##### Allowed Average Degrees

Check the limitations of the average degrees allowed.

Code is hidden by default, you can open it and see the code. Basically, we just change the parameter $\gamma$ from $0$ to $1$ and see what average degrees we obtain.
"""

# ╔═╡ 128bd907-c1d7-4154-abb9-a090ab7712a9
md"""
the numerical maximum average tends to be slightly higher than the theoretical maximum degree, this is likely due to finite size effects. You can try a larger network, such as `n=[16,16,2]` and `B=[32,2,1]` to see that the relative discrepancy reduces.
"""

# ╔═╡ df93b0fa-2ac4-4196-8708-11aad511f743
md"""
#### Average degree for fixed $k$ and varying $H$:
"""

# ╔═╡ 1ae6c208-8512-45b4-abda-05ad8f3c174c
md"""
the important thing is that the average degree is always close to the theoretical average degree for varying $H$.
"""

# ╔═╡ 5586aa3b-9614-4401-b9db-d227c6d16108
md"""
## Dynamical Model
"""

# ╔═╡ fa47c80d-6b86-40ec-9780-760d47efaa76
md"""
### Kuramoto-Sakaguchi Dynamics

Let $P$ be the partition matrix of the variation of the SBM defined in the manuscript. You can create one using the function: `_partition_mat(n, B)`.

Consider an oscillator $i$. The interaction terms between oscillators $j$ within the same block at the first layer satisfy the following condition:

$P_{1,i} = P_{1,j}\quad\text{(a)}.$

Interactions (exclusively) at the second layer instead satisfy:

$P_{2,i} = P_{2,j},\,P_{1,i} \neq P_{1,j}\quad\text{(b)},$

while at the third layer we have the condition

$P_{2,i} \neq P_{2,j}\quad\text{(c)}.$

(**note** Pluto notebooks don't allow $\LaTeX$ equations on multiple lines):

Then the dynamical equations can be written as
```math
\begin{align}
\dot\theta_i &= \omega_i - K \Big[\\
& \;\;\;\;\sum_{j \text{ s.t. (a)}}\sin\big(\theta_j(t) - \theta_i(t)\big) \\ 
&+ \sum_{j\text{ s.t. (b)}}\sin\big(\theta_j(t) - \theta_i(t) - \alpha\big) \\
&+ \sum_{j\text{ s.t. (c)}}\sin\big(\theta_j(t) - \theta_i(t) - \alpha\big)\Big]
\end{align}
```

Such that only the interactions with oscillators in the same block in the first hierarchical layer are not phase lagged, all the other connections are.
"""

# ╔═╡ 67798bae-9e1b-4467-b858-c93e1e224976
md"""
In the functions section, we use the function `KS_sim()` to simulate the dynamics given a network $A$. Note, the function uses a simple Euler method with arbitrary stepsize.
"""

# ╔═╡ 8da0d12a-d942-4672-b677-339a0dcaf9f3
md"""
### Nested Attractors and Topological Scales

Ignore the phase lag for now. How does the Kuramoto dynamics behave on hierarchical networks constructed using our variation of the nSBM?

In the images below we can see the path towards synchronization, at different timescales we have different "relative" attractors.

To see this, try the following in order (by clicking on the submit button next to the selections):
- using $H=0.6$ and simulation time $= 0.03$
- using $H = 0.2$ and simulation time $= 0.1$.
- then keep simulation time $= 0.1$ and change $H$ from $0$ to $1$ to see the differences. Notice how beyond $H = 0.4$ the simulation time is not enough to reach the final attractor. 
"""

# ╔═╡ 465c625f-d6fa-4f1d-b81e-b2c6ccda65a9
md"""
Select $H$: $(@bind H_methods_2_input confirm(NumberField(0.00:0.01:1, default=0.6)))

Select simulation time: $(@bind sim_time_methods_2_input confirm(NumberField(0.01:0.01:10, default=0.03))) 
"""

# ╔═╡ eebc3fb4-346e-46b4-94f2-3094a2580756
md"""
We can see how for low simulation times and low "hierarchicalness", i.e., low parameter $H$, the system reached the global state attractor. For high $H$ instead, longer simulation times are required. Interestingly, at low simulation times the systems seems to reach another "attractor" (the attractor specific to the fast timescale dynamics).
"""

# ╔═╡ 727c2c12-5188-45eb-a191-6362ef9e7a99
md"""
Another way to see this could be to find the number of connected components of synchronized oscillatory units, like in ["Synchronization Reveals Topological Scales in Complex Networks"](https://arxiv.org/pdf/cond-mat/0511730.pdf)

The figure below should reproduce to some degree the results from [[3]](https://arxiv.org/pdf/cond-mat/0511730.pdf).

**NOTE** We obtain similar results *only* for the case of large systems. The paper does not specify how large the system size should be since it is intuitive to see that these effects only arise in large systems.

Change the setting below to see. Start with $0.1$ simulation time and then increase. You will need a simulation of at least 1 second in order to see the correct results.
The 1 second simulation may take a couple of minutes to compute if you do so (20 seconds the simulation and a minute or so to count the Connected components).
"""

# ╔═╡ b9916053-081c-48e8-88b8-b72ac6ac2fa0
md"""
Do you want to use a larger system? $(@bind large_sys_Q_M2 confirm(CheckBox(default=false)))
"""

# ╔═╡ d260a2f7-5769-4b52-a71f-0b819d918c5c
md"""
## Renormalization Approach

In the previous section we included the nested attractors and the topological scales studies as it can give us a bit of an intuition of what the Laplacian renormalization approach can show **for the case of scale dependent systems**, such as our variation of the nSBM.

The Laplacian Renormalization Group (LRG) flow, when applied to scale dependent networks such as this one, captures very well changes in the information diffusion pathways encoded in the density operator (defined below).

**NOTE** The following is entirely based on the Laplacian, i.e., linear diffusion, but the Kuramoto is nonlinear right? Indeed, in the Kuramoto model the interaction terms are sinusoidal and additive. However, if we linearize the sine interaction term we get the Laplacian. More precisely, we may assume $\theta_j - \theta_i \ll 1$ such that $\sin(\theta_j - \theta_i) \approx \theta_j - \theta_i$. Alternatively, at the synchronziation attractor point, $i.e.,$ $\theta^\star_i = \theta^\star_j = \theta^\star$ we can perturb any nodes $i$ and $j$ via $\epsilon_i$. Then, $\sin(\theta^\star_j + \epsilon_j - (\theta^\star_i + \epsilon_i ) ) = \sin(\epsilon_j - \epsilon_i) \approx \epsilon_j - \epsilon_i$.
"""

# ╔═╡ 4415af2d-be77-456b-8955-931977328a38
md"""
### LRG flow

I will not go too much into the detail, for more context on this see [[4]](https://arxiv.org/abs/2010.04014) or more in general the work done by CoMuNeLab, for instance [here](https://manliodedomenico.com/network_information_theory.php). Here I will sum up the important functions we can study.

The "ingredients" are:
- the **graph Laplacian** $L$, with $L_{ij} = \delta_{ij}\sum_k^N A_{ik} - A_{ij}$ with eigenvalues $\lambda_i$ and associated eigenvectors $\vec{V}_i.$ Also let the matrix of eigenvectors to be $V = [\vec{V}_1, \vec{V}_2, \dots]$;

- the **propagator**: $$\hat{K} = \exp(-\tau\hat{L})=\sum_{n}\frac{(\tau\hat{L})^n}{n!}$$ acting on the states $\vec{x}(\tau) = \hat{K} \vec{x}(0)$;

- the **density operator**: $\hat\rho(\tau) = \frac{\hat{K}}{\mathrm{tr}[\hat{K}]},$ with eigenvalues $\mu_i(\tau) = \frac{\exp(-\lambda_i\tau)}{\sum_{j}\exp(-\lambda_j\tau)}.$

- the **Shannon entropy**: $S[\hat\rho(\tau)] = \frac{1}{\log(N)}\sum_{i=1}^N\mu_{i}(\tau)\log(\mu_i(\tau))\in[0,1].$ This measure, in the words of [[1]](https://doi.org/10.1103/PhysRevResearch.4.033196): $\text{``}$*can be seen as a measure of the residual information still encoded in the evolved state* $x(τ)\text{"}.$

- finally, the **specific heat**: $C = -\frac{\mathrm{d}S[\hat\rho(\tau)]}{\mathrm{d}\log\tau}.$ Which detects the rapid changes in information diffusion.

Below is an example to reproduce the results from "*Laplacian renormalization group for heterogeneous networks*" fig. 5 Supplementary Information (`N = 256` case):
"""

# ╔═╡ 480c6796-7695-408a-aca4-cf30ded12f3d
# md"""
# **Note**: here I normalized the specific heat: $C \rightarrow \frac{C}{\mathrm{max}(C)}$, otherwise the maximum value is around $1.6.$ I guess they did as well in the paper since they compare $C$ in different larger networks.
# """

# ╔═╡ fef1faad-8ab6-436c-bc63-c699c10f85d7
md"""
To identify the peaks, we can find the minima and maxima of the specific heat as $\tau$ varies.
"""

# ╔═╡ 1421bb08-7a75-48a5-9ab4-39ca0f4b626a
md"""
### Construct the Information Core

As time increases, new diffusion pathways become more dominant in complex networks as the density operator changes. These changes in information diffusion dynamics are detected by the specific heat. To visualize these changes, and eventually coarse grain the system, we can define a "rule". In [[1]](https://doi.org/10.1103/PhysRevResearch.4.033196) and [[2]](https://doi.org/10.1038/s41567-022-01866-8) they use:

```math
\begin{align}
\rho'_{ij} &= \frac{\rho'_{ij}}{\mathrm{min}\left( \rho_{ii},\rho_{jj}\right)} \\
\zeta_{ij} &= \mathcal{H}(\rho'_{ij} - 1).
\end{align}
```

Notice how $\zeta_{ij}$ captures when the information shared between two nodes $\rho_{ij}$ becomes higher than the information $\rho_{ii}$ in node $i$ or $\rho_{jj}$ in node $j$, i.e., when $\frac{\rho'_{ij}}{\mathrm{min}\left( \rho_{ii},\rho_{jj}\right)} > 1$.

Practically (and informally), the new graph $\zeta_{ij}$ identifies the subsets of nodes that are sharing the same amount of information at that given timescale $\tau$.
"""

# ╔═╡ f927e709-e51e-4744-a93f-e46e26b5c2c1
let
	n = [32, 16, 2]
	B = [32, 2, 1]
	N = prod(n)

	global N_methods_3 = N
	global n_methods_3 = n
	global B_methods_3 = B

	println("Select here network size parameters")
	println("\nParameters used:")
	println("- N = $N")
	println("- n = $n")
	println("- B = $B")
end

# ╔═╡ 7663302e-5483-4828-a25e-6978eb95ef66
md"""
below we construct this new adjacency matrix $\zeta$ at a chosen final time (you can select this below). In this subsection we use the nSBM rather than the $1$-layer SBM.
"""

# ╔═╡ 8442be53-fe2b-4f1c-ba39-81b7f7cb761e
begin
	times_avail_methods_3 = vcat([0.000001, 0.00001, 0.0001, 0.001, 0.01], collect(0.01:0.01:10), [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000])

	k_min_methods_3 = n_methods_3[1] + 0 * (n_methods_3[1]*n_methods_3[2] - n_methods_3[1]) - 1
	k_max_methods_3 = n_methods_3[1] + 1 * (n_methods_3[1]*n_methods_3[2] - n_methods_3[1]) - 1
	
	md"""
	Select control parameter $H$: $(@bind H_methods_3 confirm(Select(collect(0.00:0.01:1.00), default = 0.4)))

	Select average degree $k$: $(@bind k_methods_3 confirm(Select(collect(k_min_methods_3:1:k_max_methods_3), default = k_min_methods_3 + round(Integer, k_min_methods_3 * 0.3))))

	Select final time: $(@bind sim_time_methods_3 confirm(Select(times_avail_methods_3, default = 0.000001)))

	Show density matrices? $(@bind show_ρ_plots_methods_3 confirm(CheckBox(default=false)))
	"""
end

# ╔═╡ b68e3644-bfc5-4d89-8854-4c8d896c3b31
md"""
First try using $H = 0.4$ and $k=40$ at times: $0.000001, 0.2, 1, 2,$ and $6$. Below the plot I also added the number of connected components in $\zeta$.

**NOTE:** it takes a few seconds to run.

- At first no connections in $\zeta$ appear, all information is still along the diagonal (each single node).
- Then at $t = 0.2$ you start to see that information has been diffused within the layer one blocks. As a results, the number of connected components in $\zeta$ reaches $32$, $i.e.,$ the number of first layer blocks.
- At sim time $t = 1$, due to randomly more dense connections between layer one blocks, the first information pathways emerge within second layer blocks.
- At about sim time $t = 2$ information has completely been diffused within the second layer blocks, but not between $2$nd layer blocks.
- Only after $6$ seconds we start to see information being diffuesed between layer $2$ blocks.
- beyond this point, more and more information pathways emerge.
"""

# ╔═╡ fd20fde7-3b04-44af-948e-732b20483a9f
md"""
While keeping the same simulation times and average degree, see how the second layer changes as we change the structural parameter $H$.
"""

# ╔═╡ 0783b6d1-825a-4312-a59e-44012d6a1d23
md"""
### Supernodes construction

Consider the Laplacian $L$ and its eigenvalues and eigenvectors using the braket notation:

$L = \sum_{i=1}^{N}\lambda_{i} |\lambda_{i}\rangle.$

Recall that we have some minima and maxima peaks in specific heat at some timescale $\tau$. Let the first maxima peak be at $\tau^\star$. Then, we integrate out the $n'$ faster modes where $1/\lambda < 1/\lambda^{\star} < \tau^\star$:

$L' = \sum_{i}^{N-n'} \lambda_{i}| \lambda_{i}\rangle.$

After doing this, we can rescale the network by the $\tau^\star$ in order to change the time unit:

$L'' = \tau^\star L'.$

Note, this is still a $N\times{N}$ matrix, to reduce it, we need to define a new orthogonal basis $\{|\alpha \rangle\}$ for our $n'$ modes. In this case, for most parameter values the charactersitic timescale of the first peak in specific heat corresponds to the second layer SBM. Then, in real space, we can obtain the renormalized matrix $A'$ (or the matrix of supernodes) using

$A'_{\alpha \beta} = - \langle\alpha| L'' |\beta\rangle = - L''_{\alpha \beta},$

and then set $A'_{\alpha \alpha} = 0, \forall \alpha.$

Below is an example:
"""

# ╔═╡ 7b30c5bd-34ff-469b-983c-07af53d1b371
md"""
In the cell below we compute the average coupling between nodes in the same population and nodes in distict populations.
"""

# ╔═╡ cd30d2fc-2801-4d32-b7c5-2871a0cd0a30
md"""
as we can see, similarly to the two population model [[5]](https://arxiv.org/pdf/0806.0594.pdf), we have higher coupling between nodes in the same population than nodes in distinct populations. Hence, it is possible to find a relation between the parameter $H$ and the "effective" control parameter $A$ used in [[5]](https://arxiv.org/pdf/0806.0594.pdf).
"""

# ╔═╡ 7dba8a1f-7aae-450f-8e81-8ad85c0d8c8b
md"""
### Connected Components Size

Here we show how the number of connected components of $\zeta$ as a function of time for varying $H$.

This subsection is not included in the manuscript, these are just some additional tests. Without the loaded results (email me if you need the data I have collected), the function below will take a while to compute (about 30 seconds on my machine). To collect more data change the `H_range` to `H_range = 0.0:0.1:0.6` and `t_range` to `t_range = 0.0:0.01:5`.

You should be able to see that as $H$ increases the $2$nd layer blocks take longer to start to share information and form a connected component in $\zeta$. Interestingly, it takes less for the information to be shared in first layer blocks as $H$ increases.
"""

# ╔═╡ f3e72899-b046-4d47-b9a3-c56db5174b66
colors_dict = Dict(
		.0 => :blue,
		.1 => :pink,
		.2 => :purple,
		.3 => :red,
		.4 => :orange,
		.5 => :magenta,
		.6 => :lightblue,
		1 => :green
	)

# ╔═╡ c15a7fe4-cf16-44e5-8d68-09c881c68a47
md"""
Load results? $(@bind results_load_methods_32 confirm(CheckBox(default=false)))
"""

# ╔═╡ 2fbe9f94-7eba-4f8f-8c3d-9107aca6fd03
md"""
# Main Results
"""

# ╔═╡ e690de5e-b60d-4e3a-9f31-e0f0daf2388d
md"""
## Analysis for Varying $H$
"""

# ╔═╡ 6f25633b-648a-40ff-9565-8be41f89f372
md"""
In this section, we see what happens as we use the dynamical system defined in the methods section on top of our variation of the nested SBM. First, we focus on the structural parameter $H$ as it is varied. Secondly, we look at what happens as the mean degree $k$ is changed.

We use a simple local order parameter measure to investigate the levels of synchronization at different layers. Consider a subset of nodes $s$, then the local order parameter is:

$Z_s = R_s e^{e\psi_{s}} = \frac{1}{|s|}\sum_{j\in s}\exp(i\theta_{j})$

In all cases explored so far we find the following dynamical regimes at the second layer:
- stable chimera for intermediate values of $H$;
- breathing chimera (fast and slow) for slighly higher values of $H$;
- metastability, unstable chimeras, and alternating chimeras as $H$ increases further.

For most values of $H$, we have highly metastable layer 1 modules when $k$ is high enough. For lower values of $k$ instead, the layer $1$ blocks display almost perfect synchronization.
"""

# ╔═╡ 0d4fb75f-ea5b-4aec-bf1a-7d7bfa122781
md"""
Some recommended parameters to explore the system:

- control parameter $H \in [0.2, 0.6]$;
- coupling $K = 50$ (or $1$ if not normalized, should be just time rescaling);
- system's size: `n = [16, 8, 2]`, `B = [16, 2, 1]` (should be quite fast);
- lag parameter $\beta \in [0.05, 0.2]$;
- average degree between $1/3$ and $1/20$ of the size of of the network.
"""

# ╔═╡ ad443a3d-dfd3-44e4-a912-f1c142d5c247
md"""
Select here simulation parameters:

Select $H$ $\quad$ $(@bind H_KS_1 confirm(NumberField(0.00:0.01:1, default=0.30)))

Select $K$ $\quad$ $(@bind K_KS_1 confirm(NumberField(1:0.01:10000, default=1)))

Select $\beta$ $\quad$ $(@bind β_KS_1 confirm(NumberField(0.0:0.01:1, default=0.1)))

Select simulation time $\quad$ $(@bind sim_time_KS_1 confirm(NumberField(10:1:1000, default=10)))

Select seed value $\quad$ $(@bind seedval_KS_1 confirm(NumberField(1:1:1000, default=10)))

Identical oscillators? $(@bind identical_KS_1Q confirm(CheckBox(default=true)))
"""

# ╔═╡ 1db6074c-7a29-4ee4-8d5b-7a59f5b71bbe
if !identical_KS_1Q

	md"""
	Select Normal distr $\sigma$ for the natural frequencies ($\mu = 1$) $\quad$ $(@bind nat_f_σ_KS_1 confirm(Select([0.001, 0.01, 0.1, 1, 2,3,4,5,6,7,8,9, 10, 100], default=0.01)))
	"""
end

# ╔═╡ 8f38c6cc-16e7-4871-819e-67140d7cb4a1
let
	md"""
	If you need to change multiple parameters, set this to false: 
	$(@bind run_sim_KS_1Q confirm(Select([false, true], default=true)))

	Then the simulation will start only when it is set back to true
	"""
end

# ╔═╡ 7228c6c5-3209-43b5-820b-aef8337c7ec1
let
	n = [16, 8, 2]
	B = [16, 2, 1]
	# k = prod(n) / 10
	k = 51
	global Δt_KS_1 = 1e-3
	global n_KS_1 = n
	global B_KS_1 = B
	global k_KS_1 = k

	println("Change here network size parameters\n")
	println("Network size: $(prod(n))")
	println("n = $(n)")
	println("B = $(B)")
	println("Average degree : $(k)")
end

# ╔═╡ ec62a9ea-d2d7-4132-9767-0da16460c969
md"""
Normalize the coupling i.e., $K\rightarrow K/k$ where $k$ is the mean degree?

$(@bind normalize_KS_1Q confirm(Select([false, true], default=false)))
"""

# ╔═╡ 018add25-80fb-492b-9402-95e7f5a8b7a1
md"""
Only figure second layer KOPs? $(@bind save_results_1 confirm(CheckBox(default=false)))
"""

# ╔═╡ 2ba0a96e-a38f-4c91-a0e2-d8917c7babe7
md"""
Below the results of the simulation (using default values it should take about 10 seconds). We differentiate between modules belonging to populations $1$ and $2$.
"""

# ╔═╡ 919c8b41-1cf8-426a-894e-c1e6b49739ae
# savefig(plot_to_save_results_1, "/Users/ec627/Documents/Sussex/papers/CompetitionAcrossTimescales/PostProcessing/main/subplots/populations_example_H_0.4.png")

# ╔═╡ e7370d3e-1086-4e4c-9ecc-86cd001f75b7
md"""
Below you can check the coalition entropy of the modules: either for a specific population or for the entire first layer.
"""

# ╔═╡ c185b2fd-1283-443d-b572-043b8426f0f5
md"""
Compute the populations' coalition entropy? $(@bind CE_KS_1_Q confirm(CheckBox(default=false)))
"""

# ╔═╡ 3f422269-c4c9-40b0-96b3-8eeb60d12ce7
md"""
Compute the coalition entropy of the whole first layer? $(@bind CE_KS_1_Q2 confirm(CheckBox(default=false)))
"""

# ╔═╡ 0cb9f939-8e26-4cd3-b6b9-84b260ee6db6
# let
# 	#####################################
# 	# here we just show the eigenvalues #
# 	#####################################
	
# 	Random.seed!(1)
	
# 	# get parameters from previous simulation
# 	n=n_KS_1; B=B_KS_1
# 	# n = [32, 16, 2]
# 	# B = [32, 2, 1]
# 	H = 0.4
# 	N=prod(n); k=k_KS_1; K=K_KS_1; β=β_KS_1

# 	# recreate graphs again
# 	A, P, p = SBMvar(n, B, H, k; K = K)
# 	steps = 0+1e-3:1e-3:sim_time_KS_1

# 	L = Laplacian(A)
# 	λs, vecs = eigen(L)
# 	evec = 2
# 	heatmap(
# 		vecs[:, evec] * vecs[:, evec]',
# 		title = "eigenvector visualization",
# 		yflip = true
# 	)
# end

# ╔═╡ 546f983a-2db6-4c1d-be78-259fa24a0c10
# md"""
# Save the above figure? $(@bind save_results_1Q confirm(CheckBox(default=false)))
# """

# ╔═╡ 96a53aff-0df0-4ba1-a85e-656fbc66caa3
# if save_results_1Q
# 	savefig(plot_to_save_results_1, path_results_1 * filename_to_save_results_1 * ".png")
# else
# 	println(path_results_1 * filename_to_save_results_1 * ".png")
# end

# ╔═╡ 3e4fac71-69c4-49dd-856c-dce0c3057f1c
# savefig(modules_plt_KS_1, "/Users/ec627/Documents/Sussex/CompleNet/presentation/images/" * "intro_image.png")

# ╔═╡ 2726622b-147c-4d5d-9955-055b7244b67c
md"""
Save the above figure? $(@bind save_results_2Q confirm(CheckBox(default=false)))
"""

# ╔═╡ 21c40ce1-745e-4257-b775-a151feeea431
md"""
Compute the connected components? $(@bind CC_KS_1_Q confirm(CheckBox(default=false)))
"""

# ╔═╡ ffe085ee-801f-439a-b61a-1d4fc5818cf4
md"""
For reference, this is the adjacency matrix of the system analysed above and the natural frequencies
"""

# ╔═╡ 661a5343-608d-4067-bbbd-b29b213c2688
md"""
## Analysis for Varying $k$

In this section the data was collected using a high performance computer cluster. Hence, the results peresented here are the averaged datat over multiple initial conditions.

The code is available on the dedicated Github page.
"""

# ╔═╡ 42848af2-779f-4894-9972-fc91170c2a07
md"""
Below, we report the degree of metastability of all modules in the first layer for varying $H$ and $k$.
"""

# ╔═╡ 824dd135-5340-4ee5-8926-619be27fe8bd
let
	folderpath = "/Users/ec627/Documents/Data/HierarchicalChimera/DataCollect/PhaseSpace/"

	cd(folderpath)
	subfolders = readdir()[2:end]
	subfolders = map(x -> folderpath * x, subfolders)
	
	# Define a regular expression pattern to match the numbers
	# example: seed_2_beta_0.1_H_0.0_k_24.jld2
	pattern = r"seed_(\d+)_beta_([\d.]+)_H_([\d.]+)_k_([\d.]+)\.jld2"

	# storage
	k_range = collect(20:2:126)
    H_range = collect(0.0:0.025:1)
	seed_range = collect(2:1:9)

	modules_metastability = zeros(length(H_range), length(k_range), length(seed_range))

	for subfolder in subfolders
		cd(subfolder)
		filenames = readdir()[1:end]

		for filename in filenames

			# Match the pattern in the filename
		    match_result = match(pattern, filename)
		
		    # Extract values if there is a match
		    if match_result !== nothing
		
		        # Extract values from the matched groups
		        seed = parse(Int, match_result[1])
		        β = parse(Float64, match_result[2])
		        H = parse(Float64, match_result[3])
		        k = parse(Float64, match_result[4])

				# checks[round(Integer, H/0.025 + 1), round(Integer, k/2 - 9), seed - 1] = 1

				results = load_object(filename)
				modules_metastability[round(Integer, H/0.025 + 1), round(Integer, k/2 - 9), seed - 1] = mean(results[8])
			end

		end
	end

	yticks = ((round.(Integer, k_range ./ 2) .- 9)[1:5:end], string.(k_range)[1:5:end])
	xticks = ((H_range ./ 0.025 .+ 1)[1:5:end], (string.(H_range))[1:5:end])

	heatmap(
		mean(modules_metastability, dims = 3)[:,:,1]',
		xlabel = L"H", ylabel = L"k", zlabel = L"\sigma_{\mathrm{met}}(R_{\mu_i})",
		xticks = xticks, yticks = yticks,
		# left_margin = 4Plots.mm,
		right_margin = 8Plots.mm,
		# bottom_margin = 4Plots.mm,
		# top_margin = 4Plots.mm,
		tickfontsize = 16,
		legendfontsize = 16,
		xlabelfontsize = 16,
		ylabelfontsize = 16,
		size = (700, 500)
	) # L"\sigma_{\mathrm{met}}"
end

# ╔═╡ 65261365-bbc7-4103-bc4d-12a6198e0792
md"""
we can see how for low $k$ the degree of metastability is zero for any $H$. As $k$ increases metastability increases rapidly before reaching a plateau. As $k$ increases even further metastability decreases very slowly.

We will look at this more in details in the next section.
"""

# ╔═╡ 4fe35c9d-b971-4c53-9e9b-04c64bae520d
md"""
## Laplacian Harmonics
"""

# ╔═╡ f56b7c7f-1d2a-4461-b10c-21a7a453264f
md"""
### Laplacian Spectrum study for varying $H$ and $k$
"""

# ╔═╡ ccbad8b3-4bd3-4164-97d3-435d62b96c52
let
	println("This cell is used to plot data from local file")
	
	plt = plot(
		xlabel = L"\lambda_i^{-1}",
		ylabel = L"\mathrm{mode}\;i",
		tickfontsize = 16,
		legendfontsize = 16,
		xlabelfontsize = 16,
		ylabelfontsize = 16,
		left_margin = 5Plots.mm,
		right_margin = 5Plots.mm,
		bottom_margin = 5Plots.mm,
		top_margin = 5Plots.mm,
		xticks = ([1e-2, 1e-1, 1e0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
		yticks = ([1e0, 1e1, 1e2], [L"10^{0}", L"10^{1}", L"10^{2}"])
	)
	
	folderpath = "/Users/ec627/Documents/Data/HierarchicalChimera/DataCollect/n1608fast/plots/"
	colors = [:red, :orange, :cyan, :green, :blue]

	for k in [16, 32, 64]
	
		filename = folderpath * "LaplacianPlot_k_" * string(k) * ".jld2"
		λs = load_object(filename)
		
		
		for i in 1:5
			if k == 32
				plot!(
					plt,
					1 ./ λs[i, :],
					collect(1:lastindex(λs[i, :])),
					xaxis = :log10,
					yaxis = :log10,
					label = "", # "H = $H, k = $(round(Integer, k))",
					marker = true,
					markersize = 2,
					ylims = (1, 600),
					c = colors[i]
				)
			else
				plot!(
					plt,
					1 ./ λs[i, :],
					collect(1:lastindex(λs[i, :])),
					xaxis = :log10,
					yaxis = :log10,
					label = "", # "H = $H, k = $(round(Integer, k))",
					marker = true,
					markersize = 2,
					ylims = (1, 600),
				)
			end
		end

		if k == 32
			offest = 0.004
		elseif k == 16 
			offest = 0.02
		else
			offest = -0.001
		end
		
		annotate!(
			1 ./ λs[1, end] + offest,
			350,
			(L"k = %$(k)", 16)
		)
	end
		
	plot(plt)

	# filename = folderpath * "LaplacianPLot.png"
	# savefig(plt, filename)
end

# ╔═╡ 3206a25e-d386-47a3-9792-c9710405afb7
let
	println("This cell plots the inset from local data")
	folderpath = "/Users/ec627/Documents/Data/HierarchicalChimera/DataCollect/n1608fast/plots/"

	k = 32
	filename = folderpath * "LaplacianPlot_k_" * string(k) * ".jld2"
	λs = load_object(filename)
	H_range = [0.3, 0.35, 0.4, 0.45, 0.5]
	colors = [:red, :orange, :cyan, :green, :blue]

	plt = plot(
		tickfontsize = 24,
		legendfontsize = 24,
		xlabelfontsize = 24,
		ylabelfontsize = 24,
		xticks = [],
		yticks = []
	)
	
	for (i, H) in enumerate(H_range)
		plot!(plt,
			1 ./ λs[i, 1:3],
			collect(1:lastindex(λs[i, 1:3])),
			xaxis = :log10,
			yaxis = :log10,
			label = L"H = %$H", # "H = $H, k = $(round(Integer, k))",
			marker = true,
			markersize = 9,
			# ylims = (1, 600),
			c = colors[i],
			lw = 5
		)
	end

	plot(plt)
	plot_filename = folderpath * "LaplacianPLot_inset.png"
	# savefig(plt, plot_filename)
end

# ╔═╡ 7e010234-48f1-4eb4-b9ac-1b6f49a6b392
md"""
### Size of the $2^{\text{nd}}$ spectral gap
"""

# ╔═╡ 53a88fb8-6247-49e2-99c6-4d825624341e
md"""
Select here simulation parameters:

Select $n_1$: $(@bind n1_results_LRG_1 confirm(NumberField(1:1:64, default=16)))

Select $n_2$: $(@bind n2_results_LRG_1 confirm(NumberField(1:1:32, default=8)))

Select $H$ $\quad$ $(@bind H_results_LRG_1 confirm(NumberField(0.00:0.01:1, default=0.30)))

Select $K$ $\quad$ $(@bind K_results_LRG_1 confirm(NumberField(1:0.01:10000, default=1)))

Select $\beta$ $\quad$ $(@bind β_results_LRG_1 confirm(NumberField(0.0:0.01:1, default=0.1)))

Select simulation timestep $\Delta{t}$ $\quad$ $(@bind Δt_results_LRG_1 confirm(Select([0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01], default=0.001)))

Select simulation time $\quad$ $(@bind sim_time_results_LRG_1 confirm(NumberField(1:1:100, default=5)))
"""

# ╔═╡ 5c0135a5-d378-4b7e-af40-886e03665e9c
let
	n_results_LRG_1 = [n1_results_LRG_1, n2_results_LRG_1, 2]
	
	k_min_results_LRG_1 = n_results_LRG_1[1] + 0 * (n_results_LRG_1[1]*n_results_LRG_1[2] - n_results_LRG_1[1]) - 1
	k_max_results_LRG_1 = n_results_LRG_1[1] + 1 * (n_results_LRG_1[1]*n_results_LRG_1[2] - n_results_LRG_1[1]) - 1

	k_possible_results_LRG_1 = collect(k_min_results_LRG_1:1:k_max_results_LRG_1)

	md"""Select average degree $k$: $(@bind k_results_LRG_1 confirm(Select(k_possible_results_LRG_1, default = k_possible_results_LRG_1[round(Integer, end*0.1)])))"""
	
end

# ╔═╡ 58e5226d-39ed-45f3-8b8f-865498e2a49e
md"""
To reproduce the SBM results from "Laplacian paths in complex networks: Information core emerges from entropic transitions" use the following:

- network size: $n_1 = 64, n_2 = 2, k < 70$
- network parameters: $H = 0.0, K = 1.0$
"""

# ╔═╡ 8fa9443b-acde-41de-9ba2-724d624c32e2
md"""
# Discussion

There is no order yet in this section.
"""

# ╔═╡ 0e454201-0cc7-43ee-bfcc-d17cdca79dfa
md"""
## Initial transient and High Correlation


"""

# ╔═╡ a4de6034-00e9-48be-b984-068275677571
md"""
# Functions
"""

# ╔═╡ 3eddd420-edef-437d-9d2c-adac079e4ee9
md"""
### Network Construction
"""

# ╔═╡ ac1ff98c-ae22-4386-a998-0717366c7c79
"""
	function _partition_mat(n, B)

	Inputs:
	
		n::AbstractArray 		vector of length 3, number of l-1 blocks in each l block
		B::AbstractArray 		number of blocks in layer l

	Output:

		P::AbstractMatrix 		partition matrix

"""
function _partition_mat(n, B)

	if B[1] != B[2]*n[2]
		error("B₁ != B₂ × n₂, check model conditions")
	end
	if B[2] != B[3]*n[3]
		error("B₂ != B₃ × n₃, check model conditions")
	end

	N = prod(n)
	L = length(n)
	P = zeros(L - 1, N)
	
	for i in 1:(L-1)
		P[i, :] = vcat(
			[repeat([j], prod(n[1:i])) for j in 1:B[i]]...
		)
	end

	return P
	
end

# ╔═╡ d7c56086-4956-43c5-a9e4-f512bca1c440
"""
	function _SBMvar_constructor(n::AbstractArray, B::AbstractArray, H::Number, γ::Number; K = 1)

Function to crate the nested SBM variation with 3 hierarchical levels.

Inputs:

	n::AbstractArray 		vector of length 3, number of l-1 blocks in each l block
	B::AbstractArray 		number of blocks in layer l
	H::Number 				value of hierarchical parameter
	γ::Number 				parameter to control the average degree, γ ∈ [0,1]
	K::Number 				set the coupling (weighted matrix output)

Outputs:

	A::AbstractMatrix 		 		adjacency matrix
	P::AbstractMatrix 				partition matrix
	[p₁, p₂, p₃]::AbstractArray 	probabilities of connection at layer l


"""
function _SBMvar_constructor(n::AbstractArray, B::AbstractArray, H::Number, γ::Number; K = 1)

	if length(n) != length(B)
		error("vectors n and B mismatch")
	end
	if γ < 0 || γ > 1
		error("require 0 ≤ γ ≤ 1")
	end
	if B[1] != B[2]*n[2]
		error("B₁ != B₂ × n₂, check model conditions")
	end
	if B[2] != B[3]*n[3]
		error("B₂ != B₃ × n₃, check model conditions")
	end

	N = prod(n) # number of nodes
	L = length(n) # number of layers
	
	# store partition and adjaecncy matrix
	P = _partition_mat(n, B)
	A = zeros(N, N)

	# compute edge probabilities
	p₁ = (n[1]*H*γ) / (2*(n[1]-1)) + (1 - (n[1]*1*γ) / (2*(n[1]-1)))
	p₂ = (1/2 + H/2) * γ
	p₃ = (1/2 - H/2) * γ

	for i in 1:N
		for j in i+1:N

			# connect if in the same l = 1 block only
			if P[:, i] == P[:, j] && rand() < p₁
				A[i, j] = K
				A[j, i] = K
				
			# connect if in the same l = 2 block only
			elseif P[2, i] == P[2, j] && rand() < p₂
				A[i, j] = K
				A[j, i] = K

			# connect if in the same l = 3 block only
			elseif P[2, i] != P[2, j] && rand() < p₃
				A[i, j] = K
				A[j, i] = K
				
			end
		end
	end

	return A, P, [p₁, p₂, p₃]
end

# ╔═╡ db4be42d-0cb6-4737-bd4f-e38f51f76e5a
let
	# Random.seed!(123)
	n = [16, 8, 2]
	B = [16, 2, 1]
	N = prod(n) # number of nodes
	K = 1
	H = 0.3

	γ_range = 0.0:0.1:1
	store_k = zeros(length(γ_range))
	
	for (i, γ) in enumerate(γ_range)
		
		A, P, p = _SBMvar_constructor(n, B, H, γ; K = 1)
		store_k[i] = mean(sum(eachcol(A)))
		
	end

	println("Number of nodes: $N")
	println("n = $n")
	println("B = $(B)\n")
	println("Theoretical minimum: $(n[1] + 0 * (n[1]*n[2] - n[1]) - 1)\nTheoretical maximum: $(n[1] + 1 * (n[1]*n[2] - n[1]) - 1)\n")

	println("Numerical minimum: $(minimum(store_k))\nNumerical maximum: $(maximum(store_k))")
	
	# A, P, p = SBMvar(n, B, H, k; K = K)
end

# ╔═╡ b7abb166-2daf-4594-9d14-068f6e0051f2
"""
	function _SBMvar_constructor(n::AbstractArray, B::AbstractArray, H::Number, γ::Number; K = 1)

Function to crate the nested SBM variation with 3 hierarchical levels.

Inputs:

	n::AbstractArray 		vector of length 3, number of l-1 blocks in each l block
	B::AbstractArray 		number of blocks in layer l
	H::Number 				value of hierarchical parameter
	k::Number 				desired average degree, (n₁ - 1) ≤ k ≤ (n₁n₂ - 1)
	K::Number 				set the coupling (weighted matrix output)

Outputs:

	A::AbstractMatrix 		 		adjacency matrix
	P::AbstractMatrix 				partition matrix
	[p₁, p₂, p₃]::AbstractArray 	probabilities of connection at layer l


"""
function SBMvar(n::AbstractArray, B::AbstractArray, H::Number, k::Number; K = 1)

	γ = (k + 1 - n[1]) / (n[1]*n[2] - n[1])

	return _SBMvar_constructor(n, B, H, γ; K = K)
	
end

# ╔═╡ d6902f4c-a8f1-4e13-a048-53fdf5f5a4b1
let
	Random.seed!(123)
	
	n = n_methods_1
	B = B_methods_1
	N = N_methods_1 # number of nodes
	K = K_methods_1
	H = H_methods_1
	k = k_methods_1
	γ = (k + 1 - n[1]) / (n[1]*n[2] - n[1])
	
	A, P, p = SBMvar(n, B, H, k; K = K_methods_1)
	p₁ = p[1]
	p₂ = p[2]
	p₃ = p[3]
	
	k_analytical = (n[1] - 1)*p₁ + (n[2] - 1)*n[1]*p₂ + (N - n[1]*n[2])*p₃

	println(
"Average k (computation): $(k),
Average k (analytical): $(n[1]-1-n[1]*γ+γ*n[1]*n[2]),
or from the full equation (k_analytical): $(k_analytical)"
	)
	println("Numerical average deregree: $(mean(sum(eachcol(A))))")
	println()
	println("Other network parameters:")
	println("Number of nodes: $(N)")
	println("coupling K = $K")
	println("γ = $γ")
	println("p₁ = $(p₁)")
	println("p₂ = $(p₂)")
	println("p₃ = $(p₃)")
	println("n = $(n)")
	println("B = $(B)")
	
	plt = heatmap(A,
		yflip = true, size = (256, 256),
		cbar = false, aspect_ratio = :equal,
		framestyle = :box, xticks = [], yticks = [],
		c = cgrad([:white, :black])
	)

	if save_methods_1Q
		filepath = "/Users/ec627/Documents/Sussex/CompleNet/presentation/images/matrices/"
		# "/Users/ec627/Documents/Sussex/papers/CompetitionAcrossTimescales/RawFigures/AdjacencyMatrices/"
		
		filename = "_n_" * string(n[1]) * "_" * string(n[2]) * "_" * string(n[3]) * "_H_" * string(H) * "_k_" * string(k)[1:2]
		savefig(plt, filepath * filename * ".png")
	end

	plot(plt)
end

# ╔═╡ 1e8113b3-6f7b-4113-b749-c4888e0d1d36
let
	# Random.seed!(123)
	
	n = [16, 8, 2]
	B = [16, 2, 1]
	N = prod(n) # number of nodes
	K = 1
	k = 25

	H_range = 0.0:0.1:1
	no_tests = 10
	store_k = zeros(length(H_range), no_tests)
	
	for (i, H) in enumerate(H_range)
		
		for j in 1:no_tests
			
			A, P, p = SBMvar(n, B, H, k; K = K)
			store_k[i, j] = mean(sum(eachrow(A)))
		end
		
	end

	println("Number of nodes: $N")
	println("n = $n")
	println("B = $(B)")
	println("k = $(k)")
	
	plot(
		mean.(eachrow(store_k)),
		yerr = std.(eachrow(store_k)),
		label = "",
		seriestype = :scatter,
		c = :magenta,
		ylims = (k-5, k+5),
		ylabel = "Numerical average k",
		xlabel = "H"
	)
	plot!(
		[0, 11],
		[k, k],
		label = "Theoretical average k",
		size = (700, 250)
	)
end

# ╔═╡ c2f7c2ce-584e-42f0-ada7-b07c51a08229
let
	save_results_2 = false
	seedval = 2
	Random.seed!(seedval)
	
	# hSBM parameters
	H = H_methods_2_input
	if large_sys_Q_M2
		n = [32, 16, 2]
		B = [32, 2, 1]
	else
		n = [16, 8, 2]
		B = [16, 2, 1]
	end
	N = prod(n)
	if large_sys_Q_M2
		k = N / 20
	else
		k = N / 5
	end
	K = 3.0

	# build network model
	A, P, p = SBMvar(n, B, H, k; K = K)
	# A = A ./ k # this line normalizes K or not

	# simulation parameters
	ω = repeat([0], N)
	sim_time = sim_time_methods_2_input
	Δt = 1e-3
	save_ratio = 1
	
	# simulation parameters
	steps = (0.0+Δt):Δt:sim_time
	no_steps = length(steps)
	
	# storing parameters
	no_saves = round(Integer, no_steps / save_ratio)
	# θ_now = θ₀ = sort(rand(Uniform(0 + 0.2, 2*π - 0.2), N))  # random init conditions
	θ_now = θ₀ = collect(0+6/N:6/N:6) .+ 0.1
	θs = zeros(no_saves, N)
	θs[1, :] = θ_now
	
	save_counter = 1
	
	for t in 2:no_steps
	    
	    # update phases
	    θj_θi_mat = (repeat(θ_now',N) - repeat(θ_now',N)')
	    setindex!.(Ref(θj_θi_mat), 0.0, 1:N, 1:N) # set diagonal elements to zero 
	
	    k1 = map(sum, eachrow(A .* sin.(θj_θi_mat)))
	    θ_now += Δt .* (ω + k1)
	    save_counter += 1
	
	    # save θ
	    if save_counter % save_ratio == 0
	        θs[round(Integer, save_counter / save_ratio), :] = θ_now
	    end
	    
	end
	
	# layer 1 data (modules)
	modules_KOP = zeros(B[1])
	modules_KOP_std = zeros(B[1])
	modules_macro_op = zeros(length(θs[:, 1]), B[1])
	
	for i in 1:B[1]
		modules_macro_op[:, i] = macro_op(θs[:, n[1]*(i-1)+1:n[1]*i])
		modules_KOP[i] = mean(modules_macro_op[:, i])
		modules_KOP_std[i] = std(modules_macro_op[:, i])
	end

	# layer 2 data (populations)
	# relax = 500
	# mean_pop1 = round(mean(macro_op(θs[relax+1:end, 1:Integer(N/2)])), digits = 3)
	# std_pop1 = round(std(macro_op(θs[relax+1:end, 1:Integer(N/2)])), digits = 3)
	# mean_pop2 = round(mean(macro_op(θs[relax+1:end, Integer(N/2):N])), digits = 3)
	# std_pop2 = round(std(macro_op(θs[relax+1:end, Integer(N/2):N])), digits = 3)
	
	# plots
	global θs_KS_2 = θs
	global mat_plot_KS_2 = heatmap(A, c = cgrad([:white, :black]), yflip = true, size = (300, 300), frame = :box, cbar = false, xticks = [], yticks = [])

	plot(
		steps,
		θs_KS_2,
		xaxis = :log10,
		label = "",
		size = (900, 450),
		xticks = [],
		yticks = ([0, π, π*2], [L"""0""", L"""π""", L"""2π"""]),
		tickfontsize = 24,
		legendfontsize = 24,
		xlabelfontsize = 24,
		ylabelfontsize = 24,
		left_margin = 8Plots.mm,
		right_margin = 8Plots.mm,
		bottom_margin = 8Plots.mm,
		top_margin = 8Plots.mm,
		frame = :box
	)
end

# ╔═╡ f5f239d4-6537-4d77-a6df-a0dbc9afcdbb
let
	Random.seed!(123)
	
	n = [8, 4, 2]
	B = [8, 2, 1]
	N = prod(n) # number of nodes
	K = 1
	H = 0.4
	k = N / 3
	
	A, P, p = SBMvar(n, B, H, k; K = 1)

	α = π/2 - 0.05
	α_mat = zeros(N, N) # phase lag matrix
	for i in 1:N
	    for j in i+1:N
			# add lag if different partition at layer 1
	        if A[i,j] != 0 && P[1, i] != P[1, j]
	            α_mat[i, j] = α
	            α_mat[j, i] = α
	        end
	    end
	end
	
	
	# simulation parameters
	Δt = 0.001
	sim_time = 2
	save_ratio = 1
	steps = (0.0+Δt):Δt:sim_time
	no_steps = length(steps)
	# ω = rand(Uniform(0, 2), N)
	ω = repeat([1], N)
	noise_scale = 0.
	
	# storing parameters
	no_saves = round(Integer, no_steps / save_ratio)
	θ_now = rand(Uniform(-π, π), N)  # random init conditions
	θs = zeros(no_saves, N)
	θs[1, :] = θ_now
	
	save_counter = 1
	
	for t in 2:no_steps
	    
	    # update phases
	    θj_θi_mat = (repeat(θ_now',N) - repeat(θ_now',N)') - α_mat
	    setindex!.(Ref(θj_θi_mat), 0.0, 1:N, 1:N) # set diagonal elements to zero 
	
	    k1 = map(sum, eachrow(A .* sin.(θj_θi_mat)))
	    θ_now += Δt .* (ω + k1) + noise_scale*(rand(Normal(0,1),N))*sqrt(Δt)
	    save_counter += 1
	
	    # save θ
	    if save_counter % save_ratio == 0
	        θs[round(Integer, save_counter / save_ratio), :] = θ_now
	    end
	    
	end

	global n_discussion_1 = n
	global B_discussion_1 = B
	global N_discussion_1 = N
	global K_discussion_1 = K
	global H_discussion_1 = H
	global k_discussion_1 = k
	global θs_discussion_1 = θs
	global steps_discussion_1 = steps
	
	println("This cell contains the simulation for the plots below")

	heatmap(
		A, yflip = true, c = cgrad([:white, :black]),
		cbar = false, xticks = [], yticks = [], size = (300, 300)
	)
end

# ╔═╡ c4d4a990-5924-489f-bf5a-4e96e2c6aff8
md"""
Plot 2 random subsets of nodes KOP? $(@bind rand_discussion_1 confirm(CheckBox(default=false)))

Select time-windown:

Start: $(@bind t_span_start_discussion_1 confirm(NumberField(1:1:size(θs_discussion_1, 1), default=1)))

End: $(@bind t_span_end_discussion_1 confirm(NumberField(1:1:size(θs_discussion_1, 1), default=size(θs_discussion_1, 1))))
"""

# ╔═╡ 3d894d8a-f067-4318-a082-0624ee51d5ba
let
	n = n_discussion_1
	B = B_discussion_1
	N = N_discussion_1
	K = K_discussion_1
	H = H_discussion_1
	k = k_discussion_1
	θs = θs_discussion_1
	t_span = (t_span_start_discussion_1, t_span_end_discussion_1)
	steps = steps_discussion_1[t_span[1]:t_span[2]]
	
	# plots
	if rand_discussion_1
		
		all = collect(1:N)
		a = zeros(Integer(N/2))
		b = []
		
		for i in 1:round(Integer, N/2)
			new = rand(all)
			while new ∈ a
				new = rand(all)
			end
			a[i] = new
		end
		for new in 1:N
			if !(new ∈ a)
				append!(b, new)
			end
		end
	
		to_check_pop_1 = zeros(size(θs)[1], Integer(N/2))
		to_check_pop_2 = zeros(size(θs)[1], Integer(N/2))
		
		for (i, id) in enumerate(a)
			to_check_pop_1[:, i] = θs[:, round(Integer, id)]
		end
		for (i, id) in enumerate(b)
			to_check_pop_2[:, i] = θs[:, round(Integer, id)]
		end

		plot(
			steps, macro_op(θs[t_span[1]:t_span[2], :]),
			linestyle = :dash, c = :black, size = (900, 450), label = "", ylims = (0,1.1), grid=false, lw = 2
			)
		plot!(
			steps, macro_op(to_check_pop_1[t_span[1]:t_span[2], :]), c = :purple, label = "", lw = 2
		)
		plot!(
			steps, macro_op(to_check_pop_2[t_span[1]:t_span[2], :]), c = :darkorange, label = "", lw = 2
		)

	else
		plot1 = plot(
			steps,
			macro_op(θs[t_span[1]:t_span[2], 1:Integer(N/2)]),
			ylims = (0,1.1),
			# xlabel = "Timesteps",
			# xticks = [],
			# ylabel = "KOP",
			c = :darkorange,
			# label = "Population 1",
			# tickfontsize = 24,
			# legendfontsize = 24,
			# xlabelfontsize = 24,
			# ylabelfontsize = 24,
			size = (900, 450),
			lw = 2,
			grid = false,
			label = "",
			# xaxis = :log10,
			# fmt = plot_format
		)
	
		plot!(
			steps,
			macro_op(θs[t_span[1]:t_span[2], Integer(N/2)+1:end]),
			c = :purple,
			# label = "Population 2",
			label = "",
			lw = 2
		)
	
		plot!(
			steps,
			macro_op(θs[t_span[1]:t_span[2], :]),
			c = :black,
			label = "",
			# label = "Global",
			lw = 2,
			linestyle = :dash
		)
	end

end

# ╔═╡ eca7c878-63ef-4014-83ce-0992d3f49dfd
let
	n = n_discussion_1
	B = B_discussion_1
	N = N_discussion_1
	K = K_discussion_1
	H = H_discussion_1
	k = k_discussion_1
	θs = θs_discussion_1
	t_span = (t_span_start_discussion_1, t_span_end_discussion_1)
	steps = steps_discussion_1[t_span[1]:t_span[2]]

	
	modules_1_KOP = zeros(size(θs, 1), B[1])
	for i in 1:B[1]
		modules_1_KOP[:, i] = macro_op(θs[:, n[1]*(i-1)+1:n[1]*i])
	end
		
	plt1 = plot(
		steps,
		modules_1_KOP[t_span[1]:t_span[2], 1:round(Integer, B[1]/2)],
		title = "Modules pop 1",
		label = "",
		lw = 0.5,
		# size = (900, 450),
	)

	plot!(
		steps,
		macro_op(θs)[t_span[1]:t_span[2]],
		lw = 3,
		linestyle = :dash,
		label = "",
		c = :black
	)

	plot!(
		steps,
		macro_op(θs[t_span[1]:t_span[2], 1:Integer(N/2)]),
		c = :darkorange,
		# label = "Population 2",
		label = "",
		lw = 2
		)

	plt2 = plot(
		steps,
		modules_1_KOP[t_span[1]:t_span[2], round(Integer, B[1]/2)+1:end],
		title = "Modules pop 2",
		label = "",
		lw = 0.5,
		# size = (900, 450),
	)

	plot!(
		steps,
		macro_op(θs)[t_span[1]:t_span[2]],
		lw = 3,
		linestyle = :dash,
		label = "",
		c = :black
	)

	plot!(
		steps,
		macro_op(θs[t_span[1]:t_span[2], Integer(N/2)+1:end]),
		c = :purple,
		# label = "Population 2",
		label = "",
		lw = 2
		)

	plots_vec = [plt1, plt2]
	plot(
		plots_vec...,
		size = (900, length(plots_vec)*450),
		layout = (length(plots_vec), 1)
	)
end

# ╔═╡ d4252280-3be2-424d-979c-1295e352e80a
# function SBMvar(n::AbstractArray, B::AbstractArray, H::Number, γ; K = 1)

# 	return _SBMvar_constructor(n, B, H, γ; K = K)
	
# end

# ╔═╡ 4e9c1763-9062-4e31-907d-85e08861c6d5
md"""
### Dynamical Model
"""

# ╔═╡ d7e6062c-b280-4636-8c19-fb9cc6a35e63
"""
	function KS_sim(A::AbstractArray, P::AbstractArray, β::Real, K::Real, ω::AbstractArray, Δt::Real, sim_time::Real; seedval = 123, noiseQ = false, noise_scale = 0.1, save_ratio = 1)

Function to crate the nested SBM variation with 3 hierarchical levels.

Inputs:

	A::AbstractArray 		adjacency matrix
	P::AbstractArray 		partition matrix
	β::Real 				phase lag parameter (s.t. α = π/2 - β)
	K::Real 				coupling
	ω::AbstractArray 		natural frequencies vector
	Δt::Real 				time step size
	sim_time::Real 			simulation time
	seedval = 123 			seed value
	noiseQ = false 			set to true to use noise
	noise_scale = 0.1 		set the noise scale factor
	save_ratio = 1 			set to 1 outputs the phases at all time steps
 							set to 2 saves every other iteration etc

Outputs:

	θs::AbstractArray 		phase at each time step (if save_ratio = 1)
	params::Dict 			dictionary of parameters


"""
function KS_sim(A::AbstractArray, P::AbstractArray, β::Real, K::Real, ω::AbstractArray, Δt::Real, sim_time::Real; seedval = 123, noiseQ = false, save_ratio = 1)

	N = size(A)[1]
	
	# simulations settings
	Random.seed!(seedval)
	
	# lag parameter
	α = π / 2 - β  # lag parameter in the equations
	
	# construct lag matrix
	α_mat = zeros(N, N) # phase lag matrix
	for i in 1:N
	    for j in i+1:N
			# add lag if different partition at layer 1
	        if A[i,j] != 0 && P[1, i] != P[1, j]
	            α_mat[i, j] = α
	            α_mat[j, i] = α
	        end
	    end
	end
	
	# oscillators parameters
	if noiseQ
	    noise_scale = noise_scale
	else
	    noise_scale = 0
	end
	
	# simulation parameters
	steps = (0.0+Δt):Δt:sim_time
	no_steps = length(steps)
	
	# storing parameters
	no_saves = round(Integer, no_steps / save_ratio)
	θ_now = rand(Uniform(-π, π), N)  # random init conditions
	θs = zeros(no_saves, N)
	θs[1, :] = θ_now
	
	save_counter = 1
	
	for t in 2:no_steps
	    
	    # update phases
	    θj_θi_mat = (repeat(θ_now',N) - repeat(θ_now',N)') - α_mat
	    setindex!.(Ref(θj_θi_mat), 0.0, 1:N, 1:N) # set diagonal elements to zero 
	
	    k1 = map(sum, eachrow(A .* sin.(θj_θi_mat)))
	    θ_now += Δt .* (ω + k1) + noise_scale*(rand(Normal(0,1),N))*sqrt(Δt)
	    save_counter += 1
	
	    # save θ
	    if save_counter % save_ratio == 0
	        θs[round(Integer, save_counter / save_ratio), :] = θ_now
	    end
	    
	end
	
	params = Dict(
	    "α" => α,
	    "β" => β,
	    "K" => K,
	    "A" => A,
		"ω" => ω,
	    "seedval" => seedval,
	    "Δt" => Δt,
	    "sim_time" => sim_time,
	    "noiseQ" => noiseQ,
	)

	return θs, params
end

# ╔═╡ 5cf64b7c-4cd2-46bd-985a-c8af72419a44
let
	if run_sim_KS_1Q
		Random.seed!(seedval_KS_1)
		
		# hSBM parameters
		H = H_KS_1
		n = n_KS_1
		B = B_KS_1
		N = prod(n)
		k = k_KS_1
		K = K_KS_1
	
		# build network model
		A, P, p = SBMvar(n, B, H, k; K = K)
		if normalize_KS_1Q
			A = A ./ k # this line normalizes K or not
		end
	
		# simulation parameters
		β = β_KS_1
		if identical_KS_1Q
			ω = repeat([1], N)
		else
			ω = rand(Normal(1, nat_f_σ_KS_1), N)
		end
		global ω_KS_1 = ω
		sim_time = sim_time_KS_1
		Δt = Δt_KS_1
		save_ratio = 1
	
		# simulation
		θs, params = KS_sim(A, P, β, K, ω, Δt, sim_time; seedval = seedval_KS_1, noiseQ = false, save_ratio = save_ratio)
		
		# layer 1 data (modules)
		global modules_KOP_KS_1 = zeros(B[1])
		global modules_KOP_std_KS_1 = zeros(B[1])
		global modules_macro_op_KS_1 = zeros(length(θs[:, 1]), B[1])
		
		for i in 1:B[1]
			modules_macro_op_KS_1[:, i] = macro_op(θs[:, n[1]*(i-1)+1:n[1]*i])
			modules_KOP_KS_1[i] = mean(modules_macro_op_KS_1[:, i])
			modules_KOP_std_KS_1[i] = std(modules_macro_op_KS_1[:, i])
		end
	
		# layer 2 data (populations)
		relax = round(Integer, (sim_time_KS_1 / Δt) * 0.25)
		global mean_pop1_KS_1 = round(mean(macro_op(θs[relax+1:end, 1:Integer(N/2)])), digits = 3)
		global std_pop1_KS_1 = round(std(macro_op(θs[relax+1:end, 1:Integer(N/2)])), digits = 3)
		global mean_pop2_KS_1 = round(mean(macro_op(θs[relax+1:end, Integer(N/2):N])), digits = 3)
		global std_pop2_KS_1 = round(std(macro_op(θs[relax+1:end, Integer(N/2):N])), digits = 3)
		
		# plots
		global θs_KS_1 = θs
		global mat_plot_KS_1 = heatmap(A, c = cgrad([:white, :black]), yflip = true, size = (300, 300), frame = :box, cbar = false, xticks = [], yticks = [])
		global A_KS_1 = A
	
		println("This cell computes the simulation")
	end
end

# ╔═╡ e0f47bf1-47c1-4e3d-8ed9-b09a5e8ddd37
let
	seedval = 10
	Random.seed!(seedval)
	relax = 5000
	
	# hSBM parameters from previous cells
	H=H_KS_1; n=n_KS_1; B=B_KS_1; N=prod(n); k=k_KS_1; K=K_KS_1; β=β_KS_1
	θs = θs_KS_1
	A = A_KS_1
	steps = 0+Δt_KS_1:Δt_KS_1:sim_time_KS_1

	global filename_to_save_results_1 = "_n1_" * string(n[1]) * "_n2_" * string(n[2]) * "_H_" * string(H) * "_k_" * string(k) * "_beta_" * string(β) * "_seed_" * string(seedval)

	global path_results_1 = "/Users/ec627/Documents/Sussex/papers/CompetitionAcrossTimescales/RawFigures/NewMain/"
	
	
	if save_results_1

		plot1 = plot(
			steps,
			macro_op(θs[:, 1:Integer(N/2)]),
			ylims = (0,1.1),
			# xlabel = "Timesteps",
			xticks = [],
			yticks = ([0.0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.5", L"0.75", L"1.0"]),
			ylabel = L"R_{\rho_i}",
			c = :darkorange,
			label = L"\rho_1",
			tickfontsize = 24,
			legendfontsize = 24,
			xlabelfontsize = 24,
			ylabelfontsize = 24,
			size = (900, 350),
			lw = 2,
			grid=false,
			# xaxis = :log10,
			# fmt = plot_format
		)
	
		plot!(
			steps,
			macro_op(θs[:, Integer(N/2)+1:end]),
			c = :blue,
			label = L"\rho_2",
			lw = 2
		)
	
		plot!(
			steps,
			macro_op(θs[:, :]),
			c = :black,
			label = "",
			# label = "Global",
			lw = 2,
			linestyle = :dash
		)
			
		local plots_vec = [
			plot1
		]
	
		# println(L"Z_0 = %$(mean(macro_op(θs[relax:end, :])))")
		# println(L"Z_21 = %$(mean(macro_op(θs[relax:end, 1:n[1]*n[2]])))")
		# println(L"Z_22 = %$(mean(macro_op(θs[relax:end, n[1]*n[2]+1:end])))")
		
		global plot_to_save_results_1 = plot(
			plots_vec...,
			layout = (length(plots_vec), 1),
			left_margin = 8Plots.mm,
			right_margin = 8Plots.mm,
			bottom_margin = 8Plots.mm,
			top_margin = 8Plots.mm,
			size = (900, 300)
		)

	else
		whole_plt = plot(
			steps,
			macro_op(θs),
			ylims = (0.0, 1.0),
			label = "",
			title = "Whole System",
			# seriestype = :scatter
		)
	
		# population 1 plot
		pop_1_plt = plot(
			steps,
			macro_op(θs[:, 1:Integer(N/2)]),
			ylims = (0.0, 1.0),
			label = "",
			title = "Population 1"
		)
		
		# annotate!(- 15, 1.25, text("Mean KOP: $(mean_pop1_KS_1) ± $(std_pop1_KS_1)", :top, :left, 10, :red))
	
		# population 2 plot
		pop_2_plt = plot(
			steps,
			macro_op(θs[:, Integer(N/2)+1:N]),
			ylims = (0.0, 1.0),
			label = "",
			title = "Population 2"
		)
		
		# annotate!(- 15, 1.25, text("Mean KOP: $(mean_pop2_KS_1) ± $(std_pop1_KS_1)", :top, :left, 10, :red))
	
		modules_pop1_plot = plot(
			steps,
			modules_macro_op_KS_1[:, 1:round(Integer, B[1]/2)],
			label = "",
			title = "Modules l = 1, pop 1"
		)
		modules_pop2_plot = plot(
			steps,
			modules_macro_op_KS_1[:, round(Integer, B[1]/2)+1:end],
			label = "",
			title = "Modules l = 1, pop 2"
		)
	
		# print info
		println("variance of KOP_L1 population 1: ", mean(modules_KOP_std_KS_1[1:Integer(B[1]/2)])^2)
		println("variance of KOP_L1 population 2: ", mean(modules_KOP_std_KS_1[Integer(B[1]/2)+1:end])^2)
		
		# plot everything
		all_plots = [whole_plt, pop_1_plt, pop_2_plt, modules_pop1_plot, modules_pop2_plot]
		plot(
			all_plots...,
			layout = (length(all_plots), 1),
			size = (700, length(all_plots) * 120)
		)
	end
end

# ╔═╡ 334985f4-7349-4501-8d0d-1c4cb7e4998a
let
	seedval = 10
	Random.seed!(seedval)

	steps = 0+1e-3:1e-3:sim_time_KS_1
	final_step_id = round(Integer, sim_time_KS_1/1e-3)
	t_span = (final_step_id - 2500, final_step_id - 800)
	# t_span = (1, 2500)
	
	# hSBM parameters
	H = H_KS_1
	n = n_KS_1
	B = B_KS_1
	N = prod(n)
	k = k_KS_1
	K = K_KS_1

	# build network model
	A, P, p = SBMvar(n, B, H, k; K = K)
	# A = A ./ k # this line normalizes K or not

	# simulation parameters
	β = β_KS_1
	ω = repeat([1], N)
	sim_time = sim_time_KS_1
	Δt = 1e-3
	save_ratio = 1
	steps = 0+Δt:Δt:sim_time

	# get data from previous cell
	θs = θs_KS_1
	
	# layer 1 data (modules)
	modules_KOP = zeros(B[1])
	modules_KOP_std = zeros(B[1])
	modules_macro_op = zeros(length(θs[:, 1]), B[1])
	
	for i in 1:B[1]
		modules_macro_op[:, i] = macro_op(θs[:, n[1]*(i-1)+1:n[1]*i])
		modules_KOP[i] = mean(modules_macro_op[:, i])
		modules_KOP_std[i] = std(modules_macro_op[:, i])
	end

	modules_per_pop = round(Integer, B[1]/2) # number of modules in a population
	yticks = ([0, 0.5, 1], ["0", "0.5", "1"])
	
	plt_modules_1 = plot(
		steps[t_span[1]:t_span[2]],
		modules_macro_op[t_span[1]:t_span[2], 1:modules_per_pop],
		label = "",
		frame = :box,
		ylims = (0,1),
		xticks = [],
		yticks = [], # yticks,
		lw = 3,
		# seriestype = :scatter
	)

	plt_modules_2 = plot(
		steps[t_span[1]:t_span[2]],
		modules_macro_op[t_span[1]:t_span[2], modules_per_pop+1:end],
		label = "",
		ylims = (0,1),
		yticks = yticks,
		lw = 3
	
	)

	global filename_to_save_results_2 = "_n1_" * string(n[1]) * "_n2_" * string(n[2]) * "_H_" * string(H) * "_k_" * string(k) * "_beta_" * string(β) * "_seed_" * string(seedval) * "_modules_plot_"

	# print info
	println("variance of KOP_L1 population 1: ", mean(modules_KOP_std_KS_1[1:Integer(B[1]/2)])^2)
	println("variance of KOP_L1 population 2: ", mean(modules_KOP_std_KS_1[Integer(B[1]/2)+1:end])^2)
	
	global modules_plt_KS_1 = plot(
		[
			plt_modules_1 #, plt_modules_2
		]...,
		layout = (2,1),
		size = (900, 350),
		# left_margin = 8Plots.mm,
		# right_margin = 8Plots.mm,
		# bottom_margin = 8Plots.mm,
		# top_margin = 8Plots.mm,
		# tickfontsize = 16,
		# legendfontsize = 16,
		# xlabelfontsize = 16,
		# ylabelfontsize = 16,
	)
end

# ╔═╡ ace0bc9d-2fef-4f91-b9f8-ef791058afa7
let
	if save_results_2Q
		savefig(modules_plt_KS_1, path_results_1 * filename_to_save_results_2 * ".png")
	else
		println(path_results_1 * filename_to_save_results_2 * ".png")
	end
end

# ╔═╡ 089e4c85-023a-4c4c-94f6-f09afbf0900b
let
	# hSBM parameters from previous cells
	H=H_KS_1; n=n_KS_1; B=B_KS_1; N=prod(n); k=k_KS_1; K=K_KS_1; β=β_KS_1
	θs = θs_KS_1
	A = A_KS_1
	steps = 0+1e-3:1e-3:sim_time_KS_1

	heatmap(transpose(θs .% (2*π)), yflip = true, size = (700, 350))
end

# ╔═╡ 547861de-d3fe-4950-bde4-5b2cb976d4a8
let
	if CC_KS_1_Q
		θs = θs_KS_1
		N = size(θs_KS_1)[2]
		A = A_KS_1
		no_steps = size(θs_KS_1)[1]
		threshold = 0.9
		window = 5
		
		CC_sizes = zeros(no_steps)
		DC_sizes = zeros(no_steps)
		
		# compute iPL matrices
		for t in window+1:no_steps-window
			t_span = (t-window, t+window)
			iPL_mat = fun_pattern(A, θs, t_span)
			ζ = iPL_mat .> threshold
	
			# compute number of connected components
			CC = connected_components(Graph(ζ))
			CC_sizes[t] = size(CC)[1]
	
			# DC_sizes[t] = count(x -> x < 0.01, eigen(Laplacian(ζ)).values)
		end
	
		global CC_sizes_KS_1 = CC_sizes
	end

end

# ╔═╡ 1b18093e-3ee1-4dea-b8a5-aea0ed3a04e9
let
	if CC_KS_1_Q
		plot(
			CC_sizes_KS_1,
			# xaxis = :log10,
			# yaxis = :log10,
			seriestype = :scatter,
			title = "Connected Components iPL matrix",
			label = ""
		)
	end
end

# ╔═╡ 86b8e8f1-47a3-47ba-96ce-c596b59d1d1e
plot(
	[mat_plot_KS_1,
	plot(ω_KS_1, seriestype = :scatter, ylabel = L"\omega", label = "", xlabel = "node id")]...,
	layout = (1,2), 
	size = (700, 350)
)

# ╔═╡ 7a43b227-ceeb-4238-b52e-7c57c05d81f7
md"""
### LRG flow
"""

# ╔═╡ cae6447f-0a68-4cb4-822b-475b8b55c358
"""
	function Laplacian(A::AbstractMatrix; method = :graph)

Function used to compute the Laplacian of the graph defined by the
adjacency matrix `A`.

If `method == :graph` --> compute the graph Laplacian;

If `method == :normalized` --> compute the normalized graph Laplacian.

"""
function Laplacian(A::AbstractMatrix; method = :graph)

	N = length(A[1, :])
	
	if method == :graph
		
		D = diagm([sum(A[i, :]) for i in 1:N])
		
		return D - A
		
	elseif method == :normalized

		𝐼 = diagm(ones(N))
		k = [sum(A[i, :]) for i in 1:N]
		A′ = zeros(N, N)
		
		for i in 1:N
			for j in 1:N
				if A[i,j] != 0
					A′[i,j] = A[i,j] / √(k[i]*k[j])
				end
			end
		end

		return 𝐼 - A′
	end
	
end

# ╔═╡ ed1167d1-3ec1-4d1d-a4a2-2a38aee53b77
let
	# set parameters
	θs = θs_KS_2
	N = size(θs_KS_2)[2]
	no_steps = size(θs_KS_2)[1]
	H = H_methods_2_input
	if large_sys_Q_M2
		k = N / 20
	else
		k = N / 5
	end
	if large_sys_Q_M2
		n = [32, 16, 2]
		B = [32, 2, 1]
	else
		n = [16, 8, 2]
		B = [16, 2, 1]
	end
	
	A, P, p = SBMvar(n, B, H, k; K = 1)
	
	threshold = 0.99999
	CC_sizes = zeros(no_steps)
	DC_sizes = zeros(no_steps)
	
	# compute iPL matrix
	for t in 1:no_steps
		ζ = zeros(N, N)
		for i in 1:N
			for j in 1+i:N
				
				if cos(θs[t, i] - θs[t, j]) > threshold
					ζ[i, j] = 1
					ζ[j, i] = 1
				end
				
			end
		end

		# compute number of connected components
		CC = connected_components(Graph(ζ))
		CC_sizes[t] = size(CC)[1]

		# DC_sizes[t] = count(x -> x < 0.01, eigen(Laplacian(ζ)).values)
	end

	L = Laplacian(A)
	λs = eigen(L).values[2:end]

	plt1 = plot(
		CC_sizes,
		xaxis = :log10,
		yaxis = :log10,
		seriestype = :scatter,
		title = "Connected Components iPL matrix",
		label = ""
	)

	plt2 = plot(
		1 ./ λs,
		collect(2:lastindex(λs)+1),
		xaxis = :log10,
		yaxis = :log10,
		title = "inverse eigenvalues log plot"
	)

	plot(
		[plt1, plt2]...,
		size = (600, 480),
		layout = (2,1),
		label = "",
		marker = true,
		markersize = 3
	)

end

# ╔═╡ b7b01c1f-3360-4bf8-a1ab-3cb068f165d9
let
	println("This cell is used to obtain data")
	# set parameters
	n = [16, 8, 2]
	B = [16, 2, 1]
	N=prod(n)
	k = 16;

	# parameter exploration
	H_range = [0.3, 0.35, 0.4, 0.45, 0.5]
	seed_range = collect(1:10)

	store_λs = zeros(lastindex(H_range), N-1)
	
	for seed in seed_range
		for (i, H) in enumerate(H_range)
			Random.seed!(seed)
			
			 K = 1; β = 0.1
	
			# construct graphs
			A, P, p = SBMvar(n, B, H, k; K = K)
			steps = 0+1e-3:1e-3:10
	
			# compute Laplacian
			L = Laplacian(A)
			store_λs[i, :] += eigen(L).values[2:end]
		end
	end

	folderpath = "/Users/ec627/Documents/Data/HierarchicalChimera/DataCollect/n1608fast/plots/"
	filename = folderpath * "LaplacianPlot_k_" * string(k) * ".jld2"
	
	# save_object(filename, store_λs ./ lastindex(seed_range))
end

# ╔═╡ e7265301-ef09-4a8a-8c30-1ee1904e9aee
let
	plt = plot()
	
	for H in [0.3, 0.35, 0.45, 0.5]
		Random.seed!(seedval_KS_1)

		# set parameters
		n = [16, 8, 2]
		B = [16, 2, 1]
		N=prod(n); k= n[1]*2; K=1; β=0.1

		# construct graphs
		A, P, p = SBMvar(n, B, H, k; K = K)
		steps = 0+1e-3:1e-3:10

		# compute Laplacian
		L = Laplacian(A)
		λs = eigen(L).values
	
		plot!(
			plt,
			1 ./ λs[2:end],
			collect(1:lastindex(λs[2:end])),
			xaxis = :log10,
			yaxis = :log10,
			label = "H = $H, k = $(round(Integer, k))",
			marker = true,
			markersize = 2,
			xlabel = L"\lambda_i^{-1}",
			ylabel = L"\mathrm{mode}\;i"
		)
	end

	# set parameters
	n = [16, 8, 2]
	B = [16, 2, 1]
	N=prod(n);
	
	for k in [16, 64] # [n[1], n[1]*n[2]/2]
		for H in [0.3, 0.35, 0.45, 0.5]
			Random.seed!(seedval_KS_1)

			# construct graphs
			A, P, p = SBMvar(n, B, H, k; K = 1)
			steps = 0+1e-3:1e-3:10
	
			# compute Laplacian
			L = Laplacian(A)
			λs = eigen(L).values
	
			plot!(
				plt,
				1 ./ λs[2:end],
				collect(1:lastindex(λs[2:end])),
				xaxis = :log10,
				yaxis = :log10,
				label = "H = $H, k = $(round(Integer, k))",
				marker = true,
				markersize = 2
			)
		end
	end

	for H in [0.3, 0.35, 0.45, 0.5]
		Random.seed!(seedval_KS_1)

		# set parameters
		n = [32, 8, 2]
		B = [16, 2, 1]
		N=prod(n); k= 32; K=1; β=0.1

		# construct graphs
		A, P, p = SBMvar(n, B, H, k; K = K)
		steps = 0+1e-3:1e-3:10

		# compute Laplacian
		L = Laplacian(A)
		λs = eigen(L).values
	
		plot!(
			plt,
			1 ./ λs[2:end],
			collect(1:lastindex(λs[2:end])),
			xaxis = :log10,
			yaxis = :log10,
			label = "H = $H, k = $(round(Integer, k))",
			marker = true,
			markersize = 2,
			xlabel = L"\lambda_i^{-1}",
			ylabel = L"\mathrm{mode}\;i"
		)
	end

	for H in [0.3, 0.35, 0.45, 0.5]
		Random.seed!(seedval_KS_1)

		# set parameters
		n = [5, 8, 2]
		B = [16, 2, 1]
		N=prod(n); k= 32; K=1; β=0.1

		# construct graphs
		A, P, p = SBMvar(n, B, H, k; K = K)
		steps = 0+1e-3:1e-3:10

		# compute Laplacian
		L = Laplacian(A)
		λs = eigen(L).values
	
		plot!(
			plt,
			1 ./ λs[2:end],
			collect(1:lastindex(λs[2:end])),
			xaxis = :log10,
			yaxis = :log10,
			label = "H = $H, k = $(round(Integer, k))",
			marker = true,
			markersize = 2,
			xlabel = L"\lambda_i^{-1}",
			ylabel = L"\mathrm{mode}\;i",
			size = (700, 350),
			left_margin = 5Plots.mm,
			right_margin = 5Plots.mm,
			bottom_margin = 5Plots.mm,
			top_margin = 5Plots.mm,
		)
	end
		
	plot(plt)
end

# ╔═╡ 1fe169ab-b47a-4615-aac8-0c3e0d871eba
let
	# set parameters
	H = 0.5
	n = [16, 8, 2]
	B = [16, 2, 1]
	k = n[1]*3
	
	N=prod(n); K=1; β=0.1;

	# construct graphs
	A, P, p = SBMvar(n, B, H, k; K = K)
	steps = 0+1e-3:1e-3:10

	# compute Laplacian
	L = Laplacian(A)
	λs = eigen(L).values

	println("Using k = $k, the largest eigenvalues is: ", maximum(abs.(λs)))
end

# ╔═╡ 9d651701-18a7-4629-b2da-e1e3881a61b0
let
	# set parameters
	n = [16, 8, 2]
	B = [16, 2, 1]
	H = 0.5
	N=prod(n); K=1; β=0.1
	k = n[1]+10

	# simulation paramaters
	no_seeds = 10
	store_λs = zeros(no_seeds, N-1)

	for seed in 1:no_seeds
		Random.seed!(seed)
	
		# construct graphs
		A, P, p = SBMvar(n, B, H, k; K = K)
		steps = 0+1e-3:1e-3:10
	
		# compute Laplacian
		L = Laplacian(A)
		store_λs[seed, :] = eigen(L).values[2:end]
	end

	λs_to_plot = mean(store_λs, dims = 1)[1, :]
	
	plt1 = scatter(collect(1:length(λs_to_plot)), λs_to_plot, label = "")
	scatter!([n[1]-1], [λs_to_plot[n[1]-1]], label = "", markersize = 6)
	plt2 = heatmap(
		SBMvar(n, B, H, k; K = K)[1],
		cgrad = [:black, :white], yflip = true,
		cbar = false
	)

	plot(
		[
			plt1, plt2
		]...,
		layout = (1,2),
		size = (700, 320)
	)
end

# ╔═╡ 839e6c5f-281b-47d6-bd02-2635491b5bfb
let
	# set parameters
	n = [16, 8, 2]
	B = [16, 2, 1]
	H = 0.5
	N=prod(n); K=1; β=0.1

	k_range = collect(n[1]:n[2]*n[1]-1)
	no_seeds = 10
	store = zeros(length(k_range), N-2, no_seeds)
	store_gap_2 = zeros(length(k_range), no_seeds)

	for seed in 1:no_seeds
		Random.seed!(seed)
		for k in k_range
			# construct graphs
			A, P, p = SBMvar(n, B, H, k; K = K)
			steps = 0+1e-3:1e-3:10
		
			# compute Laplacian
			L = Laplacian(A) #, method = :normalized)
			λs = eigen(L).values[2:end]
			# plot(diff(λs), label = "")
			store[k-n[1]+1, :, seed] = diff(λs)
			store_gap_2[k-n[1]+1, seed] = abs(λs[15] - λs[16])
		end
	end

	global sec_spectral_gap = mean(store_gap_2, dims = 2)[:, 1]
	# heatmap(store)
	global spectral_gap_plt = scatter(
		k_range, mean(store_gap_2, dims = 2)[:, 1],
		yerror = std(store_gap_2, dims = 2)[:, 1],
		label = "", title = "size of the 2nd spectral gap",
		xticks = ([20, 40, 60, 80, 100, 120], [L"20", L"40", L"60", L"80", L"100", L"120"]),
		size = (600, 300)
	)

	println("Here the `spectral_gap_plt` is computed")
end

# ╔═╡ 5bac2e14-23c5-4c2b-bf44-52bd3a0cb671
let
	folderpath = "/Users/ec627/Documents/Data/HierarchicalChimera/DataCollect/n1608fast/n1608Var_k/H_00/"
	cd(folderpath)

	filenames = readdir()[1:end]
	
	# Define a regular expression pattern to match the numbers
	pattern = r"seed_(\d+)_beta_([\d.]+)_H_([\d.]+)_k_([\d.]+)\.jld2"

	k_range = collect(17:1:127)
	seed_range = collect(1:1:10)

	store = zeros(lastindex(seed_range), lastindex(k_range))

	for filename in filenames
		
		# Match the pattern in the filename
	    match_result = match(pattern, filename)
	
	    # Extract values if there is a match
	    if match_result !== nothing
	
	        # Extract values from the matched groups
	        seed = parse(Int, match_result[1])
	        β = parse(Float64, match_result[2])
	        H = parse(Float64, match_result[3])
	        k = parse(Float64, match_result[4])

			if k ≤ 127
				res = load_object(folderpath * "/" * filename)
				# if seed > 2
				# 	seed -= 1
				# end
				store[seed, round(Integer, k)-16] = mean(std(res[9], dims = 1))
			end
		end
	end

	plt = plot(
		k_range,
		mean(store, dims = 1)[1, :],
		ribbon = std(store, dims = 1)[1, :],
		label = "",
		xlabel = L"k",
		ylabel = L"\langle\sigma(R_{\mu_i})\rangle",
		# showaxis = :y,
		# tickfontsize = 24,
		# legendfontsize = 24,
		# xlabelfontsize = 24,
		# ylabelfontsize = 24,
		# left_margin = 5Plots.mm,
		# right_margin = 5Plots.mm,
		# bottom_margin = 5Plots.mm,
		# top_margin = 5Plots.mm,
		xticks = ([20, 40, 60, 80, 100, 120], [L"20", L"40", L"60", L"80", L"100", L"120"]),
		yticks = ([0.0, 0.1, 0.2, 0.3], [L"0.0", L"0.1", L"0.2", L"0.3"]),
		c = :purple
	)

	global meta_var_k = mean(store, dims = 1)[1, :]
	global meta_var_k_error = std(store, dims = 1)[1, :]
	
	final_plot = plot(
		[plt, spectral_gap_plt]...,
		layout = (2,1),
		size = (700, 370),
		# xaxis = :log10,
		# left_margin = 5Plots.mm,
		# right_margin = 5Plots.mm,
		# bottom_margin = 5Plots.mm,
		# top_margin = 5Plots.mm,
		# yaxis = :log10
	)

	# if abstract_plot_Q2
	# 	savefig(final_plot, "/Users/ec627/Documents/Data/HierarchicalChimera/DataCollect/n1608fast/plots/modules_metastability_varying_k.png")
	# end

	plot(final_plot)
end

# ╔═╡ 6b8247b1-000b-48b0-ab05-30421e0e533a
meta_var_k

# ╔═╡ 659f8ecd-d37e-4706-b0bb-d45c6f8db2e5
let
	# compute correlation	
	# println("Correlation: ", cor(meta_var_k[1:end], sec_spectral_gap[2:end]))

	threshold = 2

	plt1 = scatter(
		sec_spectral_gap[2:end],
		meta_var_k[1:end],
		yerror = meta_var_k_error,label="",
		xaxis=:log10,yaxis=:log10,
		xlabel=L"2^{\mathrm{nd}}\,\mathrm{spectral}\,\mathrm{gap}\,\mathrm{size}",
		ylabel=L"\sigma_{\mathrm{met}}(R_{\mu_i})",
		xlims = (1e-1, 15), ylims = (10^(-3.1), 0.2),
		xticks=([1e-1, 1e0, threshold, 1e1],[L"10^{-1}", L"0", L"2", L"10"]),
		yticks=([1e-3, 1e-2, 1e-1], [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
		tickfontsize = 16,
		legendfontsize = 12,
		xlabelfontsize = 16,
		ylabelfontsize = 16,
		size = (700, 350),
		left_margin = 5Plots.mm,
		right_margin = 5Plots.mm,
		bottom_margin = 5Plots.mm,
		top_margin = 5Plots.mm,
		framestyle = :box
	)
	# plot!(
		# [threshold, threshold], [minimum(meta_var_k), maximum(meta_var_k)], c = :red,label="",
	# )

	# savefig(plt1, "/Users/ec627/Documents/Sussex/CompleNet/presentation/images/SpectralGap.png")
end

# ╔═╡ fc29bab1-9c33-4e90-9e42-d386d57e7329
sec_spectral_gap

# ╔═╡ 7cd584d7-5330-421e-be91-8d155414a402
function density_mat(τ, L)
	
	return exp(- τ .* L) / (tr(exp(- τ .* L)))
	
end

# ╔═╡ 34138921-448d-407b-a5b2-9a4459b52b65
let
	Random.seed!(123)
	
	t = sim_time_methods_3
	
	# hierarchical SBM
	N = N_methods_3
	n = n_methods_3
	B = B_methods_3
	k = k_methods_3
	H = H_methods_3
	
	SBMvar_output = SBMvar(n, B, H, k; K = 1)
	A = SBMvar_output[1]

	# find Laplacian
	L = Laplacian(A, method = :graph)
	(λ, v) = eigen(L)

	ticks_eigen = map(x -> string(x), 1 ./ λ[2:end])
	for (i, str) in enumerate(ticks_eigen)
		ticks_eigen[i] = str[1:5]
	end

	# laplacian operator
	K = exp(- t .* L)

	# density matrix operator
	ρ = density_mat(t, L)
	(μ, u) = eigen(ρ)

	# rule to generate new graph ζ
	ρ_prime = zeros(N, N)
	for i in 1:N
		for j in i+1:N
			ρ_prime[i,j] = ρ[i,j] / (minimum([ρ[i,i], ρ[j,j]]))
			ρ_prime[j,i] = ρ[i,j] / (minimum([ρ[i,i], ρ[j,j]]))
		end
	end

	# initialize new graph with separated components (defining the supernodes)
	ζ = zeros(N,N)
	for i in 1:N
		for j in i+1:N
			if (ρ_prime[i,j] - 1) ≥ 0
				ζ[i,j] = 1
				ζ[j,i] = 1
			end
		end
	end

	ρ_plot_title = L"\rho = \frac{\hat{K}}{\mathrm{tr}[\hat{K}]}"
	ρ_prime_plot_title = L"\rho'_{ij} = \frac{\rho_{ij}}{\mathrm{min}(\rho_{ii},\rho_{jj})}'"
	ζ_plot_title = L"\zeta = \mathcal{H}(\rho'_{ij} - 1)"

	CC = connected_components(Graph(ζ))
	println("Connected components: $(size(CC))")

	if show_ρ_plots_methods_3
		plot(
			[
				heatmap(A, yflip = true, cbar = false, title = L"A", xticks = [], yticks = []),
				heatmap(ζ, yflip = true, cbar = false, title = ζ_plot_title, xticks = [], yticks = []),
				heatmap(ρ, yflip = true, cbar = false, title = ρ_plot_title, xticks = [], yticks = []),
				heatmap(ρ_prime, yflip = true, cbar = false, title = ρ_prime_plot_title, xticks = [], yticks = []),
			]...,
			layout = (2,2),
			size = (450, 560)
		)
	else
		plot(
			[
				heatmap(A, yflip = true, cbar = false, title = L"A", xticks = [], yticks = []),
				heatmap(ζ, yflip = true, cbar = false, title = ζ_plot_title, xticks = [], yticks = [])
			]...,
			layout = (1,2),
			size = (650, 360)
		)
	end

	####################
	# some other plots #
	
	# plt_eigen = plot(
	# 	1 ./ λ[2:end],
	# 	collect(2:length(λ)),
	# 	marker = :circle,
	# 	color = RGB(.1, 1., 0),
	# 	label = "",
	# 	xticks = (1 ./ λ[2:end], ticks_eigen),
	# 	xlabel = L"""\frac{1}{\lambda}""",
	# 	bg = RGB(.2,.2,.2)
	# )

	# mode = 2
	# plt_mode = plot(
	# 	v[mode, :],
	# 	seriestype = :scatter,
	# 	marker = :circle,
	# 	color = RGB(.1, 1., 0),
	# 	bg = RGB(.2,.2,.2),
	# 	label = "eigenmode: $mode"
	# )
end

# ╔═╡ 224a8039-8bff-46b0-a095-c0be26f6ce04
let
	Random.seed!(1)

	path_filename = "/Users/ec627/Documents/Data/HierarchicalChimera/paper_folder/results_methods_32.jld2"
	
	if results_load_methods_32
		
		global CC_sizes, DC_sizes = load_object(path_filename)
		println("Data loaded correctly")
		
	else
		# hierarchical SBM
		n = [16, 8, 2]
		B = [16, 2, 1]
		N = prod(n)
		k = N / 5
		H_range = 0.0:0.2:0.6
		t_range = 0.0:0.1:5
		global H_range_M3 = H_range
		global t_range_M3 = t_range
	
		# store CC values
		global CC_sizes = zeros(length(H_range), length(t_range)) # connected components
		global DC_sizes = zeros(length(H_range), length(t_range)) # disconnected components
		
		for (j, H) in enumerate(H_range)
			
			# H = choose_control_param_H
			SBMvar_output = SBMvar(n, B, H, k; K = 1)
			A = SBMvar_output[1]
		
			# find Laplacian
			L = Laplacian(A, method = :graph)
			(λ, v) = eigen(L)
		
			ticks_eigen = map(x -> string(x), 1 ./ λ[2:end])
			for (i, str) in enumerate(ticks_eigen)
				ticks_eigen[i] = str[1:5]
			end
			
			for (i, t) in enumerate(t_range)
				# laplacian operator
				K = exp(- t .* L)
			
				# density matrix operator
				ρ = density_mat(t, L)
				(μ, u) = eigen(ρ)
			
				# rule to generate new graph ζ
				ρ_prime = zeros(N, N)
				for i in 1:N
					for j in i+1:N
						ρ_prime[i,j] = ρ[i,j] / (minimum([ρ[i,i], ρ[j,j]]))
						ρ_prime[j,i] = ρ[i,j] / (minimum([ρ[i,i], ρ[j,j]]))
					end
				end
			
				# initialize new graph with separated components (defining the supernodes)
				ζ = zeros(N,N)
				for i in 1:N
					for j in i+1:N
						if (ρ_prime[i,j] - 1) ≥ 0
							ζ[i,j] = 1
							ζ[j,i] = 1
						end
					end
				end
	
				# compute number of connected components
				CC = connected_components(Graph(ζ))
				CC_sizes[j, i] = size(CC)[1]
	
				# compute number of disconnected components
				DC_sizes[j,i] = count(x -> x < 0.01, eigen(Laplacian(ζ)).values)
			end
			
		end

		# save_object("/Users/ec627/Documents/Data/HierarchicalChimera/paper_folder/results_methods_32.jld2", [CC_sizes, DC_sizes])
	end

end

# ╔═╡ c54b731f-842c-47aa-a098-e81bce7f22f6
let
	if results_load_methods_32
		H_range = 0.0:0.1:0.6
		# t_range = 0.0:0.01:5
	else
		t_range = t_range_M3
		H_range = H_range_M3
	end

	plt1 = plot(size = (400, 400)) # ylims = (0, 20)
	
	for (i, H) in enumerate(H_range)
		plot!(
			# t_range,
			CC_sizes[i, :],
			label = "H = $(H_range[i])",
			xaxis = :log10,
			yaxis = :log10,
			lw = 3,
			# c = colors_dict[H]
		)
	end

	# plt2 = plot(size = (400, 400))
	
	# for i in 1:length(H_range)
	# 	plot!(
	# 		t_range,
	# 		DC_sizes[i, :],
	# 		label = "H = $(H_range[i])",
	# 		xaxis = :log10,
	# 		yaxis = :log10,
	# 		lw = 3
	# 	)
	# end

	# plot([plt1, plt2]..., size = (900, 450))

	plot(plt1, size = (700, 350))
end

# ╔═╡ 15afb60a-76f0-45ae-aeb1-fe2f81c7713d
function spec_entropy(ρ::AbstractMatrix, N::Number, τ::Number)

	μ = eigen(ρ).values
	k₁ = sum([μ[i] * log(10, μ[i]) for i in 1:length(μ)])
	
	return - (1/log(10, N)) * k₁
end

# ╔═╡ 8a1a8183-7b79-4d77-b19a-45e1195d5a9b
function spec_entropy(μ::AbstractArray)

	N = length(μ)
	# μ = μ[findall(x -> x > 0, μ)]
	
	return - (1/log(10, N)) * sum([μ[i] * log(10, μ[i]) for i in 1:length(μ)])
	
end

# ╔═╡ 954a6605-f6c8-4b83-a27b-3d52fe1d7a12
let
	# define plots
	plot_xticks = ([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], [L"""10^{-3}""",L"""10^{-2}""", L"""10^{-1}""",L"""10^{0}""", L"""10^{1}""", L"""10^{2}""", L"""10^{3}""", L"""10^{4}"""])
	plt_hsbm = plot(size = (600, 350), framestyle = :box)
	
	# construct network
	H = H_results_LRG_1
	n = [n1_results_LRG_1, n2_results_LRG_1, 2]
	B = [2*n2_results_LRG_1, 2, 1]
	N = prod(n)
	k = k_results_LRG_1
	SBMvar_output = SBMvar(n, B, H, k; K = 1)
	A = SBMvar_output[1]

	if H > 0.9
		start_λ = 3
	else
		start_λ = 2
	end
	
	# find Laplacian
	L = Laplacian(A, method = :graph)
	λs = eigen(L).values

	# simulation params
	Δt = Δt_results_LRG_1
	t = sim_time_results_LRG_1 # 1e2/2
	steps_iter = Δt:Δt:t
	inv_steps = 1 ./ collect(steps_iter)

	# entropy store
	S = zeros(length(steps_iter))
	μ_laplacian = zeros(N)

	for (i, τ) in enumerate(steps_iter)

		# calculate eigenvals of density matrix
		Z = sum([exp(- λs[i] * τ) for i in 1:N])
		μ_laplacian = [exp(- λs[j] * τ) / Z for j in 1:N]
		
		S[i] = spec_entropy(μ_laplacian)
	end

	# calculate specific heat C
	ds = diff(S)
	dx = diff(log.(collect(steps_iter)))
	SC = filter(!isnan, (- ds ./ dx)) # specific heat

	# plot Entropy
	plot!(plt_hsbm,
		collect(steps_iter),
		# 1 ./ collect(steps_iter),
		1 .- S,
		# S,
		xscale = :log10,
		legend = :topleft,
		label = "S for H = $H",
		xticks = plot_xticks,
		linestyle = :dot,
		lw = 2,
		linealpha = .6,
		# c = colors_dict[H]
	)

	# plot Specific Heat
	plot!(plt_hsbm,
		collect(steps_iter[1:length(SC)]),
		# 1 ./ collect(steps_iter[1:length(SC)]),
		SC,
		xscale = :log10,
		label = "C for H = $H",
		xticks = plot_xticks,
		lw = 2,
		linealpha = 1,
		# c = colors_dict[H]
	)

	plt2 = heatmap(
		A, yflip = true, c = cgrad([:white, :black]),
		cbar = false, xticks = [], yticks = [], size = (350, 350)
	)

	println("Lowest eigenvalues:\n$(λs[2:6])")
	println("Highest eigenvalues:\n$(λs[end-5:end])")
	
	plot(
		[plt_hsbm, plt2]...,
		size = (800, 350),
		layout = (1, 2)
	)
end

# ╔═╡ 84fd259e-6353-4015-b9bb-47e30e7bc039
function forward_finite_difference(y, x)

	return diff(y) ./ x[1:end-1]
	
end

# ╔═╡ baac38d1-5928-4949-824d-b9d533bc402c
function central_finite_difference(y, h::Number)

	return (y[3:end] - y[1:end-2]) ./ (2*h)

end

# ╔═╡ 7f26d0ef-3194-4981-9833-ee843da89da4
function central_finite_difference(y, dx::AbstractArray)

	dy = (y[3:end] - y[1:end-2])

	return dy ./ diff(dx)

end

# ╔═╡ 47c7c8be-7031-4b44-bc7e-fc76ac651ce2
let
	# define plots
	plot_xticks = ([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], [L"""10^{-3}""",L"""10^{-2}""", L"""10^{-1}""",L"""10^{0}""", L"""10^{1}""", L"""10^{2}""", L"""10^{3}""", L"""10^{4}"""])
	plt_hsbm = plot(size = (600, 350), framestyle = :box)
	colors = [:blue, :orange, :pink, :red]

	for (id, N) in enumerate([256, 256*2, 256*4])
		
		# construct network
		p_intra = 128/N
		p_inter = 1/N
		n = Integer(N / 4)
		P = reduce(vcat, [repeat([i], n) for i in 1:4]) # partition matrix
		A = zeros(N, N)
		for i in 1:N
			for j in i+1:N
				if P[i] == P[j] && rand() < p_intra
					A[i,j] = 1
					A[j,i] = 1
				elseif rand() < p_inter
					A[i,j] = 1
					A[j,i] = 1
				end
			end
		end
		
		# find Laplacian
		L = Laplacian(A, method = :graph)
		λs = eigen(L).values
	
		# simulation params
		Δt = 1e-3
		t = 20
		steps_iter = Δt:Δt:t
		inv_steps = 1 ./ collect(steps_iter)
	
		# entropy store
		S = zeros(length(steps_iter))
		μ_laplacian = zeros(N)
	
		for (i, τ) in enumerate(steps_iter)
	
			# calculate eigenvals of density matrix
			Z = sum([exp(- λs[i] * τ) for i in 1:N])
			μ_laplacian = [exp(- λs[j] * τ) / Z for j in 1:N]
			
			S[i] = spec_entropy(μ_laplacian)
		end
	
		# calculate specific heat C
		dlogt = log.(collect(steps_iter))[2:end]
		SC = - central_finite_difference(S, dlogt) # specific heat
	
		# plot Entropy
		plot!(plt_hsbm,
			collect(steps_iter),
			1 .- S,
			xscale = :log10,
			legend = :topleft,
			label = "",
			xticks = plot_xticks,
			linestyle = :dot,
			lw = 2,
			linealpha = .6,
			c = colors[id]
		)
	
		# plot Specific Heat
		plot!(plt_hsbm,
			collect(steps_iter[1:length(SC)]),
			SC, # ./ maximum(filter(!isnan, SC))
			xscale = :log10,
			label = "N = $N",
			xticks = plot_xticks,
			lw = 2,
			linealpha = 1,
			c = colors[id]
		)
	end

	# plt2 = heatmap(
	# 	A, yflip = true, c = cgrad([:white, :black]),
	# 	cbar = false, xticks = [], yticks = [], size = (350, 350)
	# )
	
	# plot(
	# 	[plt_hsbm, plt2]...,
	# 	size = (950, 350),
	# 	layout = (1, 2)
	# )

	plot(plt_hsbm)
end

# ╔═╡ 8fabe675-1932-4c2e-aa0c-f92b00b294b7
let
	# define plots
	plot_xticks = ([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], [L"""10^{-3}""",L"""10^{-2}""", L"""10^{-1}""",L"""10^{0}""", L"""10^{1}""", L"""10^{2}""", L"""10^{3}""", L"""10^{4}"""])
	plt_hsbm = plot(size = (600, 350), framestyle = :box, xlabel = L"\tau")
	colors = [:blue, :orange, :pink, :red]

	id = 1
	
	# construct network
	N = 256
	p_intra = 128/N
	p_inter = 1/N
	n = Integer(N / 4)
	P = reduce(vcat, [repeat([i], n) for i in 1:4]) # partition matrix
	A = zeros(N, N)
	for i in 1:N
		for j in i+1:N
			if P[i] == P[j] && rand() < p_intra
				A[i,j] = 1
				A[j,i] = 1
			elseif rand() < p_inter
				A[i,j] = 1
				A[j,i] = 1
			end
		end
	end
	
	# find Laplacian
	L = Laplacian(A, method = :graph)
	λs = eigen(L).values

	# simulation params
	Δt = 1e-3
	t = 20
	steps_iter = Δt:Δt:t
	inv_steps = 1 ./ collect(steps_iter)

	# entropy store
	S = zeros(length(steps_iter))
	μ_laplacian = zeros(N)

	for (i, τ) in enumerate(steps_iter)

		# calculate eigenvals of density matrix
		Z = sum([exp(- λs[i] * τ) for i in 1:N])
		μ_laplacian = [exp(- λs[j] * τ) / Z for j in 1:N]
		
		S[i] = spec_entropy(μ_laplacian)
	end

	# calculate specific heat C
	dlogt = log.(collect(steps_iter))[2:end]
	SC = - central_finite_difference(S, dlogt) # specific heat

	# find peaks
	dSC = diff(SC)
	zeros_dSC = []
	for (i, val) in enumerate(dSC[2:end])
		if sign(val) != sign(dSC[i])
			push!(zeros_dSC, i)
		end
	end

	τ_stars = collect(steps_iter)[zeros_dSC[1:3]]

	# plot Entropy
	plot!(plt_hsbm,
		collect(steps_iter),
		1 .- S,
		xscale = :log10,
		legend = :topleft,
		label = L"S",
		xticks = ([1e-3, 1e-2, 1e-1, 1e0, 1e1], [L"""10^{-3}""",L"""10^{-2}""", L"""10^{-1}""",L"""10^{0}""", L"""10^{1}"""]),
		linestyle = :dot,
		lw = 2,
		linealpha = .6,
		c = colors[id]
	)

	# plot Specific Heat
	plot!(plt_hsbm,
		collect(steps_iter[1:length(SC)]),
		SC, # ./ maximum(filter(!isnan, SC))
		xscale = :log10,
		label = L"C",
		xticks = ([1e-3, 1e-2, 1e-1, 1e0, 1e1], [L"""10^{-3}""",L"""10^{-2}""", L"""10^{-1}""",L"""10^{0}""", L"""10^{1}"""]),
		lw = 2,
		linealpha = 1,
		c = colors[id]
	)

	plot!(
		plt_hsbm,
		τ_stars,
		[SC[zeros_dSC[1]], SC[zeros_dSC[2]], SC[zeros_dSC[3]]],
		seriestype = :scatter,
		label = "maxima & minima"
	)
	annotation_values = round.([SC[zeros_dSC[1]], SC[zeros_dSC[2]], SC[zeros_dSC[3]]], digits = 3)
	
	annotate!(
		τ_stars[1] + 3*τ_stars[1],
		SC[zeros_dSC[1]],
		L"\tau = %$(τ_stars[1])"
	)
	annotate!(
		τ_stars[2] + 3*τ_stars[2],
		SC[zeros_dSC[2]],
		L"\tau = %$(τ_stars[2])"
	)
	annotate!(
		τ_stars[3] + 3*τ_stars[3],
		SC[zeros_dSC[3]],
		L"\tau = %$(τ_stars[3])"
	)

	plt2 = heatmap(
		A, yflip = true, c = cgrad([:white, :black]),
		cbar = false, xticks = [], yticks = [], size = (350, 350)
	)

	plt3 = plot(
		1 ./ λs[2:end],
		collect(2:lastindex(λs)),
		label = "",
		seriestype = :scatter,
		xaxis = :log10,
		yaxis = :log10,
		xticks = ([1e-1, 1e0, 1e1, 1e2], [L"""10^{-1}""",L"""10^{0}""", L"""10^{1}""", L"""10^{2}"""]),
		c = colors[id], markersize = 2.5,
		xlabel = L"1/\lambda_i",
		ylabel = L"i"
	)
	
	plot(
		[plt_hsbm, plt3]...,
		size = (950, 350),
		layout = (1, 2),
		left_margin = 8Plots.mm,
		right_margin = 8Plots.mm,
		bottom_margin = 8Plots.mm,
		top_margin = 8Plots.mm,
	)
end

# ╔═╡ 6bb3d9a8-da2f-4b53-9461-09c1e9956a61
md"""
### Coalition Entropy
"""

# ╔═╡ 934a7db6-ff49-40b5-aa81-32ae58fb6918
"""
	function coal(x; γ = 0.8)

returns 	`1` if `x > γ`

returns		`0` otherwise.
"""
function coal(x; γ = 0.8)
	if x > γ
		return 1
	else
		return 0
	end
end

# ╔═╡ 4beca585-082c-43af-a966-05d76f75ec60
function get_coalitions(θs, t_span, n, B; γ = 0.8)
	local_KOPs = [
		macro_op(θs[t_span[1]:t_span[2], (i-1)*n[1]+1:i*n[1]]) for i in 1:B[1]
			]			
	local_KOPs = reduce(hcat, local_KOPs)'
	return map(x -> coal(x, γ = γ), local_KOPs)
end

# ╔═╡ ca15159b-4a52-458e-839d-c66d42c3ff23
function coal_entropy(coalitions)

	# get number of modules n[2]
	n_2 = size(coalitions)[1]
	# get number of timesteps
	n0_timesteps = size(coalitions)[2]

	# define space of all possible coalitions of n[2] modules
	coalition_space = hcat([digits(i, base=2, pad=n_2) |> reverse for i in 0:(2^n_2)-1])
	
	# compute coalition empirical probability distribution
	count = zeros(2^n_2)
	for i in 1:n0_timesteps
		coal = coalitions[:, i]
		count[findfirst(x -> x == coal, coalition_space)[1]] += 1
	end
	prob_coal = count ./ n0_timesteps

	# calculate entropy of empirical probability distribution
	coal_H = 0
	for p in prob_coal
		if p != 0
			coal_H -= (p * log2(p)) / log2(2^n_2)
		end
	end

	return coal_H
end

# ╔═╡ 913e20da-66b8-4660-bc1c-a524602e6e97
let
	# define numerical analysis parameters
	steps = 0+1e-3:1e-3:sim_time_KS_1
	relaxation = round(Integer, 1 / 1e-3)
	t_span = (relaxation, round(Integer, sim_time_KS_1/1e-3))
	
	# define threshold to be part of the coalition
	γ = 0.8
	
	if CE_KS_1_Q

		# get parameters from previous simulation
		H=H_KS_1; n=n_KS_1; B=B_KS_1; N=prod(n); k=k_KS_1; K=K_KS_1; β=β_KS_1
		θs = θs_KS_1; A = A_KS_1

		coalitions = get_coalitions(θs, t_span, n, B, γ = γ)
		
		println(
			"This cell computes the coalitions from timesteps:\n$(t_span[1]) to $(t_span[2]), which correspond to the time variables \n$(steps[t_span[1]]) to $(steps[t_span[2]])"
		)

		println("\nThe coalition entropy of the modules in the first population is:\n$(coal_entropy(coalitions[1:n[2], :]))")

		println("\nThe coalition entropy of the modules in the second population is:\n$(coal_entropy(coalitions[n[2]+1:n[2]*2, :]))")
	end
end

# ╔═╡ d069618b-e710-48ec-bb78-ac503a9e7b4a
let
	# define numerical analysis parameters
	steps = 0+1e-3:1e-3:sim_time_KS_1
	relaxation = round(Integer, 1 / 1e-3)
	t_span = (relaxation, round(Integer, sim_time_KS_1/1e-3))
	
	# define threshold to be part of the coalition
	γ = 0.8
	
	if CE_KS_1_Q2

		# get parameters from previous simulation
		H=H_KS_1; n=n_KS_1; B=B_KS_1; N=prod(n); k=k_KS_1; K=K_KS_1; β=β_KS_1
		θs = θs_KS_1; A = A_KS_1

		coalitions = get_coalitions(θs, t_span, n, B, γ = γ)
		
		println(
			"This cell computes the coalitions from timesteps:\n$(t_span[1]) to $(t_span[2]), which correspond to the time variables \n$(steps[t_span[1]]) to $(steps[t_span[2]])"
		)

		println("\nThe coalition entropy of the modules in the first layer is:\n$(coal_entropy(coalitions))")

	end
end

# ╔═╡ 7820082d-8954-4104-86e1-c476d1c8b370
hcat([digits(i, base=2, pad=16) |> reverse for i in 1:(2^16)-1])

# ╔═╡ e39c4f42-5947-4074-8de0-332f1a33b97f
"""
```jldoctest
    shannon(p::AbstractArray)
```
Simple function to implement Shannon entropy given an array of probabilities `p`.
Requires `sum(p) == 1`.
"""
function shannon(p::AbstractArray)

    # check requirement
    if sum(p) != 1
        error("Probability distribution needs to add up to 1")
    end

    # compute H(p)
	out = 0
	for pi in p
		if pi > 0
			out -= pi * log2(pi)
		end
	end

	return out
end

# ╔═╡ ad57ed62-c893-4cd5-b1b0-a01c56563134
md"""
# Extras
"""

# ╔═╡ 217e296f-b587-4d32-818a-14f96ddcab72
md"""
below I use the new central diff formula to explore how the specific heat changes wrt to structural parameter changes
"""

# ╔═╡ 0e6cfa00-cff7-4889-b70b-8bf35b6de37b
let
	# define plots
	plot_xticks = ([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], [L"""10^{-3}""",L"""10^{-2}""", L"""10^{-1}""",L"""10^{0}""", L"""10^{1}""", L"""10^{2}""", L"""10^{3}""", L"""10^{4}"""])
	plt_hsbm = plot(size = (600, 350), framestyle = :box, xlabel = "τ", ylabel = "S and C" )
	
	colors_plot = []
	num_colors = 11
	for i in 0:1/num_colors:1
	    cc = RGB(i, 0, 0.5)
	    push!(colors_plot, cc)
	end

	colors_plot

	n₁ = 16
	n₂ = 16
	k = n₁ + ((n₁*n₂-1) - n₁) * 0.01
	
	# construct network
	H = 0.5
	S_label = "S for H = $H"
	C_label = "C for H = $H"
	# C_label = ""
	# S_label = ""

	n = [n₁, n₂, 2]
	B = [2*n₂, 2, 1]
	N = prod(n)
	SBMvar_output = SBMvar(n, B, H, k; K = 1)
	A = SBMvar_output[1]

	if H > 0.9
		start_λ = 3
	else
		start_λ = 2
	end
	
	# find Laplacian
	L = Laplacian(A, method = :graph)
	λs = eigen(L).values

	# simulation params
	Δt = 1e-3
	t = 10
	steps_iter = Δt:Δt:t
	inv_steps = 1 ./ collect(steps_iter)

	# entropy store
	S = zeros(length(steps_iter))
	μ_laplacian = zeros(N)

	for (i, τ) in enumerate(steps_iter)

		# calculate eigenvals of density matrix
		Z = sum([exp(- λs[i] * τ) for i in 1:N])
		μ_laplacian = [exp(- λs[j] * τ) / Z for j in 1:N]
		
		S[i] = spec_entropy(μ_laplacian)
	end

	# calculate specific heat C
	dlogt = log.(collect(steps_iter))[2:end]
	SC = - central_finite_difference(S, dlogt) # specific heat

	# plot Entropy
	plot!(plt_hsbm,
		collect(steps_iter),
		# 1 ./ collect(steps_iter),
		S,
		# S,
		xscale = :log10,
		legend = :topleft,
		label = S_label,
		xticks = plot_xticks,
		linestyle = :dot,
		lw = 2,
		linealpha = .6,
		# c = colors_plot[i]
	)

	# plot Specific Heat
	plot!(plt_hsbm,
		collect(steps_iter[1:length(SC)]),
		# 1 ./ collect(steps_iter[1:length(SC)]),
		SC,
		xscale = :log10,
		label = C_label,
		xticks = plot_xticks,
		lw = 2,
		linealpha = 1,
		# c = colors_plot[i]
	)

	plot([plt_hsbm, heatmap(A, yflip = true, cbar = false)]..., size = (900, 450))

	annotate!(
		1e-2,
		0.35,
		"N¹ = $(n₁)"
	)
	annotate!(
		1e-2,
		0.25,
		"N² = $(n₂)"
	)
	annotate!(
		1e-2,
		0.15,
		"k = $(k)"
	)
end

# ╔═╡ abfe1d56-3764-4fc5-bf0f-86bfdbd3b10b
md"""
here is the matrix that forms, I am trying to understand when $\tau^\star$ actually appears...
"""

# ╔═╡ 4f87bd3b-ee4f-4b2c-9111-6dc093c8f7af
function _n_modes_laplacian(λs, V, n)

	N = lastindex(λs)
	L_prime = zeros(N, N)

	# first eigenvalue should be zero anyway
	for i in 1:n+1
		L_prime += λs[n] .* (V[:, i] * V[:, i]')
	end

	return L_prime
end

# ╔═╡ c561d5f1-5abc-4f15-8f88-e281c6ef2bc2
function mat_from_decomposition(λ, v)

	N = lastindex(λ)
	L_prime = zeros(N, N)
	for i in 1:N

		L_prime += λ[i] .* (v[:, i] * v[:, i]')
		
	end

	return L_prime
end

# ╔═╡ cc9a610d-37ea-47c9-9f41-f16109c7eaab
let
	Random.seed!(123)
	# construct network
	H = 0.5
	n₁ = 16
	n₂ = 32
	k = n₁ + ((n₁*n₂-1) - n₁) * 0.01
	n = [n₁, n₂, 2]
	B = [2*n₂, 2, 1]
	N = prod(n)
	SBMvar_output = SBMvar(n, B, H, k; K = 1)
	A = SBMvar_output[1]

	# select time t
	t = 0.25

	# get laplacian
	L = Laplacian(A)
	λs, L_evecs = eigen(L)

	# laplacian operator at time t
	K = exp(- t .* L)

	# density matrix operator at time t
	ρ = density_mat(t, L)
	(μ, u) = eigen(ρ)

	# rule to generate new graph ζ
	ρ_prime = zeros(N, N)
	for i in 1:N
		for j in i+1:N
			ρ_prime[i,j] = ρ[i,j] / (minimum([ρ[i,i], ρ[j,j]]))
			ρ_prime[j,i] = ρ[i,j] / (minimum([ρ[i,i], ρ[j,j]]))
		end
	end

	# initialize new graph with separated components (defining the supernodes)
	ζ = zeros(N,N)
	for i in 1:N
		for j in i+1:N
			if (ρ_prime[i,j] - 1) ≥ 0
				ζ[i,j] = 1
				ζ[j,i] = 1
			end
		end
	end

	λplot = plot(
		1 ./ λs[2:end],
		collect(1:length(1 ./ λs[2:end])),
		xaxis = :log10, yaxis = :log10,
		seriestype = :scatter, label = ""
	)

	new_mat_plot = heatmap(ζ, yflip = true, cbar = false)
	
	old_laplac_plot = heatmap(mat_from_decomposition(λs, L_evecs), yflip = true, cbar = false)

	# L_prime = zeros()

	plot(
		[λplot, new_mat_plot]...,
		size = (900, 450)
	)

	
end

# ╔═╡ 34e1f64a-f73a-4c3f-8743-8bb49e83df14
function reduce_mat(n₁, n₂, L)

	N = n₁ * n₂ * 2
	N_prime = n₂ * 2

	# just a partition to contruct the vector
	vec = zeros(N_prime)
	vec[1] = 1
	for i in 2:N_prime
		vec[i] = vec[i-1]+n₁
	end
	vec = round.(Integer, vec)

	# construct vectors
	α_mat = zeros(N, N_prime)
	for i in 1:N_prime-1
		α_mat[vec[i]:vec[i+1]-1, i] = ones(n₁)
	end
	α_mat[vec[end]:end, end] = ones(n₁)

	# reduce matrix
	L_prime = zeros(N_prime, N_prime)
	for i in 1:N_prime
		for j in 1:N_prime
			α = α_mat[:, i]
			β = α_mat[:, j]

			L_prime[i, j] = α' * L * β
			L_prime[j, i] = β' * L * α
		end
	end

	return L_prime
end

# ╔═╡ c26e7f2d-7cbc-4330-8688-ee4da9db3a8b
let
	Random.seed!(1)
	
	# hierarchical SBM
	n = [32, 32, 2]
	B = [64, 2, 1]
	N = prod(n)
	k = 40
	H = 0.15

	# construct matrix
	SBMvar_output = SBMvar(n, B, H, k; K = 1)
	A = SBMvar_output[1]

	# find Laplacian
	L = Laplacian(A, method = :normalized)
	(λs, V) = eigen(L)

	# select n_prime (see results section for an actual calculation of this)
	n_prime = B[1]
	τ_star = 1
	
	L_prime = _n_modes_laplacian(λs, V, n_prime)
	L_second = τ_star .* L_prime

	L_reduced = - reduce_mat(n[1], n[2], L_second)
	A_prime = - L_reduced
	for i in 1:(B[1])
		A_prime[i,i] = 0
	end

	og_A_plt = heatmap(
		A, yflip = true, cbar = false, title = "A", framestyle = :box,
		xticks = [], yticks = [],
		c = cgrad([:white, :black])
	)
	new_A_plt = heatmap(
		A_prime, yflip = true, cbar = false, title = "A'",
		xticks = [], yticks = [],size=(500, 500)
	)

	global A_prime_M_3 = A_prime

	# savefig(new_A_plt, "/Users/ec627/Documents/Sussex/CompleNet/presentation/images/matrices/renormalized.png")
	
	plot(
		[og_A_plt, new_A_plt]...,
		size = (900, 450)
	)

	# plot(new_A_plt)
end

# ╔═╡ 6bfda9ba-8452-43f7-a4d8-b985f4c14500
let
	A = A_prime_M_3
	global A_primeprime_M_3 = zeros(size(A))
	
	N = size(A)[1]
	P = reduce(vcat, [repeat([i], round(Integer, N/2)) for i in 1:2])

	itra_connections = []
	inter_connections = []
	
	for i in 1:N
		for j in 1:N
			if P[i] == P[j] && i != j
				push!(itra_connections, A[i,j])
			elseif P[i] != P[j]
				push!(inter_connections, A[i,j])
			end
		end
	end

	println("Intra mean: ", mean(itra_connections))
	println("Inter mean: ", mean(inter_connections))

	a = mean(itra_connections)
	b = mean(inter_connections)
	c = 1 / (a+b)

	# construct A_primeprime_M_3 using mean inter- and intra-couplings
	for i in 1:N
		for j in 1:N
			if P[i] == P[j] && i != j
				A_primeprime_M_3[i,j] = a*c
				A_primeprime_M_3[j,i] = a*c
			elseif P[i] != P[j]
				A_primeprime_M_3[i,j] = b*c
				A_primeprime_M_3[j,i] = b*c
			end
		end
	end

	global intra_M_3 = a
	global intra_M_3 = b

	println("\na-b = $(a-b) while ac - bc = $((a*c) - (b*c))")
end

# ╔═╡ 1c0f6e8a-0f53-4fca-a37f-b30c40f384a3
let
	seedval = 123
	Random.seed!(seedval)
	K = 1
	
	# A = A_prime_M_3 .* K
	A = A_primeprime_M_3
	N = size(A)[1]
	P = reduce(vcat, [repeat([i], round(Integer, N/2)) for i in 1:2])

	# lag parameter
	β = 0.1
	α = π / 2 - β  # lag parameter in the equations
	
	# construct lag matrix
	α_mat = zeros(N, N) # phase lag matrix
	for i in 1:N
	    for j in i+1:N
			# add lag if different partition at layer 1
	        if A[i,j] != 0 && i != j
	            α_mat[i, j] = α
	            α_mat[j, i] = α
	        end
	    end
	end
	
	# oscillators parameters
	noiseQ = false
	if noiseQ
	    noise_scale = noise_scale
	else
	    noise_scale = 0
	end
	
	# simulation parameters
	sim_time = 25
	Δt = 1e-3
	steps = (0.0+Δt):Δt:sim_time
	no_steps = length(steps)
	ω = zeros(N)
	
	# storing parameters
	save_ratio = 1
	no_saves = round(Integer, no_steps / save_ratio)
	θ_now = rand(Uniform(-π, π), N)  # random init conditions
	θs = zeros(no_saves, N)
	θs[1, :] = θ_now
	
	save_counter = 1
	
	for t in 2:no_steps
	    
	    # update phases
	    θj_θi_mat = (repeat(θ_now',N) - repeat(θ_now',N)') - α_mat
	    setindex!.(Ref(θj_θi_mat), 0.0, 1:N, 1:N) # set diagonal elements to zero 
	
	    k1 = map(sum, eachrow(A .* sin.(θj_θi_mat)))
	    θ_now += Δt .* (ω + k1) + noise_scale*(rand(Normal(0,1),N))*sqrt(Δt)
	    save_counter += 1
	
	    # save θ
	    if save_counter % save_ratio == 0
	        θs[round(Integer, save_counter / save_ratio), :] = θ_now
	    end
	    
	end
	
	params = Dict(
	    "α" => α,
	    "β" => β,
	    "K" => K,
	    "A" => A,
		"ω" => ω,
	    "seedval" => seedval,
	    "Δt" => Δt,
	    "sim_time" => sim_time,
	    "noiseQ" => noiseQ,
	)

	# plt0 = plot(
	# 	steps,
	# 	macro_op(θs),
	# 	ylims = (0,1),label = ""
	# )

	plt1 = plot(
		steps,
		macro_op(θs[:, 1:round(Integer, N/2)]),
		ylims = (0,1),label = L"\rho_{1}",
		xticks = [], yticks = ([0,0.5,1], [L"0.0", L"0.5", L"1.0"]),
		ylabel = L"R_{\rho_i}",
		c = :darkorange,
		tickfontsize = 24,
		legendfontsize = 24,
		xlabelfontsize = 24,
		ylabelfontsize = 24,
		size = (900, 350),
		left_margin = 8Plots.mm,
		right_margin = 8Plots.mm,
		bottom_margin = 8Plots.mm,
		top_margin = 8Plots.mm,
	)

	# plt2 = plot(
	# 	steps,
	# 	macro_op(θs[:, round(Integer, N/2):end]),
	# 	ylims = (0,1),label = "population 2"
	# )

	plot!(
		steps,
		macro_op(θs[:, round(Integer, N/2):end]),
		ylims = (0,1),label = L"\rho_{2}",
		c = :blue,
	)
		
	# plot(
	# 	[
	# 		plt1,plt2
	# 	]...,
	# 	layout = (2,1),
	# 	size = (900, 500)
	# )
end

# ╔═╡ 02091044-b223-49d6-8bf9-13b9ed521219
(count(x -> x<0, A_prime_M_3), count(x -> x>0, A_prime_M_3))

# ╔═╡ 0380177d-d207-4001-b354-d44556578d95
(maximum(A_prime_M_3), minimum(A_prime_M_3))

# ╔═╡ 33d1e622-5b8a-4c68-9855-d31c7bafa9d8
let
	Random.seed!(123)

	# cosntruct matrix
	H = 0.3
	n₁ = 16
	n₂ = 16
	k = n₁ + ((n₁*n₂-1) - n₁) * 0.1
	n = [n₁, n₂, 2]
	B = [2*n₂, 2, 1]
	N = prod(n)
	SBMvar_output = SBMvar(n, B, H, k; K = 1)
	A = SBMvar_output[1]

	# get laplacian
	L = Laplacian(A)
	λs, L_evecs = eigen(L)
	
	# Renormalize Laplacian
	τ_star = 10^(-1.2)
	findfirst(x-> x < τ_star, 1 ./ λs)
	1 ./ λs
	# L_prime = mat_from_decomposition(λs[1:33], L_evecs)

	L_prime = - reduce_mat(n₁, n₂, L)
	A_prime = - L_prime
	for i in 1:n₂*2
			A_prime[i,i] = 0
	end

	# λplot = plot(
	# 	1 ./ λs[2:end],
	# 	collect(1:length(1 ./ λs[2:end])),
	# 	xaxis = :log10, yaxis = :log10,
	# 	label = ""
	# )
	
	original_A_plot = heatmap(A, yflip = true, cbar = false)
	new_A_plot = heatmap(A_prime, yflip = true, cbar = false)

	plot(
		[original_A_plot, new_A_plot]...,
		size = (900, 450)
	)
	
	# A_prime
end

# ╔═╡ 6601bb01-466a-4925-92fa-24c9c4274c5e
md"""
Below I calculate the spatial coherence measure $g_0(|\hat{D}|)$
"""

# ╔═╡ 08e2b6fa-0b8e-4027-962e-6b14f57a15ee
let
	function local_curvature_op(f, x)
		return f[x+1] - 2*f[x] + f[x-1]
	end

	t_span = 1:100:10000
	g_vec = []
	for t in t_span
	
		θ = mod2pi.(θs_KS_1)
		f = θ[t, :]
		vec = []
		for x in 2:(lastindex(f)-1)
			push!(vec, local_curvature_op(f, x))
		end
		
		plot(collect(1:lastindex(vec)), abs.(vec) ./ maximum(abs.(vec)), seriestype = :scatter, label = "")
	
		g_hist = fit(Histogram, sort(abs.(vec)))
		g_distr = UvBinnedDist(g_hist)
		
		plot_pdf = []
		δ = 0.01*maximum(vec)
		for i in 0:0.0001:maximum(vec)
			push!(plot_pdf, pdf(g_distr, i))
		end
		plot(plot_pdf)
	
		# maximum(vec) * 0.01
		# fieldnames(typeof(g_distr))
		# g_distr._bin_pdf
		
		push!(g_vec, pdf(g_distr, δ))
	end

	plot(g_vec, label = "", title = "spatial coherence")
end

# ╔═╡ bef11bdc-d145-4cad-9f98-d8ec1eefdeb4


# ╔═╡ Cell order:
# ╟─08f9424c-87cc-442e-9b6e-5f1951597f50
# ╟─2c3103e0-ad76-11ee-0f91-392fd41b20f4
# ╟─b199813e-ac0f-40a1-b823-dbc8d11399a7
# ╟─cb0f05ab-4b84-43e0-8e10-6486aa719645
# ╟─5e06750a-c50d-4d16-897a-0aec36bb135a
# ╟─b3fdf394-2da6-4fed-b787-db5e8460e436
# ╟─e280f836-d7b4-4694-85ff-cc5369c6af85
# ╟─81e9e58d-474c-4b3c-84e9-4e41586b6088
# ╟─4f3e3b64-4568-405f-93d3-863309b7f4df
# ╟─b37f1ede-c82c-448e-8e21-f9cdb62c1eb7
# ╟─77440c0b-490f-4279-b55d-3030ffadfecd
# ╟─090fa835-acd3-4532-b557-e527ea5e6de5
# ╟─5645229e-0cb7-4f8a-a40f-74884f642c2b
# ╟─d6902f4c-a8f1-4e13-a048-53fdf5f5a4b1
# ╟─a6e25db2-c9b5-4278-9f70-678098863bad
# ╟─67bd2260-c60d-438c-a7f5-3ac214ac792d
# ╟─db4be42d-0cb6-4737-bd4f-e38f51f76e5a
# ╟─128bd907-c1d7-4154-abb9-a090ab7712a9
# ╟─df93b0fa-2ac4-4196-8708-11aad511f743
# ╟─1e8113b3-6f7b-4113-b749-c4888e0d1d36
# ╟─1ae6c208-8512-45b4-abda-05ad8f3c174c
# ╟─5586aa3b-9614-4401-b9db-d227c6d16108
# ╟─fa47c80d-6b86-40ec-9780-760d47efaa76
# ╟─67798bae-9e1b-4467-b858-c93e1e224976
# ╟─8da0d12a-d942-4672-b677-339a0dcaf9f3
# ╟─465c625f-d6fa-4f1d-b81e-b2c6ccda65a9
# ╟─c2f7c2ce-584e-42f0-ada7-b07c51a08229
# ╟─eebc3fb4-346e-46b4-94f2-3094a2580756
# ╟─727c2c12-5188-45eb-a191-6362ef9e7a99
# ╟─b9916053-081c-48e8-88b8-b72ac6ac2fa0
# ╟─ed1167d1-3ec1-4d1d-a4a2-2a38aee53b77
# ╟─d260a2f7-5769-4b52-a71f-0b819d918c5c
# ╟─4415af2d-be77-456b-8955-931977328a38
# ╟─47c7c8be-7031-4b44-bc7e-fc76ac651ce2
# ╟─480c6796-7695-408a-aca4-cf30ded12f3d
# ╟─fef1faad-8ab6-436c-bc63-c699c10f85d7
# ╟─8fabe675-1932-4c2e-aa0c-f92b00b294b7
# ╟─1421bb08-7a75-48a5-9ab4-39ca0f4b626a
# ╟─f927e709-e51e-4744-a93f-e46e26b5c2c1
# ╟─7663302e-5483-4828-a25e-6978eb95ef66
# ╟─8442be53-fe2b-4f1c-ba39-81b7f7cb761e
# ╟─b68e3644-bfc5-4d89-8854-4c8d896c3b31
# ╟─34138921-448d-407b-a5b2-9a4459b52b65
# ╟─fd20fde7-3b04-44af-948e-732b20483a9f
# ╟─0783b6d1-825a-4312-a59e-44012d6a1d23
# ╟─c26e7f2d-7cbc-4330-8688-ee4da9db3a8b
# ╟─7b30c5bd-34ff-469b-983c-07af53d1b371
# ╟─6bfda9ba-8452-43f7-a4d8-b985f4c14500
# ╟─cd30d2fc-2801-4d32-b7c5-2871a0cd0a30
# ╠═02091044-b223-49d6-8bf9-13b9ed521219
# ╠═0380177d-d207-4001-b354-d44556578d95
# ╟─1c0f6e8a-0f53-4fca-a37f-b30c40f384a3
# ╟─7dba8a1f-7aae-450f-8e81-8ad85c0d8c8b
# ╟─f3e72899-b046-4d47-b9a3-c56db5174b66
# ╟─c15a7fe4-cf16-44e5-8d68-09c881c68a47
# ╟─224a8039-8bff-46b0-a095-c0be26f6ce04
# ╟─c54b731f-842c-47aa-a098-e81bce7f22f6
# ╟─2fbe9f94-7eba-4f8f-8c3d-9107aca6fd03
# ╟─e690de5e-b60d-4e3a-9f31-e0f0daf2388d
# ╟─6f25633b-648a-40ff-9565-8be41f89f372
# ╟─0d4fb75f-ea5b-4aec-bf1a-7d7bfa122781
# ╟─ad443a3d-dfd3-44e4-a912-f1c142d5c247
# ╟─1db6074c-7a29-4ee4-8d5b-7a59f5b71bbe
# ╟─8f38c6cc-16e7-4871-819e-67140d7cb4a1
# ╟─7228c6c5-3209-43b5-820b-aef8337c7ec1
# ╟─ec62a9ea-d2d7-4132-9767-0da16460c969
# ╟─5cf64b7c-4cd2-46bd-985a-c8af72419a44
# ╟─018add25-80fb-492b-9402-95e7f5a8b7a1
# ╟─2ba0a96e-a38f-4c91-a0e2-d8917c7babe7
# ╟─e0f47bf1-47c1-4e3d-8ed9-b09a5e8ddd37
# ╠═919c8b41-1cf8-426a-894e-c1e6b49739ae
# ╟─e7370d3e-1086-4e4c-9ecc-86cd001f75b7
# ╟─c185b2fd-1283-443d-b572-043b8426f0f5
# ╟─913e20da-66b8-4660-bc1c-a524602e6e97
# ╟─3f422269-c4c9-40b0-96b3-8eeb60d12ce7
# ╟─d069618b-e710-48ec-bb78-ac503a9e7b4a
# ╟─0cb9f939-8e26-4cd3-b6b9-84b260ee6db6
# ╟─546f983a-2db6-4c1d-be78-259fa24a0c10
# ╟─96a53aff-0df0-4ba1-a85e-656fbc66caa3
# ╟─334985f4-7349-4501-8d0d-1c4cb7e4998a
# ╠═3e4fac71-69c4-49dd-856c-dce0c3057f1c
# ╟─2726622b-147c-4d5d-9955-055b7244b67c
# ╟─ace0bc9d-2fef-4f91-b9f8-ef791058afa7
# ╟─089e4c85-023a-4c4c-94f6-f09afbf0900b
# ╟─21c40ce1-745e-4257-b775-a151feeea431
# ╟─547861de-d3fe-4950-bde4-5b2cb976d4a8
# ╟─1b18093e-3ee1-4dea-b8a5-aea0ed3a04e9
# ╟─ffe085ee-801f-439a-b61a-1d4fc5818cf4
# ╟─86b8e8f1-47a3-47ba-96ce-c596b59d1d1e
# ╟─661a5343-608d-4067-bbbd-b29b213c2688
# ╟─42848af2-779f-4894-9972-fc91170c2a07
# ╟─824dd135-5340-4ee5-8926-619be27fe8bd
# ╟─65261365-bbc7-4103-bc4d-12a6198e0792
# ╟─4fe35c9d-b971-4c53-9e9b-04c64bae520d
# ╟─f56b7c7f-1d2a-4461-b10c-21a7a453264f
# ╟─b7b01c1f-3360-4bf8-a1ab-3cb068f165d9
# ╟─ccbad8b3-4bd3-4164-97d3-435d62b96c52
# ╟─3206a25e-d386-47a3-9792-c9710405afb7
# ╟─e7265301-ef09-4a8a-8c30-1ee1904e9aee
# ╟─1fe169ab-b47a-4615-aac8-0c3e0d871eba
# ╟─7e010234-48f1-4eb4-b9ac-1b6f49a6b392
# ╟─9d651701-18a7-4629-b2da-e1e3881a61b0
# ╟─839e6c5f-281b-47d6-bd02-2635491b5bfb
# ╟─5bac2e14-23c5-4c2b-bf44-52bd3a0cb671
# ╟─659f8ecd-d37e-4706-b0bb-d45c6f8db2e5
# ╠═6b8247b1-000b-48b0-ab05-30421e0e533a
# ╠═fc29bab1-9c33-4e90-9e42-d386d57e7329
# ╟─53a88fb8-6247-49e2-99c6-4d825624341e
# ╟─5c0135a5-d378-4b7e-af40-886e03665e9c
# ╟─58e5226d-39ed-45f3-8b8f-865498e2a49e
# ╟─954a6605-f6c8-4b83-a27b-3d52fe1d7a12
# ╟─8fa9443b-acde-41de-9ba2-724d624c32e2
# ╟─0e454201-0cc7-43ee-bfcc-d17cdca79dfa
# ╟─f5f239d4-6537-4d77-a6df-a0dbc9afcdbb
# ╟─c4d4a990-5924-489f-bf5a-4e96e2c6aff8
# ╟─3d894d8a-f067-4318-a082-0624ee51d5ba
# ╟─eca7c878-63ef-4014-83ce-0992d3f49dfd
# ╟─a4de6034-00e9-48be-b984-068275677571
# ╟─3eddd420-edef-437d-9d2c-adac079e4ee9
# ╟─b7abb166-2daf-4594-9d14-068f6e0051f2
# ╟─d7c56086-4956-43c5-a9e4-f512bca1c440
# ╟─ac1ff98c-ae22-4386-a998-0717366c7c79
# ╟─d4252280-3be2-424d-979c-1295e352e80a
# ╟─4e9c1763-9062-4e31-907d-85e08861c6d5
# ╟─d7e6062c-b280-4636-8c19-fb9cc6a35e63
# ╟─7a43b227-ceeb-4238-b52e-7c57c05d81f7
# ╟─cae6447f-0a68-4cb4-822b-475b8b55c358
# ╟─7cd584d7-5330-421e-be91-8d155414a402
# ╟─15afb60a-76f0-45ae-aeb1-fe2f81c7713d
# ╟─8a1a8183-7b79-4d77-b19a-45e1195d5a9b
# ╟─84fd259e-6353-4015-b9bb-47e30e7bc039
# ╟─baac38d1-5928-4949-824d-b9d533bc402c
# ╟─7f26d0ef-3194-4981-9833-ee843da89da4
# ╟─6bb3d9a8-da2f-4b53-9461-09c1e9956a61
# ╟─934a7db6-ff49-40b5-aa81-32ae58fb6918
# ╟─4beca585-082c-43af-a966-05d76f75ec60
# ╟─ca15159b-4a52-458e-839d-c66d42c3ff23
# ╟─7820082d-8954-4104-86e1-c476d1c8b370
# ╟─e39c4f42-5947-4074-8de0-332f1a33b97f
# ╟─ad57ed62-c893-4cd5-b1b0-a01c56563134
# ╟─217e296f-b587-4d32-818a-14f96ddcab72
# ╟─0e6cfa00-cff7-4889-b70b-8bf35b6de37b
# ╟─abfe1d56-3764-4fc5-bf0f-86bfdbd3b10b
# ╟─cc9a610d-37ea-47c9-9f41-f16109c7eaab
# ╟─33d1e622-5b8a-4c68-9855-d31c7bafa9d8
# ╟─4f87bd3b-ee4f-4b2c-9111-6dc093c8f7af
# ╟─c561d5f1-5abc-4f15-8f88-e281c6ef2bc2
# ╟─34e1f64a-f73a-4c3f-8743-8bb49e83df14
# ╟─6601bb01-466a-4925-92fa-24c9c4274c5e
# ╟─08e2b6fa-0b8e-4027-962e-6b14f57a15ee
# ╠═bef11bdc-d145-4cad-9f98-d8ec1eefdeb4
