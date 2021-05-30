### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 9eb2aac0-b730-11eb-3a05-57b0c52cff00
begin
	using CSV, DataFrames, LinearAlgebra, Plots, PlutoUI, Statistics
	gr()
	TableOfContents()
end

# ╔═╡ 0256ac05-b444-4810-9d3a-474958431560
md"
# Red Wine Analysis Supplemental
### Imports and Cell Settings
Imports and cell settings hidden in next two cells
"

# ╔═╡ 37a66add-7a25-454f-81e4-f6f3599d425d
html"""<style>
main {
    max-width: 1200px;
}
"""

# ╔═╡ 50fbcf46-75d4-4cd0-949d-cf0ddca61c28
md"# Loading the data

Data source: [UCI: Wine Quality](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)
"

# ╔═╡ a4415b17-b0ce-44cd-964d-856e0cb61647
@bind wine_csv PlutoUI.FilePicker() 

# ╔═╡ b2eef98e-1db8-4259-8ee9-330a8d1f5645
md"
## Dataset Information
Reproduced from included file
```
Citation Request:
  This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
  Please include this citation if you plan to use this database:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

1. Title: Wine Quality 

2. Sources
   Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009
   
3. Past Usage:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  In the above reference, two datasets were created, using red and white wine samples. The inputs include objective tests (e.g. PH values) and the output is based on sensory data (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model these datasets under a regression approach. The support vector machine model achieved the best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T), etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity analysis procedure).
 
4. Relevant Information:

   The two datasets are related to red and white variants of the Portuguese \"Vinho Verde\" wine.
   For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
   Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

   These datasets can be viewed as classification or regression tasks.
   The classes are ordered and not balanced (e.g. there are munch more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods. 

5. Number of Instances: red wine - 1599; white wine - 4898. 

6. Number of Attributes: 11 + output attribute
  
   Note: several of the attributes may be correlated, thus it makes sense to apply some sort of feature selection.

7. Attribute information:

   For more information, read [Cortez et al., 2009].

   Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)

8. Missing Attribute Values: None
```

As stated above, the dataset contains 1599 red wines and 4898 white wines. It seems reasonable to assume that the desirable qualities of red and white wines differ. Therefore, the author's preference for red wine has provided focus to the analysis.

The dataset has been seperated into a 1299 item training set, and a 300 item testing set.
"

# ╔═╡ 2ef59fc3-9732-4374-a2f6-ddb2e378ff04
wine_df = CSV.File(wine_csv["data"]) |> DataFrame

# ╔═╡ 78546077-f8bc-4cda-85b0-b59e918db284
md"## Dataframe Description
Cursory analysis reveals a complete dataset."

# ╔═╡ 414a54e3-33ae-4047-8432-9538b7eb4bf9
describe(wine_df)

# ╔═╡ 948ca440-5d96-4d1a-b834-b1cc7a36f5fd
md"
# Histograms
## One Variable
"

# ╔═╡ 6fee6de7-23a5-44fe-96d3-a99f487faaed
begin
	gr()
	
	h1 = histogram(
		wine_df."fixed acidity",
		legend = :none,
		title = "Fixed Acidity\n(tartaric acid)",
		xlabel = "g/L",
		ylabel = "Count"
	)
	
	h2 = histogram(
		wine_df."volatile acidity",
		legend = :none,
		title = "Volatile Acidity\n(acetic acid)",
		xlabel = "g/L",
		ylabel = "Count"
	)
	
	h3 = histogram(
		wine_df."citric acid",
		legend = :none,
		title = "Citric Acid",
		xlabel = "g/L",
		ylabel = "Count"
	)
	
	h4 = histogram(
		wine_df."residual sugar",
		legend = :none,
		title = "Residual Sugar",
		xlabel = "g/L",
		ylabel = "Count"
	)
	
	h5 = histogram(
		wine_df."chlorides",
		legend = :none,
		title = "Chlorides\n(Sodium Chloride)",
		xlabel = "g/L",
		ylabel = "Count"
	)
	
	h6 = histogram(
		wine_df."free sulfur dioxide",
		legend = :none,
		title = "Free Sulfur Dioxide",
		xlabel = "mg/L",
		ylabel = "Count"
	)
	
	h7 = histogram(
		wine_df."total sulfur dioxide",
		legend = :none,
		title = "Total Sulfur Dioxide",
		xlabel = "mg/L",
		ylabel = "Count"
	)
	
	h8 = histogram(
		wine_df."density",
		legend = :none,
		title = "Density",
		xlabel = "g/mL",
		ylabel = "Count"
	)
	
	h9 = histogram(
		wine_df."pH",
		legend = :none,
		title = "pH",
		xlabel = "pH",
		ylabel = "Count"
	)
	
	h10 = histogram(
		wine_df."sulphates",
		legend = :none,
		title = "Sulphates\n(potassium sulphate)",
		xlabel = "g/L",
		ylabel = "Count"
	)
	
	h11 = histogram(
		wine_df."alcohol",
		legend = :none,
		title = "Alcohol",
		xlabel = "% Alcohol",
		ylabel = "Count"
	)
	
	h12 = histogram(
		wine_df."quality",
		legend = :none,
		title = "Quality",
		xlabel = "Score",
		ylabel = "Count"
	)
	
	plot(
		h1, h2, h3, h4, 
		h5, h6, h7, h8, 
		h9, h10, h11, h12, 
		layout = (6,2), 
		size = (800, 1400))
end

# ╔═╡ 3fff9fc5-1ad0-4225-962a-4592a4ff8eaa
begin
	hist2d_1_slider = @bind hist2d_1 html"""<input value="1" type="range" min="1" max="12"/>"""
	hist2d_2_slider = @bind hist2d_2 html"""<input value="2" type="range" min="1" max="12"/>"""
	
	
	
	md"""
	## Two Variable
	2D Histogram Attributes:\
	Attribute 1: $(hist2d_1_slider)\
	Attribute 2: $(hist2d_2_slider)
	"""
end

# ╔═╡ 34e23619-609b-41ab-b4cf-1e14f4367d80
begin
	attr1 = names(wine_df)[hist2d_1];
	attr2 = names(wine_df)[hist2d_2];

	md"""
	Attribute1: $(attr1)\
	Attribute2: $(attr2)
	"""
end

# ╔═╡ 57a06743-48d2-4bf3-b069-a08be6015b86
begin
	gr()
	
	histogram2d(
		wine_df[:, hist2d_1],
		wine_df[:, hist2d_2],
		colorbar_title = "Count",
		size = (800,600),
		title = "Combined Attribute Histogram",
		xlabel = attr1,
		ylabel = attr2,
	)
end

# ╔═╡ 299f731b-ba22-47c8-9fde-55475be8e1e2
md"
# PCA
## Prediction Label Removal
We begin by removing the quality label and transforming the dataframe to a datamatrix. This involves transposing the matrix such that each row represents an attribute.
"

# ╔═╡ ecf2ff8a-338a-45e6-a619-93df25399ab3
begin
	wine_df
	W = Matrix(wine_df[:, 1:11])'
end

# ╔═╡ dbf65d79-f867-4e1c-b4c0-114ed83c8797
k = size(W)[2]

# ╔═╡ 630ad160-5993-4d6c-90a3-446a045df7d8
md"
## MDF
The datamatrix is put into mdf by subtracting the row mean from each row.
"

# ╔═╡ 870ea699-a93a-482d-a8fd-72adffe0c470
# mdf Matrix
W_mdf = W .- mean(W, dims=2)

# ╔═╡ 504d8449-5164-4c0d-a1cf-fecad7b3c0a9
md"
## Covariance
The covariance matrix is constructed such that 

$Cov(W_{mdf}, W_{mdf}) = \frac{1}{k-1}W{_{mdf}}W{_{mdf}}{^T}$
"

# ╔═╡ ee6370ef-f35d-475d-b6d0-ca6e7f5e880e
# Covariance Matrix
W_cov = 1/(size(W_mdf)[2]-1) * W_mdf * W_mdf'

# ╔═╡ fc1103e3-7a39-4272-951f-f7d4c30a69f3
md"
## Eigenvalues and Eigenvectors
"

# ╔═╡ 411dbbbb-3c78-4598-b896-87b1587cb26d
W_eigs = eigen(W_cov)

# ╔═╡ 435aacc3-1cc6-4299-b73c-2025ce37950b
begin
	pca_bv_1_slider = @bind pca_bv_1 html"""<input value="1" type="range" min="1" max="11"/>"""
	pca_bv_2_slider = @bind pca_bv_2 html"""<input value="2" type="range" min="1" max="11"/>"""
	
	md"""
	## Projection
	Basis vectors:\
	[1:11]: $(pca_bv_1_slider)\
	[1:11]: $(pca_bv_2_slider)
	"""
end

# ╔═╡ 19000645-c156-42f9-bb50-28135915a52a
md"""
x-axis basis vector: $(pca_bv_1)\
y-axis basis vector: $(pca_bv_2)
"""

# ╔═╡ 2ac9313a-08e6-45c5-bfb2-3810a9a600d9
begin
	plotly()
	
	# Sort weighting vectors
	pca_vecs = W_eigs.vectors[:, end:-1:1]
	
	# Need to reshape vectors for grouping labels
	pca_x = reshape(pca_vecs[:, pca_bv_1]' * W_mdf, k,1)
	pca_y = reshape(pca_vecs[:, pca_bv_2]' * W_mdf, k,1)

	scatter(
		pca_x,
		pca_y,
		group=wine_df.quality,
		hovermode = "closest",
		legend = :bottomright,
		size = (800,600),
		title = "PCA Quality Scores, All Groups",
		xlabel = "Basis Vector $(pca_bv_1)",
		ylabel = "Basis Vector $(pca_bv_2)",
	)
end

# ╔═╡ 81528a2f-3973-4dae-996f-f7c0c83def2d
md"
Note: Plotly legend items can be clicked to add/remove the dataset on the plot.
You probably know this, but just in case.
"

# ╔═╡ 090ee70c-da68-4fe2-8868-7240df3866c8
md"
## Explained Variance
Plotting explained variance, 

$e(c) = \sum_{i=1}^{c} \lambda _i / \sum _{i=1}^{N} \lambda _i$ 

, shows we've reached $99.5%$ of the variance with the first two principal components. 
"

# ╔═╡ bdc55463-42a6-4f8c-b558-7f18356940c1
begin
	eigs_total = sum(W_eigs.values)
	eigs_value = zeros(length(W_eigs.values))
	
	for (index, value) in enumerate(reverse(W_eigs.values))
		if index == 1
			eigs_value[index] = value
		else
			eigs_value[index] = eigs_value[index-1] + value
		end
	end
	
	plotly()
	plot(
		1:length(W_eigs.values),
		eigs_value ./ eigs_total,
		legend = :none,
		lw = 3,
		size = (800,600),
		title = "Explained Variance",
		xlabel = "Principal Component",
		ylabel = "% Total Sum",
	)	
end

# ╔═╡ ba76af24-646b-4c54-b9c8-960493e5d1c3
md"
## Correlation Matrix
Since we have almost all the requisite components, we calculate the Correlation Matrix where 

$Corr(W_{mdf}, W_{mdf}) = \frac{W_{cov}}{\sigma_{Wmdf} * \sigma_{Wmdf}}$

It could have been useful to see correlations with the prediction target, so the quality labels were reintroduced.
"

# ╔═╡ 9404aa33-6e2c-4761-8843-c96e9693ac0c
begin
		quality_row = reshape(wine_df.quality, 1, k)
		q_mdf = quality_row .- mean(quality_row)
		
		comb_mdf = [W_mdf; q_mdf]
		comb_cov = 1/(size(comb_mdf)[2]-1) * comb_mdf * comb_mdf'
		comb_corr = comb_cov ./ (sqrt.(diag(comb_cov) * diag(comb_cov)'))
end

# ╔═╡ 88544d24-4526-45d2-9871-658b6af7bf6d
begin	
	gr()
	heatmap(
		comb_corr,
		colorbar_title = "Correlation Coefficient",
		size=(800,600), 
		tick_direction = :out,
		title = "Quality Correlation Matrix Heatmap", 
		xtickfontvalign = :top,
		xticks = (1:12, names(wine_df)),
		xrotation = 45,
		yflip = true,
		yticks = (1:12, names(wine_df)),
	)
end

# ╔═╡ 5858718e-7451-4072-9155-a9ffa51bc00e
md"
# Kernel PCA
PCA wasn't very insightful, so we continue with Kernel PCA.
"

# ╔═╡ f3a564fe-204f-4292-86ea-b088b71cbfdf
md"
## Polynomial Kernel
Given 

$\kappa(X,Y) = (X^TY + \alpha)^n$

we first choose our values for $n$ and $\alpha$:
"

# ╔═╡ 60645d09-7bf4-4031-a787-f4fbd548d8a0
begin
	kpca_n_slider = @bind kpca_n Slider(2:8)
	kpca_alpha_slider = @bind kpca_alpha Slider(1:10)
	
	md"""
	n [2:8]: $(kpca_n_slider) \
	alpha [0:10]: $(kpca_alpha_slider)
	"""
end

# ╔═╡ 918e69fe-6be9-47dd-9f2f-5e32b066518c
md"""
n: $(kpca_n) \
alpha: $(kpca_alpha)
"""

# ╔═╡ 7e252c01-b98c-415e-a253-fe067fc44428
begin
	# Preallocate dissimilarity matrix
	kpca_poly_D = zeros((k,k))
	
	for i in 1:k
		for j in 1:k
			kpca_poly_D[i,j] = (W[:,i]'*W[:,j] + kpca_alpha)^kpca_n
		end
	end
	
	# Normalize
	kpca_poly_H = I - 1/k * ones((k,k))
	
	# Normalized dissimilarity matrix
	kpca_poly_D_h = -1/2 * kpca_poly_H * kpca_poly_D * kpca_poly_H
end

# ╔═╡ 18a39244-8910-4088-9a22-945e79f9e423
eigen(kpca_poly_D_h)

# ╔═╡ df5def35-713a-48a5-a0fc-ccd0ac0ef74e
md"
## Gaussian Kernel
Given 

$\kappa(X, Y) = exp(-\gamma ||X - Y||^2)$

we first choose our $\gamma$:
"

# ╔═╡ 51d2b81d-9375-4533-9992-14d18487034b
begin
	kpca_gamma_slider = @bind kpca_gamma html"""<input value="1" type="range" min="1" max="20"/>"""
	
	md"""
	[1:10]: $(kpca_gamma_slider)
	"""
end

# ╔═╡ aad70bb9-437e-453a-9e9b-189d71d6095f
md"""
gamma, $\gamma$: $(kpca_gamma)\
"""

# ╔═╡ c0026113-2668-45e7-9114-b1a02b86e89f
begin
	# Preallocate dissimilarity matrix
	kpca_rbf_D = zeros((k,k))
	
	for i in 1:k
		for j in 1:k
			kpca_rbf_D[i,j] = exp(-kpca_gamma * norm(W[:,i]-W[:,j])^2)
		end
	end
	
	# Normalize
	kpca_rbf_H = I - 1/k * ones((k,k))
	
	# Normalized dissimilarity matrix
	kpca_rbf_D_h = -1/2 * kpca_rbf_H * kpca_rbf_D * kpca_rbf_H
end

# ╔═╡ d27abe4b-b11b-4721-a762-da3a54951628
eigen(kpca_rbf_D_h)

# ╔═╡ f200a9b4-8d31-49d7-a935-492950005112
sum(kpca_rbf_D_h[1,:])

# ╔═╡ Cell order:
# ╟─0256ac05-b444-4810-9d3a-474958431560
# ╟─9eb2aac0-b730-11eb-3a05-57b0c52cff00
# ╟─37a66add-7a25-454f-81e4-f6f3599d425d
# ╟─50fbcf46-75d4-4cd0-949d-cf0ddca61c28
# ╟─a4415b17-b0ce-44cd-964d-856e0cb61647
# ╟─b2eef98e-1db8-4259-8ee9-330a8d1f5645
# ╟─2ef59fc3-9732-4374-a2f6-ddb2e378ff04
# ╟─78546077-f8bc-4cda-85b0-b59e918db284
# ╟─414a54e3-33ae-4047-8432-9538b7eb4bf9
# ╟─dbf65d79-f867-4e1c-b4c0-114ed83c8797
# ╟─948ca440-5d96-4d1a-b834-b1cc7a36f5fd
# ╟─6fee6de7-23a5-44fe-96d3-a99f487faaed
# ╟─3fff9fc5-1ad0-4225-962a-4592a4ff8eaa
# ╟─34e23619-609b-41ab-b4cf-1e14f4367d80
# ╟─57a06743-48d2-4bf3-b069-a08be6015b86
# ╟─299f731b-ba22-47c8-9fde-55475be8e1e2
# ╠═ecf2ff8a-338a-45e6-a619-93df25399ab3
# ╟─630ad160-5993-4d6c-90a3-446a045df7d8
# ╠═870ea699-a93a-482d-a8fd-72adffe0c470
# ╟─504d8449-5164-4c0d-a1cf-fecad7b3c0a9
# ╠═ee6370ef-f35d-475d-b6d0-ca6e7f5e880e
# ╟─fc1103e3-7a39-4272-951f-f7d4c30a69f3
# ╠═411dbbbb-3c78-4598-b896-87b1587cb26d
# ╟─435aacc3-1cc6-4299-b73c-2025ce37950b
# ╟─19000645-c156-42f9-bb50-28135915a52a
# ╟─2ac9313a-08e6-45c5-bfb2-3810a9a600d9
# ╟─81528a2f-3973-4dae-996f-f7c0c83def2d
# ╟─090ee70c-da68-4fe2-8868-7240df3866c8
# ╟─bdc55463-42a6-4f8c-b558-7f18356940c1
# ╟─ba76af24-646b-4c54-b9c8-960493e5d1c3
# ╠═9404aa33-6e2c-4761-8843-c96e9693ac0c
# ╟─88544d24-4526-45d2-9871-658b6af7bf6d
# ╟─5858718e-7451-4072-9155-a9ffa51bc00e
# ╟─f3a564fe-204f-4292-86ea-b088b71cbfdf
# ╟─60645d09-7bf4-4031-a787-f4fbd548d8a0
# ╟─918e69fe-6be9-47dd-9f2f-5e32b066518c
# ╠═7e252c01-b98c-415e-a253-fe067fc44428
# ╠═18a39244-8910-4088-9a22-945e79f9e423
# ╟─df5def35-713a-48a5-a0fc-ccd0ac0ef74e
# ╟─51d2b81d-9375-4533-9992-14d18487034b
# ╟─aad70bb9-437e-453a-9e9b-189d71d6095f
# ╠═c0026113-2668-45e7-9114-b1a02b86e89f
# ╠═d27abe4b-b11b-4721-a762-da3a54951628
# ╠═f200a9b4-8d31-49d7-a935-492950005112
