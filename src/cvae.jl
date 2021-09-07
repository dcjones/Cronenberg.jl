
import Zygote
using Flux
using Flux: @functor, onehotbatch
using Flux.Optimise: update!
import BSON

const device = gpu
# const device = cpu


struct Encoder
    shared_layers
    μ_layer
    logσ_layer
end

@functor Encoder

Flux.params(enc::Encoder) = params(enc.layers)


function Encoder(ngenes::Int, ncelltypes::Int, hidden_dim::Int, z_dim::Int)
    shared_layers = Chain(
        Dense(ngenes + ncelltypes, hidden_dim, leakyrelu),
        Dense(hidden_dim, hidden_dim, leakyrelu))

    μ_layer = Dense(hidden_dim, z_dim)
    logσ_layer = Dense(hidden_dim, z_dim)

    return Encoder(shared_layers, μ_layer, logσ_layer)
end


function (enc::Encoder)(x, labels)
    h = enc.shared_layers(cat(x, labels, dims=1))
    return (enc.μ_layer(h), enc.logσ_layer(h))
end

struct Decoder
    layers
end

@functor Decoder

Flux.params(dec::Decoder) = params(dec.layers)


function Decoder(z_dim::Int, ncelltypes::Int, hidden_dim::Int, ngenes::Int)
    layers = Chain(
        Dense(z_dim + ncelltypes, hidden_dim, leakyrelu),
        Dense(hidden_dim, hidden_dim, leakyrelu),
        Dense(hidden_dim, ngenes))
    return Decoder(layers)
end


function (dec::Decoder)(z, labels)
    return dec.layers(cat(z, labels, dims=1))
end


function train_step(enc, dec, opt, x, labels)
    ps = params(enc, dec)
    loss, back = Zygote.pullback(ps) do
        μ, logσ = enc(x, labels)
        σ = exp.(logσ) .+ 1f-6
        z = device(randn(Float32, size(μ))) .* σ
        x̄ = dec(z, labels)

        logp = -sum((x .- x̄).^2)
        kl = 0.5f0 * sum(σ.^2 + μ.^2 .- 1f0 .- 2f0 .* logσ)
        elbo = logp - kl

        return -elbo
    end
    grads = back(1f0)
    update!(opt, ps, grads)
    return loss
end


function fit_expression_model_cvae(
        input_h5ad_filename::String,
        output_params_filename::String;
        seed::Int=0, nepochs::Int=10000,
        hidden_dim::Int=80, z_dim::Int=10)

    adata = read(input_h5ad_filename, AnnData)

    x = device(Matrix{Float32}(adata.X))
    ncelltypes = maximum(adata.obs.label .+ 1)
    labels = device(Matrix{Float32}(onehotbatch(adata.obs.label .+ 1, 1:ncelltypes)))
    ngenes, ncells = size(x)

    enc = Encoder(ngenes, ncelltypes, hidden_dim, z_dim) |> device
    dec = Decoder(z_dim, ncelltypes, hidden_dim, ngenes) |> device
    opt = ADAM(1e-3, (0.9,  0.999))

    for epoch in 1:nepochs
        neg_elbo = train_step(enc, dec, opt, x, labels)
        if epoch % 100 == 0
            println("Epoch: ", epoch, "  elbo: ", -neg_elbo)
        end
    end

    # TODO: do debug this I want to pass some data through the decoder
    # and plot the latent space

    μ, logσ = enc(x, labels)
    σ = exp.(logσ) .+ 1f-6
    z = device(randn(Float32, size(μ))) .* σ
    x̄ = dec(z, labels)

    open("debugging-cvae.csv", "w") do output
        labels_oc = Flux.onecold(labels)
        println(output, "label,z1,z2,μ1,logσ1")
        for i in 1:ncells
            println(output, labels_oc[i], ",", z[1,i], ",", z[2,i], ",", μ[1,i], ",", logσ[1,i])
        end
    end

    dec_cpu = cpu(dec)
    BSON.@save output_params_filename z_dim dec_cpu
end


function sample_expression_model_cvae(
        input_params_filename::String,
        input_h5ad_filename::String,
        output_h5ad_filename::String)

    z_dim_, dec_cpu_ = BSON.load(input_params_filename, @__MODULE__)
    z_dim = last(z_dim_)
    dec = last(dec_cpu_) |> device

    @show z_dim
    adata = read(input_h5ad_filename, AnnData)
    ncells = size(adata.X, 2)
    @show ncells
    ncelltypes = maximum(adata.obs.label .+ 1)
    labels = device(Matrix{Float32}(onehotbatch(adata.obs.label .+ 1, 1:ncelltypes)))

    z = randn(Float32, (z_dim, ncells)) |> device
    x̄ = dec(z, labels) |> cpu
    ngenes = size(x̄, 1)
    @show ngenes

    var = DataFrame(
        "_index" => String[string(i-1) for i in 1:ngenes])

    adata = AnnData(
        x̄,
        adata.obsm,
        adata.obsp,
        adata.uns,
        adata.obs,
        var)

    write(output_h5ad_filename, adata)
end

# # This is just a wrapper for the python code. It might be nice to rewrite this
# # in julia someday, but it would be a pain and probably end up running slower.


# function fit_expression_model(
#         input_h5ad_filename::String,
#         output_params_filename::String, seed::Int=0)

#     adata = anndata_py.read_h5ad(input_h5ad_filename)
#     cvae_py.fit_cvae(
#         adata, adata.obs.label, output_params_filename, seed=seed)
# end


# function sample_expression_model(
#         input_params_filename::String,
#         input_h5ad_filename::String,
#         output_h5ad_filename::String)

#     adata = anndata_py.read_h5ad(input_h5ad_filename)
#     cvae_py.sample_cvae(
#         adata, adata.obs.label-1, input_params_filename, output_h5ad_filename)
# end
