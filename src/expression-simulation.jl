

import Zygote
using Flux
using Flux: @functor, onehotbatch
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
import BSON

const device = gpu
# const device = cpu


struct ExprGANGenerator
    layers
end
@functor ExprGANGenerator


Flux.params(gen::ExprGANGenerator) = params(gen.layers)


function ExprGANGenerator(ncelltypes::Int, noise_dim::Int, hidden_dim::Int, ngenes::Int)
    # layers = Chain(
    #     Dense(ncelltypes + noise_dim, hidden_dim, leakyrelu),
    #     Dense(hidden_dim, hidden_dim, leakyrelu),
    #     # Dense(hidden_dim, hidden_dim),
    #     # BatchNorm(hidden_dim, relu),
    #     Dense(hidden_dim, ngenes))

    layers = Chain(
        Dense(ncelltypes + noise_dim, 30, leakyrelu),
        Dense(30, 40, leakyrelu),
        Dense(40, 50, leakyrelu),
        Dense(50, 40, leakyrelu),
        Dense(40, 30, leakyrelu),
        Dense(30, ngenes))

    return ExprGANGenerator(layers)
end


function (gen::ExprGANGenerator)(labels, noise)
    return gen.layers(cat(labels, noise, dims=1))
end


struct ExprGANDiscriminator
    layers
end
@functor ExprGANDiscriminator


Flux.params(dis::ExprGANDiscriminator) = params(dis.layers)


function ExprGANDiscriminator(ncelltypes::Int, ngenes::Int, hidden_dim::Int)
    # layers = Chain(
    #     Dense(ncelltypes + ngenes, hidden_dim, relu),
    #     Dense(hidden_dim, hidden_dim, relu),
    #     # Dense(hidden_dim, hidden_dim),
    #     # BatchNorm(hidden_dim, relu),
    #     Dense(hidden_dim, 1))

    # layers = Chain(
    #     Dense(ncelltypes + ngenes, 128, relu),
    #     # Dropout(0.5),
    #     Dense(128, 64, relu),
    #     # Dropout(0.5),
    #     Dense(64, 32, relu),
    #     Dense(32, 16, relu),
    #     Dense(16, 8, relu),
    #     Dense(8, 1))

    layers = Chain(
        Dense(ncelltypes + ngenes, 512, leakyrelu),
        Dense(512, 256, leakyrelu),
        Dense(256, 128, leakyrelu),
        Dense(128, 1))

    return ExprGANDiscriminator(layers)
end


function (dis::ExprGANDiscriminator)(labels, x)
    return dis.layers(cat(labels, x, dims=1))
end


function expr_gan_discriminator_loss(real_disc, fake_disc)
    real_loss = Flux.Losses.logitbinarycrossentropy(real_disc, 1f0)
    fake_loss = Flux.Losses.logitbinarycrossentropy(fake_disc, 0f0)
    return real_loss + fake_loss
end


function expr_gan_train_discriminator(
        dis_opt, dis::ExprGANDiscriminator,
        labels::AbstractArray, x::AbstractArray, x̄::AbstractArray)
    ps = params(dis)
    loss, back = Zygote.pullback(ps) do
        real_disc = dis(labels, x)
        fake_disc = dis(labels, x̄)
        return expr_gan_discriminator_loss(real_disc, fake_disc)
    end
    grads = back(1f0)
    update!(dis_opt, ps, grads)
    return loss
end

Zygote.@nograd expr_gan_train_discriminator


function expr_gan_generator_loss(
        x::AbstractArray, x̃::AbstractArray,
        fake_disc::AbstractArray, λ::Float32)
    gan_loss = logitbinarycrossentropy(fake_disc, 1f0)
    # l1_loss = mae(real_imgs, fake_imgs)
    # l1_loss = mse(real_imgs, fake_imgs)

    return gan_loss # + (λ * l1_loss)
end


"""
Take one gradient descent step on one batch of training data.
"""
function train_step(
        gen_opt, dis_opt,
        gen::ExprGANGenerator, dis::ExprGANDiscriminator,
        labels::AbstractArray, x::AbstractArray, noise_dim::Int;
        λ::Float32=1f0)

    ncells = size(labels, 2)
    noise = randn(Float32, (noise_dim, ncells)) |> device

    ps = params(gen)
    loss = Dict()
    loss["gen"], back = Zygote.pullback(ps) do
        x̃ = gen(labels, noise)
        loss["dis"] = expr_gan_train_discriminator(dis_opt, dis, labels, x, x̃)
        return expr_gan_generator_loss(x, x̃, dis(labels, x̃), λ)
    end
    grads = back(1f0)
    update!(gen_opt, ps, grads)
    return loss
end


function fit_expression_model_cgan(
        input_h5ad_filename::String,
        output_params_filename::String;
        seed::Int=0, nepochs::Int=10000,
        noise_dim::Int=40, hidden_dim::Int=80,
        maxcells::Int=20000)

    adata = read(input_h5ad_filename, AnnData)

    ncelltypes = maximum(adata.obs.label .+ 1)
    ngenes, ncells = size(adata.X)

    p = shuffle(1:ncells)[1:min(maxcells, ncells)]

    x =Matrix{Float32}(adata.X)[:,p] |> device
    labels = device(Matrix{Float32}(onehotbatch(adata.obs.label[p] .+ 1, 1:ncelltypes)))

    @show size(x)

    gen = ExprGANGenerator(ncelltypes, noise_dim, hidden_dim, ngenes) |> device
    dis = ExprGANDiscriminator(ncelltypes, ngenes, hidden_dim) |> device

    # gen_opt = ADAM(2e-4, (0.5, 0.999))
    # dis_opt = ADAM(2e-4, (0.5, 0.999))
    gen_opt = ADAM(1e-4, (0.5, 0.999))
    dis_opt = ADAM(1e-4, (0.5, 0.999))

    # TODO: I should try doing batches. May actaully help things.

    for epoch in 1:nepochs
        loss = train_step(gen_opt, dis_opt, gen, dis, labels, x, noise_dim)
        if epoch % 250 == 0
            println("Epoch: ", epoch)
            @show (loss["gen"], loss["dis"])
        end
    end

    gen_cpu = gen |> cpu
    dis_cpu = dis |> cpu
    BSON.@save output_params_filename noise_dim gen_cpu dis_cpu
end


function sample_expression_model_cgan(
        input_params_filename::String,
        input_h5ad_filename::String,
        output_h5ad_filename::String)

    # noise_dim_, gen_cpu_, dis_cpu_ = BSON.load(input_params_filename, @__MODULE__)
    # noise_dim_, gen_cpu_, dis_cpu_ = BSON.load(input_params_filename, @__MODULE__)
    # @show typeof(last(noise_dim_))
    # @show typeof(last(gen_cpu_))
    # @show typeof(last(dis_cpu_))

    values = BSON.load(input_params_filename, @__MODULE__)
    noise_dim = values[:noise_dim]
    gen = values[:gen_cpu] |> device

    # for (key, val) in BSON.load(input_params_filename, @__MODULE__)
    #     if key == :noise_dim
    #         noise_dim = val
    #     elseif key == :gen_cpu
    #         gen = val |> device
    #     end
    # end

    # noise_dim = last(noise_dim_)
    # gen = last(gen_cpu_) |> device

    adata = read(input_h5ad_filename, AnnData)
    ncells = size(adata.X, 2)
    ncelltypes = maximum(adata.obs.label .+ 1)

    noise = randn(Float32, (noise_dim, ncells)) |> device
    labels = device(Matrix{Float32}(onehotbatch(adata.obs.label .+ 1, 1:ncelltypes)))

    x̄ = gen(labels, noise) |> cpu

    ngenes = size(x̄, 1)

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
