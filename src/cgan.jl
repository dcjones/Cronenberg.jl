# Experimenting with fitting a cgan to try to generate realistic looking images.

using Colors
import ColorSchemes
import CSV
import JSON
import LightXML
import MultivariateStats
import TiffImages
import Zygote
using ColorVectorSpace
using Flux
using Flux.Losses: logitbinarycrossentropy, mae
using Flux.Optimise: update!
using Statistics

const device = gpu
# const device = cpu


"""
Match up cells json files to their accompanying tiff files. Not every tiff file
necessarily has a json file. It can be missing if there are no cells detected.
"""
function match_cells_imgs_filenames(
        cells_filenames::Vector{String},
        imgs_filenames::Vector{String})

    pat = r"R\d+_X\d+_Y\d+"

    cells_filenames_dict = Dict{String, String}(
        match(pat, filename).match => filename for filename in cells_filenames)
    imgs_filenames_dict = Dict{String, String}(
        match(pat, filename).match => filename for filename in imgs_filenames)

    @assert isempty(setdiff(keys(cells_filenames), keys(imgs_filenames)))

    extra_imgs = setdiff(keys(imgs_filenames), keys(cells_filenames))
    if !isempty(extra_imgs)
        println("Tiff files without accompanying .cells.json files:")
        for key in sort(extra_imgs)
            println("  ", imgs_filenames[key])
        end
    end

    cells_imgs_filenames = Tuple{String, String}[]
    for key in intersect(keys(cells_filenames_dict), keys(imgs_filenames_dict))
        push!(cells_imgs_filenames, (cells_filenames_dict[key], imgs_filenames_dict[key]))
    end

    return cells_imgs_filenames
end


"""
Read the XML metadata stored in OME TIFF files produced by the hubmap cytokit
pipeline. This gives us the channel names, and for each layer, its channel and
focal plane.
"""
function read_ome_tiff_description(img::TiffImages.AbstractTIFF)
    doc = LightXML.parse_string(first(img.ifds)[TiffImages.IMAGEDESCRIPTION].data)
    root = LightXML.root(doc)
    imgdesc = LightXML.find_element(LightXML.find_element(root, "Image"), "Pixels")

    nchannels = parse(Int, LightXML.attribute(imgdesc, "SizeC"))
    nfocuses = parse(Int, LightXML.attribute(imgdesc, "SizeZ"))
    channel_names = Vector{String}(undef, nchannels)
    channel_id_pat = r"Channel:(\d+):(\d+)"

    layer_channel = fill(typemin(Int), nchannels*nfocuses)
    layer_focus = fill(typemin(Int), nchannels*nfocuses)

    for node in LightXML.child_nodes(imgdesc)
        if !LightXML.is_elementnode(node)
            continue
        end

        el = LightXML.XMLElement(node)

        if LightXML.name(el) == "Channel"
            mat = match(channel_id_pat, LightXML.attribute(el, "ID"))
            @assert mat.captures[1] == "0"
            channel_names[parse(Int, mat.captures[2])+1] = LightXML.attribute(el, "Name")
        elseif LightXML.name(el) == "TiffData"
            i = parse(Int, LightXML.attribute(el, "IFD")) + 1
            channel = parse(Int, LightXML.attribute(el, "FirstC"))
            focus = parse(Int, LightXML.attribute(el, "FirstZ"))
            layer_channel[i] = channel
            layer_focus[i] = focus
        end
    end

    for i in 1:length(layer_channel)
        if layer_channel[i] == typemin(Int)
            error("Missing channel information for layer $(i)")
        end
        if layer_focus[i] == typemin(Int)
            error("Missing focus information for layer $(i)")
        end
    end

    return channel_names, layer_channel, layer_focus
end


"""
Read the optimal focal plane from the  hubmap/cytokit data.json file.
"""
function read_focal_planes(data_filename::String)
    data = open(data_filename) do input
        JSON.parse(input)
    end
    focal_planes = data["focal_plane_selector"]

    z_best = Dict{NamedTuple{(:r, :x, :y), Tuple{Int, Int, Int}}, Int}()
    for focal_plane in focal_planes
        tile = (r=focal_plane["region_index"]+1, x=focal_plane["tile_x"]+1, y=focal_plane["tile_y"]+1)
        @assert !haskey(z_best, tile)
        z_best[tile] = focal_plane["best_z"]
    end

    return z_best
end


"""
Read cell type labels from a csv file with at least a `cell_id` and `label` column.
"""
function read_labels(labels_filename::String)
    labels_df = CSV.read(labels_filename, DataFrame)
    labels = Dict{String, Int}()
    ncelltypes = maximum(labels_df.label) + 1

    for row in eachrow(labels_df)
        labels[row.cell_id] = row.label + 1
    end

    return labels, ncelltypes
end


function select_focal_planes(img::Array{Float32, 3}, layer_focus, best_focus)
    # throw out channels that are suboptimal focal lengths
    channel_index = Int[]
    # for (i, (channel, focus)) in enumerate(zip(layer_channel, layer_focus))
    for (i, focus) in enumerate(layer_focus)
        if best_focus == focus
            push!(channel_index, i)
        end
    end

    if isempty(channel_index)
        error("""
        No usable layers found. This is probably some kind of bug or a mismatch
        between the tiff files and the 'data.json' file.
        """)
    end

    img = img[:, :, channel_index]

    return img
end


"""
Build CGAN training examples from cytokit processed CODEX data. Includeing
`cells.json` files giving segmentation and expression data, TIFF files for each
tile, and cell type labels.
"""
function make_cgan_tile_training_examples(
        cells_filenames::Vector{String},
        imgs_filenames::Vector{String},
        data_filename::String,
        labels_filename::String;
        random_crops::Bool=true,
        crop_width::Int=256,
        crop_height::Int=256,
        crop_coverage::Int=1,
        min_crop_masked_prop::Float64=0.05,
        max_images::Int=typemax(Int),
        h5_output_filename::Union{Nothing, String}="training-examples.h5",
        png_output_path::Union{Nothing, String}=nothing)

    z_best = read_focal_planes(data_filename)
    labels, ncelltypes = read_labels(labels_filename)

    pat = r"R(\d+)_X(\d+)_Y(\d+)"

    training_examples = Tuple{Array{Float32, 3}, Array{Float32, 3}}[]

    cells_imgs_filenames = match_cells_imgs_filenames(cells_filenames, imgs_filenames)

    channel_names = nothing

    count = 0
    for (cells_filename, imgs_filename) in cells_imgs_filenames
        count += 1
        if count > max_images
            break
        end

        cells_mat = match(pat, cells_filename)
        imgs_mat = match(pat, imgs_filename)
        @assert cells_mat.captures == imgs_mat.captures

        tile_r = parse(Int, imgs_mat.captures[1])
        tile_x = parse(Int, imgs_mat.captures[2])
        tile_y = parse(Int, imgs_mat.captures[3])
        tile = (r=tile_r, x=tile_x, y=tile_y)

        println("Reading ", imgs_filename)
        tiffimg = TiffImages.load(imgs_filename)
        img_channel_names, layer_channel, layer_focus = read_ome_tiff_description(tiffimg)

        if channel_names === nothing
            channel_names = img_channel_names
        else
            @assert channel_names == img_channel_names
        end

        img = select_focal_planes(
            Array{Float32}(tiffimg.data),
            layer_focus, z_best[tile])

        height, width, nchannels = size(img)
        mask = zeros(Float32, (height, width, ncelltypes))

        celldata = open(cells_filename) do input
            JSON.parse(input)
        end

        for (cellnum, cell) in celldata
            cell_id = @sprintf(
                "R%s_X%s_Y%s_cell%s",
                cells_mat.captures[1], cells_mat.captures[2], cells_mat.captures[3], cellnum)
            label = labels[cell_id]

            poly = [(Float32(x), Float32(y)) for (x, y) in cell["poly"]]
            draw_polygon!(mask, poly, label)
        end

        if random_crops
            # TODO: it would be cool to also do e.g. random rotations, reflections, etc.
            ncrops = round(Int, crop_coverage * (width * height) / (crop_width * crop_height), RoundUp)
            for _ in 1:ncrops
                x = rand(1:(width - crop_width + 1))
                y = rand(1:(height - crop_height + 1))

                mask_crop = mask[y:y+crop_height-1, x:x+crop_width-1, :]
                img_crop = img[y:y+crop_height-1, x:x+crop_width-1, :]

                # Don't bother training on examples with few or any cells
                # (may have to revisit this, but we probably want to at least
                # control how much empty training data we have)
                if sum(mask_crop)/(crop_height*crop_width) > min_crop_masked_prop
                    push!(training_examples, (img_crop, mask_crop))
                end
            end
        else
            push!(training_examples, (img, mask))
        end
    end

    if isempty(training_examples)
        error("No training examples. Possibly the input images are too sparsely populated.")
    end

    if png_output_path !== nothing
        write_training_examples_png(png_output_path, training_examples)
    end

    if h5_output_filename !== nothing
        write_training_examples_hdf5(h5_output_filename, training_examples, channel_names)
    end

    return training_examples
end


"""
Make 64x64 images of cells to train on.
"""
function make_cgan_cell_training_examples(
        cells_filenames::Vector{String},
        imgs_filenames::Vector{String},
        data_filename::String,
        labels_filename::String;
        ex_width::Int=64, ex_height::Int=64,
        max_images::Int=typemax(Int),
        h5_output_filename::Union{Nothing, String}="cell-training-examples.h5",
        png_output_path::Union{Nothing, String}=nothing)

    z_best = read_focal_planes(data_filename)
    labels, ncelltypes = read_labels(labels_filename)

    pat = r"R(\d+)_X(\d+)_Y(\d+)"

    training_examples = Tuple{Array{Float32, 3}, Array{Float32, 3}}[]

    cells_imgs_filenames = match_cells_imgs_filenames(cells_filenames, imgs_filenames)

    channel_names = nothing

    count = 0
    for (cells_filename, imgs_filename) in cells_imgs_filenames
        if count > max_images
            break
        end

        cells_mat = match(pat, cells_filename)
        imgs_mat = match(pat, imgs_filename)
        @assert cells_mat.captures == imgs_mat.captures

        tile_r = parse(Int, imgs_mat.captures[1])
        tile_x = parse(Int, imgs_mat.captures[2])
        tile_y = parse(Int, imgs_mat.captures[3])
        tile = (r=tile_r, x=tile_x, y=tile_y)

        println("Reading ", imgs_filename)
        tiffimg = TiffImages.load(imgs_filename)
        img_channel_names, layer_channel, layer_focus = read_ome_tiff_description(tiffimg)

        if channel_names === nothing
            channel_names = img_channel_names
        else
            @assert channel_names == img_channel_names
        end

        img = select_focal_planes(
            Array{Float32}(tiffimg.data),
            layer_focus, z_best[tile])

        height, width, nchannels = size(img)

        celldata = open(cells_filename) do input
            JSON.parse(input)
        end

        for (cellnum, cell) in celldata
            count += 1
            if count > max_images
                break
            end

            cell_id = @sprintf(
                "R%s_X%s_Y%s_cell%s",
                cells_mat.captures[1], cells_mat.captures[2], cells_mat.captures[3], cellnum)
            label = labels[cell_id]

            poly = [(Float32(x), Float32(y)) for (x, y) in cell["poly"]]

            xmin, xmax = extrema([x for (x, y) in poly])
            ymin, ymax = extrema([y for (x, y) in poly])

            @assert xmax - xmin + 1 <= ex_width
            @assert ymax - ymin + 1 <= ex_height

            xoff = round(Int, xmin - (ex_width - (xmax - xmin))/2 - 1)
            yoff = round(Int, ymin - (ex_height - (ymax - ymin))/2 - 1)

            offset_polygon = [(x-xoff, y-yoff) for (x,y) in poly]

            cell_mask = zeros(Float32, (ex_height, ex_width, ncelltypes))
            draw_polygon!(cell_mask, offset_polygon, label)

            cell_img = zeros(Float32, (ex_height, ex_width, nchannels))
            copy_polygon!(cell_img, img, poly, xoff, yoff)

            push!(training_examples, (cell_img, cell_mask))
        end
    end

    if isempty(training_examples)
        error("No training examples. Possibly the input images are too sparsely populated.")
    end

    if png_output_path !== nothing
        write_training_examples_png(png_output_path, training_examples)
    end

    if h5_output_filename !== nothing
        write_training_examples_hdf5(h5_output_filename, training_examples, channel_names)
    end

    return training_examples
end


"""
Dump training examples to one big hdf5 file.
"""
function write_training_examples_hdf5(
        filename::String,
        training_exampes::Vector{Tuple{Array{Float32, 3}, Array{Float32, 3}}},
        channel_names::Vector{String})

    n = length(training_exampes)
    output = h5open(filename, "w")

    imgs = cat([img for (img, mask) in training_exampes]..., dims=4)
    masks = cat([mask for (img, mask) in training_exampes]..., dims=4)

    output["imgs"] = imgs
    output["masks"] = masks
    output["channel_names"] = channel_names

    # Or we could write each to a seperate dataset, which might save memory
    # but will be less convenient.

    # img_group = create_group(output, "img")
    # mask_group = create_group(output, "mask")

    # for (i, (img, mask))  in enumerate(training_exampes)
    #     key = @sprintf("%09d", i)
    #     img_group[key] = img
    #     mask_group[key] = mask
    # end

    close(output)
end


"""
Dump training data to a bunch of png images so we can inspect it.
"""
function write_training_examples_png(
        path::String, training_exampes::Vector{Tuple{Array{Float32, 3}, Array{Float32, 3}}})
    n = length(training_exampes)
    mkpath(path)
    for (i, (img, mask)) in enumerate(training_exampes)
        img = sum(img, dims=3)[:,:,1]
        img ./= max(1f0, maximum(img))
        img = (c -> RGB{Float64}(c,c,c)).(img)
        mask = false_color_mapped(mask)

        open(joinpath(path, @sprintf("img-%09d.png", i)), "w") do output
            save(Images.Stream{format"PNG"}(output), cat(mask, img, dims=2))
        end

        # for k in 1:size(img, 3)
        #     open(joinpath(path, @sprintf("img-%09d-%03d.png", i, k)), "w") do output
        #         # save(Images.Stream{format"PNG"}(output), img[:,:,k] ./ maximum(img[:,:,k]))
        #         save(Images.Stream{format"PNG"}(output), img[:,:,k])
        #     end
        # end

        # for k in 1:size(mask, 3)
        #     open(joinpath(path, @sprintf("mask-%09d-%03d.png", i, k)), "w") do output
        #         save(Images.Stream{format"PNG"}(output), mask[:,:,k])
        #     end
        # end
    end
end


"""
Scan line polygon filling algorithm. This is used for generating a mask image that
one-hot encodes each pixel with the cell type (if any) it belongs to.
"""
function draw_polygon!(
        img::Array{Float32, 3}, poly::Vector{Tuple{Float32, Float32}}, fill::Int)

    ymin, ymax = extrema([round(Int, p[2]) for p in poly])
    n = length(poly)

    xintersections = Int[]

    function push_if_changed!(xintersections::Vector{Int}, x::Int)
        if isempty(xintersections) || xintersections[end] != x
            push!(xintersections, x)
        end
    end

    for y in ymin:ymax
        empty!(xintersections)
        for i in 1:n-1
            x1, y1 = poly[i]
            x2, y2 = poly[i+1]

            if y < min(y1, y2) || y > max(y1, y2)
                continue
            end

            if x1 == x2 # vertical line
                push_if_changed!(xintersections, round(Int, x1))
            elseif y1 == y2 # horizontal line
                # This makes no real difference whether we include it or not
                # push_if_changed!(xintersections, round(Int, x1))
                # push_if_changed!(xintersections, round(Int, x2))
            else
                slope = (y2 - y1) / (x2 - x1)
                x = x1 + (y - y1) / slope
                push_if_changed!(xintersections, round(Int, x))
            end
        end

        sort!(xintersections)

        # drawing outline (for debugging)
        # for x in xintersections
        #     img[y+1, x+1, fill] = 1f0
        # end

        for i in 1:2:length(xintersections)-1
            x1, x2 = xintersections[i], xintersections[i+1]

            for x in x1:x2
                img[y+1, x+1, fill] = 1f0
            end
        end
    end
end


"""
Scan line polygon filling algorithm but copying from one array to another rather
than setting to some fill.
"""
function copy_polygon!(
        dest::Array{Float32, 3}, src::Array{Float32, 3},
        poly::Vector{Tuple{Float32, Float32}}, xoff::Int, yoff::Int)

    ymin, ymax = extrema([round(Int, p[2]) for p in poly])
    n = length(poly)

    xintersections = Int[]

    function push_if_changed!(xintersections::Vector{Int}, x::Int)
        if isempty(xintersections) || xintersections[end] != x
            push!(xintersections, x)
        end
    end

    for y in ymin:ymax
        empty!(xintersections)
        for i in 1:n-1
            x1, y1 = poly[i]
            x2, y2 = poly[i+1]

            if y < min(y1, y2) || y > max(y1, y2)
                continue
            end

            if x1 == x2 # vertical line
                push_if_changed!(xintersections, round(Int, x1))
            elseif y1 == y2 # horizontal line
                # This makes no real difference whether we include it or not
                # push_if_changed!(xintersections, round(Int, x1))
                # push_if_changed!(xintersections, round(Int, x2))
            else
                slope = (y2 - y1) / (x2 - x1)
                x = x1 + (y - y1) / slope
                push_if_changed!(xintersections, round(Int, x))
            end
        end

        sort!(xintersections)

        for i in 1:2:length(xintersections)-1
            x1, x2 = xintersections[i], xintersections[i+1]

            for x in x1:x2
                dest[y+1-yoff, x+1-xoff, :] = src[y+1, x+1, :]
            end
        end
    end
end


# Normalize each depth channel (3rd dimension) to lie in [-1, 1]
function normalize!(xs)
    # lower = minimum(xs, dims=(1,2,4))
    # upper = maximum(xs, dims=(1,2,4))

    # TODO: What if we don't do per-channel levels
    lower = minimum(xs, dims=(1,2,3,4))
    upper = maximum(xs, dims=(1,2,3,4))

    xs .-= lower
    xs ./= (upper .- lower)./2
    xs .-= 1f0

    # TODO: Let's keep track of the transformation so we can invert it when
    # generating images for display.
end


function read_training_data(training_data_filename::String, batchsize::Int)
    input = h5open(training_data_filename)

    masks = read(input["masks"])
    imgs = read(input["imgs"])
    channel_names = read(input["channel_names"])

    imgs .-= 0.5f0
    masks .*= 2f0
    masks .-= 1f0

    # TODO: Let's find the most variable channels and map those to RGB

    # normalize!(masks)
    # normalize!(imgs)

    trainingdata = Flux.Data.DataLoader((masks, imgs), batchsize=batchsize)

    close(input)

    return trainingdata, channel_names, masks, imgs
end


# U-Net downsample block
# Applies a `filters`-by-`filters` convolution changing input depth `indepth` to
# output depth `outdepth`
function downsample(filters::Int, indepth::Int, outdepth::Int; batchnorm=true, activate=true)
    layers = Any[]
    push!(layers, Conv(
        (filters, filters),
        indepth => outdepth,
        stride=2,
        # pad=SamePad(),
        pad=1,
        init=(dims...) -> 0.02f0 .* randn(Float32, dims...),
        bias=false))

    if batchnorm
        push!(layers, BatchNorm(outdepth))
    end

    if activate
        push!(layers, x -> leakyrelu.(x))
    end

    return Chain(layers...)
end


# U-Net upsample block
# Does transpose convolutions to double the size of the input.
function upsample(filters::Int, indepth::Int, outdepth::Int; dropout=false)
    layers = Any[]
    push!(layers, ConvTranspose(
        (filters, filters),
        indepth => outdepth,
        stride=2,
        pad=SamePad(),
        init=(dims...) -> 0.02f0 .* randn(Float32, dims...),
        bias=false))

    push!(layers, BatchNorm(outdepth))

    if dropout
        push!(layers, Dropout(0.5))
    end

    push!(layers, x -> relu.(x))

    return Chain(layers...)
end


struct Discriminator
    downsample_block
    patch_block
    output_layer
end


function Discriminator(maskdepth::Int, imgdepth::Int)
    # mask and image are concatenated to get an input of [256, 256, maskdepth+imgdepth, B]
    downsample_block = Chain(
        downsample(4, maskdepth+imgdepth, 64, batchnorm=false), # [128, 128, 64, B]
        downsample(4, 64, 128), # [64, 64, 128, B]
        downsample(4, 128, 256)) |> device # [32, 32, 256, B]

    # pad with zeros to get [34, 34, 256, B]

    patch_block = Chain(
        Conv(
            (4,4),
            256 => 512,
            stride=1,
            init=(dims...) -> 0.02f0 .* randn(Float32, dims...),
            bias=false), # [31, 31, 512, B]
        BatchNorm(512), # [31, 31, 512, B]
        x -> leakyrelu.(x)) |> device

    # pad with zeros to get [33, 33, 512, B]

    output_layer = Conv(
        (4, 4), 512 => 1,  stride=1,
        init=(dims...) -> 0.02f0 .* randn(Float32, dims...)) |> device # [30, 30, 1, B]

    return Discriminator(downsample_block, patch_block, output_layer)
end


function (disc::Discriminator)(mask::AbstractArray, img::AbstractArray)
    input = cat(mask, img, dims=3)
    input = disc.downsample_block(input) # [32, 32, 512, B]
    input = pad_zeros(input, (1, 1, 1, 1, 0, 0, 0, 0)) # [34, 34, 512, B]
    input = disc.patch_block(input) # [31, 31, 512, B]
    input = pad_zeros(input, (1, 1, 1, 1, 0, 0, 0, 0)) # [33, 33, 512, B]
    return disc.output_layer(input) # [30, 30, 1, B]
end


function discriminator_loss(real_disc, fake_disc)
    real_loss = Flux.Losses.logitbinarycrossentropy(real_disc, 1f0)
    fake_loss = Flux.Losses.logitbinarycrossentropy(fake_disc, 0f0)
    return real_loss + fake_loss
end

struct Generator
    downsample_layers
    upsample_layers
    output_layer::ConvTranspose
end


function Generator(maskdepth::Int, imgdepth::Int)
    # Input assumed to be [256, 256, indepth, B]
    downsample_layers = [
        downsample(4, maskdepth, 64, batchnorm=false), # [128, 128, 64, B]
        downsample(4, 64, 128),  # [64, 64, 128, B]
        downsample(4, 128, 256), # [32, 32, 256, B]
        downsample(4, 256, 512), # [16, 16, 512, B]
        downsample(4, 512, 512), # [8, 8, 512, B]
        downsample(4, 512, 512), # [4, 4, 512, B]
        downsample(4, 512, 512), # [2, 2, 512, B]
        downsample(4, 512, 512) # [1, 1, 512, B]
    ] .|> device

    # Note, input depths are doubled due to skip connections
    upsample_layers = [
        upsample(4, 512, 512, dropout=true), # [2, 2, 512, B]
        upsample(4, 1024, 512, dropout=true), # [4, 4, 512, B]
        upsample(4, 1024, 512, dropout=true), # [8, 8, 512, B]
        upsample(4, 1024, 512), # [16, 16, 512, B]
        upsample(4, 1024, 256), # [32, 32, 256, B]
        upsample(4, 512, 128), # [64, 64, 128, B]
        upsample(4, 256, 64), # [128, 128, 64, B]
    ] .|> device

    output_lyr = ConvTranspose(
        (4, 4),
        128 => imgdepth,
        tanh,
        stride=2,
        pad=SamePad(),
        init=(dims...) -> 0.02f0 .* randn(Float32, dims...)) |> device

    return Generator(
        downsample_layers, upsample_layers, output_lyr)
end


function (gen::Generator)(masks::AbstractArray)
    # @show extrema(masks)
    down128 = gen.downsample_layers[1](masks)
    # @show extrema(down128)
    down64  = gen.downsample_layers[2](down128)
    # @show extrema(down64)
    down32  = gen.downsample_layers[3](down64)
    # @show extrema(down32)
    down16  = gen.downsample_layers[4](down32)
    # @show extrema(down16)
    down8   = gen.downsample_layers[5](down16)
    # @show extrema(down8)
    down4   = gen.downsample_layers[6](down8)
    # @show extrema(down4)
    down2   = gen.downsample_layers[7](down4)
    # @show extrema(down2)
    # @show gen.downsample_layers[8].layers[2].μ
    # @show gen.downsample_layers[8].layers[2].σ²
    down1   = gen.downsample_layers[8](down2)
    # @show gen.downsample_layers[8].layers[2].μ
    # @show gen.downsample_layers[8].layers[2].σ²
    # @show extrema(down1)

    # TODO: Seems like everything goes to zero on this last downsample step.
    # WTF? Why is that?

    # TODO: It seems like the batchnorm sends it to zero somehow...
    # Why would that be? Considering we have a batch size of 1,
    # how else is it supposed to work?

    up2   = gen.upsample_layers[1](down1)
    # @show extrema(up2)
    up4   = gen.upsample_layers[2](cat(up2, down2, dims=3))
    # @show extrema(up4)
    up8   = gen.upsample_layers[3](cat(up4, down4, dims=3))
    # @show extrema(up8)
    up16  = gen.upsample_layers[4](cat(up8, down8, dims=3))
    # @show extrema(up16)
    up32  = gen.upsample_layers[5](cat(up16, down16, dims=3))
    # @show extrema(up32)
    up64  = gen.upsample_layers[6](cat(up32, down32, dims=3))
    # @show extrema(up64)
    up128 = gen.upsample_layers[7](cat(up64, down64, dims=3))
    # @show extrema(up128)

    out =  gen.output_layer(cat(up128, down128, dims=3))
    # @show extrema(out)

    return out
end


function generator_loss(
        real_imgs::AbstractArray, fake_imgs::AbstractArray,
        fake_disc::AbstractArray, λ::Float32)
    gan_loss = logitbinarycrossentropy(fake_disc, 1f0)
    l1_loss = mae(real_imgs, fake_imgs)

    return gan_loss + (λ * l1_loss)
end


function train_discriminator(
        disc_opt, disc::Discriminator,
        masks::AbstractArray, real_imgs::AbstractArray, fake_imgs::AbstractArray)
    ps = params(disc.downsample_block, disc.patch_block, disc.output_layer)
    loss, back = Zygote.pullback(ps) do
        real_disc = disc(masks, real_imgs)
        fake_disc = disc(masks, fake_imgs)
        return discriminator_loss(real_disc, fake_disc)
    end
    grads = back(1f0)
    update!(disc_opt, ps, grads)
    return loss
end

Zygote.@nograd train_discriminator


"""
Take one gradient descent step on one batch of training data.
"""
function train_step(
        gen_opt, disc_opt,
        gen::Generator, disc::Discriminator,
        masks::AbstractArray, real_imgs::AbstractArray;
        λ::Float32=100f0)

    ps = params(gen.downsample_layers, gen.upsample_layers, gen.output_layer)
    loss = Dict()
    loss["gen"], back = Zygote.pullback(ps) do
        fake_imgs = gen(masks)
        loss["disc"] = train_discriminator(disc_opt, disc, masks, real_imgs, fake_imgs)
        return generator_loss(real_imgs, fake_imgs, disc(masks, fake_imgs), λ)
    end
    grads = back(1f0)
    update!(gen_opt, ps, grads)
    return loss
end


"""
For debugging. Generate images using a trained generator and write them to a png.
"""
function gen_and_write_examples(
        gen::Generator, path::String, trainingdata,
        r_idx::Int, g_idx::Int, b_idx::Int)
    k = 0
    mkpath(path)

    for (masks, real_imgs) in trainingdata
        fake_imgs = gen(masks |> device) |> cpu

        masks .+= 1f0
        masks ./= 2f0

        @show extrema(fake_imgs)
        fake_imgs .+= 0.5f0
        clamp!(fake_imgs, 0f0, 1f0)

        for i in 1:size(fake_imgs, 4)
            mask_falsecolor = false_color_mapped(masks[:,:,:,i])
            real_img_falsecolor = false_color_subset(real_imgs[:,:,:,i], r_idx, g_idx, b_idx)
            fake_img_falsecolor = false_color_subset(fake_imgs[:,:,:,i], r_idx, g_idx, b_idx)

            open(joinpath(path, @sprintf("fake-img-%09d.png", k)), "w") do output
                img = cat(mask_falsecolor, real_img_falsecolor, fake_img_falsecolor, dims=2)
                save(Images.Stream{format"PNG"}(output), img)
            end

            k += 1

            # mask_stack = cat([masks[:,:,j,i] for j in 1:size(masks, 3)]..., dims=2)

            # # make this rgb
            # mask_stack = cat(mask_stack, mask_stack, mask_stack, dims=3)
            # k += 1

            # open(joinpath(path, @sprintf("fake-img-%09d.png", k)), "w") do output
            #     fcimg = false_color_image_pca(fake_imgs[:,:,:,i])
            #     img = cat(mask_stack, fcimg, dims=2)
            #     save(Images.Stream{format"PNG"}(output), img)
            #     # save(Images.Stream{format"PNG"}(output), fcimg)
            # end
#=  =#
            # for j in 1:size(fake_imgs, 3)
            #     open(joinpath(path, @sprintf("fake-img-%09d-%03d.png", k, j)), "w") do output
            #         img = fake_imgs[:,:,j,i]
            #         low = minimum(img)
            #         high = maximum(img)
            #         span = high - low
            #         img .-= low
            #         img ./= span

            #         img = cat(mask_stack, img, dims=2)

            #         save(Images.Stream{format"PNG"}(output), img)
            #     end
            # end
        end
   end
end


function false_color_mapped(img::Array{Float32, 3})
    h, w, d = size(img)

    color_scheme = ColorSchemes.colorschemes[:diverging_rainbow_bgymr_45_85_c67_n256]
    colors = ColorSchemes.get(color_scheme, 1:d, (1,d))

    fakeimg = Matrix{RGB{Float64}}(undef, (h, w))
    for j in 1:w, i in 1:h
        fakeimg[i, j] = sum(colors .* img[i, j, :]) ./ max(1e-6, sum(img[i, j, :]))
    end

    return fakeimg
end


"""
Take a k-channel image and do PCA to reduce dimensionality to 3 and produce
a color image.
"""
function false_color_image_pca(img::Array{Float32, 3})
    h, w, d = size(img)
    imgflat = transpose(reshape(img, (h*w, d)))
    pca = MultivariateStats.fit(MultivariateStats.PCA, imgflat)

    img = reshape(transpose(MultivariateStats.transform(pca, imgflat)[1:3,:]), (h, w, 3))

    lower = minimum(img, dims=(1,2))
    upper = maximum(img, dims=(1,2))
    img .-= lower
    img ./= upper.-lower
    return img
end


"""
Map three of the markers to R,G,B.
"""
function false_color_subset(img::Array{Float32, 3}, r::Int, g::Int, b::Int)
    h, w, d = size(img)
    fakeimg = Matrix{RGB{Float64}}(undef, (h, w))
    for j in 1:w, i in 1:h
        fakeimg[i, j] = RGB{Float64}(
            img[i, j, r],
            img[i, j, g],
            img[i, j, b])
    end

    return fakeimg
end


function top3_variable_channels(imgs::Array{Float32, 4})
    vs = var(imgs, dims=(1,2,4))[1,1,:,1]
    p = sortperm(vs, rev=true)

    return (p[1], p[2], p[3])
end


"""
Train a pix2pix style cGAN on paired training data saved to an HDF5 file. Write
parameters to another hdf5 file.
"""
function train_cgan(
        training_data_filename::String;
        # nepochs::Int=100,
        nepochs::Int=100,
        batchsize::Int=5,
        λ::Float32=100f0)

    # TODO: BatchNorm produces NaNs when batch size is 1.

    trainingdata, channel_names, masks, imgs = read_training_data(
        training_data_filename, batchsize)

    r_idx, g_idx, b_idx = top3_variable_channels(imgs)
    println(
        "Top 3 variable channels: ",
        channel_names[r_idx], ", ",
        channel_names[g_idx], ", ",
        channel_names[b_idx])

    maskdepth = size(masks, 3)
    imgdepth = size(imgs, 3)

    # Find the three most variable channels to map to RGB

    gen = Generator(maskdepth, imgdepth)
    disc = Discriminator(maskdepth, imgdepth)

    # Make sure I can run these
    # (masks, real_imgs) = first(trainingdata)
    # @show (maskdepth, imgdepth)
    # @show size(masks)
    # @show size(gen(masks |> device))
    # exit()

    gen_opt = ADAM(2e-4, (0.5, 0.999))
    disc_opt = ADAM(2e-4, (0.5, 0.999))

    for epoch in 1:nepochs
        println("Epoch: ", epoch)
        total_loss = 0.0
        for (masks, real_imgs) in trainingdata
            loss = train_step(
                gen_opt, disc_opt, gen, disc,
                masks |> device, real_imgs |> device,
                λ=λ)

            total_loss += loss["gen"] + loss["disc"]
        end
        @show total_loss
    end

    gen_and_write_examples(
        gen, "fake-imgs", trainingdata, r_idx, g_idx, b_idx)

    # TODO: remember we have to use `testmode!` to disable dropout once
    # we actually use the model.
end

