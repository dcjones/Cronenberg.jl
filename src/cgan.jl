
# Experimenting with fitting a cgan to try to generate realistic looking images.

import CSV
import JSON
import TiffImages
import LightXML


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
Build CGAN training examples from cytokit processed CODEX data. Includeing
`cells.json` files giving segmentation and expression data, TIFF files for each
tile, and cell type labels.
"""
function make_cgan_training_examples(
        cells_filenames::Vector{String},
        imgs_filenames::Vector{String},
        data_filename::String,
        labels_filename::String;
        random_crops::Bool=true,
        crop_width::Int=256,
        crop_height::Int=256,
        crop_coverage::Int=1,
        max_images::Int=typemax(Int),
        h5_output_filename::Union{Nothing, String}="training-examples.h5",
        png_output_path::Union{Nothing, String}=nothing)

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

    labels_df = CSV.read(labels_filename, DataFrame)
    labels = Dict{String, Int}()
    ncelltypes = maximum(labels_df.label) + 1

    for row in eachrow(labels_df)
        labels[row.cell_id] = row.label + 1
    end

    pat = r"R(\d+)_X(\d+)_Y(\d+)"
    cells = Dict{String, }

    training_examples = Tuple{Array{Float32, 3}, Array{Float32, 3}}[]

    cells_imgs_filenames = match_cells_imgs_filenames(cells_filenames, imgs_filenames)

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
        channel_names, layer_channel, layer_focus = read_ome_tiff_description(tiffimg)

        img = Array{Float32}(tiffimg.data)

        # throw out channels that are suboptimal focal lengths
        channel_index = Int[]
        for (i, (channel, focus)) in enumerate(zip(layer_channel, layer_focus))
            if z_best[tile] == focus
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

                # Don't bother training on examples with no cells
                # (may have to revisit this, but we probably want to at least
                # control how much empty training data we have)
                if sum(img_crop) > 0
                    push!(training_examples, (img_crop, mask_crop))
                end
            end
        else
            push!(training_examples, (img, mask))
        end

    end

    if png_output_path !== nothing
        write_training_examples_png(png_output_path, training_examples)
    end

    if h5_output_filename !== nothing
        write_training_examples_hdf5(h5_output_filename, training_examples)
    end

    return training_examples
end


"""
Dump training examples to one big hdf5 file.
"""
function write_training_examples_hdf5(
        filename::String, training_exampes::Vector{Tuple{Array{Float32, 3}, Array{Float32, 3}}})

    n = length(training_exampes)
    output = h5open(filename, "w")

    img_group = create_group(output, "img")
    mask_group = create_group(output, "mask")

    for (i, (img, mask))  in enumerate(training_exampes)
        key = @sprintf("%09d", i)
        img_group[key] = img
        mask_group[key] = mask
    end

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
        # k = argmax(sum(img, dims=(1,2))[1,1,:])
        for k in 1:size(img, 3)
            open(joinpath(path, @sprintf("img-%09d-%03d.png", i, k)), "w") do output
                # save(Images.Stream{format"PNG"}(output), img[:,:,k] ./ maximum(img[:,:,k]))
                save(Images.Stream{format"PNG"}(output), img[:,:,k])
            end
        end

        for k in 1:size(mask, 3)
            open(joinpath(path, @sprintf("mask-%09d-%03d.png", i, k)), "w") do output
                save(Images.Stream{format"PNG"}(output), mask[:,:,k])
            end
        end
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