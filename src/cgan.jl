
# Experimenting with fitting a cgan to try to generate realistic looking images.

import CSV
import JSON
import TiffImages


"""
Build CGAN training examples from cytokit processed CODEX data. Includeing
`cells.json` files giving segmentation and expression data, TIFF files for each
tile, and cell type labels.
"""
function make_cgan_training_examples(
        cells_filenames::Vector{String},
        imgs_filenames::Vector{String},
        labels_filename::String,
        crop_width::Int=256,
        crop_height::Int=256,
        crop_coverage::Int=2)

    labels_df = CSV.read(labels_filename, DataFrame)
    labels = Dict{String, Int}()
    ncelltypes = maximum(labels.label) + 1

    for row in eachrow(labels_df)
        labels[row.cell_id] = row.label + 1
    end

    pat = r"^R(\d+)_X(\d+)_Y(\d+)$"
    cells = Dict{String, }

    training_examples = Tuple{Array{Float32, 3}, Array{Float32, 3}}[]

    for (cells_filename, imgs_filename) in zip(cells_filenames, imgs_filenames)
        cells_mat = match(pat, cells_filename)
        imgs_mat = match(pat, imgs_filename)
        @assert cells_mat.captures == imgs_mat.captures

        img = Array{Float32}(TiffImages.load(imgs_filename))
        height, width, nchannels = size(img)
        mask = zeros(Float32, (height, width, ncelltypes))

        for (cellnum, cell) in celldata
            cell_id = @sprintf(
                "R%s_X%s_Y%s_cell%s",
                mat.captures[1], mat.captures[2], mat.captures[3], cellnum)
            label = labels[cell_id]

            poly = [(Float32(x), Float32(y)) (x, y) in cell["poly"]]
            draw_polygon!(mask, poly, label)

        end

        ncrops = round(Int, crop_coverage * (width * height) / (crop_width * crop_height))
        for _ in 1:ncrops
            x = rand(1:(width - crop_width + 1))
            y = rand(1:(height - crop_height + 1))

            mask_crop = mask[y:y+crop_height-1, x:x+crop_width-1, :]
            img_crop = img[y:y+crop_height-1, x:x+crop_width-1, :]

            push!(training_examples, (img_crop, mask_crop))
        end
    end

    return training_examples
end


function draw_polygon!(
        img::Array{Float32, 3}, poly::Vector{Tuple{Float32, Float32}}, fill::Int)

    ymin, ymax = extrema([round(Int, p[2]) for p in poly])
    n = length(poly)

    xintersections = Int[]
    for y in ymin:ymax
        empty!(xintersections)
        for i in 1:n-1
            x1, y1 = poly[i]
            x2, y2 = poly[i+1]

            if y < min(y1, y2) || y > max(y1, y2)
                continue
            end

            if x1 == x2
                push!(xintersections, x1)
            else
                slope = (y2 - y1) / (x2 - x1)
                x = x1 + (y - y1) / slope
                if min(y1, y2) <= y <= max(y1, y2)
                    push!(xintersections, x)
                end
            end
        end

        sort!(xintersections)
        for i in 1:2:length(xintersections)-1
            x1, x2 = xintersections[i], xintersections[i+1]
            for x in x1:x2
                img[y+1, x+1, fill] = 1f0
            end
        end
    end

    # TODO: really need to test this. I'm sure I fucked up somewhere.
end
