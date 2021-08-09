module Cronenberg

import Blink
import ColorSchemes
using ArgParse
using Colors: hex, LCHab, RGB
using Distributions: Categorical
using FileIO
using Images
using NearestNeighbors
using Printf: @sprintf
using Random: shuffle, shuffle!
using StaticArrays

const Tile = Tuple{UnitRange{Int}, UnitRange{Int}}

# Constant encoding neighbor positions

# With diagonals
const NEIGHBORS = SA[
    (-1,  0),
    (-1,  1),
    ( 0,  1),
    ( 1,  1),
    ( 1,  0),
    ( 1, -1),
    ( 0, -1),
    (-1, -1) ]

const NEIGHBORS_AND_SELF = SA[
    ( 0,  0),
    (-1,  0),
    (-1,  1),
    ( 0,  1),
    ( 1,  1),
    ( 1,  0),
    ( 1, -1),
    ( 0, -1),
    (-1, -1) ]


# Without diagonals
# const NEIGHBORS = SA[
#     (-1,  0),
#     ( 0,  1),
#     ( 1,  0),
#     ( 0, -1) ]


softmax(xs) = exp.(xs) ./ sum(exp.(xs))


"""
Hyperparameters determining the objective function that is optimized or sampled
with respect to.
"""
struct RuleSet
    # number of cell types
    ncelltypes::Int

    # cell adhesion scores, by cell types + 1
    adhesion::Matrix{Float32}

    self_adhesion::Float32

    # target volume (by cell type)
    target_volume::Vector{Int32}
    rigidity_volume::Vector{Float32}

    # target surface area (by cell type)
    target_area::Vector{Int32}
    rigidity_area::Vector{Float32}

    # Maximimum act values, set when pixels flip, then decayed.
    act_max::Int32
end


"""
Generate a random "reasonable" rule set with the given number of celltypes.
"""
function RuleSet(ncelltypes::Int)

    target_volume = Int32[rand(30:100) for _ in 1:ncelltypes]
    rigidity_volume = fill(0.5f0, ncelltypes)

    # Target area is a sphere
    target_area = Int32[round(Int, π*sqrt(vol), RoundUp) for vol in target_volume]
    rigidity_area = fill(0.05f0, ncelltypes)

    adhesion = zeros(Float32, (ncelltypes+1, ncelltypes+1))
    adhesion = zeros(Float32, (ncelltypes+1, ncelltypes+1))

    for i in 2:ncelltypes+1
        # adhesion[i, i] = -2.0

        candidates = shuffle((i+1):ncelltypes+1)

        # choose some celltypes disliked by i
        for j in candidates
            # adhesion[i, j] = adhesion[j, i] = 1.0
            if rand() < 0.3
                adhesion[i, j] = adhesion[j, i] = 3.0
            elseif rand() < 0.1
                adhesion[i, j] = adhesion[j, i] = -1.5
            end
        end
    end

    self_adhesion = -2.0
    # act_max = 10
    act_max = 5

    return RuleSet(
      ncelltypes,
      adhesion,
      self_adhesion,
      target_volume,
      rigidity_volume,
      target_area,
      rigidity_area,
      act_max)
end


"""
Find the radius at which points have on average `avg_neighbors_target` neighbors.
"""
function calibrate_radius(kdtree::KDTree{A, B, T}, avg_neighbors_target::Int) where {A, B, T}

    xmin = T(Inf)
    xmax = T(-Inf)
    ymin = T(Inf)
    ymax = T(-Inf)

    for pt in kdtree.data
        xmin = min(xmin, pt[1])
        xmax = max(xmax, pt[1])
        ymin = min(ymin, pt[2])
        ymax = max(ymax, pt[2])
    end

    max_radius = sqrt((xmax - xmin)^2 + (ymax - ymin)^2)
    max_radius /= 4 # try to find a more reasonable starting place
    eps = max_radius / T(1e6)
    min_radius = eps

    # binary search to find a good radius
    println("calibrating neighborhood radius...")
    while max_radius - min_radius > eps
        avg_neighbors = 0
        radius = (min_radius + max_radius) / 2
        for pt in kdtree.data
            avg_neighbors += length(inrange(kdtree, pt, radius, false))
        end
        avg_neighbors /= length(kdtree.data)

        @show (min_radius, max_radius, avg_neighbors)

        if avg_neighbors < avg_neighbors_target
            min_radius = radius
        else
            max_radius = radius
        end
    end

    radius = (min_radius + max_radius) / 2
    println("radius: $(radius)")

    return radius
end


"""
Construct a rule set from a table file containing cell positions and types.
"""
function RuleSet(xs::Vector, ys::Vector, labels::Vector, avg_neighbors_target::Int=10;
        adhesion_scale::Float32=1f0)

    n = length(xs)
    @assert length(ys) == n
    @assert length(labels) == n

    kdtree = KDTree(Float64.(transpose(hcat(xs, ys))), leafsize=10)
    radius = calibrate_radius(kdtree, avg_neighbors_target)
    @show radius

    ncelltypes = maximum(labels)
    affinity = zeros(Int, (ncelltypes, ncelltypes))

    for i in 1:n
        for j in inrange(kdtree, [xs[i], ys[i]], radius, false)
            affinity[labels[i], labels[j]] += 1
            affinity[labels[j], labels[i]] += 1
        end
    end

    type_counts = zeros(Int, ncelltypes)
    for label in labels
        type_counts[label] += 1
    end

    nedges = sum(affinity)
    type_proportions = type_counts / sum(type_counts)
    expected_edge_proportions = type_proportions * transpose(type_proportions)

    relative_log2_affinity = log2.(affinity ./ (expected_edge_proportions .* nedges))

    adhesion = zeros(Float32, (ncelltypes+1, ncelltypes+1))
    for i in 1:ncelltypes
        for j in i+1:ncelltypes
            adhesion[i+1, j+1] = adhesion[j+1, i+1] = -adhesion_scale * relative_log2_affinity[i, j]
        end
    end

    self_adhesion = -2.0
    act_max = 5

    target_volume = Int32[rand(30:100) for _ in 1:ncelltypes]
    rigidity_volume = fill(0.5f0, ncelltypes)

    target_area = Int32[round(Int, π*sqrt(vol), RoundUp) for vol in target_volume]
    rigidity_area = fill(0.05f0, ncelltypes)

    return RuleSet(
      ncelltypes,
      adhesion,
      self_adhesion,
      target_volume,
      rigidity_volume,
      target_area,
      rigidity_area,
      act_max)
end


"""
Data structure represented the current state of the world.
"""
struct World
    # State of each pixel. Which is 0 for unoccupied, or 1..ncells when occupied
    # by a specific cell.
    state::Matrix{Int32}

    # Used to keep track of changes to redraw world efficiently
    prev_state::Matrix{Int32}

    # For each pixel, record whether each of its 8 neighbors has the same
    # state, and encode in a UInt8
    neighbors::Matrix{UInt8}

    # Used to keep track of the previous state
    prev_neighbors::Matrix{UInt8}

    # Actin value, used for the Act model of cell migration
    act::Matrix{Int32}

    # For each quadrant, a vector of tiles
    tiles::Matrix{Tile}

    # Used to compute the number of candidate source pixels in each tile,
    # so we can try to balance tho number of proposals per tile.
    ncandidates::Vector{Int}

    # cell type indexed by cell state + 1 (unoccpied is represented as its own
    # state)
    types::Vector{Int32}

    # cell properties (indexed by cell)
    volumes::Vector{Int32} # technically area in 2d world
    areas::Vector{Int32} # technically circumfrence in 2d world

    # used to test for connected components
    visited::BitMatrix

    # used to keep track of change in energy at each tile
    ΔEs::Array{Float32}
end



function World(rules::RuleSet, m::Int, n::Int, ncells::Int; nominal_tile_size::Int=40)
    # Generate a bunch of single pixel cells, placed uniformly at random.
    state = zeros(Int32, (m, n))
    types = zeros(Int32, ncells)
    volumes = zeros(Int32, ncells)
    areas = zeros(Int32, ncells)

    # # Uniform random initialization
    # coords = [(i,j) for i in 1:m, j in 1:n]
    # shuffle!(coords)

    # for k in 1:ncells
    #     i, j = coords[k]
    #     state[i, j] = k
    #     types[k] = rand(1:rules.ncelltypes)
    #     volumes[k] = 1
    #     areas[k] = 1
    # end

    # Produce random layers of cells
    coords = [(i,j) for i in 1:m, j in 1:n]
    shuffle!(coords)

    freqs = [(0.2 * m + rand() * 0.5 * m) for _ in 1:rules.ncelltypes] ./ (2*π)
    offsets = [π*rand() for _ in 1:rules.ncelltypes]

    for k in 1:ncells
        i, j = coords[k]
        state[i, j] = k
        type = rand(Categorical(softmax(sin.(offsets .+ i ./ freqs))))
        types[k] = type
        volumes[k] = 1
        areas[k] = 1
    end

    # # Clumpier initialization: choose random rectangles, choose a random
    # # cell type, put some cells in there
    # ncells_remaining = ncells
    # max_cells_per_rect = 200
    # while ncells_remaining > 0
    #     h = rand(1:m)
    #     w = rand(1:n)
    #     i0 = rand(1:(m - h + 1))
    #     j0 = rand(1:(n - w + 1))
    #     i1 = i0 + h - 1
    #     j1 = j0 + w - 1

    #     type = rand(1:rules.ncelltypes)
    #     nrectcells = min(ncells_remaining, rand(1:max_cells_per_rect))

    #     for i in 1:nrectcells
    #         i = rand(i0:i1)
    #         j = rand(j0:j1)
    #         state[i, j] = ncells_remaining
    #         types[ncells_remaining] = type
    #         volumes[ncells_remaining] = 1
    #         areas[ncells_remaining] = 1
    #         ncells_remaining -= 1
    #     end
    # end

    return World(state, types, volumes, areas, nominal_tile_size=nominal_tile_size)
end

function World(
        state::Matrix{Int32}, types::Vector{Int32},
        volumes::Vector{Int32}, areas::Vector{Int32};
        nominal_tile_size::Int=40)

    m, n = size(state)

    # Build neighbors matrix
    neighbors = findneighbors(state)

    # Divide the space into tiles.

    # Chop things into a reasonable number of x y tiles.
    nytiles = max(1, div(m, nominal_tile_size, RoundUp))
    nxtiles = max(1, div(n, nominal_tile_size, RoundUp))
    ntiles = nxtiles * nytiles
    tiles = Matrix{Tile}(undef, 4, ntiles)

    tile = 0
    for ytile in 1:nytiles, xtile in 1:nxtiles
        tile += 1

        ya = (ytile-1)*nominal_tile_size+1
        yb = min(ya + div(nominal_tile_size, 2) - 1, m)
        yc = min(ya + nominal_tile_size - 1, m)

        xa = (xtile-1)*nominal_tile_size+1
        xb = min(xa + div(nominal_tile_size, 2) - 1, n)
        xc = min(xa + nominal_tile_size - 1, n)

        tiles[1, tile] = (ya:yb, xa:xb)
        tiles[2, tile] = (ya:yb, (xb+1):xc)
        tiles[3, tile] = ((yb+1):yc, xa:xb)
        tiles[4, tile] = ((yb+1):yc, (xb+1):xc)
    end

    ΔEs = Array{Float32}(undef, ntiles)

    visited = BitMatrix(undef, (m, n))
    fill!(visited, false)

    act = zeros(Int32, (m, n))

    ncandidates = zeros(Int, ntiles)

    return World(
        state, similar(state), neighbors, similar(neighbors),
        act, tiles, ncandidates, types, volumes, areas, visited, ΔEs)
end


"""
Initialize a world from csv file containing cell positions and types.
"""
function World(xs::Vector, ys::Vector, labels::Vector; nominal_tile_size::Int=40)
    ncelltypes = maximum(labels)
    ncells = length(labels)

    # TODO: We don't know the scale of xs, ys. Maybe avoid just rounding to integers?
    # Furthermore, even without scaling, we may have two cells in the same location.

    m = round(Int, maximum(ys), RoundUp)
    n = round(Int, maximum(xs), RoundUp)

    state = zeros(Int32, (m, n))
    types = zeros(Int32, ncells)
    volumes = zeros(Int32, ncells)
    areas = zeros(Int32, ncells)
    for k in 1:ncells
        i = max(1, round(Int, ys[k]))
        j = max(1, round(Int, xs[k]))
        state[i, j] = k
        types[k] = labels[k]
        volumes[k] = 1
        areas[k] = 1
    end

    return World(state, types, volumes, areas, nominal_tile_size=nominal_tile_size)
end


"""
Store the current (world.state, world.neighbors) in (world.prev_state, world.prev_neighbors)
"""
function savestate!(world::World)
    copy!(world.prev_state, world.state)
    copy!(world.prev_neighbors, world.neighbors)
end


Base.size(world::World) = size(world.state)


"""
Return the cell state at (i,j), and 0 if it's out of bounds.
"""
function getstate(world::World, i::Int, j::Int)
    return getstate(world.state, i, j)
end


function getstate(world_state::Matrix{Int32}, i::Int, j::Int)
    m, n = size(world_state)
    return ((1 <= i <= m) && (1 <= j <= n)) ? world_state[i,j] : Int32(0)
end


function gettype(world::World, i::Int, j::Int)
    state = getstate(world, i, j)
    return state == 0 ? 0 : world.types[state]
end


function gettype(world::World, state::Int32)
    return state == 0 ? 0 : world.types[state]
end


function getact(world::World, state::Int32, i::Int, j::Int)
    m, n = size(world)
    return getstate(world, i, j) == state && 1 <= i <= m && 1 <= j <= n ?
        world.act[i, j] : 0
end


"""
Check if the position is a border.
"""
function isborder(neighbors::Matrix{UInt8}, i::Int, j::Int)
    m, n = size(neighbors)
    return neighbors[i, j] != 0 || i == 1 || i == m || j == 1 || j == n
end


"""
Recompute the neighbors array from scratch. Shouldn't be necessary, since we
update in incrementally.
"""
function findneighbors!(world::World)
    m, n = size(world)
    fill!(world.neighbors, 0)
    Threads.@threads for i in 1:m
        for j in 1:n
            world.neighbors[i, j] = check_neighbors(world.state, i, j)
        end
    end
end


"""
Decay act values by 1, saturating at 0.
"""
function decayact!(world::World)
    m, n = size(world)
    Threads.@threads for i in 1:m
        for j in 1:n
            world.act[i, j] = max(0, world.act[i, j]-1)
        end
    end
end


"""
Geometric mean act value in a moore neighborhood.
"""
function gmact(world::World, i::Int, j::Int)
    m, n = size(world)
    state = getstate(world, i, j)
    gm = Int32(1)
    neighborhood_size = 0

    for (i_off, j_off) in NEIGHBORS_AND_SELF
        i_neighbor, j_neighbor = i + i_off, j + j_off
        if getstate(world, i_neighbor, j_neighbor) == state
            neighborhood_size += 1
            if 1 <= i_neighbor <= m && 1 <= j_neighbor <= n
                gm *= world.act[i_neighbor, j_neighbor]
            end
        end
    end

    return Float32(gm)^(1/neighborhood_size)
end


"""
Check each neighbor of (i,j) and record whether the share a state in a UInt8
"""
function findneighbors(world_state::Matrix{Int32}, i::Int, j::Int)
    m, n = size(world_state)

    # check neighbors numbered clockwise
    state = world_state[i, j]

    neighbors = UInt8(0)
    for (k, (i_off, j_off)) in enumerate(NEIGHBORS)
        i_neighbor, j_neighbor = i+i_off, j+j_off
        if 1 <= i_neighbor <= m && 1 <= j_neighbor <= n && world_state[i_neighbor, j_neighbor] != state
            neighbors |= 1 << (k-1)
        end
    end

    return neighbors
end


"""
Compute and return a full neighbor matrix from scratch.
"""
function findneighbors(world_state::Matrix{Int32})
    m, n = size(world_state)
    neighbors = zeros(UInt8, (m, n))

    Threads.@threads for i in 1:m
        for j in 1:n
            neighbors[i,j] = findneighbors(world_state, i, j)
        end
    end

    return neighbors
end


"""
Evaluate the free energy function from scratch across the whole world
"""
function loss(world::World, rules::RuleSet)
    E_area = sum(rules.rigidity_area[world.types] .* (rules.target_area[world.types] .- world.areas).^2)
    E_vol = sum(rules.rigidity_volume[world.types] .* (rules.target_volume[world.types] .- world.volumes).^2)

    E_adhesion = 0.0f0
    m, n = size(world.state)
    for i in 1:m, j in 1:n
        if world.neighbors[i, j] == 0
            continue
        end

        state = getstate(world, i, j)
        type = gettype(world, i, j)
        for (k, (i_off, j_off)) in enumerate(NEIGHBORS)
            if getstate(world, i+i_off, j+j_off) == state
                E_adhesion += rules.self_adhesion
            else
                E_adhesion += rules.adhesion[type+1, gettype(world, i+i_off, j+j_off)+1]
            end
        end
    end
    # everything gets counted in both directions in the above. Correct for this.
    E_adhesion /= 2

    # return E_area + E_vol + E_adhesion
    return E_vol + E_adhesion
end


"""
Compute the change in area of the cell occupying (i_source, j_source) from copying
that state to (i_dest, j_dest.)
"""
function Δsource_area(world::World, i_source::Int, j_source::Int, i_dest::Int, j_dest::Int)
    m, n = size(world.state)
    source_state = world.state[i_source, j_source]

    Δ = 0

    # We lose one area for every neighbor that only had one exposed direction.
    neighbor_count = 0
    for (i_off, j_off) in NEIGHBORS
        i, j = i_dest + i_off, j_dest + j_off
        if 1 <= i <= m && 1 <= j <= n
            if world.state[i, j] == source_state
                neighbor_count += 1
                if count_ones(world.neighbors[i, j]) == 1
                    Δ -= 1
                end
            end
        end
    end

    # Add one for the new pixel, if it's not surrounded.
    if neighbor_count != length(NEIGHBORS)
        Δ += 1
    end

    return Δ
end


function Δdest_area(world::World, i_source::Int, j_source::Int, i_dest::Int, j_dest::Int)
    m, n = size(world.state)
    dest_state = world.state[i_dest, j_dest]

    Δ = 0

    # Gain one area for any dest_state pixels that were surrounded but will be
    # no longer.
    neighbor_count = 0
    for (i_off, j_off) in NEIGHBORS
        i, j = i_dest + i_off, j_dest + j_off
        if 1 <= i <= m && 1 <= j <= n
            if world.state[i, j] == dest_state
                neighbor_count += 1
                if count_ones(world.neighbors[i, j]) == 0
                    Δ += 1
                end
            end
        end
    end

    # Lose one area if (i_dest, j_dest) was not surrounded
    if neighbor_count != length(NEIGHBORS)
        Δ -= 1
    end

    return Δ
end


"""
Compute the change in energy from copying stat from (i_source, j_soruce) to
(i_dest, j_dest)
"""
function Δloss(
        world::World, rules::RuleSet, i_source::Int, j_source::Int, i_dest::Int, j_dest::Int)

    m, n = size(world.state)

    source_state = getstate(world, i_source, j_source)
    source_type = gettype(world, i_source, j_source)

    dest_state = getstate(world, i_dest, j_dest)
    dest_type = gettype(world, i_dest, j_dest)

    ΔE_vol = 0f0
    # ΔE_area = 0f0

    if source_type != 0
        ΔE_vol += rules.rigidity_volume[source_type] * (
            (rules.target_volume[source_type] - (world.volumes[source_state] + 1))^2 -
            (rules.target_volume[source_type] - world.volumes[source_state])^2)

        # new_area = world.areas[source_state] + Δsource_area(world, i_source, j_source, i_dest, j_dest)
        # ΔE_area += rules.rigidity_area[source_type] * (
        #     (rules.target_area[source_type] - new_area)^2 -
        #     (rules.target_area[source_type] - world.areas[source_state])^2)
    end

    if dest_type != 0
        ΔE_vol += rules.rigidity_volume[dest_type] * (
            (rules.target_volume[dest_type] - (world.volumes[dest_state] - 1))^2 -
            (rules.target_volume[dest_type] - world.volumes[dest_state])^2)

        # new_area = world.areas[dest_state] + Δdest_area(world, i_source, j_source, i_dest, j_dest)
        # ΔE_area += rules.rigidity_area[dest_type] * (
        #     (rules.target_area[dest_type] - new_area)^2 -
        #     (rules.target_area[dest_type] - world.areas[dest_state])^2)

        # Don't let any cells disappear entirely
        if world.volumes[dest_state] == 1
            return Inf32, Inf32
        end

        # TODO: This is super expensive, and maybe not necessary
        # Don't let any cells break into pieces
        # if !remains_connected(world, i_dest, j_dest)
        #     return Inf32
        # end
    end

    # recompute adhesion scores in light of (i_dest, j_dest) being set to type `source_state`
    ΔE_adhesion = 0f0
    for (i_off, j_off) in NEIGHBORS
        i, j = i_dest + i_off, j_dest + j_off
        neighbor_type = gettype(world, i, j)
        neighbor_state = getstate(world, i, j)

        if neighbor_state == dest_state
            ΔE_adhesion -= rules.self_adhesion
        else
            ΔE_adhesion -= rules.adhesion[neighbor_type + 1, dest_type + 1]
        end

        if neighbor_state == source_state
            ΔE_adhesion += rules.self_adhesion
        else
            ΔE_adhesion += rules.adhesion[neighbor_type + 1, source_type + 1]
        end
    end

    # If dest site is not very active, and source site is very active,
    # energy in reduced in the copy.
    ΔE_act = gmact(world, i_dest, j_dest) - gmact(world, i_source, j_source)

    # return ΔE_area + ΔE_vol + ΔE_adhesion + ΔE_act
    return ΔE_vol + ΔE_adhesion, ΔE_act
end


"""
Checks whether a cell remains a connected components after removing it's pixel
at (i, j)
"""
function remains_connected(world::World, i::Int, j::Int)
    state = getstate(world, i, j)
    @assert state != 0

    i_min, i_max = 1, 0
    j_min, j_max = 1, 0

    # traverse the cell starting at an arbitrary neighbor, keeping track of
    # visited pixels
    for (i_off, j_off) in NEIGHBORS
        i_next, j_next = i + i_off, j + j_off
        if getstate(world, i_next, j_next) == state
            i_min, i_max, j_min, j_max = traverse_cell(
                world, state, i_next, j_next, i, j, i_min, i_max, j_min, j_max)
             break
        end
    end

    # check that each neighbor of the same state was visited
    for (i_off, j_off) in NEIGHBORS
        i_next, j_next = i + i_off, j + j_off
        if getstate(world, i_next, j_next) == state && !world.visited[i_next, j_next]
            world.visited[i_min:i_max,j_min:j_max] .= false
            return false
        end
    end

    # clear visited rect
    world.visited[i_min:i_max,j_min:j_max] .= false
    return true
end


function traverse_cell(
        world::World, state::Int32, i::Int, j::Int, i_excluded::Int, j_excluded::Int,
        i_min::Int, i_max::Int, j_min::Int, j_max::Int)

    world.visited[i, j] = true
    i_min, i_max = min(i_min, i), max(i_max, i)
    j_min, j_max = min(j_min, j), max(j_max, j)

    for (i_off, j_off) in NEIGHBORS
        i_next, j_next = i + i_off, j + j_off
        if getstate(world, i_next, j_next) == state &&
                !world.visited[i_next, j_next] &&
                !(i_next == i_excluded && j_next == j_excluded)
            i_min, i_max, j_min, j_max =
                traverse_cell(
                    world, state, i_next, j_next, i_excluded, j_excluded,
                    i_min, i_max, j_min, j_max)
        end
    end

    return i_min, i_max, j_min, j_max
end


"""
Count the number of possible source pixels for monte carlo proposals in the
given tile.
"""
function count_candidates(tile::Tile, neighbors::Matrix{UInt8})
    yrange, xrange = tile
    ncandidates_k = 0
    @inbounds for i in yrange, j in xrange
        ncandidates_k += neighbors[i, j] != 0
    end
    return ncandidates_k
end


"""
Make a single proposal in every tile.
"""
function tick(world::World, rules::RuleSet, E::Float32, T::Float64=1.0)
    m, n = size(world)

    expected_proposals = 0.0

    for quadrant in 1:size(world.tiles, 1)
        fill!(world.ΔEs, 0f0)

        # Compute the number of candidates in each tile
        fill!(world.ncandidates, 0)
        Threads.@threads for k in 1:size(world.tiles, 2)
        # for k in 1:size(world.tiles, 2)
            world.ncandidates[k] = count_candidates(
                world.tiles[quadrant, k], world.neighbors)
        end

        max_candidates = maximum(world.ncandidates)
        expected_proposals = 0.0
        for ncandidates in world.ncandidates
            expected_proposals += ncandidates / max_candidates
        end

        Threads.@threads for k in 1:size(world.tiles, 2)
        # for k in 1:size(world.tiles, 2)
            # Randomly skip sparsely populated tiles
            if rand() > world.ncandidates[k] / max_candidates
                continue
            end

            yrange, xrange = world.tiles[quadrant, k]

            # This can happen in tiles occuring on the edges
            if isempty(yrange) || isempty(xrange)
                continue
            end

            # Count the number of pixels with a neighbor of a different state
            # then select one uniformly at random
            nborder_pixels = world.ncandidates[k]

            if nborder_pixels == 0
                continue
            end

            source_border_pos = rand(1:nborder_pixels)
            i_source, j_source = 0, 0
            @inbounds for i in yrange, j in xrange
                if world.neighbors[i, j] != 0
                    source_border_pos -= 1
                    if source_border_pos == 0
                        i_source, j_source = i, j
                        break
                    end
                end
            end
            @assert i_source != 0 && j_source != 0

            # Count the number of neighbors of (i_source, j_source) with
            # a different state and choose one uniformly at random.
            neighbors = world.neighbors[i_source, j_source]
            nneighbors = count_ones(neighbors)
            dest_pos = rand(1:nneighbors)
            i_dest, j_dest = 0, 0
            for l in 1:length(NEIGHBORS)
                if (neighbors >> (l-1)) & 0x1 == 1
                    dest_pos -= 1
                    if dest_pos == 0
                        i_off, j_off = NEIGHBORS[l]
                        i_dest, j_dest = i_source + i_off, j_source + j_off
                        break
                    end
                end
            end
            @assert i_dest != 0 && j_dest != 0

            # Avoid setting any pixels at the very edge, otherwise they get stuck
            # because proposals are never made from an out of bounds source.
            if i_dest == 1 || i_dest == m || j_dest == 1 || j_dest == n
                continue
            end

            # Evaluate the energy of copying our state to the neighbors state
            ΔE, ΔE_act = Δloss(world, rules, i_source, j_source, i_dest, j_dest)
            ΔE_total = ΔE + ΔE_act

            # accept?
            if ΔE_total < 0 || rand() < exp(-ΔE_total/T)
                source_state = getstate(world, i_source, j_source)
                dest_state = getstate(world, i_dest, j_dest)

                if source_state != 0
                    world.volumes[source_state] += 1
                    # world.areas[source_state] += Δsource_area(world, i_source, j_source, i_dest, j_dest)
                end

                if dest_state != 0
                    world.volumes[dest_state] -= 1
                    # world.areas[dest_state] += Δdest_area(world, i_source, j_source, i_dest, j_dest)
                end

                world.state[i_dest, j_dest] = source_state
                if source_state != 0
                    world.act[i_dest, j_dest] = rules.act_max
                end

                # Set neighbors for (i_dest, j_dest)
                world.neighbors[i_dest, j_dest] = findneighbors(world.state, i_dest, j_dest)

                # Set neighbors for each of (i_dest, j_dests)'s neighbors
                for (k, (i_off, j_off)) in enumerate(NEIGHBORS)
                    i, j = i_dest + i_off, j_dest + j_off
                    if 1 <= i <= m && 1 <= j <= n
                        world.neighbors[i, j] = findneighbors(world.state, i, j)
                    end
                end

                world.ΔEs[k] = ΔE
            else
                world.ΔEs[k] = 0f0
            end
        end

        E += sum(world.ΔEs)
    end

    return (E, round(Int, expected_proposals))
end


"""
Function for debugging and testing. Make sure `volumes` and `areas` arrays
are correct.
"""
function check_area_volume(world::World)
    volumes = similar(world.volumes)
    fill!(volumes, 0)

    areas = similar(world.areas)
    fill!(areas, 0)

    m, n = size(world)

    # for (s, neighbors) in zip(world.state, world.neighbors)
    for i in 1:m, j in 1:n
        s = world.state[i, j]
        neighbors = world.neighbors[i, j]

        if s != 0
            volumes[s] += 1
            @assert neighbors == findneighbors(world.state, i, j)
            if (length(NEIGHBORS) - count_ones(neighbors)) > 0
                areas[s] += 1
            end
        end
    end

    @assert volumes == world.volumes
    @assert areas == world.areas
end


const blink_window = Ref{Union{Nothing, Blink.Window}}(nothing)


function clear_world(win::Blink.Window, borderpixels::Int, bgcolor="white", bordercolor="#ddd")
    Blink.js(win, Blink.JSString("clearcells(\"$(bgcolor)\")"))
    Blink.js(win, Blink.JSString("drawborder($(borderpixels), \"$(bordercolor)\")"))
end


"""
Draw the world from scratch.
"""
function draw_act_world(win::Blink.Window, world::World, act_max::Int32)
    # TODO: Draw an act heatmap
end


"""
Draw the world from scratch.
"""
function draw_world!(win::Blink.Window, world::World, colors::Vector, ncelltypes::Int)
    m, n = size(world)
    xs = Int[]
    ys = Int[]

    function collect_and_draw(color, type, border)
        empty!(xs)
        empty!(ys)

        for i in 1:m, j in 1:n
            if gettype(world, i, j) == type && (border == isborder(world.neighbors, i, j))
                push!(ys, i-1)
                push!(xs, j-1)
            end
        end

        if !isempty(xs)
            jscall = Blink.JSString("drawcells($m, $n, \"#$(hex(color))\", $xs, $ys)")
            Blink.js(win, jscall)
        end
    end

    for type in 1:ncelltypes
        # draw border cells by darkening the color
        color = colors[type]
        border_color = borderize_color(color)

        collect_and_draw(border_color, type, true)
        collect_and_draw(color, type, false )
    end
end


function redraw_world!(win::Blink.Window, world::World, colors::Vector, ncelltypes::Int)
    m, n = size(world)
    xs = Int[]
    ys = Int[]

    function collect_and_draw(color, type, border)
        empty!(xs)
        empty!(ys)

        for i in 1:m, j in 1:n
            changed =
                world.state[i, j] != world.prev_state[i, j] ||
                isborder(world.neighbors, i, j) != isborder(world.prev_neighbors, i, j)

            if changed && gettype(world, i, j) == type && (border == isborder(world.neighbors, i, j))
                push!(ys, i-1)
                push!(xs, j-1)
            end
        end

        if !isempty(xs)
            jscall = Blink.JSString("drawcells($m, $n, \"#$(hex(color))\", $xs, $ys)")
            Blink.js(win, jscall, callback=false)
        end
    end

    for type in 0:ncelltypes
        if type > 0
            color = colors[type]
            border_color = borderize_color(color)
        else
            border_color = color = RGB(1,1,1)
        end

        collect_and_draw(border_color, type, true)
        collect_and_draw(color, type, false)
    end
end


function borderize_color(c::Colorant, Δl=-30)
    c_lchab = convert(LCHab, c)
    return LCHab(c_lchab.l - Δl, c_lchab.c, c_lchab.h)
end


function draw_world!(img::Matrix{RGB24}, world::World, colors::Vector, ncelltypes::Int)
    height, width = size(img)
    m, n = size(world)
    @assert height % m == 0
    @assert width % n == 0
    y_pixelsize = div(height, m)
    x_pixelsize = div(width, n)

    colors_rgb24 = RGB24[convert(RGB24, c) for c in colors]
    border_colors_rgb24 = RGB24[convert(RGB24, borderize_color(c)) for c in colors]

    fill!(img, RGB24(1,1,1))
    Threads.@threads for i in 1:m
        for j in 1:n
            type = gettype(world, i, j)
            if type != 0
                yrange = ((i-1)*y_pixelsize+1):(i*y_pixelsize)
                xrange = ((j-1)*x_pixelsize+1):(j*x_pixelsize)
                if isborder(world.neighbors, i, j)
                    img[yrange, xrange] .= border_colors_rgb24[type]
                else
                    img[yrange, xrange] .= colors_rgb24[type]
                end
            end
        end
    end
end


function run(world::World, rules::RuleSet;
        nsteps::Int=10000, pixelsize::Int=3, T::Float64=1.0,
        output_dir::Union{Nothing, String}=nothing,
        watch::Bool=true,
        wait_when_done::Bool=true)
    m, n = size(world)

    color_scheme = ColorSchemes.colorschemes[
        :diverging_rainbow_bgymr_45_85_c67_n256]
    colors = [ColorSchemes.get(color_scheme, i, (1,rules.ncelltypes)) for i in 1:rules.ncelltypes]

    if watch && (blink_window[] === nothing || !Blink.active(blink_window[]))
        blink_window[] = Blink.Window()
    end

    if watch
        scripts = String(read(joinpath(dirname(pathof(Cronenberg)), "draw.js")))
        Blink.body!(
            blink_window[],
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Cronenberg</title>
                <script>$(scripts)</script>
            </head>
            <body>
                <canvas id="canvas" width="$(pixelsize*n)" height="$(pixelsize*m)"></canvas>
            </body>
            </html>
            """,
            async=false)

        clear_world(blink_window[], pixelsize, "white")
        draw_world!(blink_window[], world, colors, rules.ncelltypes)
        savestate!(world)
    end

    E = loss(world, rules)

    ntiles = length(world.tiles)
    # nticks_per_mcs = round(Int, m*n/ntiles)

    img = Matrix{RGB24}(undef, (m*pixelsize, n*pixelsize))

    step = 0
    imgnum = 0
    mkpath("imgs")

    for mcs in 1:nsteps
        expected_proposals = 0.0
        while expected_proposals < m*n
            E, expected_proposals_k = tick(world, rules, E, T)
            expected_proposals += expected_proposals_k
            step += 1
            if step % 1000 == 0
                imgnum += 1

                if watch
                    redraw_world!(blink_window[], world, colors, rules.ncelltypes)
                    savestate!(world)
                end

                if output_dir !== nothing
                    filename = joinpath(output_dir, @sprintf("frame-%09d.png", imgnum))
                    draw_world!(img, world, colors, rules.ncelltypes)
                    open(filename, "w") do output
                        save(Images.Stream{format"PNG"}(output), img)
                    end
                end

                @show (step, E)
            end
        end
        decayact!(world)
    end

    # TODO: Not actually sure what to do here.
    if wait_when_done
        while true
            yield()
        end
    end

    # TODO: What I really want is an interface to advance the simulation,
    # and some kind of thing to save it.
end


end # module