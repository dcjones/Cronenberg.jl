module Cronenberg

using StaticArrays
using Random: shuffle!

const Tile = Tuple{UnitRange{Int}, UnitRange{Int}}

# Constant encoding neighbor positions
const NEIGHBORS = SA[
    (-1,  0),
    (-1,  1),
    ( 0,  1),
    ( 1,  1),
    ( 1,  0),
    ( 1, -1),
    ( 0, -1),
    (-1, -1) ]


"""
Hyperparameters determining the objective function that is optimized or sampled
with respect to.
"""
struct RuleSet
    # number of cell types
    ntypes::Int

    # cell adhesion scores, by cell types + 1
    adhesion::Matrix{Float32}

    # target volume (by cell type)
    target_volume::Vector{Int32}
    elasticity_volume::Vector{Float32}

    # target surface area (by cell type)
    target_area::Vector{Int32}
    elasticity_area::Vector{Float32}
end


"""
Data structure represented the current state of the world.
"""
struct World
    # State of each pixel. Which is 0 for unoccupied, or 1..ncells when occupied
    # by a specific cell.
    state::Matrix{Int32}

    # For each quadrant, a vector of tiles
    tiles::Vector{Vector{Tile}}

    # For each pixel, record whether each of its 8 neighbors has the same
    # state, and encode in a UInt8
    neighbors::Matrix{UInt8}

    # cell type indexed by cell state + 1 (unoccpied is represented as its own
    # state)
    types::Vector{Int32}

    # cell properties (indexed by cell)
    volumes::Vector{Int32} # technically area in 2d world
    areas::Vector{Int32} # technically circumfrence in 2d world

    # used to keep track of change in energy at each tile
    ΔEs::Array{Float32}
end


function World(m::Int, n::Int, ncelltype::Int, ncells::Int; nominal_tile_size::Int=80)
    # Generate a bunch of single pixel cells, placed uniformly at random.
    state = zeros(Int32, (m, n))
    types = zeros(Int32, ncells)
    volumes = zeros(Int32, ncells)
    areas = zeros(Int32, ncells)

    coords = [(i,j) for i in 1:m, j in 1:n]
    shuffle!(coords)

    for k in 1:ncells
        i, j = coords[k]
        state[i, j] = k
        types[k] = rand(1:ncelltype)
        volumes[k] = 1
        areas[k] = 1
    end

    # Build neighbors matrix
    neighbors = check_neighbors(state)

    # Divide the space into tiles.
    tiles = Vector{Vector{Tile}}(undef, 4)

    # Chop things into a reasonable number of x y tiles.
    nytiles = max(1, div(m, nominal_tile_size))
    nxtiles = max(1, div(n, nominal_tile_size))
    ntiles = nxtiles * nytiles

    for quadrant in 1:4
        tiles[quadrant] = Vector{Tile}(undef, ntiles)
    end

    tile = 0
    for ytile in 1:nytiles, xtile in 1:nxtiles
        tile += 1

        ya = (ytile-1)*nominal_tile_size+1
        yb = min(ya + div(nominal_tile_size, 2) - 1, m)
        yc = min(ya + nominal_tile_size - 1, m)

        xa = (xtile-1)*nominal_tile_size+1
        xb = min(xa + div(nominal_tile_size, 2) - 1, n)
        xc = min(xa + nominal_tile_size - 1, n)

        tiles[1][tile] = (ya:yb, xa:xb)
        tiles[2][tile] = (ya:yb, (xb+1):xc)
        tiles[3][tile] = ((yb+1):yc, xa:xb)
        tiles[4][tile] = ((yb+1):yc, (xb+1):xc)
    end

    ΔEs = Array{Float32}(undef, ntiles)

    return World(
        state, tiles, neighbors, types, volumes, areas, ΔEs)
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


"""
Check each neighbor of (i,j) and record whether the share a state in a UInt8
"""
function check_neighbors(world_state::Matrix{Int32}, i::Int, j::Int)
    m, n = size(world_state)

    # check neighbors numbered clockwise
    state = world_state[i, j]

    neighbors = UInt8(0)
    for (k, (i_off, j_off)) in enumerate(NEIGHBORS)
        neighbors |= (state == getstate(world_state, i+i_off, j+j_off)) << (k-1)
    end

    # neighbors |= (state != getstate(world_state, i-1, j))
    # neighbors |= (state != getstate(world_state, i-1, j+1)) << 1
    # neighbors |= (state != getstate(world_state, i,   j+1)) << 2
    # neighbors |= (state != getstate(world_state, i+1, j+1)) << 3
    # neighbors |= (state != getstate(world_state, i+1, j))   << 4
    # neighbors |= (state != getstate(world_state, i+1, j-1)) << 5
    # neighbors |= (state != getstate(world_state, i,   j-1)) << 6
    # neighbors |= (state != getstate(world_state, i-1, j-1)) << 7

    return neighbors
end


function check_neighbors(world_state::Matrix{Int32})
    m, n = size(world_state)
    neighbors = zeros(UInt8, (m, n))

    Threads.@threads for i in 1:m
        for j in 1:n
            neighbors[i,j] = check_neighbors(world_state, i, j)
        end
    end

    return neighbors
end


"""
Evaluate the free energy function from scratch across the whole world
"""
function loss(world::World, rules::RuleSet)
    E_area = sum(rules.elasticity_area[world.types] .* (rules.target_area[world.types] .- world.areas).^2)
    E_vol = sum(rules.elasticity_volume[world.types] .* (rules.target_volume[world.types] .- world.volumes).^2)

    E_adhesion = 0.0f0
    m, n = size(world.neighbors)
    for i in 1:m, j in 1:n
        neighbors = world.neighbors[i, j]
        type = gettype(world, i, j)

        if neighbors != 0
            for (k, (i_off, j_off)) in enumerate(NEIGHBORS)
                E_adhesion += rules.adhesion[type+1, gettype(world, i+i_off, j+j_off)+1]
            end
        end
    end
    # everything gets counted in both directions in the above. Correct for this.
    E_adhesion /= 2

    return E_area + E_vol + E_adhesion
end


function random_neighbor(neighbors::UInt8)
    nth_zero = rand(1:count_zeros(neighbors))
    for i in 1:8
        if (neighbors >> (i-1)) & 0x1 == 0
            nth_zero -= 1
            if nth_zero == 0
                return i
            end
        end
    end
    error("Pixel has no neighbors with different state.")
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
    neighbour_count = 0
    for (i_off, j_off) in NEIGHBORS
        i, j = i_dest + i_off, j_dest + j_off
        if 1 <= i <= m && 1 <= j <= n
            if world.state[i, j] == source_state
                neighbour_count += 1
                if count_ones(world.neighbors[i, j]) == 1
                    Δ -= 1
                end
            end
        end
    end

    # Add one for the new pixel, if it's not surrounding.
    if neighbour_count != 8
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
    neighbour_count = 0
    for (i_off, j_off) in NEIGHBORS
        i, j = i_dest + i_off, j_dest + j_off
        if 1 <= i <= m && 1 <= j <= n
            if world.state[i, j] == dest_state
                neighbour_count += 1
                if count_ones(world.neighbors[i, j]) == 8
                    Δ += 1
                end
            end
        end
    end

    # Lose one area if (i_dest, j_dest) was not surrounded
    if neighbour_count != 8
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
    ΔE_area = 0f0

    if source_type != 0
        ΔE_vol += rules.elasticity_volume[source_type] * (
            (rules.target_volume[source_type] - (world.volumes[source_state] + 1))^2 -
            (rules.target_volume[source_type] - world.volumes[source_state])^2)

        new_area = world.areas[source_state] + Δsource_area(world, i_source, j_source, i_dest, j_dest)
        ΔE_area += rules.elasticity_area[source_type] * (
            (rules.target_area[source_type] - new_area)^2 -
            (rules.target_area[source_type] - world.areas[source_state])^2)

    end

    if dest_type != 0
        ΔE_vol += rules.elasticity_volume[dest_type] * (
            (rules.target_volume[dest_type] - (world.volumes[dest_state] - 1))^2 -
            (rules.target_volume[dest_type] - world.volumes[dest_state])^2)

        new_area = world.areas[dest_state] + Δdest_area(world, i_source, j_source, i_dest, j_dest)
        ΔE_area += rules.elasticity_area[dest_type] * (
            (rules.target_area[dest_type] - new_area)^2 -
            (rules.target_area[dest_type] - world.areas[dest_state])^2)


    end

    # recompute adhesion scores in light of (i_dest, j_dest) being set to type `source_state`
    ΔE_adhesion = 0f0
    for (i_off, j_off) in NEIGHBORS
        i, j = i_dest + i_off, j_dest + j_off
        neighbor_type = gettype(world, i, j)
        ΔE_adhesion -= rules.adhesion[neighbor_type + 1, dest_type + 1]
        ΔE_adhesion += rules.adhesion[neighbor_type + 1, source_type + 1]
    end

    # @show (source_type, dest_type, ΔE_area, ΔE_vol, ΔE_adhesion)

    return ΔE_area + ΔE_vol + ΔE_adhesion
end


"""
Advance the world by one iteration.
"""
function tick(world::World, rules::RuleSet, E::Float32)
    m, n = size(world)

    for quadrant in 1:4
        # Threads.@threads for (k, (yrange, xrange)) in enumerate(world.tiles[quadrant])
        Threads.@threads for k in 1:length(world.tiles[quadrant])
            yrange, xrange = world.tiles[quadrant][k]

            # Choose a random pixel with a border. (we do this just by rejection
            # sampling)
            i_source, j_source = rand(yrange), rand(xrange)
            attempts = 1000
            while count_zeros(world.neighbors[i_source, j_source]) == 0 && attempts > 0
                i_source, j_source = rand(yrange), rand(xrange)
                attempts -= 1
            end

            if attempts == 0
                continue
            end

            # select the random neighbor
            i_off, j_off = NEIGHBORS[random_neighbor(world.neighbors[i_source, j_source])]
            i_dest, j_dest = i_source + i_off, j_source + j_off

            # Evaluate the energy of copying our state to the neighbors state
            ΔE = Δloss(world, rules, i_source, j_source, i_dest, j_dest)

            # accept?
            if ΔE < 0 || rand() < exp(-ΔE)
                source_state = getstate(world, i_source, j_source)
                dest_state = getstate(world, i_dest, j_dest)

                if source_state != 0
                    world.volumes[source_state] += 1
                    world.areas[source_state] += Δsource_area(world, i_source, j_source, i_dest, j_dest)
                end

                if dest_state != 0
                    world.volumes[dest_state] -= 1
                    world.areas[dest_state] += Δdest_area(world, i_source, j_source, i_dest, j_dest)
                end

                world.state[i_dest, j_dest] = source_state

                # Set neighbors for (i_dest, j_dest)
                world.neighbors[i_dest, j_dest] = check_neighbors(world.state, i_dest, j_dest)

                # Set neighbors for each of (i_dest, j_dests)'s neighbors
                for (k, (i_off, j_off)) in enumerate(NEIGHBORS)
                    i, j = i_dest + i_off, j_dest + j_off
                    if 1 <= i <= m && 1 <= j <= n
                        world.neighbors[i, j] = check_neighbors(world.state, i, j)
                    end
                end

                world.ΔEs[k] = ΔE
            else
                world.ΔEs[k] = 0f0
            end
        end

        E += sum(world.ΔEs)
    end

    return E
end


# TODO: Figure out how to implement a Blink gui, where we send the cell state
# matrix to the javascript code which then draws it using canvas or whatever.


end # module
