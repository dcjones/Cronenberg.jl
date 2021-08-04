module Cronenberg

import Blink
import ColorSchemes
using Colors: hex, LCHab, RGB
using Random: shuffle!
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


# Without diagonals
# const NEIGHBORS = SA[
#     (-1,  0),
#     ( 0,  1),
#     ( 1,  0),
#     ( 0, -1) ]


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

    # Used to keep track of changes to redraw world efficiently
    prev_state::Matrix{Int32}

    # True if the the pixel is occupied and a border
    border::BitMatrix

    # Used to keep track of previous state
    prev_border::BitMatrix

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

    # used to test for connected components
    visited::BitMatrix

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

    #     type = rand(1:ncelltype)
    #     nrectcells = min(ncells_remaining, rand(1:max_cells_per_rect))
    #     @show nrectcells

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

    # Build neighbors matrix
    neighbors = check_neighbors(state)

    # Divide the space into tiles.
    tiles = Vector{Vector{Tile}}(undef, 4)

    # Chop things into a reasonable number of x y tiles.
    nytiles = max(1, div(m, nominal_tile_size, RoundUp))
    nxtiles = max(1, div(n, nominal_tile_size, RoundUp))
    ntiles = nxtiles * nytiles

    @show ntiles

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

    visited = BitMatrix(undef, (m, n))
    fill!(visited, false)

    border = BitMatrix(undef, (m, n))
    fill!(border, false)

    return World(
        state, similar(state), border, similar(border), tiles,
        neighbors, types, volumes, areas, visited, ΔEs)
end


"""
Store the current (world.state, world.border) in (world.prev_state, world.prev_border)
"""
function savestate!(world::World)
    copy!(world.prev_state, world.state)
    copy!(world.prev_border, world.border)
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


function isborder(world::World, i::Int, j::Int)
    return count_ones(world.neighbors[i, j]) < length(NEIGHBORS)
end


function findborders!(world::World)
    m, n = size(world)
    fill!(world.border, false)
    Threads.@threads for i in 1:m
        for j in 1:n
            world.border[i, j] = isborder(world, i, j)
        end
    end
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

    # @show (E_area, E_vol, E_adhesion)

    return E_area + E_vol + E_adhesion
end


function random_neighbor(neighbors::UInt8)
    nth_zero = rand(1:(length(NEIGHBORS) - count_ones(neighbors)))
    for i in 1:length(NEIGHBORS)
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
    neighbor_count = 0
    for (i_off, j_off) in NEIGHBORS
        i, j = i_dest + i_off, j_dest + j_off
        if 1 <= i <= m && 1 <= j <= n
            if world.state[i, j] == source_state
                neighbor_count += 1
                if (length(NEIGHBORS) - count_ones(world.neighbors[i, j])) == 1
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
                if count_ones(world.neighbors[i, j]) == length(NEIGHBORS)
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

        # Don't let any cells disappear entirely
        if world.volumes[dest_state] == 1
            return Inf32
        end

        # Don't let any cells break into pieces
        if !remains_connected(world, i_dest, j_dest)
            return Inf32
        end
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
Advance the world by one iteration.
"""
function tick(world::World, rules::RuleSet, E::Float32)
    m, n = size(world)

    for quadrant in 1:4
        fill!(world.ΔEs, 0f0)

        # Threads.@threads for (k, (yrange, xrange)) in enumerate(world.tiles[quadrant])
        Threads.@threads for k in 1:length(world.tiles[quadrant])
            yrange, xrange = world.tiles[quadrant][k]

            # This can happen in tiles occuring on the edges
            if isempty(yrange) || isempty(xrange)
                continue
            end

            # Choose a random pixel with a border. (we do this just by rejection
            # sampling)
            i_source, j_source = rand(yrange), rand(xrange)
            attempts = 1000
            while (length(NEIGHBORS) - count_ones(world.neighbors[i_source, j_source])) == 0 && attempts > 0
                i_source, j_source = rand(yrange), rand(xrange)
                attempts -= 1
            end

            # @show (k, i_source, j_source, getstate(world, i_source, j_source), count_ones(world.neighbors[i_source, j_source]))

            if attempts == 0
                continue
            end

            # select the random neighbor (keep trying if it's out of bounds)
            i_off, j_off = NEIGHBORS[random_neighbor(world.neighbors[i_source, j_source])]
            i_dest, j_dest = i_source + i_off, j_source + j_off
            attempts = 20
            while !(1 <= i_dest <= m && 1 <= j_dest <= n) && attempts > 0
                i_off, j_off = NEIGHBORS[random_neighbor(world.neighbors[i_source, j_source])]
                i_dest, j_dest = i_source + i_off, j_source + j_off
                attempts -= 1
            end

            if attempts == 0
                continue
            end

            # Evaluate the energy of copying our state to the neighbors state
            ΔE = Δloss(world, rules, i_source, j_source, i_dest, j_dest)

            # @show (gettype(world, i_source, j_source), gettype(world, i_dest, j_dest))
            # @show ΔE

            # accept?
            T = 1.0
            if ΔE < 0 || rand() < exp(-ΔE/T)
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
            @assert neighbors == check_neighbors(world.state, i, j)
            if (length(NEIGHBORS) - count_ones(neighbors)) > 0
                areas[s] += 1
            end
        end
    end

    @assert volumes == world.volumes
    @assert areas == world.areas
end


const blink_window = Ref{Union{Nothing, Blink.Window}}(nothing)


function clear_world(win::Blink.Window, bgcolor="white")
    Blink.js(win, Blink.JSString("clearcells(\"$(bgcolor)\")"))
end



"""
Draw the world from scratch.
"""
function draw_world(win::Blink.Window, world::World, colors::Vector, ntypes::Int)
    m, n = size(world)
    xs = Int[]
    ys = Int[]

    function collect_and_draw(color, type, border)
        empty!(xs)
        empty!(ys)

        for i in 1:m, j in 1:n
            if gettype(world, i, j) == type && (border == world.border[i, j])
                push!(ys, i-1)
                push!(xs, j-1)
            end
        end

        if !isempty(xs)
            jscall = Blink.JSString("drawcells($m, $n, \"#$(hex(color))\", $xs, $ys)")
            Blink.js(win, jscall)
        end
    end

    for type in 1:ntypes
        # draw border cells by darkening the color
        color = colors[type]
        border_color = convert(LCHab, color)
        border_color = LCHab(border_color.l - 30.0, border_color.c, border_color.h)

        collect_and_draw(border_color, type, true)
        collect_and_draw(color, type, false )
    end
end


function redraw_world(win::Blink.Window, world::World, colors::Vector, ntypes::Int)
    m, n = size(world)
    xs = Int[]
    ys = Int[]

    function collect_and_draw(color, type, border)
        empty!(xs)
        empty!(ys)

        for i in 1:m, j in 1:n
            changed =
                world.state[i, j] != world.prev_state[i, j] ||
                world.border[i, j] != world.prev_border[i, j]

            if changed && gettype(world, i, j) == type && (border == world.border[i, j])
                push!(ys, i-1)
                push!(xs, j-1)
            end
        end

        if !isempty(xs)
            jscall = Blink.JSString("drawcells($m, $n, \"#$(hex(color))\", $xs, $ys)")
            Blink.js(win, jscall)
        end
    end

    for type in 0:ntypes
        if type > 0
            color = colors[type]
            border_color = convert(LCHab, color)
            border_color = LCHab(border_color.l - 30.0, border_color.c, border_color.h)
        else
            border_color = color = RGB(1,1,1)
        end

        collect_and_draw(border_color, type, true)
        collect_and_draw(color, type, false)
    end
end


function run(world::World, rules::RuleSet; nsteps::Int=10000, pixelsize::Int=3)
    m, n = size(world)

    if blink_window[] === nothing || !Blink.active(blink_window[])
        blink_window[] = Blink.Window()
    end

    color_scheme = ColorSchemes.colorschemes[
        :diverging_rainbow_bgymr_45_85_c67_n256]
    colors = [ColorSchemes.get(color_scheme, i, (1,rules.ntypes)) for i in 1:rules.ntypes]

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

    findborders!(world)
    clear_world(blink_window[], "white")
    draw_world(blink_window[], world, colors, rules.ntypes)
    savestate!(world)

    E = loss(world, rules)

    for step in 1:nsteps
        E = tick(world, rules, E)
        # @show (world.volumes[1], world.areas[1])
        # @show (maximum(world.volumes), maximum(world.areas))
        # @show (E, loss(world, rules))
        # sleep(0.05)
        if step % 100 == 0
            findborders!(world)
            redraw_world(blink_window[], world, colors, rules.ntypes)

            # clear_world(blink_window[], "white")
            # draw_world(blink_window[], world, colors, rules.ntypes)

            savestate!(world)
            @show (step, E)
        end
    end

    while true
        yield()
    end

    # TODO: What I really want is an interface to advance the simulation,
    # and some kind of thing to save it.
end


end # module