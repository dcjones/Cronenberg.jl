
Plan:

We simulate a a `m,n` lattice. Each entry is given a cell identifier, or zero
representing an unpopulated region.

We have to also have per-cell arrays keeping track of area, circumfrence, and
I suppose contact to adjacent cells.

We then are doing a standard Metropolis algorithm, where proposals take the
form of copying a cell state to a neighbor.

This should be pretty easy, but gets more complicated if we want to allow cell
division and death.

The other main complication is that we want to simulated on the order of 10^5
cells at a relativelry high resolution. Doing so requires many updated, so a
sequential algorithm may be unbearably slow. Instead we need to a use a
parallel scheme.

It seems like the best way is to divide the lattice into squares, and each
square into quadrants 1..4, then simulateously make proposals in all the
quadrant k's concurrently. If the squares are larger then the range of
interaction of the energy function, this should be a good approximation.

We should also look into using Blink and webgl to have a nice view of this
simulation as it runs, unlike morpheous which only does post-hoc video files.

