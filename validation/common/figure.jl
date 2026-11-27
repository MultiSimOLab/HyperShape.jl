# SVG figures without a plotting package: 0/1 designs side by side (the reference first) and histories of the
# objective. Used by otomori_levelset88.jl; the benchmarks use it for every figure with a reference.

"""
    design_figure(path; designs, curves, vlines = [], ylabel, xlabel, note = "", xmax = 200,
                  xticks = (1, 50, 100, 150, 200), ystep = 50, cell = 4, stack = false, yfrom = 1, legend_top = false)

Writes an SVG with the 0/1 designs of `designs` (pairs `(str, (title1, title2))`, `str[i, j]` true where cell `(i, j)`
is solid, `i` along x, `j` along y from the bottom; the reference goes first) and a plot of `curves`
(`(; x, y, color, dash, label)`, `dash` `nothing` or an SVG dash pattern) with vertical dashed lines `vlines`
(`(x, color)`), the x axis from 1 to `xmax`. With `stack = true` the designs go one above the other (for wide
domains) and the plot beside them has their height. The range of the y axis is that of the points with `x ≥ yfrom`
(the first iterations, far above the rest, are clipped at the top). The legend goes at the bottom of the plot, or at
the top with `legend_top = true`.
"""
function design_figure(path; designs, curves, vlines = (), ylabel, xlabel, note = "", xmax = 200,
                       xticks = (1, 50, 100, 150, 200), ystep = 50, cell = 4, stack = false, yfrom = 1, legend_top = false)
  W0 = 20
  NX, NY = size(first(designs)[1])
  nd = length(designs)
  pw, ph = NX * cell, NY * cell
  io = IOBuffer()
  ncol = stack ? 1 : nd
  dh = stack ? nd * (ph + 40) + (nd - 1) * 10 : ph + 40          # bottom of the designs
  gh = dh - 70
  W, H = (ncol + 1) * W0 + ncol * pw + 560, dh + 50
  println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="sans-serif" font-size="13" shape-rendering="crispEdges">""")
  println(io, """<rect width="$W" height="$H" fill="white"/>""")
  function design(x0, str, title, yo = 0)
    println(io, """<text x="$x0" y="$(15 + yo)">$(title[1])</text><text x="$x0" y="$(31 + yo)">$(title[2])</text>""")
    println(io, """<rect x="$x0" y="$(40 + yo)" width="$pw" height="$ph" fill="none" stroke="#999"/>""")
    for j in 1:NY
      y = 40 + yo + (NY - j) * cell
      i = 1
      while i <= NX                                       # runs of material along a row
        str[i, j] || (i += 1; continue)
        i0 = i
        while i <= NX && str[i, j]
          i += 1
        end
        # a slight overlap with the next row: no gaps between rows
        println(io, """<rect x="$(x0 + (i0 - 1) * cell)" y="$y" width="$((i - i0) * cell)" height="$(cell + 0.5)" fill="black"/>""")
      end
    end
  end
  for (k, (str, title)) in enumerate(designs)
    stack ? design(W0, str, title, (k - 1) * (ph + 50)) : design(k * W0 + (k - 1) * pw, str, title)
  end
  x0, y0, gw = (ncol + 1) * W0 + ncol * pw + 100, 40, 420
  Jmin = minimum(minimum(y for (x, y) in zip(c.x, c.y) if x >= yfrom) for c in curves)
  Jmax = maximum(maximum(y for (x, y) in zip(c.x, c.y) if x >= yfrom) for c in curves)
  lo, hi = ystep * floor(Jmin / ystep), ystep * ceil(Jmax / ystep)
  X(it) = x0 + gw * (it - 1) / (xmax - 1)
  Y(J) = y0 + gh * (hi - J) / (hi - lo)
  println(io, """<text x="$x0" y="22">$ylabel</text>""")
  println(io, """<rect x="$x0" y="$y0" width="$gw" height="$gh" fill="none" stroke="#999"/>""")
  for J in lo:ystep:hi
    println(io, """<line x1="$(x0 - 4)" x2="$x0" y1="$(Y(J))" y2="$(Y(J))" stroke="#999"/><text x="$(x0 - 8)" y="$(Y(J) + 4)" text-anchor="end">$(Int(J))</text>""")
  end
  for it in xticks
    println(io, """<line x1="$(X(it))" x2="$(X(it))" y1="$(y0 + gh)" y2="$(y0 + gh + 4)" stroke="#999"/><text x="$(X(it))" y="$(y0 + gh + 18)" text-anchor="middle">$it</text>""")
  end
  println(io, """<text x="$(x0 + gw / 2)" y="$(y0 + gh + 36)" text-anchor="middle">$xlabel</text>""")
  pts(xs, ys) = join((@sprintf("%.1f,%.1f", X(a), Y(min(b, hi))) for (a, b) in zip(xs, ys)), " ")
  dash(c) = c.dash === nothing ? "" : " stroke-dasharray=\"$(c.dash)\""
  for c in curves
    println(io, """<polyline fill="none" stroke="$(c.color)" stroke-width="1.5"$(dash(c)) points="$(pts(c.x, c.y))"/>""")
  end
  for (x, color) in vlines
    println(io, """<line x1="$(X(x))" x2="$(X(x))" y1="$y0" y2="$(y0 + gh)" stroke="$color" stroke-dasharray="3,3"/>""")
  end
  nc = length(curves)
  for (k, c) in enumerate(curves)
    yl = legend_top ? y0 + 20k : y0 + gh - 20 * (nc - k + 1)
    println(io, """<line x1="$(x0 + 190)" x2="$(x0 + 215)" y1="$yl" y2="$yl" stroke="$(c.color)" stroke-width="1.5"$(dash(c))/><text x="$(x0 + 220)" y="$(yl + 4)">$(c.label)</text>""")
  end
  isempty(note) || println(io, """<text x="$x0" y="$(y0 + gh + 52)" font-size="11">$note</text>""")
  println(io, "</svg>")
  write(path, take!(io))
end
