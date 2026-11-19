"""
    tag_adjacent_cells!(labels, newtag, model, tag; layers = 1) -> labels

Adds to `labels` the tag `newtag` (arguments in the order of `add_tag_from_vertex_filter!`): the cells of `model` that
contain an entity of `tag` (a surface, an edge or a vertex: the faces where a load or a support acts; if `tag` has
cells, those cells), and `layers − 1` more layers of cells around them (cells that share a vertex with the previous
layer), with all their faces, edges and vertices. It turns a boundary tag into a frozen
zone of whole cells (see [`FieldVariable`](@ref)), for P0 and P1 design spaces and any mesh:

```julia
tag_adjacent_cells!(labels, "load_zone", model, "load")           # the layer of cells under the load
Vρ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity = :H1, dirichlet_tags = "load_zone")
```
"""
function tag_adjacent_cells!(labels::Gridap.Geometry.FaceLabeling, newtag, model::Gridap.Geometry.DiscreteModel, tag;
                             layers::Integer = 1)
  layers >= 1 || throw(ArgumentError("tag_adjacent_cells!: layers must be at least 1"))
  newtag = String(newtag)
  newtag in labels.tag_to_name && throw(ArgumentError("tag_adjacent_cells!: the tag $newtag exists already"))
  for t in (tag isa AbstractVector ? tag : (tag,))
    String(t) in labels.tag_to_name || throw(ArgumentError("tag_adjacent_cells!: no tag named $t in the labels"))
  end
  topo = Gridap.Geometry.get_grid_topology(model)
  D = num_cell_dims(model)
  all(d -> length(labels.d_to_dface_to_entity[d+1]) == Gridap.Geometry.num_faces(topo, d), 0:D) ||
    throw(ArgumentError("tag_adjacent_cells!: the labels are not those of this model"))
  cellmask = Gridap.Geometry.get_face_mask(labels, tag, D)
  cells = falses(Gridap.Geometry.num_faces(topo, D))
  if any(cellmask)                       # a tag with cells: those cells
    cells .= cellmask
  else
    for d in 0:D-1
      mask = Gridap.Geometry.get_face_mask(labels, tag, d)
      any(mask) || continue
      dface_to_cells = Gridap.Geometry.get_faces(topo, d, D)
      for f in findall(mask), c in dface_to_cells[f]
        cells[c] = true
      end
    end
  end
  any(cells) || throw(ArgumentError("tag_adjacent_cells!: the tag $tag has no entity in the model"))
  vertex_to_cells = Gridap.Geometry.get_faces(topo, 0, D)
  cell_to_vertices = Gridap.Geometry.get_faces(topo, D, 0)
  for _ in 2:layers
    grown = copy(cells)
    for c in findall(cells), v in cell_to_vertices[c], c2 in vertex_to_cells[v]
      grown[c2] = true
    end
    cells = grown
  end
  # a labeling with one entity: the cells and everything they contain; merged with the existing tags
  unset = Gridap.Geometry.UNSET
  d_to_dface_to_entity = [fill(Int32(unset), Gridap.Geometry.num_faces(topo, d)) for d in 0:D]
  for d in 0:D
    cell_to_dfaces = Gridap.Geometry.get_faces(topo, D, d)
    e = d_to_dface_to_entity[d+1]
    for c in findall(cells), f in cell_to_dfaces[c]
      e[f] = Int32(1)
    end
  end
  merge!(labels, Gridap.Geometry.FaceLabeling(d_to_dface_to_entity, [Int32[1]], [newtag]))
  labels
end
