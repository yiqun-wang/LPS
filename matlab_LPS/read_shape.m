function shape = read_shape( shape_path )

[vertex, face] = read_off(shape_path);
shape.VERT=vertex';
shape.TRIV=face';
shape.n = length(vertex);
shape.m = length(face);

end

