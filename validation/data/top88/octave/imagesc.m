% (part of HyperShape, MIT licence; see run_top88.m)
% headless stand-in for the plot of top88.m: keeps the image (1 - xPhys) instead of drawing it
function imagesc(A)
  global LASTIMG
  LASTIMG = A;
end
