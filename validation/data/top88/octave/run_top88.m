% Regenerates the reference files of validation/data/top88 from the user's own copy of top88.m (Andreassen et al.
% 2011; not included here), run unmodified. The functions top88.m calls to print and plot (fprintf, colormap,
% imagesc, caxis, axis, drawnow) are shadowed by the stand-ins of this folder: fprintf keeps the numbers of each
% iteration (loop, c, vol, change) at full precision and prints nothing, imagesc keeps the image 1 - xPhys.
% Run from this folder (so that the stand-ins come first), e.g. in Octave 9.1:
%   run_top88('/path/to/folder/of/top88', 60, 20, 0.5, 3, 2.4, 2, '60x20')
%   run_top88('/path/to/folder/of/top88', 150, 50, 0.5, 3, 6.0, 2, '150x50')
% It writes ../top88_<tag>_hist.csv and ../top88_<tag>_xPhys.csv (this file is part of HyperShape, MIT licence).
function run_top88(top88dir, nelx, nely, volfrac, penal, rmin, ft, tag)
  global LASTIMG HIST
  HIST = zeros(0, 4);
  addpath(top88dir);
  top88(nelx, nely, volfrac, penal, rmin, ft);
  f = fopen(['../top88_' tag '_hist.csv'], 'w');
  fputs(f, sprintf('it,c,vol,change\n'));
  for k = 1:rows(HIST)
    fputs(f, sprintf('%d,%.17g,%.17g,%.17g\n', HIST(k, 1), HIST(k, 2), HIST(k, 3), HIST(k, 4)));
  end
  fclose(f);
  xPhys = 1 - LASTIMG;
  f = fopen(['../top88_' tag '_xPhys.csv'], 'w');
  fputs(f, sprintf('%.17g\n', xPhys(:)));
  fclose(f);
  disp(sprintf('%s: %d iterations, last c = %.10f, sum(xPhys) = %.17g', tag, rows(HIST), HIST(end, 2), sum(xPhys(:))));
end
