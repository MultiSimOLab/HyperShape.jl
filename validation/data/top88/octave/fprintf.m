% (part of HyperShape, MIT licence; see run_top88.m)
% headless stand-in for the print of top88.m: keeps the numbers of each iteration (loop, c, vol, change) at full
% precision and prints nothing
function fprintf(fmt, varargin)
  global HIST
  if numel(varargin) == 4
    HIST(end+1, :) = [varargin{:}];
  end
end
