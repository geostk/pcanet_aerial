function [out] = VectorizeCellElements(in)

    NumEl = length(in);
    out = cell(NumEl, 1);

    for i = 1:NumEl
        out{i} = in{i}(:);
    end

end
