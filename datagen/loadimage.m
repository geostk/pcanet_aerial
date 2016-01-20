function [image] = loadimage (file`name)

    fileID = fopen(filename, 'r');
    read_image = fread(fileID, [256 256],'uint8=>uint8');
    read_image = im2double(read_image);

end
