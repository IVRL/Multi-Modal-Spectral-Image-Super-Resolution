% Import data from a hyperspectral FLA (flat) file. 
%
% Syntax:
%         I = FLAread(filename);
%         I = FLAread(filename, scale);
%         I = FLAread(filename, rows, cols);
%         I = FLAread(filename, rect);
%         I = FLAread(filename, xstart, ystart, xstop, ystop);
%
% Description:
%     FLAread loads the content of an ENVI standard flat file (FLA). Note that the header file of the hyperspectral
%     image has to be present along with the data file in the same folder. 
% 
% Input:
%     filename:   Name of the hyperspectral image (FLA) file to be opened, including its extension
%     rows, cols: Image cube dimensions. This effectively resizes the hyperspectral image
%     scale:      Scale up to which the image is to be resized at loading time. 
%     rect:       Used to crop the image at loading time. rect is a four-element position vector[xmin ymin width height]
%                 that specifies the size and position of the crop rectangle.  
% 
% Output:
%     I: Data structure containing header information and image cube from the flat file
%
%     If you want to avoid duplicated bands, use the following code:       
%       [I.HDR.wavelength, index] = unique(I.HDR.wavelength, 'first');
%       I.HDR.bands = length(I.HDR.wavelength);
%       I.I = I.I(:, :, index);
%
% Examples:
%
%       Read a complete image cube:
%           I = FLAread('..\shared\samples\apples_small.fla');
%
%       Load an hyperspectral I with a downsampling rate:
%           I = FLAread('..\shared\samples\apples_small.fla', 0.25); or
%           I = FLAread('..\shared\samples\apples_small.fla', 'scale', 0.25);
%
%       Load an hyperspectral I and downsize to a 100by120by30 image:
%           I = FLAread('..\shared\samples\apples_small.fla', 100, 120);
%
%       Crop a region from given hyperspectral I file (XSTART, YSTART) and (XSTOP, YSTOP) specify
%           I = FLAread('..\shared\samples\apples_small.fla', 1, 30, 1, 50); which gives 50 by 30 by 30 image
%           I = FLAread('..\shared\samples\apples_small.fla', [1, 1, 100, 200]); which crops original image with
%           starting point [1, 1] and dimension 200 by 100
%
%       Extract 20th band from given hyperspectral file
%           I = FLAread('..\shared\samples\apples_small.fla', 'band', 20);
%
%       Header only: Load header information only without reading data
%           I = FLAread('..\shared\samples\apples_small.fla', 'header'); 
%           I = FLAread('..\shared\samples\apples_small.hdr');
%
% See also
%
%   FLAwrite, HSZwrite, HSZread, SLZwrite, SLZread
%
% This computer code is subject to copyright: (c) National ICT Australia Limited (NICTA) 2014 All Rights Reserved.
% Author: Ran Wei and Antonio Robles-Kelly. 

function Image = FLAread(filename, xstart, xstop, ystart, ystop)

    switch nargin
        case 5
            [I, H] = import_fla_(filename, xstart, xstop, ystart, ystop);
        case 3
            [I, H] = import_fla_(filename, xstart, xstop);
        case 2
            [I, H] = import_fla_(filename, xstart);
        case 1
            [I, H] = import_fla_(filename);
        otherwise
            error('error in input parameters');
    end
    
    
    Image.HDR   = H;
    if isequal(H, I)
        Image.I = [];
    else
        Image.I = I;
    end

end    