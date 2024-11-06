function dataOut = loadEnhancedImage(filename)
    data = load(filename);
    
    if isfield(data, 'enhancedImage')
        dataOut = data.enhancedImage;
    elseif isfield(data, 'imSub')
        dataOut = data.imSub;
    else
        error('The file does not contain "enhancedImage" or "imSub" fields.');
    end
end