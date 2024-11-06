function data = loadPD(filename)
    % loadPD loads a .mat file and returns the 'imSub' field.
    % If the filename starts with '._', remove the prefix and load the correct file.
    
    [filePath, name, ext] = fileparts(filename);
    
    % Check if the filename starts with '._'
    if startsWith(name, '._')
        % Remove the '._' prefix from the filename
        name = erase(name, '._');
        
        % Construct the new full file path
        newFilename = fullfile(filePath, [name ext]);
        % fprintf('Renaming file: %s to %s\n', filename, newFilename);
        
        % Update the filename to the corrected one
        filename = newFilename;
    end
    
    % Load the file with the corrected filename
    dataStruct = load(filename);
    
    % Make sure the file has the 'imSub' variable
    if isfield(dataStruct, 'imSub')
        data = dataStruct.imSub;  % Return the 'imSub' field
    else
        error('Field "imSub" not found in file: %s', filename);
    end
end