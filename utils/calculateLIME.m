function feature_importance = calculateLIME(model, X, numImportantFeatures)
    % Inputs:
    % - model: Trained model
    % - X: Data matrix of size (numSubjects x numFeatures)

    % Initialize variables
    numSubjects = size(X, 1);
    numFeatures = size(X, 2);
    feature_importance = zeros(numSubjects, numFeatures);
    
    % Compute the contribution of each feature for each subject using SHAP approximation
    parfor i = 1:numSubjects
        x_instance = X(i, :); % Get the i-th subject data
        
        % Calculate SHAP values of the current subject
        explainer = lime(model, "QueryPoint", x_instance, "NumImportantPredictors", numImportantFeatures);
        feature_importance(i,:) = explainer.SimpleModel.Beta;
    end
end