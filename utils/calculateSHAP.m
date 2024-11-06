function feature_importance = calculateSHAP(model, X)
    % Inputs:
    % - model: Trained model
    % - X: Data matrix of size (numSubjects x numFeatures)

    % Initialize variables
    numSubjects = size(X, 1);
    numFeatures = size(X, 2);
    feature_importance = zeros(numSubjects, numFeatures);
    
    % Compute the contribution of each feature for each subject using SHAP approximation
    for i = 1:numSubjects
        x_instance = X(i, :); % Get the i-th subject data
        
        % Calculate SHAP values of the current subject
        explainer = shapley(model, QueryPoints=x_instance, UseParallel=true);
        feature_importance(i,:) = explainer.Shapley.AD;
    end
end