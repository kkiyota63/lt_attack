import json

def fix_json_format():
    """
    Fix XGBoost JSON format to use feature indices instead of names
    """
    # Load feature mapping
    with open('feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    feature_to_index = {name: idx for idx, name in enumerate(feature_info['features'])}
    
    # Load model JSON
    with open('base_fraud_model.json', 'r') as f:
        model_data = json.load(f)
    
    def fix_tree(node):
        if 'split' in node:
            # Convert feature name to index
            feature_name = node['split']
            if feature_name in feature_to_index:
                node['split'] = feature_to_index[feature_name]
            else:
                print(f"Warning: Unknown feature {feature_name}")
        
        if 'children' in node:
            for child in node['children']:
                fix_tree(child)
    
    # Fix all trees
    for tree in model_data:
        fix_tree(tree)
    
    # Save fixed model
    with open('base_fraud_model.json', 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print("Fixed JSON format - converted feature names to indices")
    print(f"Feature mapping: {feature_to_index}")

if __name__ == "__main__":
    fix_json_format()