import pytest
from src.feature_engineering import create_features

def test_create_features():
    # Sample input data
    input_data = {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    }
    
    expected_output = {
        'new_feature': [5, 7, 9]  # Example expected output
    }
    
    # Call the function to test
    output_data = create_features(input_data)
    
    # Assert the output matches the expected output
    assert output_data == expected_output

def test_create_features_empty_input():
    input_data = {}
    
    expected_output = {}
    
    output_data = create_features(input_data)
    
    assert output_data == expected_output

def test_create_features_invalid_input():
    input_data = {
        'feature1': [1, 2, 'invalid'],
        'feature2': [4, 5, 6]
    }
    
    with pytest.raises(ValueError):
        create_features(input_data)