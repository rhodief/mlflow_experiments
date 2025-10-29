#!/usr/bin/env python3
"""
Test script for Iris Classification API
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("=" * 60)
    print("Testing Health Check (GET /)...")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200


def test_predict():
    """Test prediction endpoint"""
    print("=" * 60)
    print("Testing Predictions (POST /predict)...")
    print("=" * 60)
    
    # Test cases for each iris species
    test_cases = [
        {
            "name": "Setosa",
            "data": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        },
        {
            "name": "Versicolor",
            "data": {
                "sepal_length": 5.9,
                "sepal_width": 3.0,
                "petal_length": 4.2,
                "petal_width": 1.5
            }
        },
        {
            "name": "Virginica",
            "data": {
                "sepal_length": 6.3,
                "sepal_width": 2.9,
                "petal_length": 5.6,
                "petal_width": 1.8
            }
        }
    ]
    
    success_count = 0
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print(f"Input: {test_case['data']}")
        
        response = requests.post(f"{API_URL}/predict", json=test_case['data'])
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Status: {response.status_code}")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Class: {result['prediction_class']}")
            print(f"  Duration: {result['duration_ms']} ms")
            print(f"  Probabilities:")
            for species, prob in result['probabilities'].items():
                print(f"    - {species}: {prob:.4f}")
            success_count += 1
        else:
            print(f"✗ Status: {response.status_code}")
            print(f"  Error: {response.text}")
    
    print()
    return success_count == len(test_cases)


def test_metrics():
    """Test metrics endpoint"""
    print("=" * 60)
    print("Testing Metrics (GET /metrics)...")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/metrics")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        metrics = response.json()
        print(f"\n✓ Metrics Retrieved:")
        print(f"  Total Predictions: {metrics['total_predictions']}")
        print(f"  Mean Latency: {metrics['mean_latency_ms']:.3f} ms")
        print(f"\n  Model Info:")
        print(f"    - Name: {metrics['model_info']['name']}")
        print(f"    - Type: {metrics['model_info']['type']}")
        print(f"    - Accuracy: {metrics['model_info']['accuracy']:.4f}")
        print(f"    - F1-Score: {metrics['model_info']['f1_score']:.4f}")
        print(f"    - Features: {metrics['model_info']['n_features']}")
        print()
        return True
    else:
        print(f"✗ Error: {response.text}")
        print()
        return False


def test_model_info():
    """Test model info endpoint"""
    print("=" * 60)
    print("Testing Model Info (GET /model-info)...")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        info = response.json()
        print(f"\n✓ Model Information:")
        print(f"  Name: {info['model_name']}")
        print(f"  Type: {info['model_type']}")
        print(f"  Format: {info['format']}")
        print(f"  Created: {info['created_at']}")
        print(f"  Features: {', '.join(info['features'])}")
        print()
        return True
    else:
        print(f"✗ Error: {response.text}")
        print()
        return False


def run_load_test(n_requests=10):
    """Run a simple load test"""
    print("=" * 60)
    print(f"Running Load Test ({n_requests} requests)...")
    print("=" * 60)
    
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    latencies = []
    start_time = time.time()
    
    for i in range(n_requests):
        request_start = time.time()
        response = requests.post(f"{API_URL}/predict", json=test_data)
        request_end = time.time()
        
        if response.status_code == 200:
            latency = (request_end - request_start) * 1000
            latencies.append(latency)
            print(f"  Request {i+1}/{n_requests}: {latency:.2f} ms")
    
    total_time = (time.time() - start_time) * 1000
    
    if latencies:
        print(f"\n✓ Load Test Results:")
        print(f"  Total Requests: {len(latencies)}")
        print(f"  Total Time: {total_time:.2f} ms")
        print(f"  Throughput: {len(latencies) / (total_time/1000):.2f} req/s")
        print(f"  Min Latency: {min(latencies):.2f} ms")
        print(f"  Max Latency: {max(latencies):.2f} ms")
        print(f"  Mean Latency: {sum(latencies)/len(latencies):.2f} ms")
        print()
        return True
    else:
        print("✗ Load test failed")
        print()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("IRIS CLASSIFICATION API - TEST SUITE")
    print("=" * 60)
    print()
    
    try:
        # Run tests
        results = {
            "Health Check": test_health(),
            "Predictions": test_predict(),
            "Metrics": test_metrics(),
            "Model Info": test_model_info(),
            "Load Test": run_load_test(10)
        }
        
        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name:.<40} {status}")
        
        total_passed = sum(results.values())
        total_tests = len(results)
        
        print()
        print(f"Total: {total_passed}/{total_tests} tests passed")
        print("=" * 60)
        print()
        
        return total_passed == total_tests
        
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API")
        print("  Make sure the API is running on http://localhost:8000")
        print("  Run: ./run.sh")
        print()
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        print()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
