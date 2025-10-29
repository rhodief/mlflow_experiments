#!/usr/bin/env python3
"""
Script de teste para validar o deploy da API Iris
"""

import requests
import json
import time
import sys


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def test_root_endpoint(base_url):
    """Test the root endpoint"""
    print_header("Testando Endpoint Raiz")
    
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Status: {data.get('status')}")
        print(f"✓ Mensagem: {data.get('message')}")
        print(f"✓ Modelo: {data.get('model')}")
        print(f"✓ Versão: {data.get('version')}")
        
        return True
    except Exception as e:
        print(f"✗ Erro: {str(e)}")
        return False


def test_metrics_endpoint(base_url):
    """Test the metrics endpoint"""
    print_header("Testando Endpoint de Métricas")
    
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Total de predições: {data.get('total_predictions')}")
        print(f"✓ Latência média: {data.get('mean_latency_ms')} ms")
        
        model_info = data.get('model_info', {})
        print(f"\nInformações do Modelo:")
        print(f"  - Nome: {model_info.get('name')}")
        print(f"  - Tipo: {model_info.get('type')}")
        print(f"  - Formato: {model_info.get('format')}")
        print(f"  - Accuracy: {model_info.get('accuracy'):.4f}")
        print(f"  - Precision: {model_info.get('precision'):.4f}")
        print(f"  - Recall: {model_info.get('recall'):.4f}")
        print(f"  - F1-Score: {model_info.get('f1_score'):.4f}")
        print(f"  - Features: {', '.join(model_info.get('features', []))}")
        
        return True
    except Exception as e:
        print(f"✗ Erro: {str(e)}")
        return False


def test_prediction_endpoint(base_url):
    """Test the prediction endpoint"""
    print_header("Testando Endpoint de Predição")
    
    # Test cases for different iris species
    test_cases = [
        {
            "name": "Iris Setosa",
            "data": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            "expected": "setosa"
        },
        {
            "name": "Iris Versicolor",
            "data": {
                "sepal_length": 5.9,
                "sepal_width": 3.0,
                "petal_length": 4.2,
                "petal_width": 1.5
            },
            "expected": "versicolor"
        },
        {
            "name": "Iris Virginica",
            "data": {
                "sepal_length": 6.3,
                "sepal_width": 2.9,
                "petal_length": 5.6,
                "petal_width": 1.8
            },
            "expected": "virginica"
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\nTestando: {test_case['name']}")
        print(f"Dados: {json.dumps(test_case['data'], indent=2)}")
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=test_case['data'],
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            prediction = data.get('prediction')
            
            print(f"✓ Predição: {prediction}")
            print(f"✓ Classe: {data.get('prediction_class')}")
            print(f"✓ Duração: {data.get('duration_ms')} ms")
            print(f"✓ Probabilidades:")
            
            for species, prob in data.get('probabilities', {}).items():
                print(f"    {species}: {prob:.4f}")
            
            if prediction == test_case['expected']:
                print(f"✓ Predição correta!")
            else:
                print(f"✗ Predição incorreta! Esperado: {test_case['expected']}, Obtido: {prediction}")
                all_passed = False
                
        except Exception as e:
            print(f"✗ Erro: {str(e)}")
            all_passed = False
    
    return all_passed


def test_model_info_endpoint(base_url):
    """Test the model info endpoint"""
    print_header("Testando Endpoint de Informações do Modelo")
    
    try:
        response = requests.get(f"{base_url}/model-info", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Nome do modelo: {data.get('model_name')}")
        print(f"✓ Tipo: {data.get('model_type')}")
        print(f"✓ Formato: {data.get('format')}")
        print(f"✓ Criado em: {data.get('created_at')}")
        print(f"✓ MLflow Run ID: {data.get('mlflow_run_id')}")
        print(f"✓ MLflow Experiment: {data.get('mlflow_experiment')}")
        
        return True
    except Exception as e:
        print(f"✗ Erro: {str(e)}")
        return False


def main():
    """Main test function"""
    # Configuration
    HOST = "localhost"
    PORT = 8884
    BASE_URL = f"http://{HOST}:{PORT}"
    
    print("\n" + "="*60)
    print("  Script de Teste - API de Classificação Iris")
    print("="*60)
    print(f"\nURL Base: {BASE_URL}")
    
    # Wait for API to be ready
    print("\nAguardando API iniciar...")
    for i in range(5):
        try:
            requests.get(f"{BASE_URL}/", timeout=2)
            print("✓ API está respondendo!")
            break
        except:
            time.sleep(2)
            print(f"  Tentativa {i+1}/5...")
    else:
        print("✗ API não está respondendo após 10 segundos")
        sys.exit(1)
    
    # Run tests
    results = {
        "Root Endpoint": test_root_endpoint(BASE_URL),
        "Metrics Endpoint": test_metrics_endpoint(BASE_URL),
        "Prediction Endpoint": test_prediction_endpoint(BASE_URL),
        "Model Info Endpoint": test_model_info_endpoint(BASE_URL)
    }
    
    # Print summary
    print_header("Resumo dos Testes")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSOU" if result else "✗ FALHOU"
        print(f"{test_name:30} {status}")
    
    print(f"\nTotal: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n✓ Todos os testes passaram!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} teste(s) falharam")
        sys.exit(1)


if __name__ == "__main__":
    main()
