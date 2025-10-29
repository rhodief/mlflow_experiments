#!/usr/bin/env python3
"""
Script para verificar os resultados do DAG relatorio-II após execução
"""

import os
import json
import pickle
import sys

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_assets():
    """Verifica os assets gerados"""
    print_section("📁 ASSETS GERADOS")
    
    assets_dir = "/home/rhodie/dev/lab/mlflow/entrega_02/dags/assets"
    
    if not os.path.exists(assets_dir):
        print("❌ Diretório de assets não encontrado!")
        return False
    
    expected_files = [
        "iris_data.pkl",
        "iris_preprocessed.pkl",
        "random_forest_model.pkl",
        "iris_random_forest.onnx",
        "model_asset_metadata.json"
    ]
    
    all_found = True
    for filename in expected_files:
        filepath = os.path.join(assets_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_str = f"{size/1024:.2f} KB" if size > 1024 else f"{size} bytes"
            print(f"✅ {filename:<35} ({size_str})")
        else:
            print(f"❌ {filename:<35} (NOT FOUND)")
            all_found = False
    
    return all_found

def show_metadata():
    """Mostra o conteúdo do metadata JSON"""
    print_section("📊 MODEL METADATA")
    
    metadata_path = "/home/rhodie/dev/lab/mlflow/entrega_02/dags/assets/model_asset_metadata.json"
    
    if not os.path.exists(metadata_path):
        print("❌ Metadata não encontrado. Execute o DAG primeiro.")
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📦 Model: {metadata.get('model_name', 'N/A')}")
        print(f"🔧 Type: {metadata.get('model_type', 'N/A')}")
        print(f"📄 Format: {metadata.get('format', 'N/A')}")
        print(f"📅 Created: {metadata.get('created_at', 'N/A')}")
        print(f"🔗 MLflow Run ID: {metadata.get('mlflow_run_id', 'N/A')}")
        
        if 'metrics' in metadata:
            print("\n📈 Metrics:")
            metrics = metadata['metrics']
            print(f"   • Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"   • Precision: {metrics.get('precision', 0):.4f}")
            print(f"   • Recall:    {metrics.get('recall', 0):.4f}")
            print(f"   • F1-Score:  {metrics.get('f1_score', 0):.4f}")
        
        if 'features' in metadata:
            print(f"\n🎯 Features ({metadata.get('n_features', 0)}):")
            for feature in metadata['features']:
                print(f"   • {feature}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao ler metadata: {e}")
        return False

def show_dataset_info():
    """Mostra informações do dataset"""
    print_section("📊 DATASET INFO")
    
    data_path = "/home/rhodie/dev/lab/mlflow/entrega_02/dags/assets/iris_data.pkl"
    
    if not os.path.exists(data_path):
        print("❌ Dataset não encontrado. Execute o DAG primeiro.")
        return False
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        X = data['X']
        y = data['y']
        target_names = data['target_names']
        
        print(f"\n📏 Shape: {X.shape}")
        print(f"🎯 Classes: {len(target_names)}")
        print(f"   {', '.join(target_names)}")
        
        print(f"\n📊 Class distribution:")
        from collections import Counter
        class_counts = Counter(y)
        for cls, count in sorted(class_counts.items()):
            print(f"   • {target_names[cls]}: {count} samples")
        
        print(f"\n🔢 Features:")
        for col in X.columns:
            print(f"   • {col}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao ler dataset: {e}")
        return False

def show_preprocessed_info():
    """Mostra informações dos dados preprocessados"""
    print_section("🔧 PREPROCESSED DATA INFO")
    
    prep_path = "/home/rhodie/dev/lab/mlflow/entrega_02/dags/assets/iris_preprocessed.pkl"
    
    if not os.path.exists(prep_path):
        print("❌ Dados preprocessados não encontrados. Execute o DAG primeiro.")
        return False
    
    try:
        with open(prep_path, 'rb') as f:
            data = pickle.load(f)
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        print(f"\n📊 Train set: {X_train.shape[0]} samples")
        print(f"📊 Test set:  {X_test.shape[0]} samples")
        print(f"📐 Features:  {X_train.shape[1]}")
        
        print(f"\n📊 Train class distribution:")
        from collections import Counter
        train_counts = Counter(y_train)
        for cls, count in sorted(train_counts.items()):
            print(f"   • Class {cls}: {count} samples")
        
        print(f"\n📊 Test class distribution:")
        test_counts = Counter(y_test)
        for cls, count in sorted(test_counts.items()):
            print(f"   • Class {cls}: {count} samples")
        
        print(f"\n🔧 Scaler: {type(data['scaler']).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao ler dados preprocessados: {e}")
        return False

def show_model_info():
    """Mostra informações do modelo"""
    print_section("🤖 MODEL INFO")
    
    model_path = "/home/rhodie/dev/lab/mlflow/entrega_02/dags/assets/random_forest_model.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Modelo não encontrado. Execute o DAG primeiro.")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        
        print(f"\n🌲 Model Type: {type(model).__name__}")
        print(f"🌳 N Estimators: {model.n_estimators}")
        print(f"📏 Max Depth: {model.max_depth}")
        print(f"🎲 Random State: {model.random_state}")
        print(f"🎯 N Classes: {model.n_classes_}")
        print(f"🔢 N Features: {model.n_features_in_}")
        
        print(f"\n🎯 Feature Importances:")
        for feature, importance in zip(model_data['feature_names'], model.feature_importances_):
            bar = "█" * int(importance * 50)
            print(f"   {feature:<30} {bar} {importance:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao ler modelo: {e}")
        return False

def check_mlflow_connection():
    """Verifica conexão com MLflow"""
    print_section("🔗 MLFLOW CONNECTION")
    
    try:
        import requests
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ MLflow está online: http://localhost:5000")
            print("\n📊 Para ver os experimentos:")
            print("   http://localhost:5000/#/experiments")
            print("\n📦 Para ver os modelos registrados:")
            print("   http://localhost:5000/#/models")
            return True
        else:
            print("⚠️  MLflow respondeu mas com status não esperado")
            return False
    except Exception as e:
        print(f"❌ MLflow não está acessível: {e}")
        print("   Execute: docker-compose restart mlflow")
        return False

def main():
    """Função principal"""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "RESULTADOS DO DAG relatorio-II" + " "*23 + "║")
    print("╚" + "═"*68 + "╝")
    
    checks = [
        ("Assets", check_assets),
        ("Metadata", show_metadata),
        ("Dataset", show_dataset_info),
        ("Preprocessed", show_preprocessed_info),
        ("Model", show_model_info),
        ("MLflow", check_mlflow_connection)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Erro ao executar check '{name}': {e}")
            results.append((name, False))
    
    # Summary
    print_section("📋 SUMMARY")
    all_ok = True
    for name, result in results:
        status = "✅ OK" if result else "❌ FAILED"
        print(f"{name:.<30} {status}")
        if not result:
            all_ok = False
    
    print("\n" + "="*70)
    
    if all_ok:
        print("\n🎉 Tudo certo! O pipeline executou com sucesso.")
        print("\n📚 Próximos passos:")
        print("   1. Acesse MLflow UI: http://localhost:5000")
        print("   2. Verifique o experimento 'relatorio-II-iris-pipeline'")
        print("   3. Veja o modelo registrado 'iris-random-forest-classifier'")
        print("   4. Analise as métricas e artefatos")
        return 0
    else:
        print("\n⚠️  Alguns checks falharam.")
        print("   Execute o DAG no Airflow se ainda não o fez:")
        print("   http://localhost:8080")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário")
        sys.exit(1)
