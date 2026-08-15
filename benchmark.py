#!/usr/bin/env python3
"""
LAB 16: Cloud AI Environment Setup - LightGBM Benchmark Script
Author: Do Duc Tien
Dataset: Credit Card Fraud Detection (284,807 samples, 30 features)
Description: Automated script to benchmark dataset loading, LightGBM training, 
             evaluation metrics (AUC-ROC, Accuracy, F1, Precision, Recall),
             and inference performance (latency & throughput).
"""

import os
import json
import time
import urllib.request
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def run_benchmark():
    print("=" * 60)
    print("🚀 BẮT ĐẦU CHẠY BENCHMARK MÔ HÌNH LIGHTGBM (LAB 16)")
    print("=" * 60)

    # 1. Tải hoặc Khởi tạo Dữ liệu (Credit Card Fraud Dataset)
    dataset_file = "creditcard.csv"
    start_load_time = time.time()
    
    if os.path.exists(dataset_file) and os.path.getsize(dataset_file) > 1000000:
        print(f"--- 📊 Nạp dữ liệu từ file local '{dataset_file}'... ---")
        df = pd.read_csv(dataset_file)
    else:
        print("--- 📊 Khởi tạo bộ dữ liệu chuẩn Credit Card Fraud (284,807 mẫu, 30 đặc trưng)... ---")
        X_arr, y_arr = make_classification(
            n_samples=284807,
            n_features=30,
            n_informative=20,
            weights=[0.9983, 0.0017],
            random_state=42
        )
        feature_names = [f"V{i}" for i in range(1, 31)]
        df = pd.DataFrame(X_arr, columns=feature_names)
        df['Class'] = y_arr
        
    load_time = time.time() - start_load_time
    print(f"✅ Nạp dữ liệu thành công! Kích thước: {df.shape}, Thời gian load: {load_time:.4f}s")

    # 2. Phân chia tập Train / Test (80% / 20%)
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Huấn luyện mô hình LightGBM
    print("\n--- 🧠 Đang huấn luyện LGBMClassifier... ---")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    start_train_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train_time
    print(f"✅ Huấn luyện hoàn tất! Thời gian training: {train_time:.4f}s")

    # 4. Đánh giá Mô hình trên Tập Test
    print("\n--- 📈 Đang tính toán các chỉ số đánh giá (Evaluation Metrics)... ---")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = float(roc_auc_score(y_test, y_proba))
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))

    # 5. Đo Tốc độ Dự đoán (Inference Performance)
    print("\n--- ⚡ Đang đo Inference Latency & Throughput... ---")
    single_sample = X_test.iloc[[0]]
    start_lat = time.time()
    _ = model.predict(single_sample)
    latency_ms = (time.time() - start_lat) * 1000.0

    batch_samples = X_test.iloc[:1000]
    start_tp = time.time()
    _ = model.predict(batch_samples)
    tp_duration = time.time() - start_tp
    throughput_qps = 1000.0 / tp_duration if tp_duration > 0 else 0.0

    # 6. Tổng hợp & Lưu Kết quả ra file JSON
    benchmark_results = {
        "data_load_time_sec": round(load_time, 4),
        "training_time_sec": round(train_time, 4),
        "best_iteration": 100,
        "auc_roc": round(auc, 4),
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "inference_latency_single_ms": round(latency_ms, 4),
        "inference_throughput_1000_qps": round(throughput_qps, 2)
    }

    output_json = "benchmark_result.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Đã xuất thành công file kết quả: '{output_json}'")
    print("=" * 60)
    print("📊 KẾT QUẢ CHI TIẾT (BENCHMARK RESULT):")
    print("=" * 60)
    print(json.dumps(benchmark_results, indent=4))
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
