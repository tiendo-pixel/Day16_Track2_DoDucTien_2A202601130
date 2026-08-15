# Day 16 readiness — 2A202601130

- **Cloud path chính:** GCP (Google Cloud Platform) / Cloud AI Compute Node
- **Cloud identity:** Đã che email / account ID / project ID riêng (`ai-lab-16-gcp-XXXXXX`)
- **Budget alert:** Đã tạo; recipient đã cấu hình ($0 threshold alert)
- **CLI:** Lệnh `gcloud`, `terraform`, `git`, `python3`, `nvidia-smi` đã chạy và output đã che thông tin nhạy cảm
- **Terraform:** Version đã kiểm tra (`terraform-gcp/` đạt `terraform validate` thành công 100%)
- **Local:** Python 3.12, Git (`tiendo-pixel`), make, curl; Docker pending trước Day 18
- **Hugging Face:** Đã tạo Read token (bảo mật, không ghi token trực tiếp ra bài nộp)
- **GPU quota:** Không thể xin tăng do Free Trial / Billing Hold trên GCP -> Đã linh hoạt chuyển đổi sử dụng máy chủ Cloud GPU Tesla T4 (15GB VRAM, CUDA 13.0, Driver 580.82.07, 50GB RAM, 8 vCPUs)
- **Fallback & Deliverables:** Đã thực thi thành công phương án Fallback theo hướng dẫn với script [`benchmark.py`](file:///d:/Day16_Track2_DoDucTien_2A202601130/benchmark.py), huấn luyện mô hình LightGBM trên bộ dữ liệu Credit Card Fraud Detection (284,807 mẫu x 30 đặc trưng), xuất file chỉ số [`benchmark_result.json`](file:///d:/Day16_Track2_DoDucTien_2A202601130/benchmark_result.json) (Training time: 1.894s, Accuracy: 99.34%, Throughput: 313,686.64 QPS) và báo cáo tài nguyên [`resource_usage.txt`](file:///d:/Day16_Track2_DoDucTien_2A202601130/resource_usage.txt)
