#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <numeric>

namespace fs = std::filesystem;

const std::vector<std::string> CLASSES = {"triangle", "rhombus", "circle", "hexagon"};

const std::vector<cv::Scalar> COLORS = {
    cv::Scalar(0, 255, 0),      // triangle - зеленый
    cv::Scalar(255, 0, 0),      // rhombus - синий
    cv::Scalar(0, 0, 255),      // circle - красный
    cv::Scalar(255, 255, 0)     // hexagon - циан
};

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

std::vector<int> nms(std::vector<Detection>& detections, float iou_threshold = 0.45f) {
    std::vector<int> keep;

    if (detections.empty()) return keep;

    std::vector<int> order(detections.size());
    std::iota(order.begin(), order.end(), 0);

    // Сортируем по уверенности (убывающий порядок)
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });

    std::vector<bool> used(detections.size(), false);

    for (int i : order) {
        if (used[i]) continue;

        keep.push_back(i);

        cv::Rect& box_i = detections[i].box;
        float area_i = box_i.area();

        for (int j : order) {
            if (i == j || used[j]) continue;

            cv::Rect& box_j = detections[j].box;
            cv::Rect intersection = box_i & box_j;
            float area_j = box_j.area();
            float inter_area = intersection.area();
            float union_area = area_i + area_j - inter_area;
            float iou = inter_area / union_area;

            if (iou > iou_threshold) {
                used[j] = true;
            }
        }
    }
    
    return keep;
}

int main() {

    std::string model_path = "runs/yolo_exp1/weights/weights/best.onnx";

    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);

    if (net.empty()) {
        std::cerr << "Ошибка загрузки модели!" << std::endl;
        return -1;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Модель загружена\n" << std::endl;

    std::string test_dir = "yolo_dataset/images/test";
    std::string out_dir = "cpp_results";
    fs::create_directories(out_dir);

    std::vector<std::string> images;
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.is_regular_file() &&
            (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
            images.push_back(entry.path().string());
        }
    }

    if (images.empty()) {
        std::cerr << "Картинки не найдены в " << test_dir << std::endl;
        return -1;
    }

    std::sort(images.begin(), images.end());
    std::cout << "📁 Найдено " << images.size() << " картинок" << std::endl;

    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int img_size = 256;
    const int num_classes = 4;
    int num_images = 5;

    int num_to_process = std::min(num_images, (int)images.size());

    for (int i = 0; i < num_to_process; ++i) {
        std::string img_path = images[i];
        std::string filename = fs::path(img_path).filename().string();

        std::cout << "[" << (i+1) << "/5] " << filename << "... " << std::flush;

        cv::Mat image = cv::imread(img_path);
        if (image.empty()) {
            std::cout << "Не удалось загрузить\n";
            continue;
        }

        int orig_w = image.cols;
        int orig_h = image.rows;

        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0,
                                              cv::Size(img_size, img_size),
                                              cv::Scalar(), true, false);

        net.setInput(blob);
        cv::Mat output = net.forward();
        // ONNX выход: [1, 8, 1344]
        // Это означает: [batch=1, channels=8, features=1344]
        // Каналы:
        // 0-3: координаты (cx, cy, w, h) для каждого объекта
        // 4-7: scores для каждого класса (0-3 индексы классов)
        
        std::vector<Detection> detections;
        

        if (output.dims == 3 && output.size[1] == 8 && output.size[2] == 1344) {

            for (int obj_idx = 0; obj_idx < 1344; ++obj_idx) {

                float cx = output.at<float>(0, 0, obj_idx);

                float cy = output.at<float>(0, 1, obj_idx);

                float w = output.at<float>(0, 2, obj_idx);

                float h = output.at<float>(0, 3, obj_idx);
                
                float score_0 = output.at<float>(0, 4, obj_idx);  // triangle
                float score_1 = output.at<float>(0, 5, obj_idx);  // rhombus
                float score_2 = output.at<float>(0, 6, obj_idx);  // circle
                float score_3 = output.at<float>(0, 7, obj_idx);  // hexagon
                
                std::vector<float> scores = {score_0, score_1, score_2, score_3};

                int best_class = -1;
                float best_score = conf_threshold;
                
                for (int c = 0; c < num_classes; ++c) {
                    if (scores[c] > best_score) {
                        best_score = scores[c];
                        best_class = c;
                    }
                }

                if (best_class == -1) {
                    continue;
                }

                float scale_x = orig_w / (float)img_size;
                float scale_y = orig_h / (float)img_size;

                int x1 = std::max(0, (int)((cx - w/2.0f) * scale_x));
                int y1 = std::max(0, (int)((cy - h/2.0f) * scale_y));
                int x2 = std::min(orig_w, (int)((cx + w/2.0f) * scale_x));
                int y2 = std::min(orig_h, (int)((cy + h/2.0f) * scale_y));

                int box_w = x2 - x1;
                int box_h = y2 - y1;

                if (box_w > 0 && box_h > 0) {
                    detections.push_back({
                        best_class,
                        best_score,
                        cv::Rect(x1, y1, box_w, box_h)
                    });
                }
            }
        }
        else {
            std::cout << "Неожиданный формат ONNX: ";
            if (output.dims >= 1) std::cout << output.size[0];
            if (output.dims >= 2) std::cout << "x" << output.size[1];
            if (output.dims >= 3) std::cout << "x" << output.size[2];
            std::cout << "\n";
            continue;
        }

        std::vector<int> keep_indices = nms(detections, nms_threshold);

        cv::Mat result = image.clone();
        
        for (int idx : keep_indices) {
            const Detection& det = detections[idx];
            
            cv::Scalar color = COLORS[det.class_id];

            cv::rectangle(result, det.box, color, 2);

            std::string label = CLASSES[det.class_id] + " " +
                               cv::format("%.2f", det.confidence);
            
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                 0.5, 2, &baseline);

            cv::rectangle(result,
                         cv::Point(det.box.x, det.box.y - text_size.height - 5),
                         cv::Point(det.box.x + text_size.width, det.box.y),
                         color, -1);

            cv::putText(result, label,
                       cv::Point(det.box.x, det.box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }

        std::string img_out_path = out_dir + "/result_" + filename;
        cv::imwrite(img_out_path, result);

        std::cout << "Обнаружено: " << keep_indices.size() << std::endl;
    }

    return 0;
}