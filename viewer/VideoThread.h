#include <iostream>
#include <opencv2/opencv.hpp>

#undef slots
#include <torch/script.h>
#include <torch/torch.h>
#define slots Q_SLOTS

#include <qthread.h>
#include <qmutex.h>
#include <QImage>
#include <QDebug>

struct DetectionBox {
    float xmin, ymin, xmax, ymax, conf;
};

// YOLOv8 letterbox: 종횡비를 유지한 채 target_size 정사각형에 맞춰 리사이즈하고
// 남는 부분은 회색(114,114,114)으로 패딩. scale/pad는 박스 좌표를 원본 프레임으로
// 되돌릴 때 필요해서 함께 반환함.
struct LetterboxInfo {
    float scale;
    int pad_x;
    int pad_y;
};

inline cv::Mat letterbox(const cv::Mat& src, int target_size, LetterboxInfo& info) {
    int w = src.cols, h = src.rows;
    float scale = std::min((float)target_size / w, (float)target_size / h);
    int new_w = static_cast<int>(std::round(w * scale));
    int new_h = static_cast<int>(std::round(h * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    cv::Mat out(target_size, target_size, src.type(), cv::Scalar(114, 114, 114));
    int pad_x = (target_size - new_w) / 2;
    int pad_y = (target_size - new_h) / 2;
    resized.copyTo(out(cv::Rect(pad_x, pad_y, new_w, new_h)));

    info.scale = scale;
    info.pad_x = pad_x;
    info.pad_y = pad_y;
    return out;
}

inline float calculate_iou(const DetectionBox& box1, const DetectionBox& box2) {
    float x1 = std::max(box1.xmin, box2.xmin);
    float y1 = std::max(box1.ymin, box2.ymin);
    float x2 = std::min(box1.xmax, box2.xmax);
    float y2 = std::min(box1.ymax, box2.ymax);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin);
    float area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin);
    float union_val = area1 + area2 - inter;

    return inter / (union_val + 1e-6f);
}

inline std::vector<DetectionBox> nms(std::vector<DetectionBox>& boxes, float iou_threshold = 0.45f) {
    if (boxes.empty()) return {};

    std::sort(boxes.begin(), boxes.end(), [](const DetectionBox& a, const DetectionBox& b) {
        return a.conf > b.conf;
        });

    std::vector<DetectionBox> keep;
    while (!boxes.empty()) {
        DetectionBox best = boxes.front();
        boxes.erase(boxes.begin());
        keep.push_back(best);

        std::vector<DetectionBox> remain;
        for (const auto& box : boxes) {
            if (calculate_iou(best, box) < iou_threshold) {
                remain.push_back(box);
            }
        }
        boxes = remain;
    }
    return keep;
}

// model.export(format='torchscript', nms=False)의 출력은 (1,5,N) = 채널별
// [cx,cy,w,h,score] (letterbox 320x320 좌표계, 픽셀 단위, score는 이미 sigmoid
// 적용된 person 클래스 확신도 — 클래스가 1개뿐이라 별도 argmax 불필요).
// nms=True 버전은 torchvision::nms C++ 연산에 의존해서 RPi4용 LibTorch만으로는
// 링크가 안 되기 때문에 raw 출력 + 아래의 가벼운 NMS 조합으로 대체함.
inline std::vector<DetectionBox> decode_yolo_output(torch::Tensor output, float conf_threshold) {
    std::vector<DetectionBox> boxes;
    output = output.cpu();
    if (output.dim() == 3) {
        output = output[0];
    }

    auto acc = output.accessor<float, 2>();
    int n = static_cast<int>(output.size(1));
    for (int i = 0; i < n; ++i) {
        float score = acc[4][i];
        if (score < conf_threshold) {
            continue;
        }

        float cx = acc[0][i];
        float cy = acc[1][i];
        float w = acc[2][i];
        float h = acc[3][i];

        boxes.push_back({ cx - w / 2.0f, cy - h / 2.0f, cx + w / 2.0f, cy + h / 2.0f, score });
    }

    return boxes;
}

class VideoThread : public QThread {
    Q_OBJECT
public:
    VideoThread(int index, QString videoPath, cv::Mat hMatrix, torch::jit::script::Module* model, bool isModelLoaded, QMutex* modelMutex, QObject* parent = nullptr)
        : QThread(parent), m_index(index), m_videoPath(videoPath), m_H(hMatrix), m_model(model), m_isModelLoaded(isModelLoaded), m_modelMutex(modelMutex), m_stopRequested(false) {}

    void stop() {
        m_stopRequested = true;
        wait();
    }

    std::vector<cv::Point2f> getLatestSatPoints() {
        QMutexLocker locker(&m_dataMutex);
        return m_latestSatPoints;
    }

signals:
    void frameReady(int index, QImage videoImg);
    void videoCompleted(int index);

protected:
    void run() override {
        cv::VideoCapture cap;
        if (!cap.open(m_videoPath.toStdString())) {
            return;
        }

        while (!m_stopRequested) {
            cv::Mat frame;
            cap >> frame;

            if (frame.empty()) {
                break;
            }

            int h = frame.rows;
            int w = frame.cols;
            std::vector<cv::Point2f> satPoints;

            if (m_isModelLoaded && m_model) {
                try {
                    const int target_size = 320;
                    LetterboxInfo lb;
                    cv::Mat img_resized = letterbox(frame, target_size, lb);
                    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
                    img_resized.convertTo(img_resized, CV_32FC3, 1.0 / 255.0);

                    torch::Tensor img_tensor = torch::from_blob(img_resized.data, { 1, target_size, target_size, 3 }, torch::kFloat32).to(torch::kCPU);
                    img_tensor = img_tensor.permute({ 0, 3, 1, 2 }).contiguous();

                    torch::Tensor output;
                    {
                        QMutexLocker locker(m_modelMutex);
                        torch::NoGradGuard no_grad;
                        auto outputs = m_model->forward({ img_tensor });
                        output = outputs.toTensor();
                    }

                    auto boxes = decode_yolo_output(output, 0.3f);
                    boxes = nms(boxes, 0.45f);

                    for (auto& box : boxes) {
                        // letterbox 좌표계 -> 원본 프레임 좌표계로 역변환
                        int r_xmin = static_cast<int>((box.xmin - lb.pad_x) / lb.scale);
                        int r_ymin = static_cast<int>((box.ymin - lb.pad_y) / lb.scale);
                        int r_xmax = static_cast<int>((box.xmax - lb.pad_x) / lb.scale);
                        int r_ymax = static_cast<int>((box.ymax - lb.pad_y) / lb.scale);
                        float conf = box.conf;

                        r_xmin = std::max(0, std::min(r_xmin, w));
                        r_ymin = std::max(0, std::min(r_ymin, h));
                        r_xmax = std::max(0, std::min(r_xmax, w));
                        r_ymax = std::min(h, std::max(0, r_ymax));

                        cv::rectangle(frame, cv::Point(r_xmin, r_ymin), cv::Point(r_xmax, r_ymax), cv::Scalar(0, 255, 0), 3);
                        std::string conf_str = cv::format("%.2f", conf);
                        cv::putText(frame, conf_str, cv::Point(r_xmin, std::max(20, r_ymin - 10)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

                        float foot_x = (r_xmin + r_xmax) / 2.0f;
                        float foot_y = static_cast<float>(r_ymax);

                        if (!m_H.empty()) {
                            std::vector<cv::Point2f> srcPoints = { cv::Point2f(foot_x, foot_y) };
                            std::vector<cv::Point2f> dstPoints;
                            cv::perspectiveTransform(srcPoints, dstPoints, m_H);

                            float sat_x = std::max(0.0f, std::min(dstPoints[0].x, 657.0f));
                            float sat_y = std::max(0.0f, std::min(dstPoints[0].y, 748.0f));
                            satPoints.push_back(cv::Point2f(sat_x, sat_y));
                        }
                    }
                }
                catch (const std::exception& e) {
                    qDebug() << "Thread processing error: " << e.what();
                }
            }

            {
                QMutexLocker locker(&m_dataMutex);
                m_latestSatPoints = satPoints;
            }

            cv::Mat rgbFrame;
            cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
            QImage img(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, rgbFrame.step, QImage::Format_RGB888);

            emit frameReady(m_index, img.copy());
            QThread::msleep(33);
        }

        cap.release();
        emit videoCompleted(m_index);
    }

private:
    int m_index;
    QString m_videoPath;
    cv::Mat m_H;
    torch::jit::script::Module* m_model;
    bool m_isModelLoaded;
    QMutex* m_modelMutex;

    QMutex m_dataMutex;
    std::vector<cv::Point2f> m_latestSatPoints;
    bool m_stopRequested;
};