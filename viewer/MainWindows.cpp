#include "MainWindows.h"
#include "H_matrix.h"

MainWindows::MainWindows(QWidget *parent)
    : QDialog(parent) {
    ui.setupUi(this);

// connect widget
    connect(ui.exit, &QPushButton::clicked,this, &MainWindows::exit);
    connect(ui.video_up, &QPushButton::clicked, this, &MainWindows::load_video);
    connect(ui.video_start, &QPushButton::clicked, this, &MainWindows::show_video);
    connect(ui.calibration, &QPushButton::clicked, this, &MainWindows::calibration);
    connect(ui.img_up, &QPushButton::clicked, this, &MainWindows::show_sat);
    connect(ui.calibration_load, &QPushButton::clicked, this, &MainWindows::run_calibration);
    connect(ui.algorithm, &QPushButton::clicked, this, &MainWindows::run_algorithm);
    connect(ui.load_model, &QPushButton::clicked, this, &MainWindows::load_model);
}

MainWindows::~MainWindows() {
    
}

void MainWindows::load_model() {
    QString filePath = QFileDialog::getOpenFileName(this, tr("Select TorchScript Model"),
        "", tr("Model Files (*.pt);;All Files (*)"));

    if (filePath.isEmpty()) {
        return;
    }

    try {
        m_model = torch::jit::load(filePath.toStdString());
        m_model.to(torch::kCPU);
        m_model.eval();
        m_isModelLoaded = true;

        QMessageBox::information(this, "Success", "Model loaded successfully!");
    }
    catch (const c10::Error& e) {
        QMessageBox::critical(this, "Error", QString("Failed to load model:\n%1").arg(e.what()));
        m_isModelLoaded = false;
        return;
    }
}

void MainWindows::run_algorithm() {
    QString filePath = QFileDialog::getOpenFileName(this, tr("Open coordinates"),
        "", tr("Text Files (*.txt);;All Files (*)"));

    if (filePath.isEmpty()) {
        return;
    }

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("error"), tr("can not opened file."));
        return;
    }

    QTextStream in(&file);

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#")) {
            continue;
        }
        QStringList parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (parts.size() >= 5) {
            // int index = parts[0].toInt();
            float sat_x = parts[1].toFloat();
            float sat_y = parts[2].toFloat();
            float img_x = parts[3].toFloat();
            float img_y = parts[4].toFloat();

            sat_gps_pts.push_back(cv::Point2f(sat_x, sat_y));
            sat_img_pts.push_back(cv::Point2f(img_x, img_y));
        }
    }

    file.close();
    if (sat_img_pts.size() < 4) {
        QMessageBox::warning(this, tr("Warning"), tr("You must do getting 4 points for homography matrix.\ncurrent point: %1").arg(sat_img_pts.size()));
        return;
    }

    m_isDetecting = !m_isDetecting;

    return;
}

void MainWindows::run_calibration() {
    QString filePath = QFileDialog::getOpenFileName(this, tr("Open coordinates"),
        "", tr("Text Files (*.txt);;All Files (*)"));

    if (filePath.isEmpty()) {
        return;
    }

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("error"), tr("can not opened file."));
        return;
    }

    QTextStream in(&file);
    std::vector<cv::Point2f> pts_img;
    std::vector<cv::Point2f> pts_sat;

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#")) {
            continue;
        }
        QStringList parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (parts.size() >= 5) {
            // int index = parts[0].toInt();
            float sat_x = parts[1].toFloat();
            float sat_y = parts[2].toFloat();
            float img_x = parts[3].toFloat();
            float img_y = parts[4].toFloat();

            pts_sat.push_back(cv::Point2f(sat_x, sat_y));
            pts_img.push_back(cv::Point2f(img_x, img_y));
        }
    }

    file.close();
    if (pts_img.size() < 4) {
        QMessageBox::warning(this, tr("Warning"), tr("You must do getting 4 points for homography matrix.\ncurrent point: %1").arg(pts_img.size()));
        return;
    }
    cv::Mat H = cv::findHomography(pts_img, pts_sat, cv::RANSAC);

    if (H.empty()) {
        QMessageBox::critical(this, tr("Failed"), tr("Failed to calculate the homography matrix."));
        return;
    }

    Homography_matrix.push_back(H);

    QMessageBox::information(this, tr("Success"), QString("Homography matrix calculated successfully. (Total %1 points)").arg(pts_img.size()));

    return;
}

void MainWindows::show_sat() {
    QString filePath = QFileDialog::getOpenFileName(this, tr("Select satellite img"),
        "", tr("Video Files (*.png *.jpg *.jpeg *.bmp)"));

    if (filePath.isEmpty()) {
        QMessageBox messagebox;
        messagebox.setText("Can not found img");
        return;
    }

    QPixmap pixmap(filePath);
    origin_sat = QPixmap(filePath);

    ui.satellite->setPixmap(pixmap.scaled(ui.satellite->size(), Qt::KeepAspectRatio,
        Qt::SmoothTransformation));

    return;
}

void MainWindows::calibration() {
    H_matrix h_matrix_window(this);

    h_matrix_window.exec();
}

void MainWindows::load_video() {
    QString filePath = QFileDialog::getOpenFileName(this,tr("Select video"),
        "",tr("Video Files (*.mp4 *.avi *.mov *.mkv)"));

    if (filePath.isEmpty()) {
        QMessageBox messagebox;
        messagebox.setText("Can not found video");
        return;
    }

    QFileInfo fileInfo(filePath);
    QString fileName = fileInfo.fileName();
    ui.video_list->addItem(fileName);
    videoPath.push_back(filePath);

    return;
}

void MainWindows::show_video() {
    for (int i = 0; i < m_videoThreads.size(); i++) {
        if (m_videoThreads[i]) {
            m_videoThreads[i]->stop();
            delete m_videoThreads[i];
        }
    }
    m_videoThreads.clear();
    m_videoThreads.resize(videoPath.size());

    if (m_satUpdateTimer) {
        m_satUpdateTimer->stop();
        delete m_satUpdateTimer;
    }

    cv::Mat h_mat = Homography_matrix.empty() ? cv::Mat() : Homography_matrix[0];

    for (int i = 0; i < videoPath.size(); i++) {
        VideoThread* thread = new VideoThread(i, videoPath[i], h_mat, &m_model, m_isModelLoaded, &m_inferenceMutex, this);
        m_videoThreads[i] = thread;

        connect(thread, &VideoThread::frameReady, this, [=](int idx, QImage vImg) {
            if (idx == 0) ui.video->setPixmap(QPixmap::fromImage(vImg).scaled(ui.video->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            else if (idx == 1) ui.video2->setPixmap(QPixmap::fromImage(vImg).scaled(ui.video2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            else if (idx == 2) ui.video3->setPixmap(QPixmap::fromImage(vImg).scaled(ui.video3->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            else if (idx == 3) ui.video4->setPixmap(QPixmap::fromImage(vImg).scaled(ui.video4->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            });

        connect(thread, &VideoThread::videoCompleted, this, [=](int idx) {
            if (idx == 0) ui.video->setText("Completed video");
            else if (idx == 1) ui.video2->setText("Completed video");
            else if (idx == 2) ui.video3->setText("Completed video");
            else if (idx == 3) ui.video4->setText("Completed video");
            });

        thread->start();
    }

    m_satUpdateTimer = new QTimer(this);
    connect(m_satUpdateTimer, &QTimer::timeout, this, [=]() {
        if (origin_sat.isNull()) return;

        QPixmap satellite_canvas = origin_sat;
        QPainter painter(&satellite_canvas);
        painter.setPen(QPen(Qt::red, 8));
        painter.setBrush(Qt::red);

        bool hasPoints = false;

        for (int i = 0; i < m_videoThreads.size(); i++) {
            if (m_videoThreads[i]) {
                std::vector<cv::Point2f> pts = m_videoThreads[i]->getLatestSatPoints();
                for (const auto& pt : pts) {
                    painter.drawEllipse(QPointF(pt.x, pt.y), 6, 6);
                    hasPoints = true;
                }
            }
        }

        painter.end();

        ui.satellite->setPixmap(satellite_canvas.scaled(
            ui.satellite->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation
        ));
        });

    m_satUpdateTimer->start(33);
}

void MainWindows::exit() {
    QMessageBox messagebox;
    messagebox.setText("Do you want to exit?");
    messagebox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
    messagebox.setDefaultButton(QMessageBox::Ok);

    for (int i = 0; i < m_videoThreads.size(); i++) {
        if (m_videoThreads[i]) {
            m_videoThreads[i]->stop();
            delete m_videoThreads[i];
            m_videoThreads[i] = nullptr;
        }
    }
    m_videoThreads.clear();
    if (m_satUpdateTimer) {
        m_satUpdateTimer->stop();
        delete m_satUpdateTimer;
        m_satUpdateTimer = nullptr;
    }

    if (messagebox.exec() == QMessageBox::Ok) {
        close();
        return;
    }
    return;
}

