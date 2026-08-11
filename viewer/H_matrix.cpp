#include "H_matrix.h"

H_matrix::H_matrix(QWidget* parent)
    : QDialog(parent)
{
    ui.setupUi(this);
    ui.sat_img->installEventFilter(this);
    ui.img->installEventFilter(this);

    // connect widget
     //   connect(m_timer, &QTimer::timeout, this, &MainWindow::updateVideoFrame);
    connect(ui.exit, &QPushButton::clicked, this, &H_matrix::exit);
    connect(ui.save, &QPushButton::clicked, this, &H_matrix::save_calibration);
    connect(ui.upload_image, &QPushButton::clicked, this, &H_matrix::load_img);
    connect(ui.upload_satellite, &QPushButton::clicked, this, &H_matrix::load_sat);
}

H_matrix::~H_matrix()
{

}

bool H_matrix::eventFilter(QObject* watched, QEvent* event) {

    int currentIndex = ui.index->value();
    if (watched == ui.sat_img && event->type() == QEvent::MouseButtonPress) {
        if (m_originalPixmap.isNull()) return true;

        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        QPoint widgetPos = mouseEvent->pos();

        QSize labelSize = ui.sat_img->size();
        QSize pixmapSize = m_originalPixmap.size();
        QSize scaledSize = pixmapSize.scaled(labelSize, Qt::KeepAspectRatio);

        int xOffset = (labelSize.width() - scaledSize.width()) / 2;
        int yOffset = (labelSize.height() - scaledSize.height()) / 2;

        if (widgetPos.x() < xOffset || widgetPos.x() >= xOffset + scaledSize.width() ||
            widgetPos.y() < yOffset || widgetPos.y() >= yOffset + scaledSize.height()) {
            return true;
        }

        double scaleFactor = (double)scaledSize.width() / pixmapSize.width();
        int realX = (widgetPos.x() - xOffset) / scaleFactor;
        int realY = (widgetPos.y() - yOffset) / scaleFactor;
        QPoint realPos(realX, realY);

        m_satPoints[currentIndex] = realPos;

        QPixmap tempPixmap = m_originalPixmap;
        QPainter painter(&tempPixmap);
        painter.setRenderHint(QPainter::Antialiasing, true);

        int realRadius = 10 / scaleFactor;
        double penWidth = qMax(1.0, 4.0 / scaleFactor);
        int fontSize = qMax(8, (int)(12 / scaleFactor));

        for (auto it = m_satPoints.begin(); it != m_satPoints.end(); ++it) {
            int idx = it.key();
            QPoint pt = it.value();

            painter.setBrush(QBrush(Qt::yellow));
            painter.drawEllipse(pt, realRadius, realRadius);

            painter.setPen(QPen(Qt::red, 2));
            painter.setFont(QFont("Arial", 10 / scaleFactor, QFont::Bold));
            painter.drawText(pt + QPoint(realRadius, -realRadius), QString::number(idx));
        }

        ui.sat_img->setPixmap(tempPixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    else if (watched == ui.img && event->type() == QEvent::MouseButtonPress) {
        if (m_originalPixmap2.isNull()) return true;

        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        QPoint widgetPos = mouseEvent->pos();

        QSize labelSize = ui.img->size();
        QSize pixmapSize = m_originalPixmap2.size();
        QSize scaledSize = pixmapSize.scaled(labelSize, Qt::KeepAspectRatio);

        int xOffset = (labelSize.width() - scaledSize.width()) / 2;
        int yOffset = (labelSize.height() - scaledSize.height()) / 2;

        if (widgetPos.x() < xOffset || widgetPos.x() >= xOffset + scaledSize.width() ||
            widgetPos.y() < yOffset || widgetPos.y() >= yOffset + scaledSize.height()) {
            return true;
        }

        double scaleFactor = (double)scaledSize.width() / pixmapSize.width();
        int realX = (widgetPos.x() - xOffset) / scaleFactor;
        int realY = (widgetPos.y() - yOffset) / scaleFactor;
        QPoint realPos(realX, realY);

        m_imgPoints[currentIndex] = realPos;

        QPixmap tempPixmap = m_originalPixmap2;
        QPainter painter(&tempPixmap);
        painter.setRenderHint(QPainter::Antialiasing, true);

        int realRadius = 10 / scaleFactor;
        double penWidth = qMax(1.0, 4.0 / scaleFactor);
        int fontSize = qMax(8, (int)(12 / scaleFactor));

        for (auto it = m_imgPoints.begin(); it != m_imgPoints.end(); ++it) {
            int idx = it.key();
            QPoint pt = it.value();

            painter.setBrush(QBrush(Qt::yellow));
            painter.drawEllipse(pt, realRadius, realRadius);

            painter.setPen(QPen(Qt::red, 2));
            painter.setFont(QFont("Arial", 10 / scaleFactor, QFont::Bold));
            painter.drawText(pt + QPoint(realRadius, -realRadius), QString::number(idx));
        }

        ui.img->setPixmap(tempPixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    return QDialog::eventFilter(watched, event);
}

void H_matrix::load_sat() {
    QString filePath = QFileDialog::getOpenFileName(this, tr("Select satellite image"),
        "", tr("Video Files (*.png *.jpg *.jpeg *.bmp)"));

    if (filePath.isEmpty()) {
        QMessageBox messagebox;
        messagebox.setText("Can not found img");
        return;
    }

    QPixmap pixmap(filePath);
    m_originalPixmap = QPixmap(filePath);

    ui.sat_img->setPixmap(pixmap.scaled(ui.sat_img->size(), Qt::KeepAspectRatio, 
        Qt::SmoothTransformation));

    return;
}

void H_matrix::load_img() {
    QString filePath = QFileDialog::getOpenFileName(this, tr("Select image or video"),
        "", tr("Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv)"));

    if (filePath.isEmpty()) {
        QMessageBox::warning(this, tr("Error"), tr("Can not found file"));
        return;
    }

    QFileInfo fileInfo(filePath);
    QString ext = fileInfo.suffix().toLower();
    QPixmap pixmap;

    if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv") {
        cv::VideoCapture cap(filePath.toStdString());
        if (!cap.isOpened()) {
            QMessageBox::warning(this, tr("Error"), tr("Can not open video file"));
            return;
        }

        cv::Mat frame;
        cap >> frame;
        cap.release();

        if (frame.empty()) {
            QMessageBox::warning(this, tr("Error"), tr("Video frame is empty"));
            return;
        }

        cv::Mat rgbFrame;
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
        QImage img(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, rgbFrame.step, QImage::Format_RGB888);
        pixmap = QPixmap::fromImage(img);
    }
    else {
        pixmap.load(filePath);
        if (pixmap.isNull()) {
            QMessageBox::warning(this, tr("Error"), tr("Failed to load image"));
            return;
        }
    }

    m_originalPixmap2 = pixmap;

    ui.img->setPixmap(pixmap.scaled(ui.img->size(), Qt::KeepAspectRatio,
        Qt::SmoothTransformation));

    return;
}

void H_matrix::save_calibration() {
    QString filePath = QFileDialog::getSaveFileName(this, tr("save coordinates"), 
        "", tr("Text Files (*.txt);;All Files (*)"));

    if (filePath.isEmpty()) {
        return;
    }

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("error"), tr("can not opened file."));
        return;
    }

    QTextStream out(&file);

    QSet<int> allKeys;
    for (int key : m_satPoints.keys()) {
        allKeys.insert(key);
    }
    for (int key : m_imgPoints.keys()) {
        allKeys.insert(key);
    }

    QList<int> sortedKeys = allKeys.values();
    std::sort(sortedKeys.begin(), sortedKeys.end());

    out << "# Index | Sat_X | Sat_Y | Img_X | Img_Y\n";

    int savedCount = 0;

    for (int key : sortedKeys) {
        bool hasSat = m_satPoints.contains(key);
        bool hasImg = m_imgPoints.contains(key);

        if (hasSat && hasImg) {
            QPoint p1 = m_satPoints[key];
            QPoint p2 = m_imgPoints[key];

            out << key << " "
                << p1.x() << " " << p1.y() << " "
                << p2.x() << " " << p2.y() << "\n";

            savedCount++;
        }
    }
    file.close();
    return;
}

void H_matrix::exit() {
    QMessageBox messagebox;
    messagebox.setText("Do you want to exit?");
    messagebox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
    messagebox.setDefaultButton(QMessageBox::Ok);

    if (messagebox.exec() == QMessageBox::Ok)
    {
        close();
        return;
    }

    return;
}
