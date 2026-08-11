#pragma once

//QT library
#include <QtWidgets/QDialog>
#include <qpushbutton.h>
#include <qmessagebox.h>
#include <qtimer.h>
#include <qfiledialog.h>
#include <qfileinfo.h>
#include <qstring.h>
#include <qpainter.h>
#include <qpen.h>
#include <qevent.h>
#include <qsize.h>
#include <qpoint.h>
#include <qmap.h>
#include <qset.h>
#include <qlist.h>
#include <qstringlist.h>
#include <qtextstream.h>
#include <qfile.h>
#include <qregularexpression.h>
#include <qthread.h>
#include <qmutex.h>

//OpenCV library
#include <opencv2/opencv.hpp>

//C++ library
#include <iostream>
#include <algorithm>
#include <vector>
#include "VideoThread.h"

//Ui header
#include "ui_MainWindows.h"

//torch header
#undef slots
#include <torch/script.h>
#include <torch/torch.h>
#define slots Q_SLOTS

using namespace std;
using namespace cv;

class MainWindows : public QDialog
{
    Q_OBJECT

public:
    MainWindows(QWidget *parent = nullptr);
    ~MainWindows();

    //Pushbutton function
    void run_algorithm();
    void run_calibration();

    void show_sat();
    void load_video();
    void show_video();
    void load_model();

    void calibration();
    void exit();

    QPixmap origin_sat;

// QT-ui private
private:
    Ui::MainWindowsClass ui;
    vector<QTimer*> timer_vector;

    vector<cv::Mat> Homography_matrix;
    cv::Mat Sat_H;

    torch::jit::script::Module m_model;
    bool m_isModelLoaded = false;
    bool m_isDetecting = false;

    std::vector<cv::Point2f> sat_img_pts;
    std::vector<cv::Point2f> sat_gps_pts;
    QVector<VideoThread*> m_videoThreads;
    QMutex m_inferenceMutex;
    QTimer* m_satUpdateTimer;

protected:
    vector<QString> videoPath;
    vector<VideoCapture> cap_vector;
};