// 원본 .ui 디자이너 파일이 없어서 MainWindows.cpp가 참조하는 위젯 이름/타입에 맞춰
// 직접 작성한 대체용 헤더입니다. Qt Designer가 생성하는 실제 레이아웃과는 다르지만,
// 기능(시그널/슬롯 연결, 위젯 이름)은 동일하게 동작합니다.
#pragma once

#include <QtWidgets/QDialog>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QGridLayout>
#include <QtCore/QString>

QT_BEGIN_NAMESPACE

class Ui_MainWindowsClass
{
public:
    QPushButton* video_up;
    QPushButton* video_start;
    QPushButton* calibration;
    QPushButton* calibration_load;
    QPushButton* img_up;
    QPushButton* algorithm;
    QPushButton* load_model;
    QPushButton* exit;

    QListWidget* video_list;
    QLabel* video;
    QLabel* video2;
    QLabel* video3;
    QLabel* video4;
    QLabel* satellite;

    void setupUi(QDialog* MainWindowsClass)
    {
        if (MainWindowsClass->objectName().isEmpty())
            MainWindowsClass->setObjectName(QString::fromUtf8("MainWindowsClass"));
        MainWindowsClass->resize(1200, 800);

        QGridLayout* mainLayout = new QGridLayout(MainWindowsClass);

        video_up = new QPushButton(MainWindowsClass);
        video_start = new QPushButton(MainWindowsClass);
        calibration = new QPushButton(MainWindowsClass);
        calibration_load = new QPushButton(MainWindowsClass);
        img_up = new QPushButton(MainWindowsClass);
        algorithm = new QPushButton(MainWindowsClass);
        load_model = new QPushButton(MainWindowsClass);
        exit = new QPushButton(MainWindowsClass);

        mainLayout->addWidget(video_up, 0, 0);
        mainLayout->addWidget(video_start, 0, 1);
        mainLayout->addWidget(calibration, 0, 2);
        mainLayout->addWidget(calibration_load, 0, 3);
        mainLayout->addWidget(img_up, 0, 4);
        mainLayout->addWidget(algorithm, 0, 5);
        mainLayout->addWidget(load_model, 0, 6);
        mainLayout->addWidget(exit, 0, 7);

        // 좌: Satellite(큼) / 중: Video 2x2 / 우: Video list(좁음) - 참고 스크린샷 배치 기준
        QLabel* satelliteHeader = new QLabel(MainWindowsClass);
        satelliteHeader->setText(QString::fromUtf8("Satellite"));
        mainLayout->addWidget(satelliteHeader, 1, 0, 1, 4);

        satellite = new QLabel(MainWindowsClass);
        satellite->setMinimumSize(400, 400);
        satellite->setFrameShape(QFrame::Box);
        mainLayout->addWidget(satellite, 2, 0, 2, 4);

        QLabel* videoHeader = new QLabel(MainWindowsClass);
        videoHeader->setText(QString::fromUtf8("Video"));
        mainLayout->addWidget(videoHeader, 1, 4, 1, 2);

        video = new QLabel(MainWindowsClass);
        video2 = new QLabel(MainWindowsClass);
        video3 = new QLabel(MainWindowsClass);
        video4 = new QLabel(MainWindowsClass);
        video->setMinimumSize(320, 240);
        video2->setMinimumSize(320, 240);
        video3->setMinimumSize(320, 240);
        video4->setMinimumSize(320, 240);
        video->setFrameShape(QFrame::Box);
        video2->setFrameShape(QFrame::Box);
        video3->setFrameShape(QFrame::Box);
        video4->setFrameShape(QFrame::Box);

        mainLayout->addWidget(video, 2, 4);
        mainLayout->addWidget(video2, 2, 5);
        mainLayout->addWidget(video3, 3, 4);
        mainLayout->addWidget(video4, 3, 5);

        QLabel* videoListHeader = new QLabel(MainWindowsClass);
        videoListHeader->setText(QString::fromUtf8("Video list"));
        mainLayout->addWidget(videoListHeader, 1, 6, 1, 2);

        video_list = new QListWidget(MainWindowsClass);
        video_list->setMinimumSize(160, 200);
        mainLayout->addWidget(video_list, 2, 6, 2, 2);

        retranslateUi(MainWindowsClass);
    }

    void retranslateUi(QDialog* MainWindowsClass)
    {
        MainWindowsClass->setWindowTitle(QString::fromUtf8("MainWindows"));
        video_up->setText(QString::fromUtf8("Get satellite"));
        load_model->setText(QString::fromUtf8("Load_Model"));
        calibration->setText(QString::fromUtf8("Load_VideoCalib"));
        algorithm->setText(QString::fromUtf8("Run_model"));
        img_up->setText(QString::fromUtf8("Video upload"));
        video_start->setText(QString::fromUtf8("Video START"));
        calibration_load->setText(QString::fromUtf8("Calculate_Calib"));
        exit->setText(QString::fromUtf8("Exit"));
    }
};

namespace Ui {
    class MainWindowsClass : public Ui_MainWindowsClass {};
}

QT_END_NAMESPACE
