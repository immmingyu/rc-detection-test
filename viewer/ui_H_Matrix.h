// 원본 .ui 디자이너 파일이 없어서 H_matrix.cpp가 참조하는 위젯 이름/타입에 맞춰
// 직접 작성한 대체용 헤더입니다. Qt Designer가 생성하는 실제 레이아웃과는 다르지만,
// 기능(시그널/슬롯 연결, 위젯 이름)은 동일하게 동작합니다.
#pragma once

#include <QtWidgets/QDialog>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtCore/QString>

QT_BEGIN_NAMESPACE

class Ui_Dialog
{
public:
    QLabel* sat_img;
    QLabel* img;
    QSpinBox* index;
    QPushButton* upload_satellite;
    QPushButton* upload_image;
    QPushButton* save;
    QPushButton* exit;

    void setupUi(QDialog* Dialog)
    {
        if (Dialog->objectName().isEmpty())
            Dialog->setObjectName(QString::fromUtf8("Dialog"));
        Dialog->resize(1000, 600);

        QGridLayout* mainLayout = new QGridLayout(Dialog);

        sat_img = new QLabel(Dialog);
        img = new QLabel(Dialog);
        sat_img->setMinimumSize(400, 400);
        img->setMinimumSize(400, 400);
        sat_img->setFrameShape(QFrame::Box);
        img->setFrameShape(QFrame::Box);
        mainLayout->addWidget(sat_img, 0, 0, 1, 2);
        mainLayout->addWidget(img, 0, 2, 1, 2);

        index = new QSpinBox(Dialog);
        index->setMinimum(0);
        index->setMaximum(999);
        mainLayout->addWidget(index, 1, 0);

        upload_satellite = new QPushButton(Dialog);
        upload_image = new QPushButton(Dialog);
        save = new QPushButton(Dialog);
        exit = new QPushButton(Dialog);
        mainLayout->addWidget(upload_satellite, 1, 1);
        mainLayout->addWidget(upload_image, 1, 2);
        mainLayout->addWidget(save, 1, 3);
        mainLayout->addWidget(exit, 2, 3);

        retranslateUi(Dialog);
    }

    void retranslateUi(QDialog* Dialog)
    {
        Dialog->setWindowTitle(QString::fromUtf8("Homography Calibration"));
        sat_img->setText(QString::fromUtf8("sat_img"));
        img->setText(QString::fromUtf8("img"));
        upload_satellite->setText(QString::fromUtf8("Upload Satellite"));
        upload_image->setText(QString::fromUtf8("Upload Image"));
        save->setText(QString::fromUtf8("Save"));
        exit->setText(QString::fromUtf8("Exit"));
    }
};

namespace Ui {
    class Dialog : public Ui_Dialog {};
}

QT_END_NAMESPACE
