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

//OpenCV library
#include <opencv2/opencv.hpp>

//C++ library
#include <iostream>
#include <algorithm>
#include <vector>

//Ui header
#include "ui_H_Matrix.h"

using namespace std;
using namespace cv;

class H_matrix : public QDialog
{
    Q_OBJECT

public:
    H_matrix(QWidget* parent = nullptr);
    ~H_matrix();

    //Pushbutton function
    void load_sat();
    void load_img();
    void save_calibration();
    void exit();
    QPixmap m_originalPixmap;
    QPixmap m_originalPixmap2;

    // QT-ui private
private:
    Ui::Dialog ui;

    QMap<int, QPoint> m_satPoints;
    QMap<int, QPoint> m_imgPoints;

protected:
    // event
    bool eventFilter(QObject* watched, QEvent* event) override;
};
