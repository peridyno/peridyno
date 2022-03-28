#pragma once


#ifndef LOCKER_BUTTON_H
#define LOCKER_BUTTON_H

#include <QWidget>
#include <QPushButton>

class QLabel;

class LockerButton : public QPushButton
{
    Q_OBJECT
public:
    explicit LockerButton(QWidget* parent = nullptr);

    //  SetImageLabel
    void SetImageLabel(const QPixmap& pixmap);

    //  SetTextLabel
    void SetTextLabel(QString text);

    // @brief GetImageHandle
    QLabel* GetImageHandle();

    // @brief GetImageHandle
    QLabel* GetTextHandle();

private:
   
    QLabel* mImageLabel;
    
    QLabel* mTextLabel;
};

#endif // LOCKER_BUTTON_H
