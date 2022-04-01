#include "LockerButton.h"

#include <QLabel>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QDoubleValidator>

LockerButton::LockerButton(QWidget* parent)
    : QPushButton(parent)
{
    mImageLabel = new QLabel;
    mImageLabel->setFixedSize(16, 16);
    mImageLabel->setScaledContents(true);
    mImageLabel->setStyleSheet("QLabel{background-color:transparent;}");

    mTextLabel = new QLabel;
    mTextLabel->setStyleSheet("QLabel{background-color:transparent;}");
    //mTextLabel->setFont(QFont("ด๓ะก", 8, QFont::Black));

    QHBoxLayout* mainLayout = new QHBoxLayout;
    mainLayout->addWidget(mImageLabel);
    mainLayout->addWidget(mTextLabel);
    
    mainLayout->setMargin(0);
    mainLayout->setSpacing(0);
    mainLayout->setAlignment(0);
    this->setLayout(mainLayout);
}

void LockerButton::SetImageLabel(const QPixmap& pixmap)
{
    mImageLabel->setPixmap(pixmap);
}

void LockerButton::SetTextLabel(QString text)
{
    mTextLabel->setText(text);
}

QLabel* LockerButton::GetImageHandle()
{
    return mImageLabel;
}

QLabel* LockerButton::GetTextHandle()
{
    return mTextLabel;
}
