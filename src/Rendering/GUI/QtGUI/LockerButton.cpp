#include "LockerButton.h"

#include <QLabel>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QDoubleValidator>

LockerButton::LockerButton(QWidget* parent)
    : QPushButton(parent)
{
    m_imageLabel = new QLabel;
    m_imageLabel->setFixedWidth(30);
    m_imageLabel->setScaledContents(true);
    m_imageLabel->setStyleSheet("QLabel{background-color:transparent;}");

    m_textLabel = new QLabel;
    m_textLabel->setStyleSheet("QLabel{background-color:transparent;}");

    QHBoxLayout* mainLayout = new QHBoxLayout;
    mainLayout->addWidget(m_imageLabel);
    mainLayout->addWidget(m_textLabel);
    mainLayout->setMargin(0);
    mainLayout->setSpacing(0);
    mainLayout->setAlignment(0);
    //this->setText("66666666666666666666666666666666666666666666666");
    this->setLayout(mainLayout);
    this->setBackgroundRole(QPalette::Dark);
}

void LockerButton::SetImageLabel(const QPixmap& pixmap)
{
    m_imageLabel->setPixmap(pixmap);
}

void LockerButton::SetTextLabel(QString text)
{
    m_textLabel->setText(text);
}

QLabel* LockerButton::GetImageHandle()
{
    return m_imageLabel;
}

QLabel* LockerButton::GetTextHandle()
{
    return m_textLabel;
}
