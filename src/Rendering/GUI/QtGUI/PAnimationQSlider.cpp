#include "PAnimationQSlider.h"

#include <QPalette>
#include <QDebug>
#include <QHBoxLayout>
namespace dyno
{
	PAnimationQSlider::PAnimationQSlider(QWidget* parent):
		QSlider(parent)
	{
		m_displayLabel = new QLabel(this);
		m_displayLabel->setFixedSize(QSize(20, 20));

		m_displayLabel->setAlignment(Qt::AlignCenter);


		this->setRange(1, 100);
		this->setSingleStep(1);
		this->setOrientation(Qt::Horizontal);

		this->setTickPosition(QSlider::TicksAbove);

	}
	PAnimationQSlider::~PAnimationQSlider() 
	{

	}
	void PAnimationQSlider::mousePressEvent(QMouseEvent* event)
	{
		QSlider::mousePressEvent(event);
	}



	void PAnimationQSlider::mouseReleaseEvent(QMouseEvent* event)
	{
		
		m_displayLabel->setVisible(false);
		QSlider::mouseReleaseEvent(event);
	}

	void PAnimationQSlider::mouseMoveEvent(QMouseEvent* event)
	{

		m_displayLabel->move(8 + (this->width() - 10) * (this->value() - this->minimum()) / (this->maximum() - this->minimum()), 6);
		m_displayLabel->setText(QString::number(this->value()));
	
		m_displayLabel->setVisible(true);
		
		QSlider::mouseMoveEvent(event);
	}
}
