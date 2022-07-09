#include "PAnimationQSlider.h"

#include <QPalette>
#include <QDebug>
#include <QHBoxLayout>
#include <QPainter>

namespace dyno
{
	PAnimationQSlider::PAnimationQSlider(QWidget* parent):
		QSlider(parent)
	{
		this->setRange(1, 1000);

		m_displayLabel = new QLabel(this);
		m_displayLabel->setFixedSize(QSize(30, 20));

		m_displayLabel->setAlignment(Qt::AlignCenter);

		this->setTickInterval(50);

		this->setOrientation(Qt::Horizontal);
		this->setMinimumWidth(180);
		this->setTickPosition(QSlider::TicksAbove);
	}

	PAnimationQSlider::PAnimationQSlider(int minimum, int maximum, QWidget* parent /*= nullptr*/)
	{
		this->setRange(minimum, maximum);

		m_displayLabel = new QLabel(this);
		m_displayLabel->setFixedSize(QSize(30, 20));

		m_displayLabel->setAlignment(Qt::AlignCenter);

		this->setTickInterval(50);

		this->setOrientation(Qt::Horizontal);
		this->setMinimumWidth(180);
		this->setTickPosition(QSlider::TicksAbove);
	}

	PAnimationQSlider::~PAnimationQSlider()
	{

	}

	//To find a integer in a form of 2^n*5^m that is no smaller than val
	int CalculteInterval(int val)
	{
		val = std::max(val, 1);
		int m = ceil(log(double(val)) / log(double(5)));
		int n = ceil(log2(double(val)));

		int minDiff = INT_MAX;
		int ret = pow(5, m) * pow(2, n);
		for (int i = 0; i <= m; i++)
		{
			int a = (int)pow(5, i);
			for (int j = 0; j <= n; j++)
			{
				int b = pow(2, j);
				int diff = a * b - val;
				if (diff >= 0 && diff < minDiff)
				{
					ret = a * b;
					minDiff = ret;
				}
			}
		}

		return ret;
	};

	void PAnimationQSlider::maximumChanged(int val)
	{
		auto rect = this->geometry();
		int numTicks = std::max(rect.width() / mMinimumTickWidth, 1);

		int minVal = this->minimum();

		int inverval = CalculteInterval((val - minVal) / (std::min(numTicks, mMaximumTickNum)));
		setRange(minVal, val);
		setSingleStep(inverval);
		setTickInterval(inverval);
	}

	void PAnimationQSlider::minimumChanged(int val)
	{
		auto rect = this->geometry();
		int numTicks = std::max(rect.width() / mMinimumTickWidth, 1);

		int maxVal = this->maximum();

		int inverval = CalculteInterval((maxVal - val) / (std::min(numTicks, mMaximumTickNum)));
		setRange(val, maxVal);
		setSingleStep(inverval);
		setTickInterval(inverval);
	}

	void PAnimationQSlider::paintEvent(QPaintEvent* ev)
	{
		QSlider::paintEvent(ev);

		auto painter = new QPainter(this);
		painter->setPen(QPen(Qt::black));

		auto rect = this->geometry();

		int numTicks = std::max((maximum() - minimum()) / tickInterval(), 1);

		QFontMetrics fontMetrics = QFontMetrics(this->font());
		
		if (this->orientation() == Qt::Horizontal) {

			int fontHeight = fontMetrics.height();

			for (int i = 0; i <= numTicks; i++) {

				int tickNum = minimum() + (tickInterval() * i);

				auto tickX = (((rect.width() - 10.0f) / (maximum() - minimum()))* tickInterval() * i) - (fontMetrics.width(QString::number(tickNum)) / 2);

				auto tickY = (rect.height() + fontHeight) / 2;

				auto tickMarkX = (((rect.width() - 10.0f) / (maximum() - minimum())) * tickInterval() * i);

				painter->drawText(QPoint(tickX + 6, tickY - 5), QString::number(tickNum));
				painter->drawLine(QPoint(tickMarkX + 6, rect.height() - 3), QPoint(tickMarkX + 6, rect.height()));
			}
		}
		else if (this->orientation() == Qt::Vertical) {
			//do something else for vertical here, I haven't implemented it because I don't need it
		}
		else {
			return;
		}

		//painter->drawRect(rect);

		painter->end();
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
		QFontMetrics fontMetrics = QFontMetrics(this->font());

		int labelWidth = fontMetrics.width(QString::number(this->value()));

		m_displayLabel->move(12 + (this->width() - 10) * (this->value() - this->minimum()) / (this->maximum() - this->minimum()), 0);
		m_displayLabel->setText(QString::number(this->value()));
		m_displayLabel->setFixedWidth(labelWidth);
	
		m_displayLabel->setVisible(true);
		m_displayLabel->setStyleSheet("QLabel{background:#FFFFFF;}");
		m_displayLabel->setAlignment(Qt::AlignVCenter);
		QSlider::mouseMoveEvent(event);
	}

	void PAnimationQSlider::resizeEvent(QResizeEvent* event)
	{
		auto rect = this->geometry();
		int numTicks = std::max(rect.width() / mMinimumTickWidth, 1);

		int inverval = CalculteInterval(maximum() / (std::min(numTicks, mMaximumTickNum)));

		setSingleStep(inverval);
		setTickInterval(inverval);
	}
}
