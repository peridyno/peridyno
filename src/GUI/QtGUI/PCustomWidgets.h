#pragma once

#include <QSlider>
#include <QDoubleSpinBox>

namespace dyno
{
	class QDoubleSlider : public QSlider
	{
		Q_OBJECT

	public:
		QDoubleSlider(QWidget* pParent = NULL);

		void setRange(double Min, double Max);
		void setMinimum(double Min);
		double minimum() const;
		void setMaximum(double Max);
		double maximum() const;
		double value() const;

	public slots:
		void setValue(int value);
		void setValue(double Value, bool BlockSignals = false);

	private slots:

	signals:
		void valueChanged(double Value);
		void rangeChanged(double Min, double Max);

	private:
		double	m_Multiplier;
	};

	class QDoubleSpinner : public QDoubleSpinBox
	{
		Q_OBJECT

	public:

		QDoubleSpinner(QWidget* pParent = NULL);;

		virtual QSize sizeHint() const;
		void setValue(double Value, bool BlockSignals = false);
	};
}

