#ifndef PANIMATIONQSLIDER_H
#define PANIMATIONQSLIDER_H

#include <QWidget>
#include <QMouseEvent>
#include <QSlider>
#include <QLabel>
#include <QSpinBox>
namespace dyno
{
	QT_FORWARD_DECLARE_CLASS(PSimulationThread)

	class PAnimationQSlider : public QSlider
	{
		Q_OBJECT

	public:
		PAnimationQSlider(QWidget* parent = nullptr);
		PAnimationQSlider(int minimum, int maximum, QWidget* parent = nullptr);
		~PAnimationQSlider();

	signals:

	public slots:
		void maximumChanged(int val);
		void minimumChanged(int val);

	protected:
		void paintEvent(QPaintEvent* ev) override;
		void mousePressEvent(QMouseEvent* event) override;
		void mouseReleaseEvent(QMouseEvent* event) override;
		void mouseMoveEvent(QMouseEvent* event) override;

		void resizeEvent(QResizeEvent* event);

	private:
		QLabel* m_displayLabel;

		int mMaximumTickNum = 15;
		int mMinimumTickWidth = 20;
	};
}

#endif // PANIMATIONQSLIDER_H
