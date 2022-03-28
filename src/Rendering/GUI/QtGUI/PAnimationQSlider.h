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
		~PAnimationQSlider();

	signals:

	public slots:

	protected:
		virtual void mousePressEvent(QMouseEvent* event);
		virtual void mouseReleaseEvent(QMouseEvent* event);
		virtual void mouseMoveEvent(QMouseEvent* event);

	private:
		QLabel* m_displayLabel;
	};
}

#endif // PANIMATIONQSLIDER_H
