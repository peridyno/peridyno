#ifndef PANIMATIONWIDGET_H
#define PANIMATIONWIDGET_H

#include <QWidget>
#include <QMouseEvent>
#include <QSlider>
#include <QLabel>
#include "PAnimationQSlider.h"

QT_FORWARD_DECLARE_CLASS(QSpinBox)
QT_FORWARD_DECLARE_CLASS(QScrollBar)
QT_FORWARD_DECLARE_CLASS(QPushButton)

namespace dyno
{
	QT_FORWARD_DECLARE_CLASS(PSimulationThread)

	class PAnimationWidget : public QWidget
	{
		Q_OBJECT

	public:
		explicit PAnimationWidget(QWidget *parent = nullptr);
		~PAnimationWidget();

	signals:

	public slots:
		void toggleSimulation();
		void resetSimulation();

		void simulationFinished();

	public:
		QPushButton*	m_startSim;
		QPushButton*	m_resetSim;

		QSpinBox*	m_start_spinbox;
		QSpinBox*	m_end_spinbox;

		QScrollBar*	m_sim_scrollbar;

		bool m_sim_started = false;
		
		PAnimationQSlider *m_slider;
	};
}

#endif // PANIMATIONWIDGET_H
