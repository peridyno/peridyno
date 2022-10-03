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

		void updateSlider();

		void buildIconLabel(QLabel* Label, QPixmap* Icon,  QPushButton*btn, int size);

	public:
		QPushButton*	m_startSim;
		QPushButton*	m_resetSim;

		QPixmap* Starticon;
		QPixmap* Pauseicon;
		QPixmap* Reseticon;
		QPixmap* Finishicon;

		QLabel* ResetLabel;
		QLabel* StartLabel;
		
		QSpinBox* mTotalFrameSpinbox;

		QScrollBar*	m_sim_scrollbar;

		bool m_sim_started = false;
		
		PAnimationQSlider* mFrameSlider;

	private:
		int mTotalFrame;
		static const int labelSize = 6;
		QLabel* label[labelSize];
	};
}

#endif // PANIMATIONWIDGET_H
