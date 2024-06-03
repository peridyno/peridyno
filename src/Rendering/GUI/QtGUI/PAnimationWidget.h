#ifndef PANIMATIONWIDGET_H
#define PANIMATIONWIDGET_H

#include <QWidget>
#include <QMouseEvent>
#include <QSlider>
#include <QLabel>
#include <QCheckBox>
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
		void simulationStarted();
		void simulationStopped();

	public slots:
		void toggleSimulation();
		void resetSimulation();

		void simulationFinished();

		void updateSlider(int frame);

		void buildIconLabel(QLabel* Label, QPixmap* Icon,  QPushButton*btn, int size);

		void totalFrameChanged(int num);

		void runForever(int state);

	private:
		QPushButton*	mStartSim;
		QPushButton*	mResetSim;

		QPixmap* mStartIcon;
		QPixmap* mPauseIcon;
		QPixmap* mResetIcon;
		QPixmap* mFinishIcon;

		QLabel* mResetLabel;
		QLabel* mStartLabel;
		
		QCheckBox* mPersistent;
		QSpinBox* mTotalFrameSpinbox;

		PAnimationQSlider* mFrameSlider;

	private:
		int mTotalFrame;
		static const int labelSize = 6;
		QLabel* label[labelSize];
	};
}

#endif // PANIMATIONWIDGET_H
