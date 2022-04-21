#include "PAnimationWidget.h"

#include "PSimulationThread.h"

#include <QGridLayout>
#include <QPushButton>
#include <QSpinBox>
#include <QLineEdit>
#include <QIntValidator>
#include <QDebug>
namespace dyno
{
	PAnimationWidget::PAnimationWidget(QWidget *parent) : 
		QWidget(parent),
		m_startSim(nullptr),
		m_resetSim(nullptr)
	{
		totalFrame = 400;

		QHBoxLayout* layout = new QHBoxLayout();
		setLayout(layout);

		QGridLayout* frameLayout	= new QGridLayout();


		m_current_frame_spinbox = new QSpinBox();
		m_current_frame_spinbox->setFixedSize(60, 25);
		m_current_frame_spinbox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

		m_current_frame_spinbox->setMaximum(1000);
		m_current_frame_spinbox->setValue(0);

		QGridLayout* GLayout = new QGridLayout;
		m_slider = new PAnimationQSlider(this);

		m_slider->setRange(0, totalFrame);
		m_slider->setSingleStep(totalFrame/20);

		for (int i = 0; i < labelSize; i++) {
			label[i] = new  QLabel(QString::number(totalFrame/5 * i));
			if (i == 0) {
				frameLayout->addWidget(label[i], 1, 0, 1, 1, Qt::AlignTop | Qt::AlignLeft);
			}
			else if (i == labelSize - 1) {
				frameLayout->addWidget(label[i], 1, (labelSize - 1) * 2 - 1, 1, 1, Qt::AlignTop | Qt::AlignRight);
			}
			else {
				frameLayout->addWidget(label[i], 1, i * 2 - 1, 1, 2, Qt::AlignTop | Qt::AlignCenter);
			}
		}
		frameLayout->addWidget(m_slider, 0, 0, 0 , (labelSize - 1) * 2);

		// 连接信号槽（相互改变）
		connect(m_current_frame_spinbox, SIGNAL(valueChanged(int)), m_slider, SLOT(setValue(int)));
		connect(m_slider, SIGNAL(valueChanged(int)), m_current_frame_spinbox, SLOT(setValue(int)));

		QGridLayout* operationLayout = new QGridLayout();

		m_startSim = new QPushButton("Start");
		m_resetSim = new QPushButton("Reset");

		operationLayout->addWidget(m_current_frame_spinbox, 0, 0);
		operationLayout->addWidget(m_startSim, 0, 1);
		operationLayout->addWidget(m_resetSim, 0, 2);

		m_startSim->setCheckable(true);

		layout->addLayout(frameLayout, 10);
		layout->addLayout(operationLayout, 1);
		
		connect(m_startSim, SIGNAL(released()), this, SLOT(toggleSimulation()));
		connect(m_resetSim, SIGNAL(released()), this, SLOT(resetSimulation()));
		connect(PSimulationThread::instance(), SIGNAL(finished()), this, SLOT(simulationFinished()));

		connect(PSimulationThread::instance(), SIGNAL(oneFrameFinished()), this, SLOT(updateSlider()));

		PSimulationThread::instance()->start();
	}

	PAnimationWidget::~PAnimationWidget()
	{
		PSimulationThread::instance()->stop();
		PSimulationThread::instance()->deleteLater();
		PSimulationThread::instance()->wait();  //必须等待线程结束
	}



	void PAnimationWidget::toggleSimulation()
	{
		if (m_startSim->isChecked())
		{
			PSimulationThread::instance()->resume();
			m_startSim->setText("Pause");
			m_resetSim->setDisabled(true);
			m_current_frame_spinbox->setEnabled(false);
			m_slider->setEnabled(false);
		}
		else
		{
			PSimulationThread::instance()->pause();
			m_startSim->setText("Resume");
			m_resetSim->setDisabled(false);

			m_current_frame_spinbox->setEnabled(true);
			m_slider->setEnabled(true);
		}
	
	}

	void PAnimationWidget::resetSimulation()
	{
		PSimulationThread::instance()->reset();
		m_startSim->setText("Start");
		m_startSim->setEnabled(true);


		m_current_frame_spinbox->setEnabled(true);
		m_slider->setEnabled(true);
		m_slider->setValue(0);
	}

	void PAnimationWidget::simulationFinished()
	{
		m_startSim->setText("Finished");
		m_startSim->setDisabled(true);
		m_resetSim->setDisabled(false);

		m_current_frame_spinbox->setEnabled(true);
		m_slider->setEnabled(true);
	}

	void PAnimationWidget::updateSlider() {
		int CurrentFrameNum = PSimulationThread::instance()->getCurrentFrameNum();
	
		if (CurrentFrameNum > totalFrame) {
			totalFrame += 200;
			for (int i = 0; i < labelSize; i++) {
				label[i]->setText(QString::number(totalFrame / 5 * i));
			}
			m_slider->setRange(0, totalFrame);
			m_slider->setSingleStep(totalFrame / 20);
		}
		
		if(m_startSim->text()!=QString("Start"))  m_slider->setValue(CurrentFrameNum);
	}
	
}
