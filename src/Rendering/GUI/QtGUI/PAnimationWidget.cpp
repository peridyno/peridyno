#include "PAnimationWidget.h"

#include "PSimulationThread.h"

#include <QGridLayout>
#include <QPushButton>
#include <QSpinBox>
#include <QLineEdit>
#include <QSlider>
#include <QScrollBar>
#include <QIntValidator>

namespace dyno
{
	PAnimationWidget::PAnimationWidget(QWidget *parent) : 
		QWidget(parent),
		m_startSim(nullptr),
		m_resetSim(nullptr)
	{
		QHBoxLayout* layout = new QHBoxLayout();
		setLayout(layout);

		QGridLayout* frameLayout	= new QGridLayout();
		QLineEdit* startFrame		= new QLineEdit();
		QLineEdit* endFrame			= new QLineEdit();

		

		m_start_spinbox = new QSpinBox();
		m_start_spinbox->setFixedSize(60, 25);
		m_start_spinbox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

		m_end_spinbox = new QSpinBox();
		m_end_spinbox->setFixedSize(60, 25);
		m_end_spinbox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
		m_end_spinbox->setMaximum(99999);
		m_end_spinbox->setValue(999);

		m_sim_scrollbar = new QScrollBar(Qt::Horizontal, this);
		m_sim_scrollbar->setFixedHeight(25);
		m_sim_scrollbar->setPageStep(999);

		frameLayout->addWidget(m_start_spinbox, 0, 0);
		frameLayout->addWidget(m_sim_scrollbar, 0, 1);
		frameLayout->addWidget(m_end_spinbox, 0, 2);

		QGridLayout* operationLayout = new QGridLayout();

		m_startSim = new QPushButton("Start");
		m_resetSim = new QPushButton("Reset");
		operationLayout->addWidget(m_startSim, 0, 0);
		operationLayout->addWidget(m_resetSim, 0, 1);

		m_startSim->setCheckable(true);

		layout->addLayout(frameLayout, 10);
		layout->addLayout(operationLayout, 1);


		connect(m_startSim, SIGNAL(released()), this, SLOT(toggleSimulation()));
		connect(m_resetSim, SIGNAL(released()), this, SLOT(resetSimulation()));
		connect(PSimulationThread::instance(), SIGNAL(finished()), this, SLOT(simulationFinished()));

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
			PSimulationThread::instance()->setTotalFrames(m_end_spinbox->value());
			PSimulationThread::instance()->resume();
			m_startSim->setText("Pause");
			m_resetSim->setDisabled(true);
		}
		else
		{
			PSimulationThread::instance()->pause();
			m_startSim->setText("Resume");
			m_resetSim->setDisabled(false);
		}
	}

	void PAnimationWidget::resetSimulation()
	{
		PSimulationThread::instance()->reset();
	}

	void PAnimationWidget::simulationFinished()
	{
		m_startSim->setText("Finished");
		m_startSim->setDisabled(true);
	}

}
