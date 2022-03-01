#include "PAnimationWidget.h"

#include "PSimulationThread.h"

#include <QGridLayout>
#include <QPushButton>
#include <QSpinBox>
#include <QLineEdit>
#include <QSlider>
#include <QScrollBar>
#include <QIntValidator>
#include <QLabel>
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

		//m_sim_scrollbar = new QScrollBar(Qt::Horizontal, this);
		//m_sim_scrollbar->setFixedHeight(25);
		//m_sim_scrollbar->setPageStep(999);

		frameLayout->addWidget(m_start_spinbox, 0, 0);
		//frameLayout->addWidget(m_sim_scrollbar, 0, 1);
		
		//slider---------------------
		QGridLayout* GLayout = new QGridLayout;
		QSlider* slider = new QSlider(Qt::Horizontal, this);
		slider->setRange(1, 250);
		slider->setSingleStep(1);
		QLabel* label1 = new QLabel("50", this);
		QLabel* label2 = new QLabel("100", this);
		QLabel* label3 = new QLabel("150", this);
		QLabel* label4 = new QLabel("200", this);
		QLabel* label5 = new QLabel("250", this);

		frameLayout->addWidget(slider, 0, 1, 1, 4);
		frameLayout->addWidget(label1, 1, 1, 1, 1);
		frameLayout->addWidget(label2, 1, 2, 1, 1);
		frameLayout->addWidget(label3, 1, 3, 1, 1);
		frameLayout->addWidget(label4, 1, 4, 1, 1);
		frameLayout->addWidget(label5, 1, 5, 1, 1);


		frameLayout->addWidget(m_end_spinbox, 0, 5);

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
