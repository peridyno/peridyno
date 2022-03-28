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
		QHBoxLayout* layout = new QHBoxLayout();
		setLayout(layout);

		QGridLayout* frameLayout	= new QGridLayout();
		QLineEdit* startFrame		= new QLineEdit();
		QLineEdit* endFrame			= new QLineEdit();

		m_end_spinbox = new QSpinBox();
		m_end_spinbox->setFixedSize(60, 25);
		m_end_spinbox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
		m_end_spinbox->setMaximum(99999);
		m_end_spinbox->setValue(2000);
	
		QGridLayout* GLayout = new QGridLayout;

		m_slider = new PAnimationQSlider(this);
		
		QLabel* label1 = new QLabel("0");
		QLabel* label2 = new QLabel("500");
		QLabel* label3 = new QLabel("1000");
		QLabel* label4 = new QLabel("1500");
		QLabel* label5 = new QLabel("2000");

		frameLayout->addWidget(label1, 1, 0,1,1, Qt::AlignTop | Qt::AlignLeft);
		frameLayout->addWidget(label2, 1, 1,1,1, Qt::AlignTop | Qt::AlignHCenter);
		frameLayout->addWidget(label3, 1, 2,1,2, Qt::AlignTop | Qt::AlignHCenter);
		frameLayout->addWidget(label4, 1, 4,1,1, Qt::AlignTop | Qt::AlignHCenter);
		frameLayout->addWidget(label5, 1, 5,1,1, Qt::AlignTop | Qt::AlignRight);
		
	
		frameLayout->addWidget(m_slider, 0, 0, 0 ,6);
		


		connect(m_end_spinbox, static_cast<void (QSpinBox ::*)(int)>(&QSpinBox::valueChanged), this, [=]() {
				int maxValue = m_end_spinbox->value();
				label2->setText(QString::number(maxValue/4));
				label3->setText(QString::number(maxValue / 2));
				label4->setText(QString::number(maxValue / 4 *3));
				label5->setText(QString::number(maxValue));

				m_slider->setRange(0, maxValue);
		});

		connect(m_slider, &PAnimationQSlider::valueChanged, m_end_spinbox,
			[&]() {
				//m_end_spinbox->setValue(m_slider->value());
			}
		);

	

		void(QSpinBox:: * spSignal)(int) = &QSpinBox::valueChanged;
		connect(m_end_spinbox, spSignal, m_slider, &QSlider::setValue);
	
		QGridLayout* operationLayout = new QGridLayout();

		m_startSim = new QPushButton("Start");
		m_resetSim = new QPushButton("Reset");

		operationLayout->addWidget(m_end_spinbox, 0, 0);
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
			if(m_startSim->text() == "Start")
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
		m_startSim->setText("Start");
		m_startSim->setEnabled(true);

		m_end_spinbox->setValue(0);
		m_slider->setValue(0);
	}

	void PAnimationWidget::simulationFinished()
	{
		m_startSim->setText("Finished");
		m_startSim->setDisabled(true);
		m_resetSim->setDisabled(false);
	}

	void PAnimationWidget::updateSlider() {
		int CurrentFrameNum = PSimulationThread::instance()->getCurrentFrameNum();
		m_end_spinbox->setValue(CurrentFrameNum);
		m_slider->setValue(CurrentFrameNum);
	}
	
}
