#include "PStatusBar.h"

#include <QPushButton>
#include <QProgressBar>
#include <QLabel>

namespace dyno
{
	PStatusBar::PStatusBar(QWidget *parent) :
		QStatusBar(parent),
		m_progressBar(nullptr),
		m_progressLabel(nullptr)
	{
		m_progressBar = new QProgressBar();
		m_progressLabel = new QLabel();

		m_progressBar->setFixedWidth(200);

		addPermanentWidget(m_progressBar);
		addPermanentWidget(m_progressLabel);

		m_progressBar->hide();
	}

}

