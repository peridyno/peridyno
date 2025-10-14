#include "QPiecewiseSpinBox.h"

#include "Module.h"
#include "Node.h"
#include "Field.h"
#include "SceneGraphFactory.h"

#include <QVBoxLayout>

#include <memory>
#include "QValueDialog.h"

namespace dyno
{
	QPiecewiseSpinBox::QPiecewiseSpinBox(QWidget* parent)
		: QSpinBox(parent)
	{

		lineEdit()->installEventFilter(this);
		this->setKeyboardTracking(false);


	}

	void QPiecewiseSpinBox::wheelEvent(QWheelEvent* event)
	{

	}

	void QPiecewiseSpinBox::contextMenuEvent(QContextMenuEvent* event)
	{
		QSpinBox::contextMenuEvent(event);
	}


	bool QPiecewiseSpinBox::eventFilter(QObject* obj, QEvent* event) 
	{
		if (obj == lineEdit())
		{
			if (event->type() == QEvent::MouseButtonPress)
			{
				QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
				if (mouseEvent->button() == Qt::MiddleButton)
				{
					createValueDialog();
					return true;
				}
			}
		}
		return QSpinBox::eventFilter(obj, event);
	}

	void QPiecewiseSpinBox::createValueDialog() 
	{
		if (mValueDialog == nullptr)
			mValueDialog = new QValueDialog(this);
		mValueDialog->updateDialogPosition();
		mValueDialog->show();
	}

	void QPiecewiseSpinBox::mousePressEvent(QMouseEvent* event)
	{
		QSpinBox::mousePressEvent(event);

		if (event->button() == Qt::MiddleButton)
		{
			createValueDialog();
		}

	}

}

