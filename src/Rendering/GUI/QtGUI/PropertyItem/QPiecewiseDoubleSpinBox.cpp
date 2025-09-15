#include "QPiecewiseDoubleSpinBox.h"

#include "Module.h"
#include "Node.h"
#include "Field.h"
#include "SceneGraphFactory.h"

#include <QVBoxLayout>

#include <memory>
#include "QValueDialog.h"

namespace dyno
{


	QPiecewiseDoubleSpinBox::QPiecewiseDoubleSpinBox(QWidget* parent)
		: QDoubleSpinBox(parent)
	{
		this->setRange(-999999, 999999);
		this->setContentsMargins(0, 0, 0, 0);

		connect(this, &QDoubleSpinBox::editingFinished, this, &QPiecewiseDoubleSpinBox::onEditingFinished);

		this->setDecimals(decimalsMax);

		lineEdit()->installEventFilter(this);
		this->setKeyboardTracking(false);

	}

	QPiecewiseDoubleSpinBox::QPiecewiseDoubleSpinBox(Real v,QWidget* parent)
		: QDoubleSpinBox(parent)
	{
		this->setRange(-999999, 999999);
		this->setValue(v);
		this->setRealValue(v);

		this->setDecimals(decimalsMax);
		this->setKeyboardTracking(false);
	}

	void QPiecewiseDoubleSpinBox::wheelEvent(QWheelEvent* event)
	{

	}

	void QPiecewiseDoubleSpinBox::contextMenuEvent(QContextMenuEvent* event) 
	{
		QDoubleSpinBox::contextMenuEvent(event);
	}

	void QPiecewiseDoubleSpinBox::mousePressEvent(QMouseEvent* event)
	{
		if (event->button() == Qt::MiddleButton)
		{
			createValueDialog();
		}

		QDoubleSpinBox::mousePressEvent(event);
		
	}

	void QPiecewiseDoubleSpinBox::mouseReleaseEvent(QMouseEvent* event)
	{
		QDoubleSpinBox::mouseReleaseEvent(event);
	}

	void QPiecewiseDoubleSpinBox::mouseMoveEvent(QMouseEvent* event)
	{
		QDoubleSpinBox::mouseMoveEvent(event);

	}

	bool QPiecewiseDoubleSpinBox::eventFilter(QObject* obj, QEvent* event)
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
		return QDoubleSpinBox::eventFilter(obj, event);
	}

	void QPiecewiseDoubleSpinBox::createValueDialog()
	{
		if (mValueDialog == nullptr)
			mValueDialog = new QValueDialog(this);
		mValueDialog->updateDialogPosition();
		mValueDialog->show();
	}

	double QPiecewiseDoubleSpinBox::setRealValue(double val)
	{
		this->setKeyboardTracking(true);
		realValue = val;
		this->lineEdit()->setText(QString::number(realValue, 10, displayDecimals));
		this->setKeyboardTracking(false);

		return realValue;
	}

}

