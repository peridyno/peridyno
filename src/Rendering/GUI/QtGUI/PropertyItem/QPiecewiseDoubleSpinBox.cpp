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
		//this->lineEdit()->setMouseTracking(true);
		this->setRange(-999999, 999999);
		this->setContentsMargins(0, 0, 0, 0);

		connect(this, SIGNAL(valueChanged(double)), this, SLOT(ModifyValueAndUpdate(double)));
		connect(this->lineEdit(), SIGNAL(textEdited(const QString&)), this, SLOT(LineEditStart(const QString&)));

		this->setDecimals(decimalsMax);
		this->setKeyboardTracking(false);

	}

	QPiecewiseDoubleSpinBox::QPiecewiseDoubleSpinBox(Real v,QWidget* parent)
		: QDoubleSpinBox(parent)
	{
		this->setRange(-999999, 999999);
		this->setValue(v);
		this->setRealValue(v);

		connect(this, SIGNAL(valueChanged(double)), this, SLOT(ModifyValueAndUpdate(double)));
		connect(this->lineEdit(), SIGNAL(textEdited(const QString&)), this, SLOT(LineEditStart(const QString&)));

		this->setDecimals(decimalsMax);
		this->setKeyboardTracking(false);


	}

	void QPiecewiseDoubleSpinBox::LineEditStart(const QString& qStr)
	{
		auto v = this->value();
		const auto& value = qStr.toDouble();
		auto min = this->minimum();
		auto max = this->maximum();

		if (value < this->minimum()) 
		{
			this->lineEdit()->setText(QString::number(this->minimum()));
		}
		if (value > this->maximum()) 
		{
			this->lineEdit()->setText(QString::number(this->maximum()));
		}

		return;
	}

	void QPiecewiseDoubleSpinBox::LineEditFinished(double v) 
	{
		realValue = v;
		this->setValue(realValue);

		return;
	}

	void QPiecewiseDoubleSpinBox::wheelEvent(QWheelEvent* event)
	{

	}
	void QPiecewiseDoubleSpinBox::contextMenuEvent(QContextMenuEvent* event) 
	{

		if(ValueModify== nullptr)
			ValueModify = new QValueDialog(this);
		ValueModify->updateDialogPosition();
		ValueModify->show();
	}


	void QPiecewiseDoubleSpinBox::mousePressEvent(QMouseEvent* event)
	{
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

	void QPiecewiseDoubleSpinBox::ModifyValue(double v)
	{
		this->setRealValue(v);

	}

	void QPiecewiseDoubleSpinBox::ModifyValueAndUpdate(double v)
	{

		this->setKeyboardTracking(true);
		this->setRealValue(v);
		this->lineEdit()->setText(QString::number(realValue, 10, displayDecimals));
		this->setKeyboardTracking(false);
	}


}

