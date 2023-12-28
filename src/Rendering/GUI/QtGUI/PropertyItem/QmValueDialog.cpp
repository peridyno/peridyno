#include "QmValueDialog.h"

#include "Module.h"
#include "Node.h"
#include "Field.h"
#include "SceneGraphFactory.h"

#include <QVBoxLayout>

#include <memory>
#include "QmDoubleSpinBox.h"

namespace dyno
{

	ValueDialog::ValueDialog(QAbstractSpinBox* parent)
	{

		SBox1 = parent;
		//
		QVBoxLayout* VLayout = new QVBoxLayout;
		float power = 0.1;

		auto doubleSpinBox = TypeInfo::cast<mDoubleSpinBox>(parent);
		if (doubleSpinBox != nullptr)
		{
			printf("castto mDoubleSpinBox\n");
			mDSpinBox = doubleSpinBox;
			for (int i = 0; i < 5; i++)
			{
				button[i] = new ValueButton;

				power *= 0.1;

				std::string s = std::to_string(power * 1000);
				QString text = QString::fromStdString(s);

				button[i]->setText(text);//initial
				button[i]->setRealText(text);
				button[i]->setFixedWidth(200);
				button[i]->setFixedHeight(40);
				button[i]->adjustSize();
				//button[i]->setAlignment(Qt::AlignHCenter| Qt::AlignVCenter);
				button[i]->setStyleSheet("QLabel{color:white;background-color:#346792;border: 1px solid #000000;border-radius:3px; padding: 0px;}");
				button[i]->StartX = QCursor().pos().x();
				button[i]->defaultValue = power * 1000;
				button[i]->SpinBoxData = doubleSpinBox->getRealValue();
				button[i]->parentDialog = this;

				VLayout->addWidget(button[i]);

				connect(button[i], SIGNAL(ValueChange(double)), doubleSpinBox, SLOT(ModifyValueAndUpdate(double)));
				connect(button[i], SIGNAL(Release(double)), this, SLOT(initData(double)));
			}
		}

		auto mIntSpinBox = TypeInfo::cast<QSpinBox>(parent);
		if (mIntSpinBox != nullptr)
		{
			mISpinBox = mIntSpinBox;
			printf("castto QSpinBox\n");
		}

		VLayout->setSpacing(0);

		this->setLayout(VLayout);
		this->setWindowFlags(Qt::WindowStaysOnTopHint | Qt::WindowCloseButtonHint | Qt::Popup);
		this->updateDialogPosition();
		this->setMouseTracking(true);
		this->hasMouseTracking();
		this->setAttribute(Qt::WA_Hover, true);

		this->setWindowTitle("Property Editor");


	}
	void ValueDialog::updateDialogPosition() 
	{
		this->move(QCursor().pos().x() - button[1]->rect().width() / 2, QCursor().pos().y() - button[1]->rect().height() * 5 / 2 - 5);

	}

	
	void ValueDialog::mouseReleaseEvent(QMouseEvent* event)
	{

	}



	void  ValueDialog::keyPressEvent(QKeyEvent* event)
	{
		QDialog::keyPressEvent(event);
		if (event->key() == Qt::Key_Shift)
		{
			for (size_t i = 0; i < 5; i++)
			{
				this->button[i]->shiftPress = true;
			}
		}
	}
	void  ValueDialog::keyReleaseEvent(QKeyEvent* event)
	{
		QDialog::keyReleaseEvent(event);
		if (event->key() == Qt::Key_Shift)
		{
			for (size_t i = 0; i < 5; i++)
			{
				this->button[i]->shiftPress = false;
			}
		}
	}

	void ValueDialog::initData(double v)
	{	
		if (mDSpinBox != nullptr) 
		{
			printf("castto doubleSpineBox\n");
			for (int i = 0; i < 5; i++)
			{
				button[i]->SpinBoxData = v;
				button[i]->Data1 = mDSpinBox->getRealValue();
			}
		}
	}

	void ValueDialog::mouseMoveEvent(QMouseEvent* event)
	{

	}

	void ValueButton::mouseMoveEvent(QMouseEvent* event)
	{
		EndX = QCursor().pos().x();
		temp = (EndX - StartX) / 10;
		sub = defaultValue * temp;

		//str = std::to_string(sub);
		if (displayRealValue) 
			str = std::to_string(defaultValue) + "\n" + std::to_string(SpinBoxData + sub);
		else 
			str = std::to_string(sub);
		
		text = QString::fromStdString(str);
		this->setText(text);

		//if (shiftPress)
		//{
		//	double p = (SpinBoxData+sub)/SpinBoxData;
		//	
		//	double d1 = DSB1->value();
		//	double d2 = DSB2->value();
		//	double d3 = DSB3->value();

		//	DSB1->setValue(Data1 * p);
		//	DSB2->setValue(Data2 * p);
		//	DSB3->setValue(Data3 * p);
		//}

		emit ValueChange(SpinBoxData + sub);


	}

	ValueButton::ValueButton(QWidget* parent) :
		QPushButton(parent)
	{

	}

	void ValueButton::mousePressEvent(QMouseEvent* event)
	{
		StartX = QCursor().pos().x();
	}
	void ValueButton::mouseReleaseEvent(QMouseEvent* event)
	{
		str = std::to_string(defaultValue);
		text = QString::fromStdString(str);
		this->setText(text);
		SpinBoxData = SpinBoxData + sub;

		emit Release(SpinBoxData);

	}
}

