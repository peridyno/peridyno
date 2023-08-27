#include "QToggleButton.h"

#include "Module.h"
#include "Node.h"
#include "Field.h"
#include "SceneGraphFactory.h"

#include <QVBoxLayout>

namespace dyno
{
	QToggleButton::QToggleButton(QWidget* pParent):
		QPushButton(pParent)
	{
		connect(this, &QPushButton::clicked,this, &QToggleButton::ModifyText);
	}

	QToggleButton::QToggleButton(bool isChecked, QWidget* pParent) :
		QPushButton(pParent)
	{
		connect(this, &QPushButton::clicked, this, &QToggleButton::ModifyText);
	}

	void QToggleButton::ModifyText()
	{
		//std::cout << textChecked << "  :  " << textUnChecked << std::endl;
		QString t;
		std::string s;
		isPress = !isPress;
		if (isPress)
		{
			t = QString::fromStdString(textChecked);
		}
		else 
		{
			t = QString::fromStdString(textUnChecked);
		}
		this->QPushButton::setText(t);

		emit clicked();
	}

	void  QToggleButton::setText(std::string textUnCheck, std::string textCheck)
	{
		textUnChecked = textUnCheck;
		textChecked = textCheck;
		updateText();
	}



}

