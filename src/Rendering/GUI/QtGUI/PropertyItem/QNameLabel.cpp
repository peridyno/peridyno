#include "QNameLabel.h"
#include <QLabel>
#include <QString>
#include <QFontMetrics>
#include "Format.h"

QNameLabel::QNameLabel(QString str)
{
	this->setText(str);
	resizeLabel();
}

void QNameLabel::resizeLabel() 
{
	QString str = this->text();
	QFontMetrics fm(this->font());

	int textWidth = fm.horizontalAdvance(str);
	int textHeight = fm.height();

	int margin = 4;
	this->setFixedSize(textWidth + margin * 2, textHeight + margin * 2);
	this->setText(str);

	//Set label tips
	this->setToolTip(str);
}
