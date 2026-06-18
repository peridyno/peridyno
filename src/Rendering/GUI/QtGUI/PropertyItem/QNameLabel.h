#pragma once
//Qt
#include <QLabel>


class QNameLabel : public QLabel 
{
	Q_OBJECT
public:
	QNameLabel(QString str);
	~QNameLabel() {}

protected:
	void resizeLabel();

};




