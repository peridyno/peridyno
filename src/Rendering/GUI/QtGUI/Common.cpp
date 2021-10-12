#include "Common.h"

#include <QRegularExpression>
#include <QVector>

namespace dyno
{
	QString FormatFieldWidgetName(std::string name)
	{
		QString fName = QString::fromStdString(name);

		QRegularExpression regexp("[A-Z][^A-Z]*");
		QRegularExpressionMatchIterator match = regexp.globalMatch(fName);
		QVector<QString> vec;

		QString ret;
		while (match.hasNext())
		{
			ret += match.next().captured() + " ";
		}

		return ret;
	}

	QString FormatBlockPortName(std::string name)
	{
		QString fName = QString::fromStdString(name);

		QRegularExpression regexp("[A-Z][^A-Z]*");
		QRegularExpressionMatchIterator match = regexp.globalMatch(fName);
		QVector<QString> vec;

		QString ret;
		while (match.hasNext())
		{
			ret += match.next().captured() + " ";
		}

		return ret;
	}

}
