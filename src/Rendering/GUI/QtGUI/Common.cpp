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

		QRegularExpression regexp0("[A-Za-z()]*");
		QRegularExpressionMatchIterator match0 = regexp0.globalMatch(fName);

		QString subStr = match0.hasNext() ? match0.next().captured() : QString("Port");

		QRegularExpression regexp("[A-Z][^A-Z]*");
		QRegularExpressionMatchIterator match = regexp.globalMatch(subStr);

		QString ret;
		while (match.hasNext())
		{
			ret += match.next().captured() + " ";
		}

		return ret;
	}

	QString FormatBlockCaptionName(std::string name)
	{
		QString fName = QString::fromStdString(name);

		QRegularExpression regexp0("[A-Za-z()\\s]*");
		QRegularExpressionMatchIterator match0 = regexp0.globalMatch(fName);

		QString subStr = match0.hasNext() ? match0.next().captured() : QString("Port");

		QRegularExpression regexp("[A-Z][^A-Z]*");
		QRegularExpressionMatchIterator match = regexp.globalMatch(subStr);

		QString ret;
		while (match.hasNext())
		{
			ret += match.next().captured() + QString(" ");
		}

		return ret;
	}

}
