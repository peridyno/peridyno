#include "Format.h"

#include <QRegularExpression>
#include <QVector>

namespace dyno
{
	QString FormatFieldWidgetName(std::string name)
	{
		QString qName = QString::fromStdString(name.c_str());

		bool isChinese = qName.contains(QRegularExpression("[\u4e00-\u9fa5]"));

		//If the string contains Chinese, show all the original string without splitting
		if (isChinese)
		{
			return qName;
		}

		//remove the "*_" prefix
		QRegularExpression prefix("_(.+)");
		QRegularExpressionMatch prefix_match = prefix.match(qName);

		QString subtitle = prefix_match.hasMatch() ? prefix_match.captured(1) : qName;

		//Otherwise, slit the name by the space
		QRegularExpression regexp("[A-Z][^A-Z]*");
		QRegularExpressionMatchIterator match = regexp.globalMatch(subtitle);

		QString ret;
		while (match.hasNext())
		{
			ret += match.next().captured() + " ";
		}

		return ret;
	}

	QString FormatBlockPortName(std::string name)
	{
		QString qName = QString::fromStdString(name.c_str());

		bool isChinese = qName.contains(QRegularExpression("[\u4e00-\u9fa5]"));

		//If the string contains Chinese, show all the original string without splitting
		if (isChinese)
		{
			return qName;
		}

		//remove the "*_" prefix
		QRegularExpression prefix("_(.+)");
		QRegularExpressionMatch prefix_match = prefix.match(qName);

		QString subtitle = prefix_match.hasMatch() ? prefix_match.captured(1) : qName;

		QRegularExpression regexp0("[A-Za-z()]*");
		QRegularExpressionMatchIterator match0 = regexp0.globalMatch(subtitle);

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
		QString qName = QString::fromStdString(name.c_str());

		bool isChinese = qName.contains(QRegularExpression("[\u4e00-\u9fa5]"));

		//If the string contains Chinese, show all the original string without splitting
		if (isChinese)
		{
			return qName;
		}

		QRegularExpression regexp0("[A-Za-z0-9]*");
		QRegularExpressionMatchIterator match0 = regexp0.globalMatch(qName);

		QString subStr = match0.hasNext() ? match0.next().captured() : QString("Port");

		QRegularExpression regexp("[A-Za-z0-9]*");
		QRegularExpressionMatchIterator match = regexp.globalMatch(subStr);

		QString ret;
		while (match.hasNext())
		{
			ret += match.next().captured() + QString(" ");
		}

		return ret;
	}

	QString FormatDescription(std::string name)
	{
		QString desc = QString::fromStdString(name.c_str());

// 		bool isChinese = qName.contains(QRegExp("[\\x4e00-\\x9fa5]+"));
// 
// 		//If the string contains Chinese, show all the original string without splitting
// 		if (isChinese)
// 		{
// 			return qName;
// 		}

		return desc;
	}

}
