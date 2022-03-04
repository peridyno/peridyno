#ifndef TOOLBARICOANDLABEL_H
#define TOOLBARICOANDLABEL_H

#include <iostream>
#include <vector>
#include <qstring.h>
namespace dyno {
	class ToolBarIcoAndLabel {
	public:

		int num;

		std::vector<QString> ico;
		std::vector<QString> label;

	};
	
	class ToolBarPage {	
	public:
		ToolBarPage();
		~ToolBarPage();

	public:
		std::vector<ToolBarIcoAndLabel> tbl;

	
	};

}

#endif
