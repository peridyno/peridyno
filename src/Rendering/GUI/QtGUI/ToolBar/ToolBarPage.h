#ifndef TOOLBARICOANDLABEL_H
#define TOOLBARICOANDLABEL_H

#include <iostream>
#include <vector>
#include <qstring.h>
namespace dyno {
	//A tab in the toolbar
	class ToolBarIcoAndLabel {
	public:
		//number of subtabs
		int num;
		//Icons for subtabs
		std::vector<QString> ico;
		//Icons for label
		std::vector<QString> label;

	};
	
	//Add all the ToolBar Page
	class ToolBarPage {	
	public:
		ToolBarPage();
		~ToolBarPage();

	public:
		std::vector<ToolBarIcoAndLabel> tbl;

	};

}

#endif
