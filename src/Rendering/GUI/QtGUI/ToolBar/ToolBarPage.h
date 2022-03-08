#ifndef TOOLBARICOANDLABEL_H
#define TOOLBARICOANDLABEL_H

#include <iostream>
#include <vector>
#include <qstring.h>
namespace dyno {
	//A tab in the toolbar
	class ToolBarIcoAndLabel {
	public:
		// Page Name
		QString tabPageName;
		QString tabPageIco;
			
		//number of subtabs
		int subtabNum;
		//Icons for subtabs
		std::vector<QString> ico;
		//labels for subtabs
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
