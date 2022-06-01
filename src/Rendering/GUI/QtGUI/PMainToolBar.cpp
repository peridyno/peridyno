#include "PMainToolBar.h"

#include <QPainter>
#include <QToolButton>
#include <QtSvg/QSvgRenderer>

//tool bar
#include "ToolBar/Page.h"
#include "ToolBar/Group.h"
#include "ToolBar/ToolBarPage.h"

#include "NodeFactory.h"

//core
#include "Platform.h"

namespace dyno
{
	PMainToolBar::PMainToolBar(QWidget* parent, unsigned _groupMaxHeight, unsigned _groupRowCount) :
		tt::TabToolbar(parent, _groupMaxHeight, _groupRowCount)
	{
		QString mediaDir = QString::fromLocal8Bit(getAssetPath().c_str()) + "icon/";

		auto convertIcon = [&](QString path) -> QIcon
		{
			QSvgRenderer svg_render(path);
			QPixmap pixmap(48, 48);
			pixmap.fill(Qt::transparent);
			QPainter painter(&pixmap);
			svg_render.render(&painter);
			QIcon ico(pixmap);

			return ico;
		};

		//Add ToolBar page
		ToolBarPage m_toolBarPage;
		std::vector<ToolBarIcoAndLabel> v_IcoAndLabel = m_toolBarPage.tbl;

		for (int i = 0; i < v_IcoAndLabel.size(); i++) {
			ToolBarIcoAndLabel m_tbl = v_IcoAndLabel[i];

			tt::Page* MainPage = this->AddPage(QPixmap(mediaDir + m_tbl.tabPageIco), m_tbl.tabPageName);
			auto m_page = MainPage->AddGroup("");

			for (int j = 0; j < m_tbl.ico.size(); j++) {
				//Add subtabs
				QAction* art = new QAction(QPixmap(mediaDir + m_tbl.ico[j]), m_tbl.label[j]);
				m_page->AddAction(QToolButton::DelayedPopup, art);

				if (i == 2 || i == 5 || i == 3) {//add connect event 
					//connect(art, &QAction::triggered, this, [=]() {addNodeByName(m_tbl.label[j].toStdString() + "<DataType3f>"); });
				}
			}
		}

		//Add dynamic toolbar page
		auto& groups = NodeFactory::instance()->nodeGroups();
		for(auto itor = groups.begin(); itor != groups.end(); itor++)
		{
			tt::Page* page = this->AddPage(QPixmap(mediaDir + QString::fromStdString(itor->second->icon())), QString::fromStdString(itor->second->caption()));
			auto& actions = itor->second->actions();

			auto page_group = page->AddGroup("");
			for (size_t i = 0; i < actions.size(); i++)
			{
				QAction* act = new QAction(QPixmap(mediaDir + QString::fromStdString(actions[i]->icon())), QString::fromStdString(actions[i]->caption()));
				page_group->AddAction(QToolButton::DelayedPopup, act);

				auto func = actions[i]->action();

				if (func != nullptr) {
					connect(act, &QAction::triggered, [=]() {
						emit nodeCreated(func());
						});
				}
			}
		}
	}
}

