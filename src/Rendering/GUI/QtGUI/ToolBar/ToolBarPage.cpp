#include "ToolBarPage.h"

namespace dyno {
	ToolBarPage::ToolBarPage() {

		//ToolBar file----------------------------------
		ToolBarIcoAndLabel filePage;
		// Main Page name and ico
		filePage.tabPageName = "File";
		filePage.tabPageIco = "48px-Document-new.png";
		//number of subTabs
		filePage.subtabNum = 4;
		//subTabs
		filePage.label.push_back("New...");
		filePage.ico.push_back("48px-Document-new.png");

		filePage.label.push_back("Open");
		filePage.ico.push_back("48px-Document-open.png");

		filePage.label.push_back("Save");
		filePage.ico.push_back("48px-Document-save.png");

		filePage.label.push_back("Save As");
		filePage.ico.push_back("48px-Document-save-as.png");
		//--------------------------------------------------

		//ToolBar edit--------------------------------------
		ToolBarIcoAndLabel editPage;
		// Main Page name and ico
		editPage.tabPageName = "Edit";
		editPage.tabPageIco = "48px-Preferences-system.png";
		//number of subTabs
		editPage.subtabNum = 1;
		//subTabs
		editPage.label.push_back("Settings");
		editPage.ico.push_back("48px-Preferences-system.png");
		//--------------------------------------------------

		//ToolBar particle----------------------------------
		ToolBarIcoAndLabel particlePage;
		// Main Page name and ico
		particlePage.tabPageName = "Particle System";
		particlePage.tabPageIco = "dyverso/icon-emi-fill.svg";
		//number of subTabs
		particlePage.subtabNum = 4;
		//subTabs
		particlePage.label.push_back("ParticleEmitterRound");
		particlePage.ico.push_back("dyverso/icon-emi-n.svg");

		particlePage.label.push_back("ParticleFluid");
		particlePage.ico.push_back("dyverso/icon-emi-bitmap.svg");

		particlePage.label.push_back("ParticleSystem");
		particlePage.ico.push_back("dyverso/icon-emi-circle.svg");

		particlePage.label.push_back("ParticleEmitterSquare");
		particlePage.ico.push_back("dyverso/icon-emi-object.svg");
		//----------------------------------------------------
		
		//ToolBar height Field--------------------------------
		ToolBarIcoAndLabel heightPage;
		// Main Page name and ico
		heightPage.tabPageName = "Height Field";
		heightPage.tabPageIco = "icon-realwave.svg";
		//number of subTabs
		heightPage.subtabNum = 3;
		//subTabs
		heightPage.label.push_back("OceanPach");
		heightPage.ico.push_back("icon-realwave.svg");

		heightPage.label.push_back("Wave 2");
		heightPage.ico.push_back("icon-realwave-cresplash.svg");

		heightPage.label.push_back("Wave 3");
		heightPage.ico.push_back("icon-realwave-objspash.svg");
		//--------------------------------------------------
		
		//ToolBar Finite Element----------------------------
		ToolBarIcoAndLabel FinitePage;
		// Main Page name and ico
		FinitePage.tabPageName = "Finite Element";
		FinitePage.tabPageIco = "daemon/icon-demon-vortex.svg";
		//number of subTabs
		FinitePage.subtabNum = 3;
		//subTabs
		FinitePage.label.push_back("Soft Body 1");
		FinitePage.ico.push_back("daemon/icon-demon-vortex.svg");

		FinitePage.label.push_back("Soft Body 2");
		FinitePage.ico.push_back("daemon/icon-demon-heater.svg");

		FinitePage.label.push_back("Soft Body 3");
		FinitePage.ico.push_back("daemon/icon-demon-ellipsoid.svg");
		//--------------------------------------------------

		//ToolBar Rigid Body--------------------------------
		ToolBarIcoAndLabel RigidPage;
		// Main Page name and ico
		RigidPage.tabPageName = "Rigid Body";
		RigidPage.tabPageIco = "geometry/icon-geometry-cube.svg";
		//number of subTabs
		RigidPage.subtabNum = 4;
		//subTabs
		RigidPage.label.push_back("GLPointVisualNode");
		RigidPage.ico.push_back("geometry/icon-geometry-cube.svg");

		RigidPage.label.push_back("GhostFluid");
		RigidPage.ico.push_back("geometry/icon-geometry-cylinder.svg");

		RigidPage.label.push_back("GhostParticles");
		RigidPage.ico.push_back("geometry/icon-geometry-rocket.svg");

		RigidPage.label.push_back("StaticBoundary");
		RigidPage.ico.push_back("geometry/icon-geometry-multibody.svg");
		//--------------------------------------------------

		//ToolBar help--------------------------------------
		ToolBarIcoAndLabel helpPage;
		// Main Page name and ico
		helpPage.tabPageName = "Help";
		helpPage.tabPageIco = "Help-browser.png";
		//number of subTabs
		helpPage.subtabNum = 1;
		//subTabs
		helpPage.label.push_back("Help");
		helpPage.ico.push_back("Help-browser.png");
		//--------------------------------------------------

		//Add all the page
		tbl.push_back(filePage);// 0
		tbl.push_back(editPage);// 1
		tbl.push_back(particlePage);//2
		tbl.push_back(heightPage);//3
		tbl.push_back(FinitePage);//4
		tbl.push_back(RigidPage);//5
		tbl.push_back(helpPage);//6
	}


	ToolBarPage::~ToolBarPage() {

	}


}