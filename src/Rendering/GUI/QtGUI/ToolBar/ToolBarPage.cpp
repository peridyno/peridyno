#include "ToolBarPage.h"

namespace dyno {
	ToolBarPage::ToolBarPage() {

		//ToolBar file
		ToolBarIcoAndLabel filePage;
		//number of filePage
		filePage.num = 5;

		filePage.label.push_back("File");
		filePage.ico.push_back("48px-Document-open.png");

		filePage.label.push_back("New...");
		filePage.ico.push_back("48px-Document-new.png");

		filePage.label.push_back("Open");
		filePage.ico.push_back("48px-Document-open.png");

		filePage.label.push_back("Save");
		filePage.ico.push_back("48px-Document-save.png");

		filePage.label.push_back("Save As");
		filePage.ico.push_back("48px-Document-save-as.png");

		//ToolBar edit
		ToolBarIcoAndLabel editPage;
		editPage.num = 2;

		editPage.label.push_back("Edit");
		editPage.ico.push_back("48px-Preferences-system.png");

		editPage.label.push_back("Settings");
		editPage.ico.push_back("48px-Preferences-system.png");

		//ToolBar particle
		ToolBarIcoAndLabel particlePage;
		particlePage.num = 5;

		particlePage.label.push_back("Particle System");
		particlePage.ico.push_back("dyverso/icon-emi-fill.svg");

		particlePage.label.push_back("ParticleEmitterRound");
		particlePage.ico.push_back("dyverso/icon-emi-n.svg");

		particlePage.label.push_back("ParticleFluid");
		particlePage.ico.push_back("dyverso/icon-emi-bitmap.svg");

		particlePage.label.push_back("ParticleSystem");
		particlePage.ico.push_back("dyverso/icon-emi-circle.svg");

		particlePage.label.push_back("ParticleEmitterSquare");
		particlePage.ico.push_back("dyverso/icon-emi-object.svg");

		//ToolBar height Field
		ToolBarIcoAndLabel heightPage;
		heightPage.num = 4;
		heightPage.label.push_back("Height Field");
		heightPage.ico.push_back("icon-realwave.svg");

		heightPage.label.push_back("OceanPach");
		heightPage.ico.push_back("icon-realwave.svg");

		heightPage.label.push_back("Wave 2");
		heightPage.ico.push_back("icon-realwave-cresplash.svg");

		heightPage.label.push_back("Wave 3");
		heightPage.ico.push_back("icon-realwave-objspash.svg");

		//ToolBar Finite Element
		ToolBarIcoAndLabel FinitePage;
		FinitePage.num = 4;
		FinitePage.label.push_back("Finite Element");
		FinitePage.ico.push_back("daemon/icon-demon-vortex.svg");

		FinitePage.label.push_back("Soft Body 1");
		FinitePage.ico.push_back("daemon/icon-demon-vortex.svg");

		FinitePage.label.push_back("Soft Body 2");
		FinitePage.ico.push_back("daemon/icon-demon-heater.svg");

		FinitePage.label.push_back("Soft Body 3");
		FinitePage.ico.push_back("daemon/icon-demon-ellipsoid.svg");

		//ToolBar Rigid Body
		ToolBarIcoAndLabel RigidPage;
		RigidPage.num = 5;
		RigidPage.label.push_back("Rigid Body");
		RigidPage.ico.push_back("geometry/icon-geometry-cube.svg");

		RigidPage.label.push_back("GLPointVisualNode");
		RigidPage.ico.push_back("geometry/icon-geometry-cube.svg");

		RigidPage.label.push_back("GhostFluid");
		RigidPage.ico.push_back("geometry/icon-geometry-cylinder.svg");

		RigidPage.label.push_back("GhostParticles");
		RigidPage.ico.push_back("geometry/icon-geometry-rocket.svg");

		RigidPage.label.push_back("StaticBoundary");
		RigidPage.ico.push_back("geometry/icon-geometry-multibody.svg");

		//ToolBar help
		ToolBarIcoAndLabel helpPage;
		helpPage.num = 2;

		helpPage.label.push_back("Help");
		helpPage.ico.push_back("Help-browser.png");

		helpPage.label.push_back("Help");
		helpPage.ico.push_back("Help-browser.png");


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