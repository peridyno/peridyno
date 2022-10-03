#include "ToolBarPage.h"

#include "QTextCodec"

namespace dyno {
	ToolBarPage::ToolBarPage() {

		QTextCodec* codec = QTextCodec::codecForName("GB2312");

// 		//ToolBar file----------------------------------
// 		ToolBarIcoAndLabel filePage;
// 		// Main Page name and ico
// 		filePage.tabPageName = "File";
// 		filePage.tabPageIco = "ToolBarIco/File/Open.png";
// 		//number of subTabs
// 		filePage.subtabNum = 4;
// 		//subTabs
// 		filePage.label.push_back("New");
// 		filePage.ico.push_back("ToolBarIco/File/New_v2.png");
// 
// 		filePage.label.push_back("Open");
// 		filePage.ico.push_back("ToolBarIco/File/Open.png");
// 
// 		filePage.label.push_back("Save");
// 		filePage.ico.push_back("ToolBarIco/File/Save.png");
// 
// 		filePage.label.push_back("Save As");
// 		filePage.ico.push_back("ToolBarIco/File/SaveAs.png");
		//--------------------------------------------------

// 		//ToolBar edit--------------------------------------
// 		ToolBarIcoAndLabel editPage;
// 		// Main Page name and ico
// 		editPage.tabPageName = "Edit";
// 		editPage.tabPageIco = "ToolBarIco/Edit/Settings_v2.png";
// 		//number of subTabs
// 		editPage.subtabNum = 1;
// 		//subTabs
// 		editPage.label.push_back("Settings");
// 		editPage.ico.push_back("ToolBarIco/Edit/Settings_v2.png");
// 		//--------------------------------------------------

		
// 		//ToolBar particle----------------------------------
// 		ToolBarIcoAndLabel particlePage;
// 		// Main Page name and ico
// 		particlePage.tabPageName = "Particle System";
// 		particlePage.tabPageIco = "ToolBarIco/ParticleSystem/ParticleSystem.png";
// 		//number of subTabs
// 		particlePage.subtabNum = 4;
// 		//subTabs
// 		particlePage.label.push_back(codec->toUnicode("Particle Emitter Round"));
// 		particlePage.ico.push_back("ToolBarIco/ParticleSystem/ParticleEmitterRound.png");
// 
// 		particlePage.label.push_back("ParticleFluid");
// 		particlePage.ico.push_back("ToolBarIco/ParticleSystem/ParticleFluid.png");
// 
// 		particlePage.label.push_back("ParticleSystem");
// 		particlePage.ico.push_back("ToolBarIco/ParticleSystem/ParticleSystem.png");
// 
// 		particlePage.label.push_back("ParticleEmitterSquare");
// 		particlePage.ico.push_back("ToolBarIco/ParticleSystem/ParticleEmitterSquare.png");
// 		//----------------------------------------------------
// 		
// 		//ToolBar height Field--------------------------------
// 		ToolBarIcoAndLabel heightPage;
// 		// Main Page name and ico
// 		heightPage.tabPageName = "Height Field";
// 		heightPage.tabPageIco = "ToolBarIco/HeightField/HeightField.png";
// 		//number of subTabs
// 		heightPage.subtabNum = 3;
// 		//subTabs
// 		heightPage.label.push_back("OceanPatch");
// 		heightPage.ico.push_back("ToolBarIco/HeightField/OceanPatch.png");
// 
// 		heightPage.label.push_back("Ocean");
// 		heightPage.ico.push_back("ToolBarIco/HeightField/Ocean.png");
// 
// 		heightPage.label.push_back("CapillaryWave");
// 		heightPage.ico.push_back("ToolBarIco/HeightField/CapillaryWave.png");
// 		//--------------------------------------------------
// 		
// 		//ToolBar Finite Element----------------------------
// 		ToolBarIcoAndLabel FinitePage;
// 		// Main Page name and ico
// 		FinitePage.tabPageName = "Finite Element";
// 		FinitePage.tabPageIco = "ToolBarIco/FiniteElement/FiniteElement.png";
// 		//number of subTabs
// 		FinitePage.subtabNum = 3;
// 		//subTabs
// 		FinitePage.label.push_back("Soft Body 1");
// 		FinitePage.ico.push_back("ToolBarIco/FiniteElement/SoftBody1.png");
// 
// 		FinitePage.label.push_back("Soft Body 2");
// 		FinitePage.ico.push_back("ToolBarIco/FiniteElement/SoftBody2.png");
// 
// 		FinitePage.label.push_back("Soft Body 3");
// 		FinitePage.ico.push_back("ToolBarIco/FiniteElement/SoftBody3.png");
// 		//--------------------------------------------------
// 
// 		//ToolBar Rigid Body--------------------------------
// 		ToolBarIcoAndLabel RigidPage;
// 		// Main Page name and ico
// 		RigidPage.tabPageName = "Rigid Body";
// 		RigidPage.tabPageIco = "ToolBarIco/RigidBody/RigidBody.png";
// 		//number of subTabs
// 		RigidPage.subtabNum = 4;
// 		//subTabs
// 		RigidPage.label.push_back("GLPointVisualNode");
// 		RigidPage.ico.push_back("ToolBarIco/RigidBody/GLPointVisualNode.png");
// 
// 		RigidPage.label.push_back("GhostFluid");
// 		RigidPage.ico.push_back("ToolBarIco/RigidBody/GhostFluid.png");
// 
// 		RigidPage.label.push_back("GhostParticles");
// 		RigidPage.ico.push_back("ToolBarIco/RigidBody/GhostParticles.png");
// 
// 		RigidPage.label.push_back("StaticBoundary");
// 		RigidPage.ico.push_back("ToolBarIco/RigidBody/StaticBoundary.png");
// 		//--------------------------------------------------

		//ToolBar help--------------------------------------
		ToolBarIcoAndLabel helpPage;
		// Main Page name and ico
		helpPage.tabPageName = "Help";
		helpPage.tabPageIco = "ToolBarIco/Help/Help_v2.png";
		//number of subTabs
		helpPage.subtabNum = 2;
		//subTabs
		helpPage.label.push_back("ReOrder");
		helpPage.ico.push_back("ToolBarIco/Help/realign_v2.png");

		helpPage.label.push_back("Help");
		helpPage.ico.push_back("ToolBarIco/Help/Help_v2.png");
		//--------------------------------------------------

		//Add all the page
//		tbl.push_back(filePage);// 0
//		tbl.push_back(editPage);// 1
		tbl.push_back(helpPage);//6

		//dynamic toolbar
// 		tbl.push_back(particlePage);//2
// 		tbl.push_back(heightPage);//3
// 		tbl.push_back(FinitePage);//4
// 		tbl.push_back(RigidPage);//5
	}


	ToolBarPage::~ToolBarPage() {

	}


}