#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WPushButton.h>
#include <Wt/WBorderLayout.h>
#include <Wt/WAnchor.h>
#include <Wt/WResource.h>
#include <Wt/WPanel.h>
#include <Wt/WTable.h>
#include <Wt/WLineEdit.h>
#include <Wt/WFileUpload.h>
#include <Wt/WProgressBar.h>
#include <Wt/WFileResource.h>
#include <Wt/Http/Request.h>
#include <Wt/Http/Response.h>
#include <Wt/WMessageBox.h>

#include <filesystem>

#include "WMainWindow.h"
#include "SceneGraph.h"

#include <SceneGraphFactory.h>
#include <SceneLoaderFactory.h>


namespace dyno
{
	class SceneLoaderFactory;
	class SceneGraphFactory;

}

class WMainWindow;

class WSaveWidget : public Wt::WContainerWidget
{
public:
	WSaveWidget(WMainWindow* parent);
	~WSaveWidget();

private:

	void createSavePanel();

	void createUploadPanel();

	void save(std::string fileName);

	void recreate();

	bool isValidFileName(const std::string& filename);

	std::string uploadFile(Wt::WFileUpload* upload);

	std::string removeXmlExtension(const std::string& filename);

private:

	Wt::WText* mSaveOut;
	Wt::WText* mUploadOut;

	Wt::WVBoxLayout* mSaveLayout;
	WMainWindow* mParent;

	std::shared_ptr<dyno::SceneGraph> mScene;
};

class downloadResource : public Wt::WFileResource
{
public:
	downloadResource(std::string fileName) : Wt::WFileResource()
	{
		suggestFileName(fileName);
		filePath = fileName;
	}

	~downloadResource()
	{
		beingDeleted();
	}

	void handleRequest(const Wt::Http::Request& request,
		Wt::Http::Response& response) {
		response.setMimeType("text/xml");


		std::ifstream inputFile(filePath);
		if (!inputFile) {
			std::cout << "无法打开输入文件" << std::endl;
			return;
		}

		// 读取输入文件并写入输出文件
		std::string line, lines;
		while (std::getline(inputFile, line)) {
			lines = lines + line + "\n";
		}
		inputFile.close();

		response.out() << lines;
	}


private:
	std::string filePath;
};