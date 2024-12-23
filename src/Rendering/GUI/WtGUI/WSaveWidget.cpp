#include "WSaveWidget.h"

WSaveWidget::WSaveWidget(WMainWindow* parent)
	: mParent(parent)
{
	this->setLayoutSizeAware(true);
	this->setOverflow(Wt::Overflow::Auto);
	this->setHeight(Wt::WLength("100%"));
	this->setMargin(10);

	//auto layout = this->setLayout(std::make_unique<Wt::WBorderLayout>());

	createSavePanel();

	createUploadPanel();
}

WSaveWidget::~WSaveWidget() {}


void WSaveWidget::createSavePanel()
{
	auto panel = this->addNew<Wt::WPanel>();
	panel->setTitle("Save File");
	panel->setCollapsible(false);
	panel->setWidth(250);

	auto container = panel->setCentralWidget(std::make_unique<Wt::WContainerWidget>());
	mSaveLayout = container->setLayout(std::make_unique<Wt::WVBoxLayout>());

	auto saveFileText = mSaveLayout->addWidget(std::make_unique<Wt::WText>("Please Input File Name:"));

	auto saveFileNameEdit = mSaveLayout->addWidget(std::make_unique<Wt::WLineEdit>());
	saveFileNameEdit->setMaxLength(256);
	saveFileNameEdit->setMargin(10, Wt::Side::Right);


	auto downloadButton = mSaveLayout->addWidget(std::make_unique< Wt::WPushButton>("Generated"));
	downloadButton->setMargin(10, Wt::Side::Top);
	downloadButton->setStyleClass("btn-primary");

	mSaveOut = mSaveLayout->addWidget(std::make_unique<Wt::WText>());

	downloadButton->clicked().connect([=]
		{

			std::string fileName = saveFileNameEdit->text().toUTF8();
			std::cout << fileName << std::endl;
			if (!fileName.empty())
			{
				if (isValidFileName(fileName))
				{
					save(fileName);
					mSaveOut->setText("Generating Successfully, Click Download!");
					saveFileNameEdit->setText("");
				}
				else
				{
					mSaveOut->setText("File Name is InValid!");
					saveFileNameEdit->setText("");
				}

			}
			else
			{

				std::cout << fileName << std::endl;
				mSaveOut->setText("File Name Cannot Be Empty!");
				saveFileNameEdit->setText("");
			}
		});
}

void WSaveWidget::createUploadPanel()
{
	auto panel = this->addNew<Wt::WPanel>();
	panel->setTitle("Upload File");
	panel->setCollapsible(false);
	panel->setWidth(250);

	auto container = panel->setCentralWidget(std::make_unique<Wt::WContainerWidget>());
	auto layout = container->setLayout(std::make_unique<Wt::WVBoxLayout>());

	Wt::WFileUpload* fu = layout->addWidget(std::make_unique<Wt::WFileUpload>());
	fu->setProgressBar(std::make_unique<Wt::WProgressBar>());
	fu->setMargin(10, Wt::Side::Right);
	fu->setMultiple(false);

	// Provide a button to start uploading.
	Wt::WPushButton* uploadButton = layout->addWidget(std::make_unique<Wt::WPushButton>("Send"));
	uploadButton->setMargin(10, Wt::Side::Top);
	uploadButton->setStyleClass("btn-primary");

	mUploadOut = layout->addWidget(std::make_unique<Wt::WText>());

	// Upload when the button is clicked.
	uploadButton->clicked().connect([=] {
		if (uploadButton->text() == "Send")
		{
			if (fu->canUpload())
			{
				fu->upload();
				uploadButton->setText("ReUpload");
			}
			else
			{
				mUploadOut->setText("Cannot Upload File");
			}

		}
		else if (uploadButton->text() == "ReUpload")
		{
			recreate();
		}
		});

	// React to a succesfull upload.
	fu->uploaded().connect([=] {

		std::string filePath = uploadFile(fu);
		if (!filePath.empty())
		{
			auto scnLoader = dyno::SceneLoaderFactory::getInstance().getEntryByFileExtension("xml");

			auto scn = scnLoader->load(filePath);

			if (scn)
			{
				mParent->setScene(scn);
				mParent->createLeftPanel();
				mUploadOut->setText("File upload is finished.");
			}

		}
		else
		{
			mUploadOut->setText("Upload File IS Empty");
		}
		});

	// React to a file upload problem.
	fu->fileTooLarge().connect([=] {
		mUploadOut->setText("File is too large.");
		});
}

void WSaveWidget::save(std::string fileName)
{
	std::string filePath = removeXmlExtension(fileName) + ".xml";

	auto scnLoader = dyno::SceneLoaderFactory::getInstance().getEntryByFileExtension("xml");
	//scnLoader->save(dyno::SceneGraphFactory::instance()->active(), filePath);
	scnLoader->save(mParent->getScene(), filePath);

	if (std::filesystem::exists(filePath))
	{
		auto xmlResource = std::make_shared<downloadResource>(filePath);

		Wt::WLink link = Wt::WLink(xmlResource);
		link.setTarget(Wt::LinkTarget::NewWindow);
		auto anchor = mSaveLayout->addWidget(std::make_unique<Wt::WAnchor>(link, "Download File"));

		anchor->clicked().connect([=] {
			mSaveLayout->removeWidget(anchor);
			std::filesystem::remove(filePath);
			mSaveOut->setText("");
			});
	}
	else
	{
		mSaveOut->setText("Failed To Generate File!");
	}
}

std::string WSaveWidget::removeXmlExtension(const std::string& filename) {
	// ≤È’“ ".xml" µƒŒª÷√
	size_t pos = filename.rfind(".xml");
	if (pos != std::string::npos) {
		return filename.substr(0, pos);
	}
	return filename;
}

bool WSaveWidget::isValidFileName(const std::string& filename) {
	// Windows
	const std::string invalidChars = "<>:\"/\\|?*";

	return std::none_of(invalidChars.begin(), invalidChars.end(),
		[&filename](char c) { return filename.find(c) != std::string::npos; });
}

void WSaveWidget::recreate()
{
	this->clear();
	this->setLayoutSizeAware(true);
	this->setOverflow(Wt::Overflow::Auto);
	this->setHeight(Wt::WLength("100%"));
	this->setMargin(10);

	this->createSavePanel();
	this->createUploadPanel();
}

std::string WSaveWidget::uploadFile(Wt::WFileUpload* upload)
{
	if (upload->uploadedFiles().size() > 0)
	{
		for (const auto& file : upload->uploadedFiles())
		{
			std::string savePath = file.clientFileName();
			std::string tempFilePath = file.spoolFileName();
			std::ifstream src(tempFilePath, std::ios::binary);
			std::ofstream dst(savePath, std::ios::binary);

			dst << src.rdbuf();

			if (dst)
			{
				Wt::WMessageBox::show("Success", "File save path." + savePath, Wt::StandardButton::Ok);
				src.close();
				dst.close();
				return savePath;
			}
			else
			{
				Wt::WMessageBox::show("Error", "File save failure.", Wt::StandardButton::Ok);
			}
			src.close();
			dst.close();
		}
	}
}
