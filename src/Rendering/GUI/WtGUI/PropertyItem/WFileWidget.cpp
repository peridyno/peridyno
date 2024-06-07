#include "WFileWidget.h"

WFileWidget::WFileWidget(dyno::FBase* field)
{
	layout = this->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	//layout->setSpacing(0);

	setValue(field);
	mfield = field;

	upload = layout->addWidget(std::make_unique<Wt::WFileUpload>());
	//uploadButton = layout->addWidget(std::make_unique<Wt::WPushButton>("Upload"));

	upload->setMultiple(true);

	mfilename->changed().connect(this, &WFileWidget::updateField);
	//uploadButton->checked().connect(upload, &Wt::WFileUpload::upload);
	//uploadButton->checked().connect(uploadButton, &Wt::WPushButton::disable);
	upload->changed().connect(upload, &Wt::WFileUpload::upload);
	upload->uploaded().connect(this, &WFileWidget::uploadFile);
	upload->fileTooLarge().connect(this, &WFileWidget::fileTooLarge);
}

WFileWidget::~WFileWidget()
{
}

void WFileWidget::uploadFile()
{
	if (upload->uploadedFiles().size() > 0)
	{
		for (const auto& file : upload->uploadedFiles())
		{
			Wt::log("info") << file.spoolFileName();
			std::string savePath = getAssetPath() + "WebUploadFiles/" + file.clientFileName();
			std::string tempFilePath = file.spoolFileName();
			std::ifstream src(tempFilePath, std::ios::binary);
			std::ofstream dst(savePath, std::ios::binary);

			dst << src.rdbuf();

			if (dst)
			{
				Wt::WMessageBox::show("Success", "File save path WebUploadFiles/.", Wt::StandardButton::Ok);
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

void WFileWidget::fileTooLarge()
{
	Wt::WMessageBox::show("Error", "File Too Large!", Wt::StandardButton::Ok);
}

std::string WFileWidget::shortFilePath(std::string str)
{
	std::string removeStr = getAssetPath();
	size_t pos = str.find(removeStr);
	while (pos != std::string::npos)
	{
		str.erase(pos, removeStr.length());
		pos = str.find(removeStr);
	}
	return str;
}

bool WFileWidget::hasFile(std::string path)
{
	if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path))
	{
		return true;
	}
	else
	{
		return false;
	}
	return false;
}

void WFileWidget::setValue(dyno::FBase* field)
{
	dyno::FVar<dyno::FilePath>* f = TypeInfo::cast<dyno::FVar<dyno::FilePath>>(field);
	if (f == nullptr)
		return;

	mfilename = layout->addWidget(std::make_unique<Wt::WLineEdit>());
	std::string filepath = shortFilePath(f->getValue().string());
	mfilename->setText(filepath);
}

void WFileWidget::updateField()
{
	auto f = TypeInfo::cast<dyno::FVar<dyno::FilePath>>(mfield);
	if (f == nullptr)
	{
		return;
	}
	auto path = f->getValue();
	std::string filePath = getAssetPath() + mfilename->text().toUTF8();
	if (hasFile(filePath))
	{
		path.set_path(filePath);
		f->setValue(path);
		f->update();
	}
	else
	{
		Wt::WMessageBox::show("Error", "file does not exist!", Wt::StandardButton::Ok);
		return;
	}

}