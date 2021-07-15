#pragma once
#include "Node.h"
#include "SceneLoaderFactory.h"
#include "tinyxml/tinyxml2.h"

namespace dyno {

	class SceneLoaderXML : public SceneLoader
	{
	public:
		std::shared_ptr<Node> load(const std::string filename) override;

	private:
		std::shared_ptr<Node> processNode(tinyxml2::XMLElement* nodeXML);
		std::shared_ptr<Module> processModule(tinyxml2::XMLElement* moduleXML);
		bool addModule(std::shared_ptr<Node> node, std::shared_ptr<Module> module);

		std::vector<std::string> split(std::string str, std::string pattern);

		virtual bool canLoadFileByExtension(const std::string extension);
	};
}
