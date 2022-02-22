#include "SceneLoaderXML.h"
#include <algorithm>

namespace dyno
{
	std::shared_ptr<Node> SceneLoaderXML::load(const std::string filename)
	{
		tinyxml2::XMLDocument doc;
		if (doc.LoadFile(filename.c_str()))
		{
			doc.PrintError();
			return nullptr;
		}

		tinyxml2::XMLElement* scenegraph = doc.RootElement();
		tinyxml2::XMLElement* rootXML  = scenegraph->FirstChildElement("Node");

		return processNode(rootXML);
	}

	std::shared_ptr<Node> SceneLoaderXML::processNode(tinyxml2::XMLElement* nodeXML)
	{
		const char* name = nodeXML->Attribute("class");
		if (!name)
			return nullptr;

		std::shared_ptr<Node> node(dynamic_cast<Node*>(Object::createObject(name)));
		if (node == nullptr)
		{
			std::cout << name << " does not exist! " << std::endl;
			return nullptr;
		}

		const char* nodeName = nodeXML->Attribute("name");
		if (nodeName)
			node->setName(nodeName);

		tinyxml2::XMLElement* childNodeXML = nodeXML->FirstChildElement("Node");
		while (childNodeXML)
		{
			auto cNode = processNode(childNodeXML);
			if (cNode)
				node->addAncestor(cNode.get());

			std::cout << childNodeXML->Name() << std::endl;
			childNodeXML = childNodeXML->NextSiblingElement("Node");
		}

		tinyxml2::XMLElement* moduleXML = nodeXML->FirstChildElement("Module");
		while (moduleXML)
		{
			auto module = processModule(moduleXML);
			if (module == nullptr)
			{
				std::cout << "Create Module " << moduleXML->Attribute("class") << " failed!" << std::endl;
			}
			else
			{
				bool re = addModule(node, module);
				if (!re)
				{
					std::cout << "Cannot add " << moduleXML->Name() << " to the current node!" << std::endl;
				}

				const char* dependence = moduleXML->Attribute("dependence");

				if (dependence)
				{
				}
				
			}
			
			moduleXML = moduleXML->NextSiblingElement("Module");
		}

		return node;
	}

	std::shared_ptr<Module> SceneLoaderXML::processModule(tinyxml2::XMLElement* moduleXML)
	{
		const char* className = moduleXML->Attribute("class");
		if (!className)
			return nullptr;

		const char* dataType = moduleXML->Attribute("datatype");
		std::string templateClass = std::string(className);
		if (dataType)
		{
			templateClass += std::string("<")+std::string(dataType)+ std::string(">");
		}
		std::shared_ptr<Module> module(dynamic_cast<Module*>(Object::createObject(templateClass)));

		return module;
	}

	bool SceneLoaderXML::addModule(std::shared_ptr<Node> node, std::shared_ptr<Module> module)
	{
		if (module->getModuleType() == "MechanicalState")
		{
			return true;
		}

		return false;
	}

	std::vector<std::string> SceneLoaderXML::split(std::string str, std::string pattern)
	{
		std::string::size_type pos;
		std::vector<std::string> result;
		str += pattern;
		size_t size = (uint)str.size();

		for (size_t i = 0; i < size; i++)
		{
			pos = str.find(pattern, i);
			if (pos < size)
			{
				std::string s = str.substr(i, pos - i);
				result.push_back(s);
				i = pos + pattern.size() - 1;
			}
		}
		return result;
	}

	bool SceneLoaderXML::canLoadFileByExtension(const std::string extension)
	{
		std::string str = extension;
		std::transform(str.begin(), str.end(), str.begin(), ::tolower);
		return (extension == "xml");
	}

}