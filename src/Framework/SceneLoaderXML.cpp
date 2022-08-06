#include "SceneLoaderXML.h"
#include <algorithm>

#include <stdlib.h>
#include <sstream>

namespace dyno
{
	std::shared_ptr<SceneGraph> SceneLoaderXML::load(const std::string filename)
	{
		tinyxml2::XMLDocument doc;
		if (doc.LoadFile(filename.c_str()))
		{
			doc.PrintError();
			return nullptr;
		}

		for (size_t i = 0; i < mConnectionInfo.size(); i++){
			mConnectionInfo[i].clear();
		}
		mConnectionInfo.clear();

		std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

		tinyxml2::XMLElement* root = doc.RootElement();

		std::vector<std::shared_ptr<Node>> nodes;
		tinyxml2::XMLElement* nodeXML = root->FirstChildElement("Node");
		while (nodeXML)
		{
			auto node = processNode(nodeXML);
			if (node) {
				scn->addNode(node);
				nodes.push_back(node);
			}

			nodeXML = nodeXML->NextSiblingElement("Node");
		}

		for (size_t i = 0; i < mConnectionInfo.size(); i++)
		{
			auto& connections = mConnectionInfo[i];
			for (size_t j = 0; j < connections.size(); j++)
			{
				auto info = connections[j];

				auto expNode = nodes[info.src];
				auto inpNode = nodes[info.dst];

				auto& outFields = expNode->getOutputFields();

				auto& inPorts = inpNode->getImportNodes();
				auto& inFields = inpNode->getInputFields();

				if (info.id0 <= 0)
				{
					expNode->connect(inPorts[info.id1]);
				}
				else {
					outFields[info.id0 - 1]->connect(inFields[info.id1 - inPorts.size()]);
				}
			}
		}

		return scn;
	}

	bool SceneLoaderXML::save(std::shared_ptr<SceneGraph> scn, const std::string filename)
	{
		tinyxml2::XMLDocument doc;

// 		const char* declaration = "<?xml version=\"0.6.0\" encoding=\"UTF-8\">";
// 		doc.Parse(declaration);

		tinyxml2::XMLElement* root = doc.NewElement("SceneGraph");

		root->SetAttribute("LowerBound", encodeVec3f(scn->getLowerBound()).c_str());
		root->SetAttribute("UpperBound", encodeVec3f(scn->getUpperBound()).c_str());

		doc.InsertEndChild(root);

		SceneGraph::Iterator itor = scn->begin();

		std::map<ObjectId, uint> indices;
		int radix = 0;
		for (; itor != scn->end(); itor++) {
			auto node = itor.get();
			indices[node->objectId()] = radix;

			radix++;
		}

		for (itor = scn->begin(); itor != scn->end(); itor++)
		{
			auto node = itor.get();

			tinyxml2::XMLElement* nodeXml = doc.NewElement("Node");
			nodeXml->SetAttribute("Class", node->getClassInfo()->getClassName().c_str());
			nodeXml->SetAttribute("Coordinate", encodeVec2f(Vec2f(node->bx(), node->by())).c_str());
			root->InsertEndChild(nodeXml);

			tinyxml2::XMLElement* varsXml = doc.NewElement("Variables");
			nodeXml->InsertEndChild(varsXml);

			//write control variables
			auto& params = node->getParameters();
			for each (auto var in params)
			{
				tinyxml2::XMLElement* field = doc.NewElement("Field");
				field->SetAttribute("Name", var->getObjectName().c_str());

				tinyxml2::XMLText* val = doc.NewText(var->serialize().c_str());
				field->InsertEndChild(val);

				varsXml->InsertEndChild(field);
			}

			//write connections
			tinyxml2::XMLElement* connections = doc.NewElement("Connections");
			nodeXml->InsertEndChild(connections);

			auto ports = node->getImportNodes();

			auto fieldInp = node->getInputFields();
			for (int i = 0; i < fieldInp.size(); i++)
			{
				auto fieldSrc = fieldInp[i]->getSource();
				if (fieldSrc != nullptr) {
					auto parSrc = fieldSrc->parent();
					if (parSrc != nullptr)
					{
						Node* nodeSrc = dynamic_cast<Node*>(parSrc);

						auto outId = nodeSrc->objectId();
						auto fieldsOut = nodeSrc->getOutputFields();

						uint outFieldIndex = 0;
						bool fieldFound = false;
						for (auto f : fieldsOut)
						{
							if (f == fieldSrc)
							{
								fieldFound = true;
								break;
							}
							outFieldIndex++;
						}

						if (fieldFound) {
							tinyxml2::XMLElement* connection = doc.NewElement("Connection");
							connection->SetAttribute("SourceId", indices[parSrc->objectId()]);
							connection->SetAttribute("From", 1 + outFieldIndex);
							connection->SetAttribute("TargetId", indices[node->objectId()]);
							connection->SetAttribute("To", uint(i + ports.size()));
							connections->InsertEndChild(connection);
						}
					}
				}
			}


			//Insert animation pipeline
			tinyxml2::XMLElement* simulationPipelineXml = doc.NewElement("Simulation");
			nodeXml->InsertEndChild(simulationPipelineXml);

			auto simulationPipeline = node->animationPipeline();
			for each (auto m in simulationPipeline->activeModules())
			{
				tinyxml2::XMLElement* moduleXml = doc.NewElement("Module");
				moduleXml->SetAttribute("Class", m->getClassInfo()->getClassName().c_str());
				moduleXml->SetAttribute("Coordinate", encodeVec2f(Vec2f(m->bx(), m->by())).c_str());
				simulationPipelineXml->InsertEndChild(moduleXml);
			}

			//Insert graphics pipeline
			tinyxml2::XMLElement* graphicsPipelineXml = doc.NewElement("Rendering");
			nodeXml->InsertEndChild(graphicsPipelineXml);

			auto graphicsPipeline = node->graphicsPipeline();
			for each (auto m in graphicsPipeline->activeModules())
			{
				tinyxml2::XMLElement* moduleXml = doc.NewElement("Module");
				moduleXml->SetAttribute("Class", m->getClassInfo()->getClassName().c_str());
				moduleXml->SetAttribute("Coordinate", encodeVec2f(Vec2f(m->bx(), m->by())).c_str());
				graphicsPipelineXml->InsertEndChild(moduleXml);
			}
		}

		doc.SaveFile(filename.c_str());

		return true;
	}

	std::shared_ptr<Node> SceneLoaderXML::processNode(tinyxml2::XMLElement* nodeXML)
	{
		const char* name = nodeXML->Attribute("Class");
		if (!name)
			return nullptr;

		std::shared_ptr<Node> node(dynamic_cast<Node*>(Object::createObject(name)));
		if (node == nullptr)
		{
			std::cout << name << " does not exist! " << std::endl;
			return nullptr;
		}

		const char* coordStr = nodeXML->Attribute("Coordinate");

		Vec2f coord = decodeVec2f(std::string(coordStr));

		node->setBlockCoord(coord.x, coord.y);

		const char* nodeName = nodeXML->Attribute("Name");
		if (nodeName)
			node->setName(nodeName);

		std::map<std::string, FBase*> str2Field;
		auto& params = node->getParameters();
		for each (auto var in params) {
			str2Field[var->getObjectName()] = var;
		}
		tinyxml2::XMLElement* varsXmls = nodeXML->FirstChildElement("Variables");
		tinyxml2::XMLElement* fieldXml = varsXmls->FirstChildElement("Field");
		while (fieldXml)
		{
			std::string name = fieldXml->Attribute("Name");
			std::string str = fieldXml->GetText();
			str2Field[name]->deserialize(str);

			fieldXml = fieldXml->NextSiblingElement("Field");
		}
		str2Field.clear();

		std::vector<ConnectionInfo> infoVec;

		tinyxml2::XMLElement* cnnXmls = nodeXML->FirstChildElement("Connections");
		tinyxml2::XMLElement* connectionXml = cnnXmls->FirstChildElement("Connection");
		while (connectionXml)
		{
			ConnectionInfo info;

			info.src = atoi(connectionXml->Attribute("SourceId"));
			info.dst = atoi(connectionXml->Attribute("TargetId"));

			info.id0 = atoi(connectionXml->Attribute("From"));
			info.id1 = atoi(connectionXml->Attribute("To"));

			infoVec.push_back(info);

			connectionXml = connectionXml->NextSiblingElement("Node");
		}

		mConnectionInfo.push_back(infoVec);

		infoVec.clear();

// 		tinyxml2::XMLElement* childNodeXML = nodeXML->FirstChildElement("Node");
// 		while (childNodeXML)
// 		{
// 			auto cNode = processNode(childNodeXML);
// 			if (cNode)
// 			{
// 				//Xiowei He
// 				//node->addAncestor(cNode.get());
// 			}
// 				
// 
// 			std::cout << childNodeXML->Name() << std::endl;
// 			childNodeXML = childNodeXML->NextSiblingElement("Node");
// 		}

// 		tinyxml2::XMLElement* moduleXML = nodeXML->FirstChildElement("Module");
// 		while (moduleXML)
// 		{
// 			auto module = processModule(moduleXML);
// 			if (module == nullptr)
// 			{
// 				std::cout << "Create Module " << moduleXML->Attribute("Class") << " failed!" << std::endl;
// 			}
// 			else
// 			{
// 				bool re = addModule(node, module);
// 				if (!re)
// 				{
// 					std::cout << "Cannot add " << moduleXML->Name() << " to the current node!" << std::endl;
// 				}
// 
// 				const char* dependence = moduleXML->Attribute("Dependence");
// 
// 				if (dependence)
// 				{
// 				}
// 				
// 			}
// 			
// 			moduleXML = moduleXML->NextSiblingElement("Module");
// 		}

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

	std::string SceneLoaderXML::encodeVec3f(const Vec3f v)
	{
		std::stringstream ss;

		ss << v.x << " " << v.y << " " << v.z;

		return ss.str();
	}

	std::string SceneLoaderXML::encodeVec2f(const Vec2f v)
	{
		std::stringstream ss;

		ss << v.x << " " << v.y;

		return ss.str();
	}

	Vec3f SceneLoaderXML::decodeVec3f(const std::string str)
	{
		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		float x = std::stof(substr.c_str());

		ss >> substr;
		float y = std::stof(substr.c_str());

		ss >> substr;
		float z = std::stof(substr.c_str());

		return Vec3f(x, y, z);
	}

	Vec2f SceneLoaderXML::decodeVec2f(const std::string str)
	{
		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		float x = std::stof(substr.c_str());

		ss >> substr;
		float y = std::stof(substr.c_str());

		return Vec2f(x, y);
	}

	bool SceneLoaderXML::canLoadFileByExtension(const std::string extension)
	{
		std::string str = extension;
		std::transform(str.begin(), str.end(), str.begin(), ::tolower);
		return (extension == "xml");
	}

}