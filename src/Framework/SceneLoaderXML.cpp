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
			tinyxml2::XMLElement* nodeConnectionsXml = doc.NewElement("Connections");
			nodeXml->InsertEndChild(nodeConnectionsXml);

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
							nodeConnectionsXml->InsertEndChild(connection);
						}
					}
				}
			}

			auto& fields = node->getAllFields();
			std::vector<FBase*> fieldsOut;
			for (auto field : fields)
			{
				if (field->getFieldType() == dyno::FieldTypeEnum::State)
				{
					fieldsOut.push_back(field);
				}
			}

			//Insert animation pipeline
			auto savePipeline = [&](std::shared_ptr<Pipeline> pipeline, const char* tag) {
				tinyxml2::XMLElement* pipelineXml = doc.NewElement(tag);
				nodeXml->InsertEndChild(pipelineXml);

				std::map<ObjectId, uint> indices;
				std::map<ObjectId, std::shared_ptr<Module>> moduleMap;
				uint radix = 0;

				auto& activeModules = pipeline->activeModules();
				for each (auto m in activeModules){
					tinyxml2::XMLElement* moduleXml = doc.NewElement("Module");
					moduleXml->SetAttribute("Class", m->getClassInfo()->getClassName().c_str());
					moduleXml->SetAttribute("Coordinate", encodeVec2f(Vec2f(m->bx(), m->by())).c_str());
					pipelineXml->InsertEndChild(moduleXml);

					tinyxml2::XMLElement* varsModuleXml = doc.NewElement("Variables");
					moduleXml->InsertEndChild(varsModuleXml);

					//write control variables
					auto& params = m->getParameters();
					for each (auto var in params)
					{
						tinyxml2::XMLElement* field = doc.NewElement("Field");
						field->SetAttribute("Name", var->getObjectName().c_str());

						tinyxml2::XMLText* val = doc.NewText(var->serialize().c_str());
						field->InsertEndChild(val);

						varsModuleXml->InsertEndChild(field);
					}

					indices[m->objectId()] = radix;
					moduleMap[m->objectId()] = m;
					radix++;
				}

				//write connections
				tinyxml2::XMLElement* moduleConnectionsXml = doc.NewElement("Connections");
				pipelineXml->InsertEndChild(moduleConnectionsXml);

				for each (auto m in activeModules)
				{
					auto& fieldIn = m->getInputFields();
					for (int i = 0; i < fieldIn.size(); i++)
					{
						auto fieldSrc = fieldIn[i]->getSource();
						if (fieldSrc != nullptr) {
							auto parSrc = fieldSrc->parent();
							if (parSrc != nullptr)
							{
								Module* src = dynamic_cast<Module*>(parSrc);
								if (src != nullptr)
								{
									auto outId = src->objectId();
									auto fieldsOut = src->getOutputFields();

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

									if (fieldFound && moduleMap.find(outId) != moduleMap.end())
									{
										auto outBlock = moduleMap[outId];
										
										tinyxml2::XMLElement* moduleConnectionXml = doc.NewElement("Connection");
										moduleConnectionXml->SetAttribute("SourceId", indices[outBlock->objectId()]);
										moduleConnectionXml->SetAttribute("From", outFieldIndex);
										moduleConnectionXml->SetAttribute("TargetId", indices[m->objectId()]);
										moduleConnectionXml->SetAttribute("To", uint(i));
										moduleConnectionsXml->InsertEndChild(moduleConnectionXml);
									}
								}
								else {
									Node* src = dynamic_cast<Node*>(parSrc);

									if (src != nullptr)
									{
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

										if (fieldFound)
										{
											tinyxml2::XMLElement* moduleConnectionXml = doc.NewElement("Connection");
											moduleConnectionXml->SetAttribute("SourceId", -1);
											moduleConnectionXml->SetAttribute("From", outFieldIndex);
											moduleConnectionXml->SetAttribute("TargetId", indices[m->objectId()]);
											moduleConnectionXml->SetAttribute("To", uint(i));
											moduleConnectionsXml->InsertEndChild(moduleConnectionXml);
										}
									}
								}
							}
						}
					}
				}

				indices.clear();
			};

			savePipeline(node->animationPipeline(), "Simulation");

// 			//Insert graphics pipeline
// 			tinyxml2::XMLElement* graphicsPipelineXml = doc.NewElement("Rendering");
// 			nodeXml->InsertEndChild(graphicsPipelineXml);
// 
// 			auto graphicsPipeline = node->graphicsPipeline();
// 			for each (auto m in graphicsPipeline->activeModules())
// 			{
// 				tinyxml2::XMLElement* moduleXml = doc.NewElement("Module");
// 				moduleXml->SetAttribute("Class", m->getClassInfo()->getClassName().c_str());
// 				moduleXml->SetAttribute("Coordinate", encodeVec2f(Vec2f(m->bx(), m->by())).c_str());
// 				graphicsPipelineXml->InsertEndChild(moduleXml);
// 			}

			savePipeline(node->graphicsPipeline(), "Rendering");

			fieldsOut.clear();
		}

		doc.SaveFile(filename.c_str());

		indices.clear();

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

		/**
		 * Construct the node connections
		 */
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

		auto& fields = node->getAllFields();
		std::vector<FBase*> states;
		for (auto field : fields)
		{
			if (field->getFieldType() == dyno::FieldTypeEnum::State)
			{
				states.push_back(field);
			}
		}

		/**
		 * Construct the animation pipeline
		 */
		std::vector<Module*> animationModules;
		node->animationPipeline()->clear();
		tinyxml2::XMLElement* animationPipelineXml = nodeXML->FirstChildElement("Simulation");
		tinyxml2::XMLElement* animationModuleXml = animationPipelineXml->FirstChildElement("Module");
		
		auto processModule = [&](tinyxml2::XMLElement* moduleXml, std::shared_ptr<Pipeline> pipeline, std::vector<Module*>& modules) {
			const char* name = moduleXml->Attribute("Class");
			if (name) {
				std::shared_ptr<Module> module(dynamic_cast<Module*>(Object::createObject(name)));
				if (module != nullptr) {
					pipeline->pushModule(module);
					modules.push_back(module.get());

					std::map<std::string, FBase*> str2Field;
					auto& params = module->getParameters();
					for each (auto var in params) {
						str2Field[var->getObjectName()] = var;
					}

					tinyxml2::XMLElement* varsXml = moduleXml->FirstChildElement("Variables");
					tinyxml2::XMLElement* varXml = varsXml->FirstChildElement("Field");

					while (varXml)
					{
						std::string name = varXml->Attribute("Name");
						std::string str = varXml->GetText();
						str2Field[name]->deserialize(str);

						varXml = varXml->NextSiblingElement("Field");
					}

					str2Field.clear();
				}
			}
		};

		while (animationModuleXml)
		{
			processModule(animationModuleXml, node->animationPipeline(), animationModules);
// 			const char* name = animationModuleXml->Attribute("Class");
// 			if (name) {
// 				std::shared_ptr<Module> module(dynamic_cast<Module*>(Object::createObject(name)));
// 				if (module != nullptr) {
// 					node->animationPipeline()->pushModule(module);
// 					animationModules.push_back(module.get());
// 				}
// 			}

			animationModuleXml = animationModuleXml->NextSiblingElement("Module");
		}
		tinyxml2::XMLElement* animationConnectionsXml = animationPipelineXml->FirstChildElement("Connections");
		tinyxml2::XMLElement* animationConnectionXml = animationConnectionsXml->FirstChildElement("Connection");
		while (animationConnectionXml)
		{
			int src = atoi(animationConnectionXml->Attribute("SourceId"));
			int dst = atoi(animationConnectionXml->Attribute("TargetId"));

			int id0 = atoi(animationConnectionXml->Attribute("From"));
			int id1 = atoi(animationConnectionXml->Attribute("To"));

			FBase* fout = src == -1 ? states[id0] : animationModules[src]->getOutputFields()[id0];
			FBase* fin = animationModules[dst]->getInputFields()[id1];

			fout->connect(fin);

			animationConnectionXml = animationConnectionXml->NextSiblingElement("Connection");
		}
		animationModules.clear();

		/**
		 * Construct the graphics pipeline
		 */
		std::vector<Module*> renderingModules;
		node->graphicsPipeline()->clear();
		tinyxml2::XMLElement* renderingPipelineXml = nodeXML->FirstChildElement("Rendering");
		tinyxml2::XMLElement* renderingModuleXml = renderingPipelineXml->FirstChildElement("Module");
		while (renderingModuleXml)
		{
			processModule(renderingModuleXml, node->graphicsPipeline(), renderingModules);
// 			const char* name = renderingModuleXml->Attribute("Class");
// 			if (name) {
// 				std::shared_ptr<Module> module(dynamic_cast<Module*>(Object::createObject(name)));
// 				if (module != nullptr) {
// 					node->graphicsPipeline()->pushModule(module);
// 					renderingModules.push_back(module.get());
// 				}
// 			}

			renderingModuleXml = renderingModuleXml->NextSiblingElement("Module");
		}

		tinyxml2::XMLElement* renderingConnectionsXml = renderingPipelineXml->FirstChildElement("Connections");
		tinyxml2::XMLElement* renderingConnectionXml = renderingConnectionsXml->FirstChildElement("Connection");
		while (renderingConnectionXml)
		{
			int src = atoi(renderingConnectionXml->Attribute("SourceId"));
			int dst = atoi(renderingConnectionXml->Attribute("TargetId"));

			int id0 = atoi(renderingConnectionXml->Attribute("From"));
			int id1 = atoi(renderingConnectionXml->Attribute("To"));

			FBase* fout = src == -1 ? states[id0] : renderingModules[src]->getOutputFields()[id0];
			FBase* fin = renderingModules[dst]->getInputFields()[id1];

			fout->connect(fin);

			renderingConnectionXml = renderingConnectionXml->NextSiblingElement("Connection");
		}
		renderingModules.clear();

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