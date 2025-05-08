#pragma once
#include "Node.h"
#include "SceneLoaderFactory.h"
#include "tinyxml/tinyxml2.h"

namespace dyno {

	class SceneLoaderXML : public SceneLoader
	{
		struct ConnectionInfo
		{
			uint src;
			uint dst;
			uint id0;
			uint id1;
			int srcModuleId = -1;
			int srcModulePipline = -1;
		};

	public:
		std::shared_ptr<SceneGraph> load(const std::string filename) override;

		bool save(std::shared_ptr<SceneGraph> scn, const std::string filename) override;

	private:
		std::shared_ptr<Node> processNode(tinyxml2::XMLElement* nodeXML);

		void processModule(tinyxml2::XMLElement* moduleXml, std::shared_ptr<Pipeline> pipeline, std::vector<Module*>& modules);

		bool ConstructPipeline(std::shared_ptr<Node> node, std::shared_ptr<Pipeline> pipeline, tinyxml2::XMLElement* nodeXML, std::vector<std::shared_ptr<Node>> nodes);

		bool ConstructNodePipeline(std::shared_ptr<Node> node, tinyxml2::XMLElement* nodeXML, std::vector<std::shared_ptr<Node>> nodes);

		std::string encodeVec3f(const Vec3f v);
		Vec3f decodeVec3f(const std::string str);

		std::string encodeVec2f(const Vec2f v);
		Vec2f decodeVec2f(const std::string str);

		virtual bool canLoadFileByExtension(const std::string extension);

		std::vector<std::vector<ConnectionInfo>> mConnectionInfo;
	};
}
