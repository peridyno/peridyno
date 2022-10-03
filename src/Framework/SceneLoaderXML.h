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
		};

	public:
		std::shared_ptr<SceneGraph> load(const std::string filename) override;

		bool save(std::shared_ptr<SceneGraph> scn, const std::string filename) override;

	private:
		std::shared_ptr<Node> processNode(tinyxml2::XMLElement* nodeXML);
		std::shared_ptr<Module> processModule(tinyxml2::XMLElement* moduleXML);
		bool addModule(std::shared_ptr<Node> node, std::shared_ptr<Module> module);

		std::vector<std::string> split(std::string str, std::string pattern);

		std::string encodeVec3f(const Vec3f v);
		Vec3f decodeVec3f(const std::string str);

		std::string encodeVec2f(const Vec2f v);
		Vec2f decodeVec2f(const std::string str);

		virtual bool canLoadFileByExtension(const std::string extension);

		std::vector<std::vector<ConnectionInfo>> mConnectionInfo;
	};
}
