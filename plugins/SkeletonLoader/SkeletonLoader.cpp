#include "SkeletonLoader.h"
#include "GLPhotorealisticRender.h"
#include <stb/stb_image.h>
#define STB_IMAGE_IMPLEMENTATION

#define AXIS 0


namespace dyno
{
	IMPLEMENT_TCLASS(SkeletonLoader, TDataType)

	template<typename TDataType>
	SkeletonLoader<TDataType>::SkeletonLoader()
		: ParametricModel<TDataType>()
	{
		auto defaultTopo = std::make_shared<DiscreteElements<TDataType>>();
		this->stateTopology()->setDataPtr(defaultTopo);
		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->statePolygonSet()->setDataPtr(std::make_shared<PolygonSet<TDataType>>());
		this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());

		auto texmeshRender = std::make_shared<GLPhotorealisticRender>();
		this->stateTextureMesh()->connect(texmeshRender->inTextureMesh());
		this->graphicsPipeline()->pushModule(texmeshRender);

		this->stateHierarchicalScene()->setDataPtr(std::make_shared<HierarchicalScene>());
		this->setForceUpdate(false);

	}


	template<typename TDataType>
	SkeletonLoader<TDataType>::~SkeletonLoader()
	{
		this->stateTextureMesh()->getDataPtr()->clear();
		this->stateHierarchicalScene()->getDataPtr()->clear();
	}

	template<typename TDataType>
	bool SkeletonLoader<TDataType>::initFBX(const char* filepath)
	{
		FILE* fp = fopen(filepath, "rb");

		if (!fp) return false;

		fseek(fp, 0, SEEK_END);
		long file_size = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		auto* content = new ofbx::u8[file_size];
		fread(content, 1, file_size, fp);

		this->mFbxScene = ofbx::load((ofbx::u8*)content, file_size, (ofbx::u64)ofbx::LoadFlags::NONE);

		float mFbxScale = this->mFbxScene->getGlobalSettings()->UnitScaleFactor;



		int objectCount = mFbxScene->getAllObjectCount();
		int meshCount = mFbxScene->getMeshCount();
		int geoCount = mFbxScene->getGeometryCount();
		int animationStackCount = mFbxScene->getAnimationStackCount();
		int embeddedDataCount = mFbxScene->getEmbeddedDataCount();

		printf("objectCount : %d \n", objectCount);
		printf("meshCount : %d \n", meshCount);
		printf("geoCount : %d \n", geoCount);

		//printf("meshCount : %d\n",meshCount);
		std::vector<std::shared_ptr<MeshInfo>> meshs;
		for (int id = 0; id < meshCount; id++)
		{
			//printf("*****************:\n");
			//std::cout << mFbxScene->getMesh(id)->name << std::endl;

			const ofbx::Mesh* currentMesh = (const ofbx::Mesh*)mFbxScene->getMesh(id);
			//std:: cout<< "Mesh Name :  " << currentMesh->name << "\n";
			
			std::shared_ptr<MeshInfo> meshInfo = std::make_shared<MeshInfo>();

			auto geoMatrix = currentMesh->getGeometricMatrix();
			auto gloTf = currentMesh->getGlobalTransform();
			auto pivot = currentMesh->getRotationPivot();
			auto locTf = currentMesh->getLocalTransform();
			auto locR = currentMesh->getLocalRotation();
			auto locS = currentMesh->getLocalScaling();
			auto locT = currentMesh->getLocalTranslation();
			auto preR = currentMesh->getPreRotation();


			meshInfo->localTranslation = Vec3f(locT.x, locT.y, locT.z);
			meshInfo->localRotation = Vec3f(locR.x, locR.y, locR.z);
			meshInfo->localScale = Vec3f(locS.x, locS.y, locS.z);
			meshInfo->preRotation = Vec3f(preR.x, preR.y, preR.z);
			meshInfo->pivot = Vec3f(pivot.x, pivot.y, pivot.z);
			meshInfo->name = currentMesh->name;
			
			auto positionCount = currentMesh->getGeometry()->getGeometryData().getPositions().count;	
			float tempScale = 0.01;


			for (size_t i = 0; i < positionCount; i++)
			{
				auto pos = currentMesh->getGeometry()->getGeometryData().getPositions().get(i) ;
				meshInfo->vertices.push_back(Vec3f(pos.x,pos.y,pos.z) * tempScale);
			}

			for (size_t i = 0; i < positionCount; i++)
			{
				auto indices = currentMesh->getGeometry()->getGeometryData().getPositions().indices[i];		
				meshInfo->verticeId_pointId.push_back(indices);	
			}


			auto normalCount = currentMesh->getGeometry()->getGeometryData().getNormals().count;
			for (size_t i = 0; i < normalCount; i++)
			{
				auto n = currentMesh->getGeometry()->getGeometryData().getNormals().get(i);
				meshInfo->normals.push_back(Vec3f(n.x,n.y,n.z));
			}

			auto uvCount = currentMesh->getGeometry()->getGeometryData().getUVs().count;
			for (size_t i = 0; i < uvCount; i++)
			{
				auto uv = currentMesh->getGeometry()->getGeometryData().getUVs().get(i);
				meshInfo->texcoords.push_back(Vec2f(uv.x,uv.y));
			}

			auto colorCount = currentMesh->getGeometry()->getGeometryData().getColors().count;
			for (size_t i = 0; i < colorCount; i++)
			{
				auto color = currentMesh->getGeometry()->getGeometryData().getColors().get(i);
				meshInfo->verticesColor.push_back(Vec3f(color.x, color.y, color.z));
			}

			auto partitionCount = currentMesh->getGeometry()->getGeometryData().getPartitionCount();
		

			for (size_t i = 0; i < partitionCount; i++)
			{
				auto polygonCount = currentMesh->getGeometry()->getGeometryData().getPartition(i).polygon_count;

				CArrayList<uint> polygons;
				CArray<uint> counter;

				for (size_t j = 0; j < polygonCount; j++)
				{
					auto verticesCount = currentMesh->getGeometry()->getGeometryData().getPartition(i).polygons[j].vertex_count;
					counter.pushBack(verticesCount);
				}
				polygons.resize(counter);

				Vec3f boundingMax = Vec3f(-FLT_MAX);
				Vec3f boundingMin = Vec3f(FLT_MAX);

				for (size_t j = 0; j < polygonCount; j++)
				{
					auto& index = polygons[j];

					auto from = currentMesh->getGeometry()->getGeometryData().getPartition(i).polygons[j].from_vertex;
					auto verticesCount = currentMesh->getGeometry()->getGeometryData().getPartition(i).polygons[j].vertex_count;

					for (size_t k = 0; k < verticesCount; k++)
					{
						int polyId = k + from;
						index.insert(polyId);

						Vec3f pos = meshInfo->vertices[polyId];
						if (pos.x > boundingMax.x)boundingMax.x = pos.x;
						else boundingMax.x = boundingMax.x;

						if (pos.y > boundingMax.y)boundingMax.y = pos.y;
						else boundingMax.y = boundingMax.y;

						if (pos.z > boundingMax.z)boundingMax.z = pos.z;
						else  boundingMax.z = boundingMax.z;

						if (pos.x < boundingMin.x)boundingMin.x = pos.x;
						else boundingMin.x = boundingMin.x;

						if (pos.y < boundingMin.y)boundingMin.y =pos.y;
						else boundingMin.y = boundingMin.y;

						if (pos.z < boundingMin.z)boundingMin.z = pos.z;
						else boundingMin.z = boundingMin.z;


					}
				}
				meshInfo->facegroup_polygons.push_back(polygons);
				meshInfo->boundingBox.push_back(TAlignedBox3D<Real>(boundingMin, boundingMax));
				if(this->varUseInstanceTransform()->getValue())
					meshInfo->boundingTransform.push_back(Transform3f((boundingMax + boundingMin) / 2,Mat3f::identityMatrix(),Vec3f(1)));
				else
					meshInfo->boundingTransform.push_back(Transform3f());

				for (size_t j = 0; j < polygonCount; j++)
				{
					auto& index = polygons[j];

					auto from = currentMesh->getGeometry()->getGeometryData().getPartition(i).polygons[j].from_vertex;
					auto verticesCount = currentMesh->getGeometry()->getGeometryData().getPartition(i).polygons[j].vertex_count;

					for (size_t k = 0; k < verticesCount; k++)
					{
						int polyId = k + from;

						meshInfo->vertices[polyId] = meshInfo->vertices[polyId] - meshInfo->boundingTransform[i].translation();
					}
				}

				TopologyModule::Triangle tri;
				std::vector<TopologyModule::Triangle> triangles;
				for (size_t j = 0; j < polygonCount; j++)
				{
					auto from = currentMesh->getGeometry()->getGeometryData().getPartition(i).polygons[j].from_vertex;
					auto verticesCount = currentMesh->getGeometry()->getGeometryData().getPartition(i).polygons[j].vertex_count;

					for (size_t k = 0; k < verticesCount - 2; k++)
					{
						int polyId = k + from;

						tri[0] = from;
						tri[1] = k + from + 1;
						tri[2] = k + from + 2;

						triangles.push_back(tri);
					}
				}

				meshInfo->facegroup_triangles.push_back(triangles);

			}

			//Material
			auto matCount = currentMesh->getMaterialCount();
			for (size_t i = 0; i < matCount; i++)
			{
				auto mat = currentMesh->getMaterial(i);
				//std::cout << mat->name << "\n";

				auto diffuseTex = mat->getTexture(ofbx::Texture::DIFFUSE);
				if (diffuseTex)
					auto relativeFileName = diffuseTex->getRelativeFileName();

				auto normalTex = mat->getTexture(ofbx::Texture::NORMAL);
				if (normalTex)
					auto relativeFileName = normalTex->getRelativeFileName();

				std::shared_ptr<Material> material = std::make_shared<Material>();
				material->baseColor = Vec3f(mat->getDiffuseColor().r, mat->getDiffuseColor().g, mat->getDiffuseColor().b);
				material->roughness = 1;



				{
					auto texture = mat->getTexture(ofbx::Texture::TextureType::DIFFUSE);
					std::string textureName;
					if (texture)
					{
						auto it = texture->getRelativeFileName();
						for (const ofbx::u8* ptr = it.begin; ptr <= it.end; ++ptr) {
							textureName += *ptr;
						}

					}

					size_t found = textureName.find_last_of("\\");

					if (found != std::string::npos) {
						std::string filename = textureName.substr(found + 1);

						auto fbxFile = this->varFileName()->getValue();
						size_t foundPath = fbxFile.string().find_last_of("/");
						std::string path = fbxFile.string().substr(0, foundPath);


						std::string loadPath = path + std::string("\\\\") + filename;
						loadPath.pop_back();
						dyno::CArray2D<dyno::Vec4f> textureData(1, 1);
						textureData[0, 0] = dyno::Vec4f(1);

						if (loadTexture(loadPath.c_str(), textureData))
							material->texColor.assign(textureData);

					}
				}
				

				{
					auto texture = mat->getTexture(ofbx::Texture::TextureType::NORMAL);
					std::string textureName;
					if (texture)
					{
						auto it = texture->getRelativeFileName();
						for (const ofbx::u8* ptr = it.begin; ptr <= it.end; ++ptr) {
							textureName += *ptr;
						}

					}

					size_t found = textureName.find_last_of("\\");

					if (found != std::string::npos) {
						std::string filename = textureName.substr(found + 1);

						auto fbxFile = this->varFileName()->getValue();
						size_t foundPath = fbxFile.string().find_last_of("/");
						std::string path = fbxFile.string().substr(0, foundPath);


						std::string loadPath = path + std::string("\\\\") + filename;
						loadPath.pop_back();

						dyno::CArray2D<dyno::Vec4f> textureData(1, 1);
						textureData[0, 0] = dyno::Vec4f(1);


						if (loadTexture(loadPath.c_str(), textureData))
							material->texBump.assign(textureData);

					}
				}

				meshInfo->materials.push_back(material);
			}


			//Pose
			auto pose = currentMesh->getPose();
			if (pose) 
			{
				auto poseMatrix = pose->getMatrix();


				////Skin

				auto clusterCount = currentMesh->getSkin()->getClusterCount();
				for (size_t i = 0; i < clusterCount; i++)
				{

					auto cluster = currentMesh->getSkin()->getCluster(i);

					auto clusteName = cluster->name;
					//std::cout << "Object Name: " << clusteName << "\n";

					auto JointName = cluster->getLink()->name;
					//std::cout << "Link Name: " << JointName << "\n";

					////clusteName - JointName ;
					//pushBone(cluster->getLink());

					auto indicesCount = cluster->getIndicesCount();

					for (size_t j = 0; j < indicesCount; j++)
					{
						auto indices = cluster->getIndices()[j];
					}

					auto weightCount = cluster->getWeightsCount();

					for (size_t j = 0; j < weightCount; j++)
					{
						auto weights = cluster->getWeights()[j];

					}

				}
			}
			
			meshs.push_back(meshInfo);
		}

		updateTextureMesh(meshs);

		//Get Bones
		auto allObj = mFbxScene->getAllObjects();
		int objCount = mFbxScene->getAllObjectCount();

		std::map<std::string, std::string> parentTag;
		std::map<std::string, std::shared_ptr<ModelObject>> nameParentObj;

		std::vector<std::shared_ptr<Bone>> bonesInfo;
		for (size_t objId = 0; objId < objCount; objId++)
		{
			//Bone
			if (allObj[objId]->getType() == ofbx::Object::Type::LIMB_NODE)
			{
				pushBone(allObj[objId],parentTag, nameParentObj, bonesInfo);
			}
		}

		buildHierarchy(parentTag, nameParentObj);

		updateHierarchicalScene(meshs, bonesInfo);

		//Update joints animation curve;
		for (int i = 0, n = mFbxScene->getAnimationStackCount(); i < n; ++i) {
			const ofbx::AnimationStack* stack = mFbxScene->getAnimationStack(i);
			for (int j = 0; stack->getLayer(j); ++j) {
				const ofbx::AnimationLayer* layer = stack->getLayer(j);
				for (int k = 0; layer->getCurveNode(k); ++k) {
					const ofbx::AnimationCurveNode* node = layer->getCurveNode(k);
					auto nodeTrans = node->getNodeLocalTransform(0);	

					char property[32];
					node->getBoneLinkProperty().toString(property);

					getCurveValue(node);	
				}
			}
		}



		delete[] content;
		fclose(fp);




		return true;




	}





	template<typename TDataType>
	void SkeletonLoader<TDataType>::loadFBX()
	{
		auto filename = this->varFileName()->getData();
		std::string filepath = filename.string();

		this->stateHierarchicalScene()->getDataPtr()->clear();
		this->stateTextureMesh()->getDataPtr()->clear();

		initFBX(filepath.c_str());



	}

	

	template<typename TDataType>
	void SkeletonLoader<TDataType>::resetStates()
	{

		loadFBX();

	}

	template<typename TDataType>
	bool SkeletonLoader<TDataType>::loadTexture(const char* path, dyno::CArray2D<dyno::Vec4f>& img)
	{

		int x, y, comp;
		stbi_set_flip_vertically_on_load(true);

		float* data = stbi_loadf(path, &x, &y, &comp, STBI_default);
		if (data == 0)
			return false;

		if (data) {
			img.resize(x, y);
			for (int x0 = 0; x0 < x; x0++)
			{
				for (int y0 = 0; y0 < y; y0++)
				{
					int idx = (y0 * x + x0) * comp;
					for (int c0 = 0; c0 < comp; c0++) {
						img(x0, y0)[c0] = data[idx + c0];
					}
				}
			}
		}

		delete data;
		return true;
	}

	DEFINE_CLASS(SkeletonLoader);
}