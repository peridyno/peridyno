#include "FBXLoader.h"
#include "GLPhotorealisticRender.h"
#include <stb/stb_image.h>
#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"
#include <regex>
#include "ImageLoader.h"
#include "Mapping/TextureMeshToTriangleSet.h"


#define STB_IMAGE_IMPLEMENTATION

#define AXIS 0


namespace dyno
{
	IMPLEMENT_TCLASS(FBXLoader, TDataType)

	template<typename TDataType>
	FBXLoader<TDataType>::FBXLoader()
		: ParametricModel<TDataType>()
	{
		auto defaultTopo = std::make_shared<DiscreteElements<TDataType>>();
		this->stateTopology()->setDataPtr(defaultTopo);
		this->varAnimationSpeed()->setRange(0.01,10);

		this->statePolygonSet()->setDataPtr(std::make_shared<PolygonSet<TDataType>>());
		this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());

		auto texmeshRender = std::make_shared<GLPhotorealisticRender>();
		this->stateTextureMesh()->connect(texmeshRender->inTextureMesh());
		this->graphicsPipeline()->pushModule(texmeshRender);

		auto convert2TriSet = std::make_shared<TextureMeshToTriangleSet<TDataType>>();
		this->stateTextureMesh()->connect(convert2TriSet->inTextureMesh());

		this->animationPipeline()->getParentNode()->addOutputField(convert2TriSet->outTriangleSet());
		this->animationPipeline()->pushModule(convert2TriSet);

		this->stateHierarchicalScene()->setDataPtr(std::make_shared<HierarchicalScene>());
		this->setForceUpdate(false);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&FBXLoader<TDataType>::varAnimationChange, this));
		this->varImportAnimation()->attach(callback);
		this->varImportAnimation()->setValue(false);

		auto callbackImport = std::make_shared<FCallBackFunc>(std::bind(&FBXLoader<TDataType>::initFBX, this));
		this->varFileName()->attach(callbackImport);
		this->varUseInstanceTransform()->attach(callbackImport);
		this->varImportAnimation()->attach(callbackImport);


		auto callbackTransform = std::make_shared<FCallBackFunc>(std::bind(&FBXLoader<TDataType>::updateTransform, this));

		this->varLocation()->attach(callbackTransform);
		this->varScale()->attach(callbackTransform);
		this->varRotation()->attach(callbackTransform);

		//auto normalCallback = std::make_shared<FCallBackFunc>(std::bind(&SkeletonLoader<TDataType>::updateNormal, this));
		//this->varRecalculateNormal()->attach(normalCallback);
		//this->varFlipNormal()->attach(normalCallback);

		{//JointRender
			auto ptRender = std::make_shared<GLPointVisualModule>();
			this->stateJointSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
			this->stateJointSet()->connect(ptRender->inPointSet());
			ptRender->varPointSize()->setValue(0.002);
			ptRender->setColor(Color::Purple3());

			auto wireRender = std::make_shared<GLWireframeVisualModule>();
			this->stateJointSet()->connect(wireRender->inEdgeSet());
			wireRender->varRenderMode()->setCurrentKey(GLWireframeVisualModule::EEdgeMode::CYLINDER);
			wireRender->setColor(Color::Purple());

			wireRender->varLineWidth()->setValue(0.002);
			this->graphicsPipeline()->pushModule(ptRender);
			this->graphicsPipeline()->pushModule(wireRender);
		}

		{
			//this->stateShapeCenter()->setDataPtr(std::make_shared<PointSet<TDataType>>());
			//auto ptRender = std::make_shared<GLPointVisualModule>();

			//this->stateShapeCenter()->connect(ptRender->inPointSet());
			//ptRender->varPointSize()->setValue(0.01);
			//ptRender->setColor(Color::Red());

			//this->graphicsPipeline()->pushModule(ptRender);
			//ptRender->varVisible()->setValue(false);
		}

		this->stateTextureMesh()->promoteOuput();


	}


	template<typename TDataType>
	void FBXLoader<TDataType>::varAnimationChange()
	{
		auto importAnimation = this->varImportAnimation()->getValue();
		if (importAnimation) 
		{
			this->setForceUpdate(true);
		}
		else 
		{
			this->setForceUpdate(false);
		}
	}



	template<typename TDataType>
	FBXLoader<TDataType>::~FBXLoader()
	{
		this->stateTextureMesh()->getDataPtr()->clear();
		this->stateHierarchicalScene()->getDataPtr()->clear();

		initialPosition.clear();
		initialNormal.clear();
		mPoint2Vertice.clear();
		mVertice2Point.clear();
		initialShapeCenter.clear();
	}

	template<typename TDataType>
	bool FBXLoader<TDataType>::initFBX()
	{
		this->stateHierarchicalScene()->getDataPtr()->clear();
		this->stateTextureMesh()->getDataPtr()->clear();

		auto targetScene = this->stateHierarchicalScene()->getDataPtr();
		auto filename = this->varFileName()->getData();
		std::string filepath = filename.string();
		std::cout << filepath <<"\n\n";

		FILE* fp = fopen(filepath.c_str(), "rb");

		if (!fp) return false;

		fseek(fp, 0, SEEK_END);
		long file_size = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		auto* content = new ofbx::u8[file_size];
		fread(content, 1, file_size, fp);

		auto mFbxScene = ofbx::load((ofbx::u8*)content, file_size, (ofbx::u64)ofbx::LoadFlags::NONE);

		float mFbxScale = mFbxScene->getGlobalSettings()->UnitScaleFactor;
		int meshCount = mFbxScene->getMeshCount();
		float tempScale = 0.01;

		//Get Bones
		auto allObj = mFbxScene->getAllObjects();
		int objCount = mFbxScene->getAllObjectCount();

		std::map<std::string, std::string> parentTag;
		std::map<std::string, std::shared_ptr<ModelObject>> nameParentObj;
		std::map<std::string, std::shared_ptr<Bone>> name2Bone;

		std::vector<std::shared_ptr<Bone>> bonesInfo;
		for (size_t objId = 0; objId < objCount; objId++)
		{
			//Bone
			if (allObj[objId]->getType() == ofbx::Object::Type::LIMB_NODE)
			{
				name2Bone[allObj[objId]->name] = pushBone(allObj[objId], parentTag, nameParentObj, bonesInfo,tempScale);
			}
		}
		
		setBonesToScene(bonesInfo, targetScene);
		std::vector<Mat4f> currentPoseInverseMatrix;



		std::vector<std::shared_ptr<MeshInfo>> meshs;


		for (int id = 0; id < meshCount; id++)
		{
			const ofbx::Mesh* currentMesh = (const ofbx::Mesh*)mFbxScene->getMesh(id);
			
			std::shared_ptr<MeshInfo> meshInfo = std::make_shared<MeshInfo>();

			auto geoMatrix = currentMesh->getGeometricMatrix();
			auto gloTf = currentMesh->getGlobalTransform();
			auto pivot = currentMesh->getRotationPivot();
			auto locTf = currentMesh->getLocalTransform();
			auto locR = currentMesh->getLocalRotation();
			auto locS = currentMesh->getLocalScaling();
			auto locT = currentMesh->getLocalTranslation();
			auto preR = currentMesh->getPreRotation();

			meshInfo->localTranslation = Vec3f(locT.x, locT.y, locT.z) * tempScale;
			meshInfo->localRotation = Vec3f(locR.x, locR.y, locR.z);
			meshInfo->localScale = Vec3f(locS.x, locS.y, locS.z);
			meshInfo->preRotation = Vec3f(preR.x, preR.y, preR.z);
			meshInfo->pivot = Vec3f(pivot.x, pivot.y, pivot.z) * tempScale;
			meshInfo->name = currentMesh->name;
						
			if (currentMesh->parent!=NULL)
			{
				parentTag[std::string(currentMesh->name)] = std::string(currentMesh->parent->name);
				nameParentObj[std::string(currentMesh->name)] = meshInfo;
			}
			else
			{
				parentTag[std::string(currentMesh->name)] = std::string("No parent object");
				nameParentObj[std::string(currentMesh->name)] = nullptr;
			}

			auto positionCount = currentMesh->getGeometry()->getGeometryData().getPositions().count;	
			auto positionValueCount = currentMesh->getGeometry()->getGeometryData().getPositions().values_count;
			meshInfo->points.resize(positionValueCount);

			for (size_t i = 0; i < positionCount; i++)
			{
				auto pos = currentMesh->getGeometry()->getGeometryData().getPositions().get(i) ;
				meshInfo->vertices.push_back(Vec3f(pos.x,pos.y,pos.z) * tempScale);
				if (pos.x == 5.22856617)
					printf("x");
				auto indices = currentMesh->getGeometry()->getGeometryData().getPositions().indices[i];
				meshInfo->verticeId_pointId.push_back(indices);
				meshInfo->pointId_verticeId[indices].push_back(i);

				meshInfo->points[indices] = Vec3f(pos.x, pos.y, pos.z) * tempScale;
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
			//auto vertexCount = currentMesh->getGeometry()->getGeometryData().getPartition(0).polygons->vertex_count();

			// Polygons
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

						/*Vec3f pos = meshInfo->vertices[polyId];
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
						else boundingMin.z = boundingMin.z;*/


					}
				}
				meshInfo->facegroup_polygons.push_back(polygons);
				//meshInfo->boundingBox.push_back(TAlignedBox3D<Real>(boundingMin, boundingMax));

				

				TopologyModule::Triangle tri;
				std::vector<TopologyModule::Triangle> triangles;
				std::vector<TopologyModule::Triangle> triangleNormalIndex;
				for (size_t j = 0; j < polygonCount; j++)
				{
					auto from = currentMesh->getGeometry()->getGeometryData().getPartition(i).polygons[j].from_vertex;
					auto verticesCount = currentMesh->getGeometry()->getGeometryData().getPartition(i).polygons[j].vertex_count;

					for (size_t k = 0; k < verticesCount - 2; k++)
					{
						int polyId = k + from;

						tri[0] = meshInfo->verticeId_pointId[from];
						tri[1] = meshInfo->verticeId_pointId[k + from + 1] ;
						tri[2] = meshInfo->verticeId_pointId[k + from + 2];

						triangles.push_back(tri);

						tri[0] = from;
						tri[1] = k + from + 1;
						tri[2] = k + from + 2;
						triangleNormalIndex.push_back(tri);
					}
				}

				meshInfo->facegroup_triangles.push_back(triangles);
				meshInfo->facegroup_normalIndex.push_back(triangleNormalIndex);
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
						size_t foundPath = fbxFile.string().find_last_of("/")< fbxFile.string().size()? fbxFile.string().find_last_of("/") : fbxFile.string().find_last_of("\\");

						std::string path = fbxFile.string().substr(0, foundPath);


						std::string loadPath = path + std::string("/") + filename;
						loadPath.pop_back();
						dyno::CArray2D<dyno::Vec4f> textureData(1, 1);
						textureData[0, 0] = dyno::Vec4f(1);


						if (ImageLoader::loadImage(loadPath.c_str(), textureData))// loadTexture
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


						if (ImageLoader::loadImage(loadPath.c_str(), textureData))//loadTexture
							material->texBump.assign(textureData);

					}
				}

				meshInfo->materials.push_back(material);

			}

			//Skin Weights
			auto pose = currentMesh->getPose();

			if (pose)
			{
				std::map<int, int> pointId_Channel;


				auto clusterCount = currentMesh->getSkin()->getClusterCount();
				for (size_t clusterId = 0; clusterId < clusterCount; clusterId++)
				{
					auto cluster = currentMesh->getSkin()->getCluster(clusterId);

					auto it = name2Bone.find(cluster->getLink()->name);

					int boneId = it == name2Bone.end() ? -1 :it->second->id;


					auto m = cluster->getTransformMatrix().m;//inverseMatrix

					Mat4f inverseM= Mat4f(m[0], m[4], m[8], m[12] *tempScale,
									m[1], m[5], m[9], m[13] * tempScale,
									m[2], m[6], m[10], m[14] * tempScale,
									m[3], m[7], m[11], m[15]);

					bonesInfo[boneId]->inverseBindMatrix = inverseM;

					auto indicesCount = cluster->getIndicesCount();

					meshInfo->resizeSkin(meshInfo->points.size());


					for (size_t j = 0; j < indicesCount; j++)
					{
						auto indices = cluster->getIndices()[j];
						auto weights = cluster->getWeights()[j];
						if (weights <= 0.001)
							continue;
						//auto vertices = meshInfo->pointId_verticeId[indices];
						auto iter = pointId_Channel.find(indices);

						int boneDataIndex = -1;
						int channel = -1;

						if (iter != pointId_Channel.end()) 
						{
							channel = iter->second;
							pointId_Channel[indices] = iter->second + 1;
						}
						else 
						{
							channel = 0;
							pointId_Channel[indices] = 1;
						}

						boneDataIndex = channel / 4;
						channel = channel % 4;

						if (boneDataIndex == 0)
						{
							//for (auto vId : vertices)
							//{
								meshInfo->boneWeights0[indices][channel] = weights;
								meshInfo->boneIndices0[indices][channel] = boneId;
							//}
						}
						else if (boneDataIndex == 1)
						{
							//for (auto vId : vertices)
							//{
								meshInfo->boneWeights1[indices][channel] = weights;
								meshInfo->boneIndices1[indices][channel] = boneId;
							//}
						}
						else if (boneDataIndex == 2)
						{
							//for (auto vId : vertices)
							//{
								meshInfo->boneWeights2[indices][channel] = weights;
								meshInfo->boneIndices2[indices][channel] = boneId;
							//}
						}
					}
				}
			}
			
			
			meshs.push_back(meshInfo);
		}

		
		buildHierarchy(parentTag, nameParentObj);

		//const ofbx::GlobalSettings* settings = mFbxScene->getGlobalSettings();
		//hierarchicalScene->mTimeStart = settings->TimeSpanStart;
		//hierarchicalScene->mTimeEnd = settings->TimeSpanStop;


		updateHierarchicalScene(meshs, bonesInfo);

		updateTextureMesh(meshs);


		if (this->varImportAnimation()->getValue() && bonesInfo.size())
		{
			//Update joints animation curve;
			for (int i = 0, n = mFbxScene->getAnimationStackCount(); i < n; ++i) {
				const ofbx::AnimationStack* stack = mFbxScene->getAnimationStack(i);
				for (int j = 0; stack->getLayer(j); ++j) {
					const ofbx::AnimationLayer* layer = stack->getLayer(j);
					for (int k = 0; layer->getCurveNode(k); ++k) {
						const ofbx::AnimationCurveNode* node = layer->getCurveNode(k);
						auto nodeTrans = node->getNodeLocalTransform(0);

						getCurveValue(node, targetScene,tempScale);
					}
				}
			}
			targetScene->mJointAnimationData->updateTotalTime();
		}
		targetScene->showJointInfo();

		////update Normal
		//if (targetScene->mJointAnimationData->isValid())
		//{
		//	//targetScene->mJointAnimationData->updateAnimationPose(0);
		//	//targetScene->mSkinData->initialNormal.assign(targetScene->mSkinData->mesh->normals());

		//	for (size_t i = 0; i < targetScene->mSkinData->size(); i++)//
		//	{
		//		auto& bindJoint0 = targetScene->mSkinData->V_jointID_0[i];
		//		auto& bindJoint1 = targetScene->mSkinData->V_jointID_1[i];
		//		auto& bindJoint2 = targetScene->mSkinData->V_jointID_2[i];

		//		auto& bindWeight0 = targetScene->mSkinData->V_jointWeight_0[i];
		//		auto& bindWeight1 = targetScene->mSkinData->V_jointWeight_1[i];
		//		auto& bindWeight2 = targetScene->mSkinData->V_jointWeight_2[i];

		//		targetScene->getVerticesNormalInBindPose(
		//			targetScene->mSkinData->initialNormal,
		//			targetScene->mJointData->mJointInverseBindMatrix,
		//			targetScene->mJointAnimationData->mSkeleton->mJointWorldMatrix,

		//			mPoint2Vertice,
		//			bindJoint0,
		//			bindJoint1,
		//			bindJoint2,
		//			bindWeight0,
		//			bindWeight1,
		//			bindWeight2,

		//			targetScene->mSkinData->skin_VerticeRange[i]
		//		);
		//	}
		//}

		//updateNormal();

		updateTransform();

		delete[] content;
		fclose(fp);

		return true;

	}


	

	template<typename TDataType>
	void FBXLoader<TDataType>::resetStates()
	{
		updateTransform();
		Node::resetStates();
	}

	template<typename TDataType>
	void FBXLoader<TDataType>::updateStates()
	{
		if(this->varImportAnimation()->getValue())
			updateAnimation(this->stateElapsedTime()->getValue() * this->varAnimationSpeed()->getValue());
		Node::updateStates();
	}

	template<typename TDataType>
	bool FBXLoader<TDataType>::loadTexture(const char* path, dyno::CArray2D<dyno::Vec4f>& img)
	{

		int x, y, comp;
		stbi_set_flip_vertically_on_load(true);

		std::string normalizedPath = std::regex_replace(path, std::regex(R"(\\+|/+)"), "/");

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

	template<typename TDataType>
	void FBXLoader<TDataType>::coutName_Type(ofbx::Object::Type ty, ofbx::Object* obj)
	{
		{
			std::cout << obj->name << "  -  ";
			std::cout << obj->element.getFirstProperty()->getValue().toU64() << "\n";


			switch (ty)
			{
			case ofbx::Object::Type::ROOT:
				printf("0_ROOT\n");
				break;

			case ofbx::Object::Type::GEOMETRY:
				printf("1_GEOMETRY\n");
				break;

			case ofbx::Object::Type::SHAPE:
				printf("2_SHAPE\n");
				break;

			case ofbx::Object::Type::MATERIAL:
				printf("3_MATERIAL\n");
				break;

			case ofbx::Object::Type::MESH:
				printf("4_MESH\n");
				break;

			case ofbx::Object::Type::TEXTURE:
				printf("5_TEXTURE\n");
				break;

			case ofbx::Object::Type::LIMB_NODE:
				printf("6_LIMB_NODE\n");
				break;

			case ofbx::Object::Type::NULL_NODE:
				printf("7_NULL_NODE\n");
				break;

			case ofbx::Object::Type::NODE_ATTRIBUTE:
				printf("8_NODE_ATTRIBUTE\n");
				break;

			case ofbx::Object::Type::CLUSTER:
				printf("9_CLUSTER\n");
				break;

			case ofbx::Object::Type::SKIN:
				printf("10_SKIN\n");
				break;

			case ofbx::Object::Type::BLEND_SHAPE:
				printf("11_BLEND_SHAPE\n");
				break;

			case ofbx::Object::Type::BLEND_SHAPE_CHANNEL:
				printf("12_BLEND_SHAPE_CHANNEL\n");
				break;

			case ofbx::Object::Type::ANIMATION_STACK:
				printf("13_BLEND_SHAPE\n");
				break;

			case ofbx::Object::Type::ANIMATION_LAYER:
				printf("14_BLEND_SHAPE\n");
				break;

			case ofbx::Object::Type::ANIMATION_CURVE:
				printf("15_BLEND_SHAPE\n");
				break;

			case ofbx::Object::Type::ANIMATION_CURVE_NODE:
				printf("16_ANIMATION_CURVE_NODE\n");
				break;

			case ofbx::Object::Type::POSE:
				printf("17_POSE  ");
				break;

			default:
				break;
			}

		}
	}

	template<typename TDataType>
	void FBXLoader<TDataType>::getCurveValue(const ofbx::AnimationCurveNode* node, std::shared_ptr<HierarchicalScene> scene,float scale)
	{
		if (!node->getBone())
			return;

		auto objId = scene->getObjIndexByName(std::string(node->getBone()->name));


		if (objId == -1)
			return;

		auto propertyData = node->getBoneLinkProperty();

		if (propertyData == "Lcl Translation")
		{
			auto x = node->getCurve(0);
			auto y = node->getCurve(1);
			auto z = node->getCurve(2);
			if (x)
			{
				for (size_t m = 0; m < x->getKeyCount(); m++)
				{
					auto time = Real(x->getKeyTime()[m]) / REFTIME;
					auto value = x->getKeyValue()[m] * scale;
					//std::cout << "x : " << value <<std::endl;
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_tT_X[objId].push_back(time);
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_T_X[objId].push_back(value);
				}
			}
			if (y)
			{
				for (size_t m = 0; m < y->getKeyCount(); m++)
				{
					auto time = Real(y->getKeyTime()[m]) / REFTIME;
					auto value = y->getKeyValue()[m] * scale;
					//std::cout << "y : " << value << std::endl;
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_tT_Y[objId].push_back(time);
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_T_Y[objId].push_back(value);
				}
			}

			if (z)
			{
				for (size_t m = 0; m < z->getKeyCount(); m++)
				{
					auto time = Real(z->getKeyTime()[m]) / REFTIME;
					auto value = z->getKeyValue()[m] * scale;
					//std::cout << "z : " << value << std::endl;
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_tT_Z[objId].push_back(time);
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_T_Z[objId].push_back(value);
				}
			}
		}

		if (propertyData == "Lcl Rotation")
		{
			auto x = node->getCurve(0);
			auto y = node->getCurve(1);
			auto z = node->getCurve(2);

			if (x)
			{
				for (size_t m = 0; m < x->getKeyCount(); m++)
				{
					auto time = Real(x->getKeyTime()[m]) / REFTIME;
					auto value = x->getKeyValue()[m];

					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_tR_X[objId].push_back(time);
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_R_X[objId].push_back(value);

				}
			}

			if (y)
			{
				for (size_t m = 0; m < y->getKeyCount(); m++)
				{
					auto time = Real(y->getKeyTime()[m]) / REFTIME;
					auto value = y->getKeyValue()[m];

					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_tR_Y[objId].push_back(time);
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_R_Y[objId].push_back(value);

				}
			}

			if (z)
			{
				for (size_t m = 0; m < z->getKeyCount(); m++)
				{
					auto time = Real(z->getKeyTime()[m]) / REFTIME;
					auto value = z->getKeyValue()[m];

					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_tR_Z[objId].push_back(time);
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_R_Z[objId].push_back(value);

				}
			}

		}

		if (propertyData == "Lcl Scaling")
		{
			auto x = node->getCurve(0);
			auto y = node->getCurve(1);
			auto z = node->getCurve(2);

			if (x)
			{
				for (size_t m = 0; m < x->getKeyCount(); m++)
				{
					auto time = Real(x->getKeyTime()[m]) / REFTIME;
					auto value = x->getKeyValue()[m];

					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_tS_X[objId].push_back(time);
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_S_X[objId].push_back(value);

				}
			}
			if (y)
			{
				for (size_t m = 0; m < y->getKeyCount(); m++)
				{
					auto time = Real(y->getKeyTime()[m]) / REFTIME;
					auto value = y->getKeyValue()[m];

					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_tS_Y[objId].push_back(time);
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_S_Y[objId].push_back(value);
				}
			}

			if (z)
			{
				for (size_t m = 0; m < z->getKeyCount(); m++)
				{
					auto time = Real(z->getKeyTime()[m]) / REFTIME;
					auto value = z->getKeyValue()[m];

					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_tS_Z[objId].push_back(time);
					this->stateHierarchicalScene()->getDataPtr()->mJointAnimationData->mJoint_KeyId_S_Z[objId].push_back(value);

				}
			}

		}

	}


	template<typename TDataType>
	void FBXLoader<TDataType>::buildHierarchy(const std::map<std::string, std::string>& obj_parent, const std::map<std::string, std::shared_ptr<ModelObject>>& name_ParentObj)
	{
		//std::cout << parentTag.size()<< " -- " << name_ParentObj.size() << "\n";
		for (auto it : obj_parent)
		{
			//Parent
			std::pair<std::string, std::string> element = it;
			std::string objName = it.first;

			while (true)
			{
				if (element.second != std::string("No parent object"))
				{
					std::string parentName = element.second;

					std::shared_ptr<ModelObject> obj = name_ParentObj.find(objName)->second;
					std::shared_ptr<ModelObject> parent = nullptr;

					auto parentIter = name_ParentObj.find(parentName);
					if (parentIter != name_ParentObj.end())
						parent = parentIter->second;

					if (parent != nullptr)
					{
						obj->parent.push_back(parent);
						element = *obj_parent.find(parentName);
					}
					else
						break;
				}
				else
					break;
			}

			//Child
			auto parentIter = name_ParentObj.find(it.second);
			if (parentIter != name_ParentObj.end())
			{
				std::shared_ptr<ModelObject> parent = parentIter->second;
				parent->child.push_back(name_ParentObj.find(objName)->second);
			}
		}
	}

	template<typename TDataType>
	std::shared_ptr<Bone> FBXLoader<TDataType>::pushBone(const ofbx::Object* bone, std::map<std::string, std::string>& parentTag, std::map<std::string, std::shared_ptr<ModelObject>>& name_ParentObj, std::vector<std::shared_ptr<Bone>>& bonesInfo, float scale)
	{
		auto it = parentTag.find(std::string(bone->name));
		std::shared_ptr<Bone> temp = nullptr;
		if (it != parentTag.end()) {}
		else {

			temp = std::make_shared<Bone>();

			temp->name = bone->name;
			temp->preRotation = Vec3f(bone->getPreRotation().x, bone->getPreRotation().y, bone->getPreRotation().z);
			temp->localTranslation = Vec3f(bone->getLocalTranslation().x, bone->getLocalTranslation().y, bone->getLocalTranslation().z) * scale;
			temp->localRotation = Vec3f(bone->getLocalRotation().x, bone->getLocalRotation().y, bone->getLocalRotation().z);
			temp->localScale = Vec3f(bone->getLocalScaling().x, bone->getLocalScaling().y, bone->getLocalScaling().z);
			temp->pivot = Vec3f(bone->getRotationPivot().x, bone->getRotationPivot().y, bone->getRotationPivot().z);

			//temp->localTransform = Mat3f();
			bonesInfo.push_back(temp);

			if (bone->parent)
			{
				parentTag[std::string(bone->name)] = std::string(bone->parent->name);
				name_ParentObj[std::string(bone->name)] = bonesInfo[bonesInfo.size() - 1];
			}
			else
			{
				parentTag[std::string(bone->name)] = std::string("No parent object");
				name_ParentObj[std::string(bone->name)] = nullptr;
			}

		}
		return temp;
	}

	template<typename TDataType>
	void FBXLoader<TDataType>::updateTextureMesh(const std::vector<std::shared_ptr<MeshInfo>>& meshsInfo)
	{
		auto texMesh = this->stateTextureMesh()->getDataPtr();

		std::vector<int> mesh_VerticesNum;
		std::vector<int> mesh_NormalNum;
		std::vector<int> mesh_UvNum;
		std::vector<int> mesh_PointsNum;

		std::vector<Vec3f> texVertices;
		std::vector<Vec3f> texPoints;
		std::vector<Vec3f> texNormals;
		std::vector<Vec2f> texCoords;

		CArrayList<int> c_point2Vertice;

		for (auto it : meshsInfo)
		{
			mesh_VerticesNum.push_back(it->vertices.size());
			mesh_NormalNum.push_back(it->normals.size());
			mesh_UvNum.push_back(it->texcoords.size());
			mesh_PointsNum.push_back(it->points.size());

			texVertices.insert(texVertices.end(), it->vertices.begin(), it->vertices.end());
			texPoints.insert(texPoints.end(), it->points.begin(), it->points.end());
			texNormals.insert(texNormals.end(), it->normals.begin(), it->normals.end());
			texCoords.insert(texCoords.end(), it->texcoords.begin(), it->texcoords.end());			
		}

		texMesh->vertices().assign(texPoints);
		texMesh->normals().assign(texNormals);
		texMesh->texCoords().assign(texCoords);
		
		this->stateHierarchicalScene()->getDataPtr()->updatePoint2Vertice(mPoint2Vertice,mVertice2Point);
		
		std::vector<uint> shapeID;
		shapeID.resize(texPoints.size());

		int tempID = 0;
		int offset = 0;
		int verticeOffset = 0;

		std::vector<int> shapeId2MeshId;
		for (size_t i = 0; i < meshsInfo.size(); i++)
		{
			
			int meshFaceGroupNum = meshsInfo[i]->facegroup_triangles.size();

			for (size_t j = 0; j < meshFaceGroupNum; j++)
			{
				auto triangles = meshsInfo[i]->facegroup_triangles[j];
				auto normalIndex = meshsInfo[i]->facegroup_normalIndex[j];

				auto shape = std::make_shared<Shape>();
				for (size_t k = 0; k < triangles.size(); k++)
				{
					triangles[k][0] = triangles[k][0] + offset;
					triangles[k][1] = triangles[k][1] + offset;
					triangles[k][2] = triangles[k][2] + offset;
					shapeID[triangles[k][0]] = tempID;
					shapeID[triangles[k][1]] = tempID;
					shapeID[triangles[k][2]] = tempID;
				}

				for (size_t k = 0; k < normalIndex.size(); k++)
				{
					normalIndex[k][0] = normalIndex[k][0] + verticeOffset;
					normalIndex[k][1] = normalIndex[k][1] + verticeOffset;
					normalIndex[k][2] = normalIndex[k][2] + verticeOffset;
				}

				shape->vertexIndex.assign(triangles);
				shape->normalIndex.assign(normalIndex);
				shape->texCoordIndex.assign(normalIndex);

				tempID++;

				texMesh->materials().push_back(meshsInfo[i]->materials[j]);
				shape->material = meshsInfo[i]->materials[j];

				shapeId2MeshId.resize(texMesh->shapes().size() + 1);
				shapeId2MeshId[texMesh->shapes().size()] = meshsInfo[i]->id;
				
				texMesh->shapes().push_back(shape);

			}
			offset += mesh_PointsNum[i];
			verticeOffset += mesh_VerticesNum[i];
		}

		texMesh->shapeIds().assign(shapeID);

		initialPosition.assign(texMesh->vertices());
		initialNormal.assign(texMesh->normals());

		
		std::vector<Mat4f> worldMatrix = this->stateHierarchicalScene()->getDataPtr()->getObjectWorldMatrix();
		DArray<Mat4f> dWorldMatrix;

		dWorldMatrix.assign(worldMatrix);
		DArray<int> shapeId_MeshId;
		shapeId_MeshId.assign(shapeId2MeshId);

		this->stateHierarchicalScene()->getDataPtr()->shapeTransform(
			initialPosition,
			texMesh->vertices(),
			initialNormal,
			texMesh->normals(),
			dWorldMatrix,
			texMesh->shapeIds(),
			shapeId_MeshId
		);
		

		//update BoundingBox 



		this->stateHierarchicalScene()->getDataPtr()->updateSkinData(this->stateTextureMesh()->getDataPtr());

		//ToCenter
		if (varUseInstanceTransform()->getValue())
		{
			initialShapeCenter = texMesh->updateTexMeshBoundingBox();

			DArray<Vec3f> unCenterPosition;
			DArray<Vec3f> d_ShapeCenter;

			d_ShapeCenter.assign(initialShapeCenter);	// Used to "ToCenter"
			unCenterPosition.assign(this->stateTextureMesh()->getDataPtr()->vertices());
			CArray<Vec3f> cshapeCenter;
			cshapeCenter.assign(d_ShapeCenter);

			this->stateHierarchicalScene()->getDataPtr()->shapeToCenter(unCenterPosition,
				this->stateTextureMesh()->getDataPtr()->vertices(),
				this->stateTextureMesh()->getDataPtr()->shapeIds(),
				d_ShapeCenter
			);

		}

	}

	template<typename TDataType>
	void FBXLoader<TDataType>::setMeshToScene(const std::vector<std::shared_ptr<MeshInfo>>& meshsInfo, std::shared_ptr<HierarchicalScene> scene)
	{
		for (auto mesh : meshsInfo)
		{
			scene->mModelObjects.push_back(mesh);
			scene->mMeshes.push_back(mesh);
		}
	}

	template<typename TDataType>
	void FBXLoader<TDataType>::setBonesToScene(const std::vector< std::shared_ptr<Bone>>& bonesInfo, std::shared_ptr<HierarchicalScene> scene)
	{
		for (auto bone : bonesInfo)
		{
			scene->pushBackBone(bone);
		}
	}

	template<typename TDataType>
	void FBXLoader<TDataType>::updateHierarchicalScene(const std::vector<std::shared_ptr<MeshInfo>>& meshsInfo, const std::vector< std::shared_ptr<Bone>>& bonesInfo)
	{
		auto hierarchicalScene = this->stateHierarchicalScene()->getDataPtr();
		hierarchicalScene->clear();

		for (auto bone : bonesInfo)
		{
			hierarchicalScene->pushBackBone(bone);
		}

		for (auto mesh : meshsInfo)
		{
			hierarchicalScene->pushBackMesh(mesh);
		}

		if(bonesInfo.size())
			hierarchicalScene->UpdateJointData();
	}

	template<typename TDataType>
	void FBXLoader<TDataType>::updateTransform()
	{


		if(this->stateHierarchicalScene()->getDataPtr()->getBones().size()&&this->varImportAnimation()->getValue())
			updateAnimation(0);
		else
		{
			auto& shape = this->stateTextureMesh()->getDataPtr()->shapes();
			auto vertices = this->stateTextureMesh()->getDataPtr()->vertices();
			CArray<Vec3f> cv;
			cv.assign(vertices);

			if (varUseInstanceTransform()->getValue())
			{
				for (size_t i = 0; i < shape.size(); i++)
				{
					auto quat = this->computeQuaternion();
					shape[i]->boundingTransform.translation() = quat.rotate(initialShapeCenter[i]) * this->varScale()->getValue() + this->varLocation()->getValue();
					shape[i]->boundingTransform.rotation() = quat.toMatrix3x3();
					shape[i]->boundingTransform.scale() = this->varScale()->getValue();
				}
			}
			else
			{
				for (size_t i = 0; i < shape.size(); i++)
				{
					shape[i]->boundingTransform.translation() = Vec3f(0);
				}
			}

		}
	}

	template<typename TDataType>
	void FBXLoader<TDataType>::updateAnimation(float time)
	{
		auto targetScene = this->stateHierarchicalScene()->getDataPtr();

		Mat4f nodeTransform = getNodeTransformMatrix();

		if(targetScene->mJointAnimationData->isValid())
			targetScene->mJointAnimationData->updateAnimationPose(time);

		auto texMesh = this->stateTextureMesh()->getDataPtr();

		for (size_t i = 0; i < targetScene->mSkinData->size(); i++)//
		{
			auto& bindJoint0 = targetScene->mSkinData->V_jointID_0[i];
			auto& bindJoint1 = targetScene->mSkinData->V_jointID_1[i];
			auto& bindJoint2 = targetScene->mSkinData->V_jointID_2[i];

			auto& bindWeight0 = targetScene->mSkinData->V_jointWeight_0[i];
			auto& bindWeight1 = targetScene->mSkinData->V_jointWeight_1[i];
			auto& bindWeight2 = targetScene->mSkinData->V_jointWeight_2[i];



			targetScene->skinAnimation(
				targetScene->mSkinData->initialPosition,
				this->stateTextureMesh()->getDataPtr()->vertices(),
				targetScene->mJointData->mJointInverseBindMatrix,
				targetScene->mJointData->mJointWorldMatrix,

				bindJoint0,
				bindJoint1,
				bindJoint2,
				bindWeight0,
				bindWeight1,
				bindWeight2,

				nodeTransform,
				false,

				targetScene->mSkinData->skin_VerticeRange[i]
			);
		



			targetScene->skinVerticesAnimation(
				targetScene->mSkinData->initialNormal,
				targetScene->mSkinData->mesh->normals(),
				targetScene->mJointData->mJointInverseBindMatrix,
				targetScene->mJointData->mJointWorldMatrix,

				mPoint2Vertice,
				bindJoint0,
				bindJoint1,
				bindJoint2,
				bindWeight0,
				bindWeight1,
				bindWeight2,

				nodeTransform,
				true,

				targetScene->mSkinData->skin_VerticeRange[i]
			);
		}

		//draw Joint
		{
			std::vector<Vec3f> jointPoints;
			CArray<Mat4f> c_Matrix;
			c_Matrix.assign(targetScene->mJointData->mJointWorldMatrix);
			for (int i = 0; i < c_Matrix.size(); i++)
			{
				auto worldMartix = c_Matrix[i];
				Vec4f p4 = nodeTransform * (worldMartix * Vec4f(0, 0, 0, 1));
				jointPoints.push_back(Vec3f(p4.x, p4.y, p4.z));
			}
			std::vector<TopologyModule::Edge> jointEdges;
			auto joints = targetScene->getBones();
			for (auto j : joints)
			{
				if (j->parent.size())
					jointEdges.push_back(TopologyModule::Edge(j->id, j->parent[0]->id));
			}
			auto jointSet = this->stateJointSet()->getDataPtr();
			jointSet->setPoints(jointPoints);
			jointSet->setEdges(jointEdges);
		}

	}

	DEFINE_CLASS(FBXLoader);
}