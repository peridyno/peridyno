#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "Node.h"
#include "Module/ComputeModule.h"
#include "Field/FilePath.h"
#include "Topology/MaterialManager.h"

namespace dyno {

    class ImageLoader 
    {
    public:

        ImageLoader() {}
        ~ImageLoader() {}
        static bool loadImage(const char* path, CArray2D<Vec4f>& img, int req_comp = STBI_default);
    };


	class ImageLoaderModule : public MaterialManagedModule
	{
		DECLARE_CLASS(ImageLoaderModule)
		MATERIAL_MANAGER_MANAGED_CLASS

	public:

		ImageLoaderModule();
		ImageLoaderModule(std::shared_ptr<ImageLoaderModule> other)
		{
			if (this->outImage()->isEmpty())
				this->outImage()->allocate();

			mFilePath = other->mFilePath;
			this->varImagePath()->setValue(other->varImagePath()->getValue());

			if(!other->outImage()->isEmpty())
				this->outImage()->assign(other->outImage()->getData());
			
			auto IndexChange = std::make_shared<FCallBackFunc>(std::bind(&ImageLoaderModule::onVarChanged, this));
			this->setName(std::string("ImageLoader"));
		}
		~ImageLoaderModule() {};

		void onVarChanged();
		void updateImpl()override { onVarChanged(); };
		std::string caption() override { return "ImageLoader"; }

		DEF_VAR(FilePath, ImagePath, FilePath(), "");
		DEF_ARRAY2D_OUT(Vec4f, Image, DeviceType::GPU, "");

	protected:
		virtual std::shared_ptr<MaterialManagedModule> clone() const override 
		{
			std::shared_ptr<MaterialManagedModule> materialPtr = MaterialManager::getMaterialManagedModule(this->getName());
			if (!materialPtr)
			{
				printf("Error: ImageLoaderModule::clone() Failed!! \n");
				return nullptr;
			}

			std::shared_ptr<ImageLoaderModule> imageLoader = std::dynamic_pointer_cast<ImageLoaderModule>(materialPtr);
			if (!imageLoader)
			{
				printf("Error: ImageLoaderModule::clone() Cast Failed!!  \n");
				return nullptr;
			}

			std::shared_ptr<ImageLoaderModule> newImageLoader(new ImageLoaderModule(imageLoader));

			return newImageLoader;
		}

	private:
		std::string mFilePath;

	};

} 

#endif // IMAGE_LOADER_H