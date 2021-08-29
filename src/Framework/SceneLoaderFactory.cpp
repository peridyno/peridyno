#include "SceneLoaderFactory.h"
#include "SceneLoaderXML.h"

namespace dyno
{
	SceneLoaderFactory& SceneLoaderFactory::getInstance()
	{
		static SceneLoaderFactory m_instance;
		return m_instance;
	}

	SceneLoader* SceneLoaderFactory::getEntryByFileExtension(std::string extension)
	{
		SceneLoaderList::iterator it = m_loaders.begin();
		while (it != m_loaders.end())
		{
			if ((*it)->canLoadFileByExtension(extension))
				return *it;
			++it;
		}
		// if not found, return 0
		return 0;
	}

	SceneLoader* SceneLoaderFactory::getEntryByFileName(std::string filename)
	{
		SceneLoaderList::iterator it = m_loaders.begin();
		while (it != m_loaders.end())
		{
			if ((*it)->canLoadFileByName(filename))
				return *it;
			++it;
		}
		// if not found, return 0
		return 0;
	}

	SceneLoader* SceneLoaderFactory::addEntry(SceneLoader *loader)
	{
		m_loaders.push_back(loader);
		return loader;
	}

	SceneLoaderFactory::SceneLoaderFactory()
	{
		SceneLoaderXML* xmlLoder = new SceneLoaderXML();
		this->addEntry(xmlLoder);
	}

}