#include "WtApp.h"
#include "WMainApp.h"

#include <Wt/WServer.h>
#include <Wt/WLogger.h>


namespace dyno {
	WtApp::WtApp(int argc /*= 0*/, char** argv /*= NULL*/)
	{
		argc_ = argc;
		argv_ = argv;
	}

	dyno::WtApp::~WtApp()
	{

	}

	void WtApp::mainLoop()
	{
		auto createApp = [&](const Wt::WEnvironment& env)->std::unique_ptr<Wt::WApplication> {
			auto app = std::make_unique<WMainApp>(env);
			return app;
		};

		try {
			if (argc_ == 1)
			{
				std::string doc_root = getAssetPath() + "docroot";
				replace(doc_root.begin(), doc_root.end(), '/', '\\');

				// default args
				std::vector<std::string> args;
				args.push_back("--http-listen");
				args.push_back("0.0.0.0:5000");
				args.push_back("--docroot");
				args.push_back(doc_root);
				args.push_back("--config");
				args.push_back(doc_root + "\\wt_config.xml");
				Wt::log("warning") << doc_root + "\\wt_config.xml";
				Wt::WRun("", args, createApp);
			}
			else
			{
				Wt::WServer server(argc_, argv_);
				server.addEntryPoint(Wt::EntryPointType::Application, createApp);
				server.run();
			}

		}
		catch (Wt::WServer::Exception& e) {
			std::cerr << e.what() << "\n";
			return;
		}
		catch (std::exception& e) {
			std::cerr << "exception: " << e.what() << "\n";
			return;
		}
	}
}
