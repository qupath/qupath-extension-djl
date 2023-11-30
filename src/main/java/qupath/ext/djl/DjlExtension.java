/*-
 * Copyright 2022 QuPath developers, University of Edinburgh
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.djl;

import org.controlsfx.control.action.Action;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.ext.djl.ui.DjlEngineCommand;
import qupath.ext.djl.ui.LaunchScriptCommand;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.tools.MenuTools;
import qupath.opencv.dnn.DnnModelBuilder;
import qupath.opencv.dnn.DnnModels;

/**
 * Experimental extension to connect QuPath and Deep Java Library.
 * 
 * @author Pete Bankhead
 */
public class DjlExtension implements QuPathExtension, GitHubProject {
	
	private final static Logger logger = LoggerFactory.getLogger(DjlExtension.class);
	
	private final static DnnModelBuilder builder = new DjlDnnModelBuilder();
	
	static {
		// Prevent downloading engines automatically
		System.setProperty("offline", "true");
		// Opt out of tracking, see https://github.com/deepjavalibrary/djl/pull/2178/files
		System.setProperty("OPT_OUT_TRACKING", "true");
	}

	@Override
	public void installExtension(QuPathGUI qupath) {
		logger.debug("Installing Deep Java Library extension");
		logger.info("Registering DjlDnnModel");
		DnnModels.registerDnnModel(DjlDnnModel.class, DjlDnnModel.class.getSimpleName());
		// Use this instead of a ServiceLoader for now, because we can't rely upon 
		// the context class loader finding the builder
		DnnModels.registerBuilder(builder);
		var menu = qupath.getMenu("Extensions>Deep Java Library", true);
		MenuTools.addMenuItems(
				menu,
                new Action("Manage DJL engines", e -> DjlEngineCommand.showDialog(qupath))
        );
		var cmd = new LaunchScriptCommand();
		var action = new Action("Create launch script", e -> cmd.promptForScript());
		MenuTools.addMenuItems(menu, action);
	}

	@Override
	public String getName() {
		return "Deep Java Library extension";
	}

	@Override
	public String getDescription() {
		return "Add Deep Java Library support to QuPath";
	}

	@Override
	public GitHubRepo getRepository() {
		return GitHubRepo.create(getName(), "qupath", "qupath-extension-djl");
	}
	
	@Override
	public Version getQuPathVersion() {
		return Version.parse("0.5.0-SNAPSHOT");
	}
	
	
}
