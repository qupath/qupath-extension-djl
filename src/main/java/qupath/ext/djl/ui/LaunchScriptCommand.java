/*-
 * Copyright 2023 QuPath developers, University of Edinburgh
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

package qupath.ext.djl.ui;

import ai.djl.engine.Engine;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ContentDisplay;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.layout.VBox;
import javafx.scene.text.TextAlignment;
import javafx.stage.FileChooser;
import org.controlsfx.control.PropertySheet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.fx.dialogs.Dialogs;
import qupath.fx.dialogs.FileChoosers;
import qupath.fx.prefs.annotations.DirectoryPref;
import qupath.fx.prefs.annotations.FilePref;
import qupath.fx.prefs.annotations.Pref;
import qupath.fx.prefs.annotations.PrefCategory;
import qupath.fx.prefs.controlsfx.PropertyItemParser;
import qupath.fx.prefs.controlsfx.PropertySheetBuilder;
import qupath.lib.common.GeneralTools;
import qupath.lib.gui.localization.QuPathResources;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;

/**
 * Command to generate a launch script for QuPath.
 * <p>
 * This is primarily intended to help with managing CUDA/cuDNN using a conda environment.
 * The script helps with setting environment variables and system properties so that the
 * expected versions of CUDA/cuDNN are used.
 * <p>
 * It also makes it possible to set some Deep Java Library properties, e.g. to change the
 * PyTorch version that is used.
 */
public class LaunchScriptCommand {

    private static final Logger logger = LoggerFactory.getLogger(LaunchScriptCommand.class);

    private static final String BUNDLE_KEY = "qupath.ext.djl.ui.strings";

    public static final String KEY_CONDA_PATH = "CONDA_PATH";
    public static final String KEY_QUPATH_EXE = "QUPATH_EXE";

    public static final String KEY_PYTORCH_VERSION = "PYTORCH_VERSION";
    public static final String KEY_PYTORCH_LIBRARY_PATH = "PYTORCH_LIBRARY_PATH";
    public static final String KEY_PYTORCH_FLAVOR = "PYTORCH_FLAVOR";

    private final DjlLaunchParamsFx launchParams = new DjlLaunchParamsFx();

    public LaunchScriptCommand() {

    }

    public void promptForScript() {

        // Important to use QuPathResources.getLocalizedResourceManager() for this
        // to work when jar is installed as an extension
        var sheet = new PropertySheetBuilder()
                .parser(new PropertyItemParser().setResourceManager(
                        QuPathResources.getLocalizedResourceManager()))
                .addAnnotatedProperties(launchParams)
                .build();
        sheet.setSearchBoxVisible(false);
        sheet.setModeSwitcherVisible(false);
        sheet.setMode(PropertySheet.Mode.NAME);

        var pane = new VBox();
        pane.getChildren().setAll(
                createInstructionLabel("script.label1"),
                createInstructionLabel("script.label2"),
                new Separator(),
                sheet
        );
        pane.setSpacing(5);
        pane.setPadding(new Insets(5.0));

        if (!ButtonType.OK.equals(Dialogs.builder()
                .content(pane)
                .prefWidth(400)
                .title(ResourceBundle.getBundle(BUNDLE_KEY).getString("script.title"))
                .buttons(ButtonType.OK, ButtonType.CANCEL)
                .showAndWait()
                .orElse(ButtonType.CANCEL))) {
            return;
        }
        var params = new LinkedHashMap<String, String>();

        params.put(KEY_QUPATH_EXE, launchParams.qupathExecutable.get());
        params.put(KEY_CONDA_PATH, launchParams.condaPath.get());
        params.put(KEY_QUPATH_EXE, launchParams.qupathExecutable.get());
        var pytorchVersion = launchParams.pytorchVersion.get();
        if (!"Default".equals(pytorchVersion))
            params.put(KEY_PYTORCH_VERSION, pytorchVersion);
        params.put(KEY_PYTORCH_LIBRARY_PATH, launchParams.pytorchLibraryPath.get());
//        params.put(KEY_PYTORCH_FLAVOR, fxParams.pytorchFlavor.get());

        try {
            var script = createLaunchScript(params);
            logger.info("Script: \n" + script);
            FileChooser.ExtensionFilter filter;
            if (GeneralTools.isWindows())
                filter = FileChoosers.createExtensionFilter("Batch file (*.bat)", "*.bat");
            else
                filter = FileChoosers.createExtensionFilter("Shell script (*.sh)", "*.sh");
            var file = FileChoosers.promptToSaveFile(filter);
            if (file != null) {
                logger.info("Writing launch script to {}", file);
                Files.writeString(file.toPath(), script);
            } else
                logger.debug("Launch script not saved - dialog cancelled");
        } catch (Exception e) {
            Dialogs.showErrorMessage("Deep Java Library", "Error creating script: " + e.getLocalizedMessage());
            logger.error(e.getMessage(), e);
        }
    }

    private static Label createInstructionLabel(String key) {
        var label = new Label();
        label.setMaxSize(Double.MAX_VALUE, Double.MAX_VALUE);
        label.setAlignment(Pos.CENTER);
        label.setTextAlignment(TextAlignment.CENTER);
        label.setContentDisplay(ContentDisplay.CENTER);
        label.setText(ResourceBundle.getBundle(BUNDLE_KEY).getString(key));
        return label;
    }

    @PrefCategory(bundle=BUNDLE_KEY, value="script.title")
    private static class DjlLaunchParamsFx {

        @DirectoryPref(bundle=BUNDLE_KEY, value="script.conda.path")
        private final StringProperty condaPath = new SimpleStringProperty(null);

        @Pref(bundle=BUNDLE_KEY, value="script.pytorch.version", type=String.class, choiceMethod = "getPyTorchVersions")
        private final StringProperty pytorchVersion = new SimpleStringProperty(null);

        @DirectoryPref(bundle=BUNDLE_KEY, value="script.pytorch.path")
        private final StringProperty pytorchLibraryPath = new SimpleStringProperty("");

        @FilePref(bundle=BUNDLE_KEY, value="script.qupath.exe")
        private final StringProperty qupathExecutable = new SimpleStringProperty("");

        private DjlLaunchParamsFx() {
            try {
                var pytorchVersions = getPyTorchVersions();
                if (!pytorchVersions.isEmpty())
                    pytorchVersion.set(pytorchVersions.get(0));
            } catch (Exception e) {
                logger.debug("Unknown DJL version: " + Engine.getDjlVersion());
            }
            try {
                var executablePath = findQuPathExecutable();
                if (executablePath != null && !executablePath.isEmpty())
                    qupathExecutable.set(executablePath);
            } catch (Exception e) {
                logger.debug("Unable to find QuPath executable");
            }
        }

        public List<String> getPyTorchVersions() {
            var version = Engine.getDjlVersion();
            switch (version) {
                case "0.23.0":
                case "0.24.0":
                case "0.25.0":
                    return Arrays.asList("Default", "2.0.1", "1.13.1", "1.12.1");
                case "0.26.0":
                    return Arrays.asList("Default", "2.1.1", "2.0.1", "1.13.1");
                default:
                    // Exception here should be caught by the caller & a text field used as a prompt
                    throw new RuntimeException("Unknown DJL version: " + version);
            }
        }

    }

    /**
     * Generate a .bat or .sh script to launch QuPath with the given parameters.
     * @param params
     * @return
     */
    public static String createLaunchScript(Map<String, String> params) {
        // Ensure we have a mutable map
        params = new LinkedHashMap<>(params);

        // Check we can get a QuPath executable
        String qupathExecutable = params.remove(KEY_QUPATH_EXE);
        if (qupathExecutable == null)
            qupathExecutable = findQuPathExecutable();
        if (qupathExecutable == null)
            throw new IllegalArgumentException("No QuPath executable found!");

        // Set path variable from conda, if possible
        String condaPath = params.remove(KEY_CONDA_PATH);
        String pathVariable = null;
        String cudnnPath = null;
        if (condaPath != null && !condaPath.isEmpty()) {
            List<String> paths = new ArrayList();
            paths.add(condaPath);
            paths.add(condaPath + File.separator + "bin");
            paths.add(condaPath + File.separator + "lib");
            paths.add(condaPath + File.separator + "lib" + File.separator + "site-packages" + File.separator + "torch" + File.separator + "lib");
            var dirCudnn = findCuDnnDir(new File(condaPath));
            if (dirCudnn != null)
                cudnnPath = dirCudnn.getAbsolutePath();
            pathVariable = String.join(File.pathSeparator, paths);
        }

        StringBuilder sb = new StringBuilder();
        if (!GeneralTools.isWindows()) {
            sb.append("#!/usr/bin/env bash")
                    .append(System.lineSeparator())
                    .append(System.lineSeparator());
        }

        for (var entry : params.entrySet()) {
            var key = entry.getKey();
            var val = entry.getValue();
            if (val != null && !val.isEmpty()) {
                appendToEnvironment(sb, key, val);
            }
        }

        // On Windows, setting the PATH should be enough for everything
        if (pathVariable != null && !pathVariable.isEmpty()) {
            if (GeneralTools.isWindows()) {
                if (!params.containsKey("PATH"))
                    appendToEnvironment(sb, "PATH",
                            pathVariable + File.pathSeparator + "%PATH%");
            }
        }

        // On Linux we need LD_LIBRARY_PATH to find cudnn only (the rest is up to JNA)
        if (GeneralTools.isLinux() && !params.containsKey("LD_LIBRARY_PATH") && cudnnPath != null) {
            appendToEnvironment(sb, "LD_LIBRARY_PATH",
                    cudnnPath + File.pathSeparator + "$LD_LIBRARY_PATH");
        }

        // Command to launch QuPath itself
        if (!sb.toString().endsWith(System.lineSeparator() + System.lineSeparator()))
            sb.append(System.lineSeparator());
        sb.append(quoteIfNeeded(qupathExecutable));

        // On Linux, we need to set the JNA path to find CUDA
        if (pathVariable != null && !pathVariable.isEmpty() && !GeneralTools.isWindows()) {
            String jnaPath = System.getProperty("jna.library.path");
            if (jnaPath == null || jnaPath.isEmpty())
                jnaPath = pathVariable;
            else
                jnaPath = pathVariable + File.pathSeparator + jnaPath;
            sb.append(" -Djna.library.path=").append(quoteIfNeeded(jnaPath));
        }

        sb.append(System.lineSeparator());

        return sb.toString();
    }

    private static File findCuDnnDir(File dir) {
        if (!dir.exists())
            return null;
        String name = System.mapLibraryName("cudnn");
        try {
            var pathCuDnn = Files.walk(dir.toPath())
                    .filter(p -> p.getFileName().toString().startsWith(name))
                    .findFirst()
                    .orElse(null);
            logger.info("Searching for {}, found {}", name, pathCuDnn);
            return pathCuDnn == null ? null : pathCuDnn.toFile().getParentFile();
        } catch (IOException e) {
            logger.warn("Error searching for cudnn: {}", e.getMessage(), e);
            return null;
        }
    }


    private static void appendToEnvironment(StringBuilder sb, String key, String val) {
        if (val != null && !val.isEmpty()) {
            if (GeneralTools.isWindows()) {
                sb.append("set ").append(key).append("=");
            } else {
                sb.append("export ").append(key).append("=");
            }
            sb.append(quoteIfNeeded(val));
            sb.append(System.lineSeparator());
        }
    }

    /**
     * Very simple (not very robust) handling of spaces within environment variables
     * or system properties.
     * Expected to occur in some file paths, but hopefully not often.
     * @param val
     * @return
     */
    private static String quoteIfNeeded(String val) {
        if (val == null || val.isEmpty())
            return val;
        if (val.contains(" ") && !val.startsWith("\"") && !val.endsWith("\""))
            return "\"" + val + "\"";
        else
            return val;
    }



    /**
     * Try to find the QuPath executable from the running instance.
     * This is assumed to be a final build, i.e. we aren't running from an IDE or gradle.
     */
    static String findQuPathExecutable() {
        var file = new File(qupath.lib.gui.QuPathGUI.class.getProtectionDomain()
                .getCodeSource()
                .getLocation()
                .getFile());
        if (file.isDirectory())
            throw new UnsupportedOperationException("QuPath must be packaged to find executable");
        var dir = file.getParentFile();
        if (dir.getName().equals("app"))
            dir = dir.getParentFile();

        File[] executables;
        if (GeneralTools.isWindows()) {
            executables = dir.listFiles(f -> f.isFile() && f.getName().toLowerCase().endsWith(".exe"));
        } else {
            if (new File(dir.getParentFile(), "bin").exists()) {
                dir = new File(dir.getParentFile(), "bin");
            } else if (GeneralTools.isMac() && new File(dir, "MacOS").exists()) {
                dir = new File(dir, "MacOS");
            }
            executables = dir.listFiles(f -> f.isFile() &&
                    f.getName().toLowerCase().startsWith("qupath") &&
                    Files.isExecutable(f.toPath()));
        }
        if (executables.length == 0)
            throw new IllegalArgumentException("No QuPath executable found!");
        else if (executables.length > 1) {
            // Sort to have the longest name first.
            // On Windows, this should be the 'console' exe, which is better for debugging.
            Arrays.sort(executables, Comparator.comparingInt((File f) -> f.getAbsolutePath().length()).reversed());
        }
        return executables[0].getAbsolutePath();
    }

}
