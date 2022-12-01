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

import java.nio.file.Files;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.util.Utils;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableMap;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.scene.Scene;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ContentDisplay;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.control.Tooltip;
import javafx.scene.effect.DropShadow;
import javafx.scene.layout.GridPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.stage.Modality;
import javafx.stage.Stage;
import qupath.lib.common.GeneralTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.gui.tools.GuiTools;
import qupath.lib.gui.tools.PaneTools;

/**
 * Command to display Deep Java Library engines.
 * 
 * @author Pete Bankhead
 */
public class DjlEngineCommand {
	
	private static final Logger logger = LoggerFactory.getLogger(DjlEngineCommand.class);
	
	private static final String TITLE = "Deep Java Library engines";
	
	private QuPathGUI qupath;
	
	private static enum EngineStatus {
		UNAVAILABLE, UNKNOWN, PENDING, AVAILABLE, FAILED
	}
	
	private static DjlEngineCommand INSTANCE;
	
	private Stage stage;
	
	/**
	 * Engine only supported on Linux
	 */
	private static Set<String> LINUX_ONLY = Set.of(
			DjlTools.ENGINE_TENSORRT
			);
	
	/**
	 * Several engines definitely don't work on Apple Silicon.
	 */
	private static Set<String> UNSUPPORTED_APPLE_SILICON = Set.of(
			DjlTools.ENGINE_PADDLEPADDLE,
			DjlTools.ENGINE_TENSORFLOW,
			DjlTools.ENGINE_LIGHTGBM			
			);
	
	/**
	 * Maintain a map to indicate if the engine is available or not
	 */
	private ObservableMap<String, ObjectProperty<EngineStatus>> available = FXCollections.observableHashMap();
	
	DjlEngineCommand(QuPathGUI qupath) {
		this.qupath = qupath;
		init();
	}
	
	private void init() {
		var pane = new GridPane();
		pane.setHgap(5);
		pane.setVgap(5);
		pane.setPadding(new Insets(10));
		int row = 0;
		
		var label = new Label(
				"Download Deep Java Library engines for inference."
				);
		label.setWrapText(true);
		PaneTools.addGridRow(pane, row++, 0, null, label, label);
		var label2 = new Label(
				"These will be stored in the local directories shown."
				);
		label2.setWrapText(true);
		PaneTools.addGridRow(pane, row++, 0, null, label2, label2);

		var label3 = new Label(
				"Please check the original engine's website for any licensing information.\n"
				);
		label3.setWrapText(true);
		PaneTools.addGridRow(pane, row++, 0, null, label3, label3);

		for (var name : Engine.getAllEngines()) {
			var separator = new Separator(Orientation.HORIZONTAL);
			PaneTools.addGridRow(pane, row++, 0, null, separator, separator);
			PaneTools.setToExpandGridPaneWidth(separator);
			
			var status = available.computeIfAbsent(name, n -> new SimpleObjectProperty<>(EngineStatus.UNKNOWN));
			
			var path = Utils.getEngineCacheDir(name);
			
			String tooltip = "Engine: " + name;
			var labelName = new Label(name);
			labelName.setStyle("-fx-font-weight: bold;");
			PaneTools.addGridRow(pane, row++, 0, tooltip, labelName, labelName);
			
			var labelPath = new Label(path.toString());
			var btnDownload = new Button("Download");
			btnDownload.setPrefWidth(60.0);
			long timeoutMillis = 500L;
			btnDownload.setOnAction(e -> checkEngineStatus(name, status, timeoutMillis, false));
			btnDownload.disableProperty().bind(Bindings.createBooleanBinding(() -> 
				status.get() == EngineStatus.AVAILABLE || status.get() == EngineStatus.PENDING,
				status));
			btnDownload.textProperty().bind(Bindings.createStringBinding(() -> {
				switch (status.get()) {
				case AVAILABLE:
					return "Available";
				case FAILED:
					return "Try again";
				case PENDING:
					return "Pending";
				case UNKNOWN:
					return "Check / Download";
				case UNAVAILABLE:
					return "Download";
				default:
					return name + " download status is unknown";				
				}
			}, status));
			
			var tooltipDownload = new Tooltip();
			tooltipDownload.textProperty().bind(Bindings.createStringBinding(() -> {
				switch (status.get()) {
				case AVAILABLE:
					return name + " is available";
				case UNAVAILABLE:
					return name + " is not available - click to download it";
				case FAILED:
					return name + " download & initialization failed - you can try again, but it may not be supported on this platform";
				case PENDING:
					return name + " download pending";
				case UNKNOWN:
				default:
					return name + " download status is unknown";				
				}
			}, status));
			btnDownload.setTooltip(tooltipDownload);
			
			var circle = new Circle(6.0);
			circle.setEffect(new DropShadow(2, 1, 1, Color.grayRgb(128, 0.5)));
			circle.styleProperty().bind(Bindings.createStringBinding(() -> {
				switch (status.get()) {
				case AVAILABLE:
					return "-fx-fill: lightgreen; -fx-opacity: 0.8";
				case UNAVAILABLE:
					return "-fx-fill: -fx-text-base-color; -fx-opacity: 0.5;";
				case FAILED:
					return "-fx-fill: darkred; -fx-opacity: 0.8";
				case PENDING:
					return "-fx-fill: yellow; -fx-opacity: 0.8";
				case UNKNOWN:
				default:
					return "-fx-fill: orange; -fx-opacity: 0.8";
				}
			}, status));
			var labelPathLabel = new Label("Path: ");
			labelPathLabel.setGraphic(circle);
			labelPathLabel.setLabelFor(labelPath);
			labelPathLabel.setContentDisplay(ContentDisplay.LEFT);

			var tip = new Tooltip();
			status.addListener((v, o, n) -> {
				if (Files.isDirectory(path)) {
					tip.setText("Double-click to open engine path");
				} else {
					labelPath.setStyle("-fx-opacity: 0.5;");
					tip.setText("Engine path does not exist");	
				}				
			});
			labelPathLabel.setTooltip(tip);
			labelPath.setOnMouseClicked(e -> {
				if (e.getClickCount() == 2 && Files.isDirectory(path))
					GuiTools.browseDirectory(path.toFile());
				else
					logger.debug("Cannot open {} - directory does not exist", path);
			});
			PaneTools.addGridRow(pane, row++, 0, null, labelPathLabel, labelPath);
			PaneTools.addGridRow(pane, row++, 0, null, btnDownload, btnDownload);
			
			PaneTools.setToExpandGridPaneWidth(labelName, labelPath, btnDownload);

			// Update the engine status quietly
			checkEngineStatus(name, status, -1, true);
		}
		
		stage = new Stage();
		stage.setTitle(TITLE);
		stage.initOwner(QuPathGUI.getInstance().getStage());
		stage.initModality(Modality.APPLICATION_MODAL);
		stage.setScene(new Scene(pane));
	}
	
	private void showDialog() {
		if (stage.isShowing())
			stage.toFront();
		else
			stage.show();
	}
	
	private void checkEngineStatus(String name, ObjectProperty<EngineStatus> status, long timeoutMillis, boolean doQuietly) {
		// Request the engine in a background thread, triggering download if necessary
		var pool = qupath.createSingleThreadExecutor(this);
		updateStatus(status, EngineStatus.PENDING);
		var future = pool.submit((Callable<Boolean>)() -> checkEngineAvailability(name, status, doQuietly));
		if (timeoutMillis <= 0)
			return;
		
		try {
			// Wait until the timeout - engine might already be available & return quickly
			var result = future.get(timeoutMillis, TimeUnit.MILLISECONDS);
			if (result != null && result.booleanValue()) {
				Dialogs.showInfoNotification(TITLE, name + " is available!");
				return;
			} else
				logger.debug("No engine available for {}", name);
		} catch (InterruptedException e) {
			logger.error("Requesting for engine " + name + "interrupted!", e);
		} catch (ExecutionException e) {
			logger.error("Error requesting engine " + name + ": " + e.getLocalizedMessage(), e);
		} catch (TimeoutException e) {
			logger.debug("Request for engine {} timed out - likely either initializing or downloading", name);
		}
	}
	
	private void updateStatus(ObjectProperty<EngineStatus> status, EngineStatus newStatus) {
		if (Platform.isFxApplicationThread()) {
			status.set(newStatus);
		} else {
			Platform.runLater(() -> updateStatus(status, newStatus));
		}
	}
	
	
	private Boolean checkEngineAvailability(String name, ObjectProperty<EngineStatus> status, boolean doQuietly) {

		if (!GeneralTools.isLinux() && LINUX_ONLY.contains(name)) {
			if (!doQuietly)
				Dialogs.showErrorMessage(TITLE, name + " is only available on Linux, sorry");
			updateStatus(status, EngineStatus.UNAVAILABLE);
			return Boolean.FALSE;
		}

		// For Apple Silicon, we may not know if we are running under Rosetta or not
		// TODO: Check for custom aarch64 build
		if (!doQuietly && GeneralTools.isMac() && UNSUPPORTED_APPLE_SILICON.contains(name)) {
			var button = Dialogs.builder()
				.title(TITLE)
				.alertType(AlertType.WARNING)
				.contentText(name + " won't work on recent Macs with Apple Silicon.\n"
						+ "Do you want to continue anyway?")
				.buttons(ButtonType.YES, ButtonType.NO)
				.showAndWait()
				.orElse(ButtonType.NO);
			if (ButtonType.NO.equals(button)) {
				updateStatus(status, EngineStatus.UNAVAILABLE);
				return Boolean.FALSE;
			}
		}
		
		try {
			// Allow engine downloads if not doQuietly
			updateStatus(status, EngineStatus.PENDING);
			boolean isAvailable;
			if (doQuietly)
				isAvailable = DjlTools.isEngineAvailable(name);
			else
				isAvailable = DjlTools.getEngine(name, true) != null;
			if (isAvailable) {
				updateStatus(status, EngineStatus.AVAILABLE);
				if (!doQuietly)
					Dialogs.showInfoNotification(TITLE, name + " is now available!");
			} else {
				updateStatus(status, EngineStatus.UNAVAILABLE);
				if (!doQuietly) {
					Dialogs.showWarningNotification(TITLE, "Unable to initialize " + name + ", sorry");
				}
			}
			return isAvailable;
		} catch (EngineException e) {
			if (doQuietly) {
				logger.debug("Error requesting engine: " + e.getLocalizedMessage(), e);				
			} else {
				logger.error("Error requesting engine: " + e.getLocalizedMessage(), e);
				Dialogs.showErrorMessage(TITLE, "Unable to initialize " + name + "\n- engine might not be supported on this platform, sorry");
			}
		} catch (Exception e) {
			if (doQuietly) {
				logger.debug("Error requesting engine: " + e.getLocalizedMessage(), e);				
			} else {
				logger.error("Error requesting engine: " + e.getLocalizedMessage(), e);
				Dialogs.showErrorMessage(TITLE, "Exception when trying to download " + name + "\n" + e.getLocalizedMessage());
			}
		}
		updateStatus(status, EngineStatus.FAILED);
		return Boolean.FALSE;
	}
	
	/**
	 * Show dialog.
	 * @param qupath
	 */
	public static void showDialog(QuPathGUI qupath) {
		if (INSTANCE == null)
			INSTANCE = new DjlEngineCommand(qupath);
		INSTANCE.showDialog();
	}
	

}
